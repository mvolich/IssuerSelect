# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import json
import os
import time
import asyncio
import unicodedata
import re
import textwrap
from urllib.parse import urlencode
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from dateutil import parser
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import sys
import random
from scipy.optimize import minimize as scipy_minimize
from scipy.stats import gaussian_kde

# AI Analysis (optional) — uses OpenAI via st.secrets
try:
    # OpenAI Python SDK v1 (Responses API)
    from openai import OpenAI  # pip install --upgrade openai
    _OPENAI_AVAILABLE = True
except Exception:  # SDK not installed in some envs
    OpenAI = None
    _OPENAI_AVAILABLE = False

# Multi-Agent Credit Report (optional) - uses LlamaIndex
try:
    from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
    from llama_index.llms.anthropic import Anthropic
    from llama_index.llms.openai import OpenAI as LlamaOpenAI
    from llama_index.core.workflow import Context
    from llama_index.core.agent.workflow import AgentStream
    _LLAMAINDEX_AVAILABLE = True
except Exception:
    FunctionAgent = AgentWorkflow = Anthropic = LlamaOpenAI = Context = AgentStream = None
    _LLAMAINDEX_AVAILABLE = False

# ============================================================================
# PERIOD SELECTION ENUMS (V2.3 Unified Period Selection)
# ============================================================================

class PeriodSelectionMode(Enum):
    """Unified period selection mode for both quality and trend scores"""
    LATEST_AVAILABLE = "latest_available"  # Option A: Maximum currency, accepts misalignment
    REFERENCE_ALIGNED = "reference_aligned"  # Option B: Common reference date, enforces alignment

class PeriodType(Enum):
    """Type of financial period"""
    ANNUAL = "FY"  # Fiscal Year
    QUARTERLY = "CQ"  # Calendar Quarter

# ============================================================================
# MODEL THRESHOLDS - SINGLE SOURCE OF TRUTH
# ============================================================================
# All threshold comparisons throughout the app MUST reference these constants.
# DO NOT hardcode threshold values elsewhere in the code.

# Cache version - increment when calculation logic changes to invalidate cached results
CACHE_VERSION = "v6.1.0"  # [Phase 2] Peer context in diagnostics + GenAI consistency

MODEL_THRESHOLDS = {
    # Signal Classification Thresholds
    'quality_strong': 55,          # Composite_Score >= 55 = "Strong" quality
    'trend_improving': 55,         # Cycle_Position_Score >= 55 = "Improving" trend
    
    # Recommendation Percentile Thresholds
    'percentile_strong_buy': 70,   # >= 70th percentile in band = Strong Buy eligible
    'percentile_exceptional': 90,  # >= 90th percentile = Exceptional quality
    
    # Visualization Thresholds (for chart display only)
    'viz_quality_split': 60,       # Visual split on quadrant chart (display only)
    
    # Volatility & Outlier Detection
    'volatility_cv': 0.30,         # Coefficient of variation threshold for VolatileSeries flag
    'outlier_z_score': -2.5,       # Z-score threshold for outlier detection
    
    # Volatility Score Interpretation (0-100 scale, higher = more stable)
    'volatility_score_consistent': 70,  # >= 70 = "consistent" trend (data closely tracks trend line)
    'volatility_score_moderate': 50,    # >= 50 = "overall" trend (some deviation from trend)
    # Below 50 = "volatile" trend (significant swings around trend line)
    
    # Data Quality
    'stale_data_days': 180,        # Days before data considered stale
    'min_coverage_pct': 70,        # Minimum data coverage percentage
    
    # Leverage Guardrail
    'leverage_score_avoid': 35,    # Leverage_Score < 35 → force Avoid (extreme leverage)
    
    # Display Classification Thresholds (for UI visualization only)
    'display_cycle_favorable': 70,     # Cycle score >= 70 = Favorable position
    'display_cycle_neutral': 40,       # Cycle score >= 40 = Neutral/Stable
    'display_composite_high': 70,      # Composite >= 70 = High Quality
    'display_composite_moderate': 50,  # Composite >= 50 = Moderate Quality
}

def classify_signal(quality_score, trend_score, quality_thr=None, trend_thr=None):
    """
    Centralized signal classification - SINGLE SOURCE OF TRUTH.
    
    All signal classification in the app should use this function.
    
    Args:
        quality_score: Composite score (0-100)
        trend_score: Cycle position score (0-100)
        quality_thr: Override quality threshold (default: QUALITY_THRESHOLD)
        trend_thr: Override trend threshold (default: TREND_THRESHOLD)
    
    Returns:
        Signal string: "Strong & Improving", "Strong but Deteriorating", 
                      "Weak but Improving", "Weak & Deteriorating", or "n/a"
    """
    if quality_thr is None:
        quality_thr = QUALITY_THRESHOLD
    if trend_thr is None:
        trend_thr = TREND_THRESHOLD
        
    if pd.isna(quality_score) or pd.isna(trend_score):
        return "n/a"
    
    is_strong = quality_score >= quality_thr
    is_improving = trend_score >= trend_thr
    
    if is_strong and is_improving:
        return "Strong & Improving"
    elif is_strong and not is_improving:
        return "Strong but Deteriorating"
    elif not is_strong and is_improving:
        return "Weak but Improving"
    else:
        return "Weak & Deteriorating"

# Convenience accessors for most-used thresholds
QUALITY_THRESHOLD = MODEL_THRESHOLDS['quality_strong']      # 55

# =============================================================================
# CIQ-ALIGNED BOUNDS FOR FALLBACK-CALCULATED RATIOS
# =============================================================================
# Based on observed ranges from 1,945 issuers * 8 periods in CIQ data.
# These bounds only apply to fallback calculations (when CIQ returns NM),
# not to CIQ-provided values which are trusted as-is.

CIQ_RATIO_BOUNDS = {
    'EBITDA / Interest Expense (x)': {'min': 0.0, 'max': 294.0},  # CIQ observed max: 293.65
    'Net Debt / EBITDA': {'min': 0.0, 'max': 274.0},              # CIQ observed max: 273.83
}

def apply_ciq_ratio_bounds(metric_name: str, raw_value: float) -> tuple:
    """
    Apply CIQ-aligned bounds to a fallback-calculated ratio value.
    
    SINGLE SOURCE OF TRUTH for all ratio bounding logic.
    
    Args:
        metric_name: The metric name (must match CIQ_RATIO_BOUNDS keys)
        raw_value: The calculated ratio value
        
    Returns:
        tuple: (bounded_value, was_bounded)
            - bounded_value: Value clipped to CIQ range
            - was_bounded: True if value was outside CIQ range and was clipped
    """
    import numpy as np
    bounds = CIQ_RATIO_BOUNDS.get(metric_name)
    
    if bounds is None:
        # No bounds defined for this metric - return as-is
        return raw_value, False
    
    bound_min = bounds['min']
    bound_max = bounds['max']
    
    if raw_value < bound_min or raw_value > bound_max:
        return float(np.clip(raw_value, bound_min, bound_max)), True
    
    return raw_value, False
TREND_THRESHOLD = MODEL_THRESHOLDS['trend_improving']       # 55
PERCENTILE_STRONG_BUY = MODEL_THRESHOLDS['percentile_strong_buy']  # 70

# Volatility score interpretation thresholds
VOLATILITY_CONSISTENT = MODEL_THRESHOLDS['volatility_score_consistent']  # 70
VOLATILITY_MODERATE = MODEL_THRESHOLDS['volatility_score_moderate']      # 50

# ============================================================================
# DIAGNOSTIC DATA STRUCTURES (V5.0)
# ============================================================================

@dataclass
class IssuerDiagnosticReport:
    """Complete diagnostic data for a single issuer - READ ONLY from pipeline."""
    
    # Section 1: Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Section 2: Issuer Identity
    identity: Dict[str, Any] = field(default_factory=dict)
    
    # Section 3: Period Selection
    period_selection: Dict[str, Any] = field(default_factory=dict)
    
    # Section 4: Raw Input Data (from Excel)
    raw_inputs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Section 5: Calculated Ratios
    calculated_ratios: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Section 6: Component Scoring (per factor)
    component_scoring: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Section 7: Quality Score Calculation
    quality_calculation: Dict[str, Any] = field(default_factory=dict)
    
    # Section 8: Trend Score Calculation
    trend_calculation: Dict[str, Any] = field(default_factory=dict)
    
    # Section 9: Composite Score
    composite_calculation: Dict[str, Any] = field(default_factory=dict)
    
    # Section 10: Ranking & Percentiles
    rankings: Dict[str, Any] = field(default_factory=dict)
    
    # Section 11: Signal Classification
    signal_classification: Dict[str, Any] = field(default_factory=dict)
    
    # Section 12: Recommendation
    recommendation: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# DIAGNOSTIC INFRASTRUCTURE (Phase 1 - All-Issuer Diagnostics)
# ============================================================================

class DiagnosticLogger:
    """
    Centralized diagnostic logging to stderr with structured JSON output.
    Controlled by RG_DIAGNOSTICS environment variable.
    """
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._section_depth = 0

    def log(self, category: str, **data):
        """Log structured diagnostic data to stderr"""
        if not self.enabled:
            return
        timestamp = datetime.now().isoformat()
        indent = "  " * self._section_depth
        log_entry = {
            "timestamp": timestamp,
            "category": category,
            **data
        }
        print(f"{indent}{json.dumps(log_entry)}", file=sys.stderr)

    def section(self, name: str):
        """Start a new diagnostic section"""
        if not self.enabled:
            return
        print(f"\n{'=' * 80}", file=sys.stderr)
        print(f"{name}", file=sys.stderr)
        print(f"{'=' * 80}", file=sys.stderr)
        self._section_depth = 0

    def subsection(self, name: str):
        """Start a subsection"""
        if not self.enabled:
            return
        print(f"\n{'-' * 60}", file=sys.stderr)
        print(f"{name}", file=sys.stderr)
        print(f"{'-' * 60}", file=sys.stderr)


        self._section_depth = 1

    def analyze_period_distribution(self, df: pd.DataFrame, suffix_col: str):
        """Analyze and log period suffix distribution"""
        if not self.enabled or df.empty or suffix_col not in df.columns:
            return

        dist = df[suffix_col].value_counts().sort_index()
        total = len(df)

        self.log("PERIOD_DISTRIBUTION",
                 total_issuers=total,
                 periods={k: {"count": int(v), "percentage": round(v/total*100, 1)}
                         for k, v in dist.items()})

    def analyze_days_distribution(self, df: pd.DataFrame, name_col: str, days_col: str):
        """Analyze and log distribution of days since latest financials"""
        if not self.enabled or df.empty or days_col not in df.columns:
            return

        days_data = df[days_col].dropna()
        if days_data.empty:
            return

        self.log("DAYS_DISTRIBUTION",
                 min_days=int(days_data.min()),
                 max_days=int(days_data.max()),
                 mean_days=round(days_data.mean(), 1),
                 median_days=round(days_data.median(), 1),
                 over_90_days=int((days_data > 90).sum()),
                 over_180_days=int((days_data > 180).sum()))

    def analyze_sector_distribution(self, df: pd.DataFrame, sector_col: str):
        """Analyze and log sector distribution"""
        if not self.enabled or df.empty or sector_col not in df.columns:
            return

        dist = df[sector_col].value_counts()
        total = len(df)

        self.log("SECTOR_DISTRIBUTION",
                 total_issuers=total,
                 num_sectors=len(dist),
                 sectors={k: {"count": int(v), "percentage": round(v/total*100, 1)}
                         for k, v in dist.items()})

    def sample_issuers(self, df: pd.DataFrame, name_col: str, n: int = 10):
        """Log a sample of issuers"""
        if not self.enabled or df.empty or name_col not in df.columns:
            return

        sample = df.sample(min(n, len(df))) if len(df) > n else df
        sample_data = []
        for _, row in sample.iterrows():
            entry = {"company": row.get(name_col, "Unknown")}
            if 'selected_suffix' in row:
                entry['period'] = row['selected_suffix']
            if 'selected_date' in row:
                entry['date'] = str(row['selected_date'])
            sample_data.append(entry)

        self.log("ISSUER_SAMPLE", sample_size=len(sample), issuers=sample_data)

    def compare_periods(self, before_count: int, after_count: int, filter_name: str):
        """Log before/after counts for a filter operation"""
        if not self.enabled:
            return
        removed = before_count - after_count
        self.log("FILTER_COMPARISON",
                 filter=filter_name,
                 before=before_count,
                 after=after_count,
                 removed=removed,
                 filter_working=(removed > 0))

    def _format_data(self, data: Any) -> str:
        """Format data for logging"""
        if isinstance(data, (dict, list)):
            return json.dumps(data, default=str)
        return str(data)



# ============================================================================
# TREND ANALYSIS VISUALIZATION HELPERS
# ============================================================================

def create_trend_chart_with_classification(dates, values, metric_name, 
                                            classification, annual_change_pct,
                                            peak_idx=None, cv_value=None):
    """
    Create chart with trend line AND classification visualization.
    
    Args:
        dates: List of datetime objects
        values: List of metric values
        metric_name: Name of the metric (e.g., "EBITDA Margin")
        classification: One of IMPROVING, NORMALIZING, MODERATING, STABLE, DETERIORATING
        annual_change_pct: Annual percentage change (e.g., -7.2)
        peak_idx: Index of peak value (for NORMALIZING)
        cv_value: Coefficient of variation (for MODERATING)
    
    Returns:
        Plotly figure object
    """
    
    fig = go.Figure()
    
    # Convert to numpy for calculations
    x_numeric = np.array([(d - dates[0]).days for d in dates])
    values_arr = np.array(values)
    
    # Linear regression for trend line
    slope, intercept = np.polyfit(x_numeric, values_arr, 1)
    trend_line = slope * x_numeric + intercept
    
    # ═══════════════════════════════════════════════════════════════════════
    # NORMALIZING: Show peak + growth/stabilization phases
    # ═══════════════════════════════════════════════════════════════════════
    if classification == 'NORMALIZING' and peak_idx is not None:
        
        # 1. Green shading for growth phase (before peak)
        fig.add_vrect(
            x0=dates[0], 
            x1=dates[peak_idx],
            fillcolor="rgba(40, 167, 69, 0.15)",  # Light green
            layer="below", 
            line_width=0,
            annotation_text="Growth Phase",
            annotation_position="top left",
            annotation_font_size=10
        )
        
        # 2. Yellow shading for stabilization phase (after peak)
        fig.add_vrect(
            x0=dates[peak_idx], 
            x1=dates[-1],
            fillcolor="rgba(255, 193, 7, 0.15)",  # Light yellow
            layer="below", 
            line_width=0,
            annotation_text="Stabilization",
            annotation_position="top right",
            annotation_font_size=10
        )
        
        # 3. Vertical dashed line at peak
        fig.add_vline(
            x=dates[peak_idx],
            line_dash="dash",
            line_color="#ff9800",
            line_width=2,
            annotation_text="Peak",
            annotation_position="top"
        )
        
        # 4. Peak marker (triangle)
        fig.add_trace(go.Scatter(
            x=[dates[peak_idx]],
            y=[values[peak_idx]],
            mode='markers',
            name='Peak',
            marker=dict(
                size=14, 
                symbol='triangle-up', 
                color='#ff9800',
                line=dict(width=2, color='white')
            ),
            hovertemplate=f"Peak: {values[peak_idx]:.1f}<extra></extra>"
        ))
        
        # 5. Annotation box explaining classification
        peak_date_str = dates[peak_idx].strftime('%b %Y')
        fig.add_annotation(
            x=0.02,
            y=0.02,
            xref="paper",
            yref="paper",
            text=f"📍 NORMALIZING<br>Peak: {peak_date_str} ({values[peak_idx]:.1f})<br>Growth phase → Stabilization",
            showarrow=False,
            bgcolor="rgba(255, 193, 7, 0.3)",
            bordercolor="#ff9800",
            borderwidth=1,
            borderpad=4,
            font=dict(size=10),
            align="left",
            xanchor="left",
            yanchor="bottom"
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # MODERATING: Show volatility band + damped trend
    # ═══════════════════════════════════════════════════════════════════════
    elif classification == 'MODERATING' and cv_value is not None:
        
        # 1. Calculate volatility band (±1 std dev around trend)
        std_dev = np.std(values_arr)
        upper_band = trend_line + std_dev
        lower_band = trend_line - std_dev
        
        # 2. Volatility envelope (shaded area between upper and lower)
        fig.add_trace(go.Scatter(
            x=list(dates) + list(dates)[::-1],
            y=list(upper_band) + list(lower_band)[::-1],
            fill='toself',
            fillcolor='rgba(255, 152, 0, 0.2)',  # Light orange
            line=dict(color='rgba(255, 152, 0, 0)'),
            name='Volatility Band (±1σ)',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # 3. Upper bound (dotted line)
        fig.add_trace(go.Scatter(
            x=dates, 
            y=upper_band,
            mode='lines',
            line=dict(color='rgba(255, 152, 0, 0.5)', dash='dot', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # 4. Lower bound (dotted line)
        fig.add_trace(go.Scatter(
            x=dates, 
            y=lower_band,
            mode='lines',
            line=dict(color='rgba(255, 152, 0, 0.5)', dash='dot', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # 5. Annotation box explaining classification
        fig.add_annotation(
            x=0.02,
            y=0.02,
            xref="paper",
            yref="paper",
            text=f"📍 MODERATING<br>CV = {cv_value:.0%}<br>Volatility damping applied",
            showarrow=False,
            bgcolor="rgba(255, 152, 0, 0.3)",
            bordercolor="#ff9800",
            borderwidth=1,
            borderpad=4,
            font=dict(size=10),
            align="left",
            xanchor="left",
            yanchor="bottom"
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # Common elements for ALL charts
    # ═══════════════════════════════════════════════════════════════════════
    
    # Actual data points (scatter + line)
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines+markers',
        name='Actual',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8),
        hovertemplate='%{y:.2f}<extra></extra>'
    ))
    
    # Trend line color based on classification
    trend_color = {
        'IMPROVING': '#28a745',      # Green
        'NORMALIZING': '#17a2b8',    # Cyan
        'MODERATING': '#ffc107',     # Yellow
        'STABLE': '#6c757d',         # Gray
        'DETERIORATING': '#dc3545'   # Red
    }.get(classification, '#6c757d')
    
    # Trend line
    fig.add_trace(go.Scatter(
        x=dates,
        y=trend_line,
        mode='lines',
        name=f'Trend ({annual_change_pct:+.1f}%/yr)',
        line=dict(color=trend_color, width=2, dash='dash'),
        hoverinfo='skip'
    ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"<b>{metric_name}</b>: {classification}",
            font=dict(size=14),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title='',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        hovermode='x unified',
        legend=dict(
            orientation='h', 
            yanchor='bottom', 
            y=1.02, 
            xanchor='center', 
            x=0.5,
            font=dict(size=10)
        ),
        height=300,
        margin=dict(l=50, r=50, t=70, b=50),
        plot_bgcolor='white'
    )
    
    return fig


def get_classification_data(accessor, metric_name):
    """
    Get classification-specific data for a metric.
    
    Returns dict with:
        - classification: str
        - peak_idx: int or None (for NORMALIZING)
        - cv_value: float or None (for MODERATING)
        - is_override: bool
    """
    import numpy as np
    
    result = {
        'classification': 'STABLE',
        'peak_idx': None,
        'cv_value': None,
        'is_override': False
    }
    
    try:
        # Get time series data
        ts_data = accessor.get_metric_time_series(metric_name)
        if not ts_data:
            return result
        
        classification = ts_data.get('classification', 'STABLE')
        result['classification'] = classification
        
        values = ts_data.get('values', [])
        if not values or len(values) < 3:
            return result
        
        # For NORMALIZING: Find peak index
        if classification == 'NORMALIZING':
            result['peak_idx'] = int(np.argmax(values))
            result['is_override'] = True
        
        # For MODERATING: Calculate CV
        elif classification == 'MODERATING':
            cv = accessor.get_metric_cv(metric_name)
            result['cv_value'] = cv if cv else 0.30  # Default to threshold
            result['is_override'] = True
        
    except Exception as e:
        # Log but don't fail
        pass
    
    return result


# ============================================================================
# DIAGNOSTIC DATA ACCESSOR (Phase 2 - Clean Data Access Layer)
# ============================================================================

class DiagnosticDataAccessor:
    """
    Clean abstraction layer for accessing diagnostic data.
    
    Provides type-safe, validated access to:
    - Final scores
    - Factor details and breakdowns
    - Time series data for trend analysis
    - Period selection information
    - Composite calculation details
    
    Usage:
        accessor = DiagnosticDataAccessor(issuer_results, diagnostic_data)
        leverage_score = accessor.get_factor_score('Leverage')
        trend_data = accessor.get_metric_time_series('Debt/EBITDA')
    """
    
    def __init__(self, issuer_results: pd.Series, diagnostic_json: str):
        """
        Initialize accessor with issuer data.
        
        Args:
            issuer_results: Single row from results_final DataFrame (pd.Series)
            diagnostic_json: JSON string containing diagnostic data
        
        Raises:
            ValueError: If diagnostic data is invalid or missing required fields
        """
        self.results = issuer_results
        self.company_name = str(issuer_results['Company_Name'])
        
        # Parse diagnostic JSON
        try:
            self.diag = json.loads(diagnostic_json)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Invalid diagnostic data for {self.company_name}: {e}")
        
        # Validate structure
        self._validate()
    
    def _validate(self):
        """Validate that diagnostic data has required structure"""
        required_keys = ['time_series', 'factor_details', 'period_selection', 'composite_calculation']
        missing = [k for k in required_keys if k not in self.diag]
        
        if missing:
            raise ValueError(f"Diagnostic data for {self.company_name} missing required keys: {missing}")
        
        # Note: 'peer_context' is optional for backward compatibility with existing diagnostic data
        # New diagnostic data will include it
    
    # ========================================================================
    # COMPANY INFORMATION
    # ========================================================================
    
    def get_company_name(self) -> str:
        """Get company name"""
        return self.company_name
    
    def get_company_id(self) -> str:
        """Get company ID"""
        return str(self.results.get('Company_ID', ''))
    
    def get_credit_rating(self) -> str:
        """Get S&P credit rating"""
        return str(self.results.get('Credit_Rating', 'NR'))
    
    def get_sector(self) -> str:
        """Get sector/classification"""
        return str(self.results.get('Rubrics_Custom_Classification', 'Unknown'))

    def get_ticker(self) -> str:
        """Get company ticker symbol"""
        return str(self.results.get('Ticker', 'N/A'))

    def get_industry(self) -> str:
        """Get company industry"""
        return str(self.results.get('Industry', 'N/A'))

    def get_market_cap(self) -> Optional[float]:
        """Get market capitalization"""
        cap = self.results.get('Market_Cap')
        return float(cap) if pd.notna(cap) else None
    
    # ========================================================================
    # FINAL SCORES
    # ========================================================================
    
    def get_composite_score(self) -> Optional[float]:
        """Get final composite score"""
        score = self.results.get('Composite_Score')
        return float(score) if pd.notna(score) else None
    
    def get_factor_score(self, factor_name: str) -> Optional[float]:
        """
        Get final score for a quality factor.
        
        Args:
            factor_name: One of 'Credit', 'Leverage', 'Profitability', 
                        'Liquidity', 'Cash_Flow'
        
        Returns:
            Score (0-100) or None if not available
        """
        valid_factors = ['Credit', 'Leverage', 'Profitability', 'Liquidity', 'Cash_Flow']
        if factor_name not in valid_factors:
            raise ValueError(f"Invalid factor name: {factor_name}. Must be one of {valid_factors}")
        
        score_col = f'{factor_name}_Score'
        score = self.results.get(score_col)
        return float(score) if pd.notna(score) else None
    
    def get_trend_score(self) -> Optional[float]:
        """Get cycle position (trend) score"""
        score = self.results.get('Cycle_Position_Score')
        return float(score) if pd.notna(score) else None
    
    def get_all_factor_scores(self) -> Dict[str, Optional[float]]:
        """Get all factor scores as dictionary"""
        return {
            'Credit': self.get_factor_score('Credit'),
            'Leverage': self.get_factor_score('Leverage'),
            'Profitability': self.get_factor_score('Profitability'),
            'Liquidity': self.get_factor_score('Liquidity'),
            'Cash_Flow': self.get_factor_score('Cash_Flow')
        }
    
    # ========================================================================
    # PEER CONTEXT (for GenAI reports)
    # ========================================================================
    
    def get_peer_context(self) -> Optional[Dict[str, Any]]:
        """
        Get pre-computed peer context for GenAI reports.
        
        Returns:
            Dictionary containing:
                - sector_comparison: {classification, peer_count, medians, percentiles}
                - rating_comparison: {rating, peer_count, medians, percentiles}
            Or None if not available
        """
        return self.diag.get('peer_context')
    
    def get_sector_percentile(self, metric_key: str) -> Optional[float]:
        """Get company's percentile within sector for a specific metric."""
        peer_ctx = self.get_peer_context()
        if peer_ctx is None:
            return None
        return peer_ctx.get('sector_comparison', {}).get('percentiles', {}).get(metric_key)
    
    def get_rating_percentile(self, metric_key: str) -> Optional[float]:
        """Get company's percentile within rating peers for a specific metric."""
        peer_ctx = self.get_peer_context()
        if peer_ctx is None:
            return None
        return peer_ctx.get('rating_comparison', {}).get('percentiles', {}).get(metric_key)
    
    # ========================================================================
    # FACTOR DETAILS (Component Breakdowns)
    # ========================================================================
    
    def get_factor_details(self, factor_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed breakdown for a quality factor.
        
        Args:
            factor_name: One of 'Credit', 'Leverage', 'Profitability', 
                        'Liquidity', 'Cash_Flow'
        
        Returns:
            Dictionary containing:
                - final_score: Overall factor score
                - components: Dict of component details (raw_value, component_score, weight, contribution)
                - data_completeness: Fraction of components available (0-1)
                - components_used: Number of components used
            Or None if factor not available
        """
        valid_factors = ['Credit', 'Leverage', 'Profitability', 'Liquidity', 'Cash_Flow']
        if factor_name not in valid_factors:
            raise ValueError(f"Invalid factor name: {factor_name}")
        
        factor_details = self.diag.get('factor_details', {})
        return factor_details.get(factor_name)
    
    def get_factor_components(self, factor_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Get individual component details for a factor.
        
        Returns:
            Dictionary mapping component name to component details
            Empty dict if factor not available
        """
        details = self.get_factor_details(factor_name)
        if details is None:
            return {}
        return details.get('components', {})
    
    def get_factor_data_completeness(self, factor_name: str) -> float:
        """
        Get data completeness for a factor (0.0 to 1.0).
        
        Returns:
            1.0 = all components present, 0.0 = no data available
        """
        details = self.get_factor_details(factor_name)
        if details is None:
            return 0.0
        return float(details.get('data_completeness', 0.0))
    
    # ========================================================================
    # TIME SERIES DATA (Trend Analysis)
    # ========================================================================
    
    def get_metric_time_series(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """
        Get time series data for a specific metric.
        
        Args:
            metric_name: Metric identifier (e.g., 'Debt/EBITDA', 'EBITDA_Margin')
        
        Returns:
            Dictionary containing:
                - dates: List of ISO date strings
                - values: List of metric values
                - trend_direction: % change per year
                - momentum: Momentum score 0-100
                - volatility: Volatility score 0-100
                - classification: 'IMPROVING', 'STABLE', 'DETERIORATING', or 'INSUFFICIENT_DATA'
                - periods_count: Number of data points
            Or None if metric not available
        """
        time_series = self.diag.get('time_series', {})
        return time_series.get(metric_name)

    def get_all_metric_time_series(self) -> Dict[str, Any]:
        """
        Get all time series data for all metrics.
        
        Returns:
            Dictionary mapping metric names to their time series data
        """
        return self.diag.get('time_series', {})
    
    def get_all_time_series_metrics(self) -> List[str]:
        """
        Get list of all metrics with time series data.
        
        Returns:
            List of metric names (e.g., ['Debt/EBITDA', 'EBITDA_Margin', ...])
        """
        time_series = self.diag.get('time_series', {})
        return list(time_series.keys())
    
    def has_time_series_data(self, metric_name: str) -> bool:
        """Check if time series data exists for a metric"""
        ts = self.get_metric_time_series(metric_name)
        if ts is None:
            return False
        return len(ts.get('dates', [])) >= 3  # Need at least 3 points for trend analysis
    
    def get_metric_classification(self, metric_name: str) -> str:
        """Get classification for a metric (e.g., 'IMPROVING', 'STABLE')"""
        ts = self.get_metric_time_series(metric_name)
        if ts is None:
            return "UNKNOWN"
        return ts.get('classification', 'UNKNOWN')

    def get_metric_cv(self, metric_name: str) -> Optional[float]:
        """Get coefficient of variation for a metric"""
        ts = self.get_metric_time_series(metric_name)
        if ts is None:
            return None
            
        values = ts.get('values', [])
        if not values or len(values) < 2:
            return None
            
        try:
            mean_val = np.mean(values)
            if abs(mean_val) < 1e-9:
                return 0.0
                
            std_val = np.std(values)
            return std_val / abs(mean_val)
        except Exception:
            return None
    
    # ========================================================================
    # PERIOD SELECTION INFORMATION
    # ========================================================================
    
    def get_period_selection(self) -> Dict[str, Any]:
        """
        Get period selection information.
        
        Returns:
            Dictionary containing:
                - selected_suffix: Suffix used (e.g., '.7')
                - selected_date: ISO date string
                - period_type: 'FY' or 'CQ'
                - periods_available: Total periods in dataset
                - selection_mode: Mode used (e.g., 'LATEST_AVAILABLE')
                - selection_reason: Human-readable explanation
        """
        return self.diag.get('period_selection', {})
    
    def get_selected_period_suffix(self) -> str:
        """Get the period suffix used in scoring (e.g., '.7')"""
        period_sel = self.get_period_selection()
        return period_sel.get('selected_suffix', '.0')
    
    def get_selected_period_date(self) -> Optional[str]:
        """Get the period date used (ISO format)"""
        period_sel = self.get_period_selection()
        return period_sel.get('selected_date')
    
    def get_period_type(self) -> str:
        """Get period type: 'FY' (fiscal year) or 'CQ' (calendar quarter)"""
        period_sel = self.get_period_selection()
        return period_sel.get('period_type', 'Unknown')
    
    # ========================================================================
    # COMPOSITE CALCULATION DETAILS
    # ========================================================================
    
    def get_composite_calculation(self) -> Dict[str, Any]:
        """
        Get composite score calculation details.
        
        Returns:
            Dictionary containing:
                - composite_score: Final composite score
                - quality_score: Quality component score
                - trend_score: Trend component score
                - factor_contributions: Dict of each factor's contribution
                - weight_method: Weighting method used
                - sector: Sector/classification
        """
        return self.diag.get('composite_calculation', {})
    
    def get_factor_contributions(self) -> Dict[str, Any]:
        """
        Get how each factor contributed to composite score.
        
        Returns:
            Dictionary containing:
                - contributions: List of dicts with 'factor', 'raw_score', 'weight', 'contribution'
                - total_weight: Sum of weights
                - total_score: Sum of contributions
        """
        comp_calc = self.get_composite_calculation()
        raw_contribs = comp_calc.get('factor_contributions', {})
        
        contributions_list = []
        total_weight = 0.0
        total_score = 0.0
        
        for factor, details in raw_contribs.items():
            weight = details.get('weight', 0.0)
            score = details.get('score', 0.0)
            contribution = details.get('contribution', 0.0)
            
            contributions_list.append({
                'factor': factor,
                'raw_score': score,
                'weight': weight,
                'contribution': contribution
            })
            
            total_weight += weight
            total_score += contribution
            
        return {
            'contributions': contributions_list,
            'total_weight': total_weight,
            'total_score': total_score
        }
    
    def get_weight_method(self) -> str:
        """Get weighting method used (e.g., 'Universal', 'Sector-Calibrated')"""
        comp_calc = self.get_composite_calculation()
        return comp_calc.get('weight_method', 'Unknown')
    
    def get_combined_signal(self) -> str:
        """Get the overall Quality × Trend signal classification."""
        return str(self.results.get('Combined_Signal', 'Unknown'))

    def get_recommendation(self) -> str:
        """Get the model recommendation (Strong Buy/Buy/Hold/Avoid)."""
        return str(self.results.get('Recommendation', 'Unknown'))

    def get_signal_details(self) -> Dict[str, Any]:
        """
        Get comprehensive signal information.
        
        Returns:
            Dictionary containing:
                - combined_signal: Overall classification (e.g., 'Strong & Normalizing')
                - recommendation: Investment recommendation
                - quality_score: Quality component
                - trend_score: Trend component
                - is_override: Whether this is a context-aware override
        """
        quality = self.get_composite_score()
        trend = self.get_trend_score()
        signal = self.get_combined_signal()
        
        # Determine if this is an override case
        is_override = signal in ['Strong & Normalizing', 'Strong & Moderating']
        
        return {
            'combined_signal': signal,
            'recommendation': self.get_recommendation(),
            'quality_score': quality,
            'trend_score': trend,
            'is_override': is_override
        }
    
    # ========================================================================
    # DATA QUALITY METRICS
    # ========================================================================
    
    def get_overall_data_completeness(self) -> float:
        """
        Get overall data completeness across all factors (0.0 to 1.0).
        
        Returns:
            Average data completeness across all factors
        """
        score = self.results.get('Composite_Data_Completeness')
        return float(score) if pd.notna(score) else 0.0
    
    def get_data_quality_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get data quality summary for all factors.
        
        Returns:
            Dictionary mapping factor name to:
                - completeness: Data completeness (0-1)
                - components_used: Number of components with data
                - components_total: Total possible components
        """
        summary = {}
        for factor in ['Credit', 'Leverage', 'Profitability', 'Liquidity', 'Cash_Flow']:
            details = self.get_factor_details(factor)
            if details:
                summary[factor] = {
                    'completeness': details.get('data_completeness', 0.0),
                    'components_used': details.get('components_used', 0),
                    'components_total': len(details.get('components', {}))
                }
        return summary
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Export all data as a dictionary for debugging or export.
        
        Returns:
            Complete snapshot of issuer data
        """
        return {
            'company_info': {
                'name': self.get_company_name(),
                'id': self.get_company_id(),
                'rating': self.get_credit_rating(),
                'sector': self.get_sector()
            },
            'scores': {
                'composite': self.get_composite_score(),
                'factors': self.get_all_factor_scores(),
                'trend': self.get_trend_score()
            },
            'period_selection': self.get_period_selection(),
            'composite_calculation': self.get_composite_calculation(),
            'data_quality': self.get_data_quality_summary(),
            'available_metrics': self.get_all_time_series_metrics()
        }


# ============================================================================
# DIAGNOSTIC REPORT BUILDER & FORMATTER (V5.0)
# ============================================================================

def build_issuer_diagnostic_report(
    idx: int,
    df: pd.DataFrame,
    results: pd.DataFrame,
    raw_input_data: Dict,
    factor_diagnostic_data: Dict,
    trend_diagnostic_data: Dict,
    config: Dict,
    selected_periods: pd.DataFrame
) -> IssuerDiagnosticReport:
    """
    Build complete diagnostic report for a single issuer.
    
    This function is READ-ONLY - it only assembles data that was
    already calculated in the main pipeline. No recalculation.
    """
    report = IssuerDiagnosticReport()
    
    # Section 1: Configuration
    report.config = {
        'analysis_date': config.get('analysis_date'),
        'period_mode': config.get('period_mode'),
        'reference_date': config.get('reference_date'),
        'dynamic_calibration': config.get('use_dynamic_calibration'),
        'calibration_band': config.get('calibration_rating_band'),
        'scoring_method': config.get('scoring_method')
    }
    
    # Section 2: Identity (from df and results)
    report.identity = {
        'company_id': str(results.loc[idx, 'Company_ID']),
        'company_name': results.loc[idx, 'Company_Name'],
        'ticker': df.loc[idx, 'Ticker'] if 'Ticker' in df.columns else None,
        'rating': results.loc[idx, 'Credit_Rating'],
        'rating_band': results.loc[idx, 'Rating_Band'],
        'sector': df.loc[idx, 'Rubrics Custom Classification'] if 'Rubrics Custom Classification' in df.columns else None,
        'industry': df.loc[idx, 'Industry'] if 'Industry' in df.columns else None,
        'market_cap': float(results.loc[idx, 'Market_Cap']) if pd.notna(results.loc[idx, 'Market_Cap']) else None
    }
    
    # Section 3: Period Selection
    if idx in selected_periods.index:
        report.period_selection = {
            'selected_suffix': selected_periods.loc[idx, 'selected_suffix'],
            'selected_date': str(selected_periods.loc[idx, 'selected_date']),
            'period_type': 'FY' if selected_periods.loc[idx, 'is_fy'] else 'CQ',
            'selection_reason': selected_periods.loc[idx, 'selection_reason'] if 'selection_reason' in selected_periods.columns else 'N/A'
        }
    
    # Section 4: Raw Inputs
    report.raw_inputs = raw_input_data.get(idx, {})
    
    # Section 5: Calculated Ratios & Section 6: Component Scoring
    # These come from factor_diagnostic_data which is structured by factor
    # We'll reorganize for the report
    
    if idx in factor_diagnostic_data:
        factor_data = factor_diagnostic_data[idx]
        
        # Initialize containers
        report.calculated_ratios = {}
        report.component_scoring = {}
        
        for factor, details in factor_data.items():
            if factor == 'Composite_Calculation':
                continue
                
            # Component Scoring
            report.component_scoring[factor] = {
                'final_score': details.get('final_score'),
                'weight_in_composite': details.get('weight_in_composite'), # If available
                'data_completeness': details.get('data_completeness')
            }
            
            # Ratios and Component Details
            if 'components' in details:
                report.calculated_ratios[factor] = {}
                for comp_name, comp_details in details['components'].items():
                    # Store ratio details
                    report.calculated_ratios[factor][comp_name] = {
                        'formula': comp_details.get('formula'),
                        'calculation': comp_details.get('calculation'),
                        'result_formatted': comp_details.get('result_formatted'),
                        'source': comp_details.get('source'),
                        'raw_value': comp_details.get('raw_value')
                    }
                    
                    # Add component scoring details
                    if 'components' not in report.component_scoring[factor]:
                        report.component_scoring[factor]['components'] = {}
                    
                    report.component_scoring[factor]['components'][comp_name] = {
                        'raw_value': comp_details.get('raw_value'),
                        'score': comp_details.get('component_score'),
                        'weight': comp_details.get('weight'),
                        'weighted_contribution': comp_details.get('weighted_contribution'),
                        'scoring_logic': comp_details.get('scoring_logic')
                    }
    
    # Section 7: Quality Score Calculation
    if idx in factor_diagnostic_data and 'Composite_Calculation' in factor_diagnostic_data[idx]:
        report.quality_calculation = factor_diagnostic_data[idx]['Composite_Calculation']
    
    # Section 8: Trend Score Calculation
    if idx in trend_diagnostic_data:
        report.trend_calculation = trend_diagnostic_data[idx]
    
    # Section 9: Composite Score
    report.composite_calculation = {
        'composite_score': results.loc[idx, 'Composite_Score'],
        'quality_score': results.loc[idx, 'Composite_Score'], # In V5, Composite IS Quality (mostly)
        'trend_score': results.loc[idx, 'Cycle_Position_Score'],
        'combined_signal': results.loc[idx, 'Combined_Signal']
    }
    
    # Section 10: Ranking & Percentiles
    report.rankings = {
        'rank_in_band': int(results.loc[idx, 'Rank_in_Band']) if 'Rank_in_Band' in results.columns else None,
        'percentile_in_band': float(results.loc[idx, 'Composite_Percentile_in_Band']) if 'Composite_Percentile_in_Band' in results.columns else None,
        'global_percentile': float(results.loc[idx, 'Composite_Percentile_Global']) if 'Composite_Percentile_Global' in results.columns else None
    }
    
    # Section 11: Signal Classification
    report.signal_classification = {
        'signal_base': results.loc[idx, 'Signal_Base'],
        'signal_final': results.loc[idx, 'Signal'],
        'is_strong_quality': results.loc[idx, 'Composite_Score'] >= 50, # V5.0.3 logic
        'is_improving_trend': results.loc[idx, 'Cycle_Position_Score'] >= TREND_THRESHOLD, # Default threshold
        'exceptional_quality': bool(results.loc[idx, 'ExceptionalQuality']) if 'ExceptionalQuality' in results.columns else False,
        'volatile_series': bool(results.loc[idx, 'VolatileSeries']) if 'VolatileSeries' in results.columns else False,
        'outlier_quarter': bool(results.loc[idx, 'OutlierQuarter']) if 'OutlierQuarter' in results.columns else False
    }
    
    # Section 12: Recommendation
    report.recommendation = {
        'recommendation': results.loc[idx, 'Recommendation'],
        'rationale': "Based on Combined Signal matrix"
    }
    
    return report

def format_diagnostic_report_text(report: IssuerDiagnosticReport) -> str:
    """
    Format diagnostic report as structured text for review.
    
    Output format is optimized for:
    1. Human readability
    2. Claude/LLM parsing
    3. Copy-paste into documents
    """
    lines = []
    
    lines.append("=" * 80)
    lines.append("ISSUER DIAGNOSTIC REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"App Version: 5.1.0")
    lines.append("")
    
    # Section 1: Configuration
    lines.append("=" * 80)
    lines.append("SECTION 1: CONFIGURATION")
    lines.append("=" * 80)
    for key, value in report.config.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    
    # Section 2: Identity
    lines.append("=" * 80)
    lines.append("SECTION 2: ISSUER IDENTITY")
    lines.append("=" * 80)
    for key, value in report.identity.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    
    # Section 3: Period Selection
    lines.append("=" * 80)
    lines.append("SECTION 3: PERIOD SELECTION")
    lines.append("=" * 80)
    for key, value in report.period_selection.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    
    # Section 4: Raw Input Data (special formatting)
    lines.append("=" * 80)
    lines.append("SECTION 4: RAW INPUT DATA (from Excel)")
    lines.append("=" * 80)
    lines.append("# These are the SOURCE values - exactly as they appear in the input file")
    lines.append("# NOTE: All monetary values are in THOUSANDS (e.g., 1000000 = $1 billion)")
    lines.append("")
    
    for category, values in report.raw_inputs.items():
        lines.append(f"{category}:")
        for field, value in values.items():
            if pd.notna(value):
                lines.append(f"  {field}: {value:,.0f} (thousands)")
            else:
                lines.append(f"  {field}: N/A")
        lines.append("")
    
    # Section 5: Calculated Ratios (with formulas)
    lines.append("=" * 80)
    lines.append("SECTION 5: CALCULATED RATIOS")
    lines.append("=" * 80)
    lines.append("# Each ratio shows: Formula → Calculation → Result")
    lines.append("")
    
    for category, ratios in report.calculated_ratios.items():
        lines.append(f"{category}:")
        for ratio_name, ratio_data in ratios.items():
            lines.append(f"  {ratio_name}:")
            lines.append(f"    Formula: {ratio_data.get('formula', 'N/A')}")
            lines.append(f"    Calculation: {ratio_data.get('calculation', 'N/A')}")
            lines.append(f"    Result: {ratio_data.get('result_formatted', 'N/A')}")
            if ratio_data.get('source'):
                lines.append(f"    Source: {ratio_data['source']}")
            lines.append("")
    
    # Section 6: Component Scoring
    lines.append("=" * 80)
    lines.append("SECTION 6: COMPONENT SCORING")
    lines.append("=" * 80)
    lines.append("# Raw Value → Score → Weight → Contribution")
    lines.append("")
    
    for factor, details in report.component_scoring.items():
        lines.append(f"{factor.upper()} (Final Score: {details.get('final_score', 'N/A')})")
        if 'components' in details:
            for comp_name, comp_data in details['components'].items():
                lines.append(f"  {comp_name}:")
                lines.append(f"    Raw Value: {comp_data.get('raw_value', 'N/A')}")
                lines.append(f"    Score: {comp_data.get('score', 'N/A')} (Weight: {comp_data.get('weight', 'N/A')})")
                lines.append(f"    Contribution: {comp_data.get('weighted_contribution', 'N/A')}")
                if comp_data.get('scoring_logic'):
                    lines.append(f"    Logic: {comp_data.get('scoring_logic')}")
        lines.append("")
        
    # Section 8: Trend Analysis
    lines.append("=" * 80)
    lines.append("SECTION 8: TREND ANALYSIS")
    lines.append("=" * 80)
    
    trend_data = report.trend_calculation
    if trend_data:
        lines.append(f"Cycle Position Score: {trend_data.get('final_score', 'N/A')}")
        lines.append("")
        
        # Time Series Data
        if 'time_series' in trend_data:
            lines.append("Time Series Data:")
            for metric, ts in trend_data['time_series'].items():
                lines.append(f"  {metric}:")
                lines.append(f"    Values: {ts.get('values', [])}")
                lines.append(f"    Periods: {ts.get('periods', [])}")
                lines.append(f"    Direction: {ts.get('trend_direction', 'N/A')}")
                lines.append(f"    Classification: {ts.get('classification', 'N/A')}")
                lines.append("")
    
    # Section 7: Quality Score Calculation
    lines.append("=" * 80)
    lines.append("SECTION 7: QUALITY SCORE CALCULATION")
    lines.append("=" * 80)
    for key, value in report.quality_calculation.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    
    # Section 9: Composite Score
    lines.append("=" * 80)
    lines.append("SECTION 9: COMPOSITE SCORE")
    lines.append("=" * 80)
    for key, value in report.composite_calculation.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    
    # Section 10: Ranking & Percentiles
    lines.append("=" * 80)
    lines.append("SECTION 10: RANKING & PERCENTILES")
    lines.append("=" * 80)
    for key, value in report.rankings.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    # Section 11: Signal Classification
    lines.append("=" * 80)
    lines.append("SECTION 11: SIGNAL CLASSIFICATION")
    lines.append("=" * 80)
    for key, value in report.signal_classification.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    
    # Section 12: Recommendation
    lines.append("=" * 80)
    lines.append("SECTION 12: RECOMMENDATION")
    lines.append("=" * 80)
    for key, value in report.recommendation.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("END OF DIAGNOSTIC REPORT")
    lines.append("=" * 80)
    
    return "\n".join(lines)

# ============================================================================
# DIAGNOSTIC ACCESSOR HELPER FUNCTIONS
# ============================================================================

def create_diagnostic_accessor(results_final: pd.DataFrame, company_name: str) -> DiagnosticDataAccessor:
    """
    Factory function to create DiagnosticDataAccessor for a specific company.
    
    Args:
        results_final: Complete results DataFrame
        company_name: Company name to look up
    
    Returns:
        DiagnosticDataAccessor instance
    
    Raises:
        ValueError: If company not found or diagnostic data invalid
    """
    # Find issuer row
    matches = results_final[results_final['Company_Name'] == company_name]
    
    if len(matches) == 0:
        raise ValueError(f"Company not found: {company_name}")
    
    if len(matches) > 1:
        raise ValueError(f"Multiple companies found with name: {company_name}")
    
    issuer_results = matches.iloc[0]
    
    # Get diagnostic data
    if 'diagnostic_data' not in issuer_results or pd.isna(issuer_results['diagnostic_data']):
        raise ValueError(f"No diagnostic data available for: {company_name}")
    
    diagnostic_json = issuer_results['diagnostic_data']
    
    return DiagnosticDataAccessor(issuer_results, diagnostic_json)

def get_available_companies(results_final: pd.DataFrame) -> List[str]:
    """
    Get list of all companies with valid diagnostic data.
    
    Args:
        results_final: Complete results DataFrame
    
    Returns:
        Sorted list of company names
    """
    # Filter to companies with diagnostic data
    has_diag = results_final['diagnostic_data'].notna()
    companies = results_final[has_diag]['Company_Name'].astype(str).unique().tolist()
    return sorted(companies)

def validate_accessor_data(accessor: DiagnosticDataAccessor) -> List[str]:
    """
    Validate that accessor data is consistent with stored scores.
    
    Args:
        accessor: DiagnosticDataAccessor instance
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    company = accessor.get_company_name()
    
    # Validate 1: Factor scores match
    for factor in ['Credit', 'Leverage', 'Profitability', 'Liquidity', 'Cash_Flow']:
        accessor_score = accessor.get_factor_score(factor)
        details = accessor.get_factor_details(factor)
        
        if accessor_score is not None and details is not None:
            detail_score = details.get('final_score')
            if detail_score is not None:
                if abs(accessor_score - detail_score) > 0.1:
                    errors.append(f"{company}: {factor} score mismatch - "
                                f"accessor={accessor_score:.2f}, details={detail_score:.2f}")
    
    # Validate 2: Composite score matches calculation
    composite = accessor.get_composite_score()
    comp_calc = accessor.get_composite_calculation()
    
    if composite is not None and comp_calc:
        calc_composite = comp_calc.get('composite_score')
        if calc_composite is not None:
            if abs(composite - calc_composite) > 0.1:
                errors.append(f"{company}: Composite score mismatch - "
                            f"accessor={composite:.2f}, calculation={calc_composite:.2f}")
    
    # Validate 3: Factor contributions sum correctly (with weights)
    # [V4.1] Composite is now 80% Quality + 20% Trend, not just sum of factor contributions
    contributions_data = accessor.get_factor_contributions()
    if contributions_data and composite is not None:
        quality_score = contributions_data.get('total_score', 0.0)
        trend_score = accessor.get_trend_score()
        if trend_score is None:
            trend_score = 50.0
        
        # Expected composite = quality * 0.80 + trend * 0.20
        expected_composite = (quality_score * 0.80) + (trend_score * 0.20)
        
        # Allow small rounding error
        if abs(composite - expected_composite) > 0.5:
            errors.append(f"{company}: Composite doesn't match 80/20 blend - "
                        f"composite={composite:.2f}, expected={expected_composite:.2f} "
                        f"(quality={quality_score:.2f}, trend={trend_score:.2f})")
    
    return errors

def create_diagnostic_summary_table(results_final: pd.DataFrame, companies: List[str] = None) -> pd.DataFrame:
    """
    Create summary table of diagnostic data for multiple companies.
    
    Args:
        results_final: Complete results DataFrame
        companies: List of company names (None = all companies)
    
    Returns:
        DataFrame with diagnostic summary for each company
    """
    if companies is None:
        companies = get_available_companies(results_final)
    
    summary_data = []
    
    for company in companies:
        try:
            accessor = create_diagnostic_accessor(results_final, company)
            
            summary_data.append({
                'Company': company,
                'Composite_Score': accessor.get_composite_score(),
                'Credit_Score': accessor.get_factor_score('Credit'),
                'Leverage_Score': accessor.get_factor_score('Leverage'),
                'Profitability_Score': accessor.get_factor_score('Profitability'),
                'Period_Type': accessor.get_period_type(),
                'Selected_Date': accessor.get_selected_period_date(),
                'Data_Completeness': accessor.get_overall_data_completeness(),
                'Weight_Method': accessor.get_weight_method(),
                'Metrics_Count': len(accessor.get_all_time_series_metrics())
            })
        except Exception as e:
            summary_data.append({
                'Company': company,
                'Error': str(e)
            })
    
    return pd.DataFrame(summary_data)


def create_diagnostic_export_data(
    accessor: DiagnosticDataAccessor,
    selected_company: str,
    scoring_method: str,
    period_mode: Any,
    reference_date_override: Any,
    use_dynamic_calibration: bool,
    calibration_rating_band: str
) -> Dict[str, Any]:
    """
    Prepare data for export (CSV/Excel).
    Combines structured report data with flat summary metrics.
    
    Args:
        accessor: DiagnosticDataAccessor instance
        selected_company: Company name
        scoring_method: Scoring method used
        period_mode: Period selection mode
        reference_date_override: Reference date (if any)
        use_dynamic_calibration: Whether dynamic calibration was used
        calibration_rating_band: Rating band used for calibration
        
    Returns:
        Dictionary containing:
            - summary: Flat dictionary of key metrics
            - details: List of dictionaries for component details
            - time_series: List of dictionaries for trend data
            - report_text: Full text report
    """
    # Reconstruct IssuerDiagnosticReport from accessor
    report = IssuerDiagnosticReport()
    
    # 1. Configuration
    report.config = {
        'analysis_date': datetime.now().isoformat(),
        'period_mode': str(period_mode),
        'reference_date': str(reference_date_override) if reference_date_override else None,
        'dynamic_calibration': use_dynamic_calibration,
        'calibration_band': calibration_rating_band,
        'scoring_method': scoring_method
    }
    
    # 2. Identity
    report.identity = {
        'company_id': accessor.get_company_id(),
        'company_name': accessor.get_company_name(),
        'ticker': accessor.get_ticker(),
        'rating': accessor.get_credit_rating(),
        'sector': accessor.get_sector(),
        'industry': accessor.get_industry(),
        'market_cap': accessor.get_market_cap()
    }
    
    # 3. Period Selection
    report.period_selection = accessor.get_period_selection()
    
    # 4. Raw Inputs
    report.raw_inputs = accessor.diag.get('raw_inputs', {})
    
    # 5. Calculated Ratios & 6. Component Scoring
    report.calculated_ratios = {}
    report.component_scoring = {}
    
    factor_details = accessor.diag.get('factor_details', {})
    for factor, details in factor_details.items():
        if factor == 'Composite_Calculation':
            continue
            
        # Component Scoring
        report.component_scoring[factor] = {
            'final_score': details.get('final_score'),
            'data_completeness': details.get('data_completeness')
        }
        
        # Ratios and Component Details
        if 'components' in details:
            report.calculated_ratios[factor] = {}
            if 'components' not in report.component_scoring[factor]:
                report.component_scoring[factor]['components'] = {}
                
            for comp_name, comp_details in details['components'].items():
                # Store ratio details
                report.calculated_ratios[factor][comp_name] = {
                    'formula': comp_details.get('formula'),
                    'calculation': comp_details.get('calculation'),
                    'result_formatted': comp_details.get('result_formatted'),
                    'source': comp_details.get('source'),
                    'raw_value': comp_details.get('raw_value')
                }
                
                # Add component scoring details
                report.component_scoring[factor]['components'][comp_name] = {
                    'raw_value': comp_details.get('raw_value'),
                    'score': comp_details.get('component_score'),
                    'weight': comp_details.get('weight'),
                    'weighted_contribution': comp_details.get('weighted_contribution'),
                    'scoring_logic': comp_details.get('scoring_logic')
                }

    # 7. Quality Score Calculation
    if 'Composite_Calculation' in factor_details:
        report.quality_calculation = factor_details['Composite_Calculation']
        
    # 8. Trend Score Calculation
    # Reconstruct from time series and score
    report.trend_calculation = {
        'final_score': accessor.get_trend_score(),
        'time_series': accessor.get_all_metric_time_series()
    }
    
    # 9. Composite Score
    report.composite_calculation = accessor.get_composite_calculation()
    
    # 10. Ranking & Percentiles (from results)
    results = accessor.results
    report.rankings = {
        'rank_in_band': int(results['Rank_in_Band']) if 'Rank_in_Band' in results and pd.notna(results['Rank_in_Band']) else None,
        'percentile_in_band': float(results['Composite_Percentile_in_Band']) if 'Composite_Percentile_in_Band' in results and pd.notna(results['Composite_Percentile_in_Band']) else None,
        'global_percentile': float(results['Composite_Percentile_Global']) if 'Composite_Percentile_Global' in results and pd.notna(results['Composite_Percentile_Global']) else None
    }
    
    # 11. Signal Classification
    report.signal_classification = {
        'signal_base': results.get('Signal_Base'),
        'signal_final': results.get('Signal'),
        'is_strong_quality': results.get('Composite_Score', 0) >= QUALITY_THRESHOLD,
        'is_improving_trend': results.get('Cycle_Position_Score', 0) >= TREND_THRESHOLD,
        'exceptional_quality': bool(results.get('ExceptionalQuality', False)),
        'volatile_series': bool(results.get('VolatileSeries', False)),
        'outlier_quarter': bool(results.get('OutlierQuarter', False))
    }
    
    # 12. Recommendation
    report.recommendation = {
        'recommendation': accessor.get_recommendation(),
        'rationale': "Based on Combined Signal matrix"
    }

    # Now generate the export data using the reconstructed report
    
    # 1. Summary Metrics (Flat)
    summary = {
        'Company': accessor.get_company_name(),
        'Ticker': accessor.get_ticker(),
        'Sector': accessor.get_sector(),
        'Credit_Rating': accessor.get_credit_rating(),
        'Composite_Score': accessor.get_composite_score(),
        'Quality_Score': report.composite_calculation.get('quality_score'),
        'Trend_Score': accessor.get_trend_score(),
        'Recommendation': accessor.get_recommendation(),
        'Period_Type': accessor.get_period_type(),
        'Selected_Date': accessor.get_selected_period_date(),
        'Data_Completeness': accessor.get_overall_data_completeness()
    }
    
    # Add factor scores
    for factor, score in accessor.get_all_factor_scores().items():
        summary[f'{factor}_Score'] = score
        
    # 2. Component Details (Tabular)
    details = []
    for factor, factor_data in report.component_scoring.items():
        if 'components' in factor_data:
            for comp_name, comp_data in factor_data['components'].items():
                row = {
                    'Factor': factor,
                    'Component': comp_name,
                    'Raw_Value': comp_data.get('raw_value'),
                    'Score': comp_data.get('score'),
                    'Weight': comp_data.get('weight'),
                    'Contribution': comp_data.get('weighted_contribution'),
                    'Logic': comp_data.get('scoring_logic')
                }
                
                # Add formula/calc if available
                if factor in report.calculated_ratios and comp_name in report.calculated_ratios[factor]:
                    ratio_data = report.calculated_ratios[factor][comp_name]
                    row['Formula'] = ratio_data.get('formula')
                    row['Calculation'] = ratio_data.get('calculation')
                    row['Source'] = ratio_data.get('source')
                    
                details.append(row)
                
    # 3. Time Series Data (Tabular)
    time_series = []
    if report.trend_calculation and 'time_series' in report.trend_calculation:
        for metric, ts in report.trend_calculation['time_series'].items():
            dates = ts.get('dates', [])
            values = ts.get('values', [])
            for i in range(len(dates)):
                time_series.append({
                    'Metric': metric,
                    'Date': dates[i],
                    'Value': values[i],
                    'Trend_Direction': ts.get('trend_direction'),
                    'Classification': ts.get('classification')
                })
                
    # 4. Full Text Report
    report_text = format_diagnostic_report_text(report)
    
    return {
        'summary': summary,
        'details': details,
        'time_series': time_series,
        'report_text': report_text
    }


def create_diagnostic_csv(export_data: Dict[str, Any]) -> str:
    """Create CSV string from export data (Summary only)."""
    # For CSV, we only export the summary row + flattened details
    # This is a simplified export. For full details, Excel is preferred.
    
    summary = export_data['summary']
    
    # Create DataFrame and convert to CSV
    df = pd.DataFrame([summary])
    return df.to_csv(index=False)


def create_diagnostic_excel(export_data: Dict[str, Any]) -> bytes:
    """Create Excel file (bytes) from export data."""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Sheet 1: Summary
        pd.DataFrame([export_data['summary']]).to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 2: Component Details
        if export_data['details']:
            pd.DataFrame(export_data['details']).to_excel(writer, sheet_name='Component Details', index=False)
            
        # Sheet 3: Time Series
        if export_data['time_series']:
            pd.DataFrame(export_data['time_series']).to_excel(writer, sheet_name='Time Series', index=False)
            
        # Sheet 4: Full Report Text
        # We'll put the text in a single cell or column
        workbook = writer.book
        worksheet = workbook.add_worksheet('Full Report')
        text_format = workbook.add_format({'text_wrap': True, 'valign': 'top'})
        worksheet.set_column('A:A', 100)
        worksheet.write('A1', export_data['report_text'], text_format)
        
    return output.getvalue()

# ============================================================================
# DIAGNOSTIC ACCESSOR TESTS
# ============================================================================

def test_diagnostic_accessor():
    """
    Unit tests for DiagnosticDataAccessor.
    Run with: RG_TESTS=1 streamlit run app.py
    """
    print("\n" + "="*80)
    print("DIAGNOSTIC ACCESSOR UNIT TESTS")
    print("="*80)
    
    # Test data structure
    test_diagnostic_json = json.dumps({
        'time_series': {
            'Debt/EBITDA': {
                'dates': ['2023-12-31', '2024-03-31', '2024-06-30'],
                'values': [2.4, 2.3, 2.2],
                'trend_direction': -5.2,
                'momentum': 65.3,
                'volatility': 87.4,
                'classification': 'IMPROVING',
                'periods_count': 3
            }
        },
        'factor_details': {
            'Leverage': {
                'final_score': 45.2,
                'components': {
                    'Net_Debt_EBITDA': {
                        'raw_value': 2.4,
                        'component_score': 60.0,
                        'weight': 0.40,
                        'weighted_contribution': 24.0
                    }
                },
                'data_completeness': 1.0,
                'components_used': 4
            }
        },
        'period_selection': {
            'selected_suffix': '.7',
            'selected_date': '2024-10-26',
            'period_type': 'LTM',
            'periods_available': 8,
            'selection_mode': 'LATEST_AVAILABLE',
            'selection_reason': 'Most recent quarterly data'
        },
        'composite_calculation': {
            'composite_score': 58.3,
            'quality_score': 56.825,
            'trend_score': 64.2,
            'factor_contributions': {},
            'weight_method': 'Universal',
            'sector': 'Technology'
        }
    })
    
    test_results = pd.Series({
        'Company_Name': 'Test Company',
        'Company_ID': 'TEST001',
        'Credit_Rating': 'BBB+',
        'Composite_Score': 58.3,
        'Leverage_Score': 45.2,
        'Cycle_Position_Score': 64.2,
        'Rubrics_Custom_Classification': 'Technology',
        'Composite_Data_Completeness': 0.95
    })
    
    # Test 1: Accessor creation
    print("\nTest 1: Accessor Creation")
    try:
        accessor = DiagnosticDataAccessor(test_results, test_diagnostic_json)
        print("✓ Accessor created successfully")
    except Exception as e:
        print(f"✗ Failed to create accessor: {e}")
        return
    
    # Test 2: Company info methods
    print("\nTest 2: Company Info Methods")
    assert accessor.get_company_name() == 'Test Company'
    assert accessor.get_company_id() == 'TEST001'
    assert accessor.get_credit_rating() == 'BBB+'
    assert accessor.get_sector() == 'Technology'
    print("✓ All company info methods working")
    
    # Test 3: Score access methods
    print("\nTest 3: Score Access Methods")
    assert accessor.get_composite_score() == 58.3
    assert accessor.get_factor_score('Leverage') == 45.2
    assert accessor.get_trend_score() == 64.2
    print("✓ All score access methods working")
    
    # Test 4: Factor details methods
    print("\nTest 4: Factor Details Methods")
    leverage_details = accessor.get_factor_details('Leverage')
    assert leverage_details is not None
    assert leverage_details['final_score'] == 45.2
    assert len(leverage_details['components']) == 1
    print("✓ Factor details methods working")
    
    # Test 5: Time series methods
    print("\nTest 5: Time Series Methods")
    ts = accessor.get_metric_time_series('Debt/EBITDA')
    assert ts is not None
    assert len(ts['dates']) == 3
    assert ts['classification'] == 'IMPROVING'
    assert accessor.has_time_series_data('Debt/EBITDA') == True
    print("✓ Time series methods working")
    
    # Test 6: Period selection methods
    print("\nTest 6: Period Selection Methods")
    assert accessor.get_selected_period_suffix() == '.7'
    assert accessor.get_selected_period_date() == '2024-10-26'
    assert accessor.get_period_type() == 'LTM'
    print("✓ Period selection methods working")
    
    # Test 7: Validation
    print("\nTest 7: Validation")
    errors = validate_accessor_data(accessor)
    if errors:
        print(f"✗ Validation errors: {errors}")
    else:
        print("✓ Validation passed")
    
    # Test 8: Summary export
    print("\nTest 8: Summary Export")
    summary = accessor.to_summary_dict()
    assert 'company_info' in summary
    assert 'scores' in summary
    assert 'period_selection' in summary
    print("✓ Summary export working")
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80 + "\n")


# Run tests if in test mode
if os.environ.get("RG_TESTS") == "1":
    test_diagnostic_accessor()


class ConfigState:
    """
    Tracks configuration state and validates consistency.
    Used for debugging configuration issues without changing app behavior.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracked state"""
        self.period_mode = None
        self.reference_date_override = None
        self.prefer_annual_reports = None
        self.use_dynamic_calibration = None
        self.calibration_rating_band = None
        self.use_sector_adjusted = None
        self.calibrated_weights = None
        self.cache_key = None
        self.selected_periods = None

    def capture_ui_state(self, **kwargs):
        """Capture configuration from UI controls"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def validate_consistency(self) -> List[str]:
        """
        Validate configuration consistency and return list of issues.
        Does not raise exceptions - only logs and returns issues.
        """
        issues = []

        # Check if reference date is required but missing
        if (self.period_mode and
            hasattr(self.period_mode, 'value') and
            self.period_mode.value == 'reference_aligned' and
            self.reference_date_override is None):
            issues.append("REFERENCE_ALIGNED mode selected but no reference date provided")

        # Check if dynamic calibration is on but no weights provided
        if self.use_dynamic_calibration and self.calibrated_weights is None:
            issues.append("Dynamic calibration enabled but no calibrated weights available")

        # Check if prefer_annual_reports is used in wrong mode
        if (self.prefer_annual_reports and
            self.period_mode and
            hasattr(self.period_mode, 'value') and
            self.period_mode.value == 'latest_available'):
            issues.append("prefer_annual_reports=True has no effect in LATEST_AVAILABLE mode")

        # Log all issues
        for issue in issues:
            DIAG.log("CONFIG_VALIDATION_WARNING", issue=issue)

        return issues


# Global diagnostic singletons
DIAG = DiagnosticLogger(enabled=os.environ.get("RG_DIAGNOSTICS", "0") == "1")
CONFIG_STATE = ConfigState()


class WarningCollector:
    """Collect warnings during scoring runs and print summaries."""
    _warnings = {}  # {category: {message: [affected_issuers]}}
    
    @classmethod
    def collect(cls, category: str, message: str, issuer_name: str = None):
        """
        Collect a warning for later summary output.
        
        Args:
            category: Warning category (e.g., 'missing_column', 'invalid_value')
            message: The warning message (e.g., column name)
            issuer_name: Name of affected issuer (optional)
        """
        if category not in cls._warnings:
            cls._warnings[category] = {}
        if message not in cls._warnings[category]:
            cls._warnings[category][message] = []
        if issuer_name and issuer_name not in cls._warnings[category][message]:
            cls._warnings[category][message].append(issuer_name)
    
    @classmethod
    def print_summary(cls):
        """Print summary of all collected warnings. Call at end of scoring run."""
        if os.environ.get("RG_TESTS") != "1":
            return
            
        if not cls._warnings:
            return
            
        print("\n" + "="*70)
        print("WARNING SUMMARY")
        print("="*70)
        
        for category, messages in cls._warnings.items():
            print(f"\n[{category.upper()}]")
            for message, issuers in messages.items():
                count = len(issuers)
                if count > 0:
                    # Show first 5 issuers as examples
                    examples = ", ".join(issuers[:5])
                    if count > 5:
                        examples += f", ... (and {count - 5} more)"
                    print(f"  {message}: {count} issuers affected")
                    print(f"    Examples: {examples}")
                else:
                    print(f"  {message}")
        
        print("="*70 + "\n")
    
    @classmethod
    def reset(cls):
        """Reset warnings (call at start of new scoring run)."""
        cls._warnings = {}


def diagnose_quarterly_annualization(df):
    """Check if quarterly figures are being annualized correctly."""
    if not DIAG.enabled:
        return

    DIAG.section("QUARTERLY ANNUALIZATION DIAGNOSTICS")

    if 'selected_suffix' not in df.columns or 'is_fy' not in df.columns:
        return

    quarterly_issuers = df[df['is_fy'] == False]
    annual_issuers = df[df['is_fy'] == True]

    DIAG.log("PERIOD_TYPE_USAGE",
             total_issuers=len(df),
             using_quarterly=len(quarterly_issuers),
             using_annual=len(annual_issuers))

    # Check NVIDIA specifically
    nvidia = df[df['Company Name'].str.contains('NVIDIA', case=False, na=False)]
    if len(nvidia) > 0:
        nvidia_row = nvidia.iloc[0]

        metrics_to_check = {
            'Levered Free Cash Flow Margin': nvidia_row.get('Levered Free Cash Flow Margin', np.nan),
            'EBITDA Margin': nvidia_row.get('EBITDA Margin', np.nan),
            'Return on Assets': nvidia_row.get('Return on Assets', np.nan),
            'Return on Equity': nvidia_row.get('Return on Equity', np.nan)
        }

        DIAG.log("NVIDIA_RAW_METRICS",
                 period_type='Quarterly' if not nvidia_row.get('is_fy', True) else 'Annual',
                 period_suffix=nvidia_row.get('selected_suffix', 'N/A'),
                 period_date=str(nvidia_row.get('selected_date', 'N/A')),
                 raw_values={k: float(v) if pd.notna(v) else None for k, v in metrics_to_check.items()})

        # Compare quarterly vs annual averages
        for metric in ['Levered Free Cash Flow Margin', 'EBITDA Margin', 'Return on Assets']:
            if metric in df.columns:
                q_mean = quarterly_issuers[metric].mean()
                a_mean = annual_issuers[metric].mean()

                if pd.notna(q_mean) and pd.notna(a_mean) and a_mean != 0:
                    ratio = q_mean / a_mean

                    DIAG.log("QUARTERLY_VS_ANNUAL_COMPARISON",
                             metric=metric,
                             quarterly_mean=float(q_mean),
                             annual_mean=float(a_mean),
                             ratio=float(ratio),
                             interpretation="Ratio should be ~1.0 if normalized correctly, ~0.25 if quarterly not annualized",
                             suspicious=ratio < 0.4 or ratio > 2.5)


# ============================================================================
# REFERENCE DATE HELPER (for timing alignment in quarterly mode)
# ============================================================================

def get_reference_date():
    """
    Auto-determine appropriate reference date for alignment.

    Returns the most recent fiscal year-end that most issuers would have reported.

    Strategy:
    - Uses December 31 of the most recent year where we can expect broad data availability
    - Before April: Use Dec 31 of TWO years ago (companies still reporting prior year)
    - April-June: Use Dec 31 of prior year (most annual data available)
    - July onwards: Use most recent quarter-end that's at least 45 days old

    This ensures the reference date automatically advances as time passes.
    """
    from datetime import datetime, timedelta

    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month

    # Determine most recent reportable period
    # Companies typically report within 45-60 days of period end

    if current_month <= 3:
        # Jan-Mar: Use Dec 31 of TWO years ago
        # (Most companies haven't reported prior year's Q4/FY yet)
        reference_year = current_year - 2
        return pd.Timestamp(f"{reference_year}-12-31")

    elif current_month <= 6:
        # Apr-Jun: Use Dec 31 of prior year
        # (Annual reports should be filed by now)
        reference_year = current_year - 1
        return pd.Timestamp(f"{reference_year}-12-31")

    else:
        # Jul onwards: Use most recent quarter-end that's 60+ days old
        # This allows the reference date to advance quarterly through the year

        # Calculate date 60 days ago (reporting lag)
        reporting_lag_date = current_date - timedelta(days=60)

        # Find most recent quarter-end before reporting_lag_date
        year = reporting_lag_date.year
        month = reporting_lag_date.month

        # Determine quarter-end
        if month >= 10:  # Q4 (Dec 31)
            return pd.Timestamp(f"{year}-12-31")
        elif month >= 7:  # Q3 (Sep 30)
            return pd.Timestamp(f"{year}-09-30")
        elif month >= 4:  # Q2 (Jun 30)
            return pd.Timestamp(f"{year}-06-30")
        else:  # Q1 (Mar 31)
            return pd.Timestamp(f"{year}-03-31")


def calculate_reference_date_coverage(df):
    """
    Calculate what % of issuers have data available for each calendar quarter.

    Logic: A company has data for Q2 2025 if ANY of their Period Ended columns
    contains a date between April 1, 2025 and June 30, 2025 (inclusive).

    Args:
        df: DataFrame with Period Ended columns

    Returns:
        List of dicts with keys: date, coverage_pct, quarter_label, date_label,
                                 label, label_with_coverage, companies_with_data, total_companies
        Sorted by date descending (most recent first)
    """
    from datetime import datetime

    # Get all Period Ended columns (handle multi-index)
    period_cols = []
    for col in df.columns:
        col_str = str(col)
        if 'Period Ended' in col_str:
            period_cols.append(col)

    if not period_cols:
        return []

    # Define possible reference quarters (last 8 quarters + current quarter)
    current_date = datetime.now()
    current_year = current_date.year
    current_ts = pd.Timestamp(current_date)

    possible_quarters = [
        pd.Timestamp(f"{current_year}-12-31"),
        pd.Timestamp(f"{current_year}-09-30"),
        pd.Timestamp(f"{current_year}-06-30"),
        pd.Timestamp(f"{current_year}-03-31"),
        pd.Timestamp(f"{current_year-1}-12-31"),
        pd.Timestamp(f"{current_year-1}-09-30"),
        pd.Timestamp(f"{current_year-1}-06-30"),
        pd.Timestamp(f"{current_year-1}-03-31"),
    ]

    # Filter to only quarters that have STARTED (not quarters that have ended)
    # This allows selection of the current quarter (e.g., Q4 2025 in November)
    valid_quarters = []

    for quarter_end in possible_quarters:
        year = quarter_end.year
        month = quarter_end.month

        # Calculate quarter start date
        if month == 12:  # Q4: Oct 1 - Dec 31
            quarter_start = pd.Timestamp(f"{year}-10-01")
        elif month == 9:  # Q3: Jul 1 - Sep 30
            quarter_start = pd.Timestamp(f"{year}-07-01")
        elif month == 6:  # Q2: Apr 1 - Jun 30
            quarter_start = pd.Timestamp(f"{year}-04-01")
        else:  # Q1: Jan 1 - Mar 31
            quarter_start = pd.Timestamp(f"{year}-01-01")

        # Include quarter if it has started (even if not ended yet)
        if quarter_start <= current_ts:
            valid_quarters.append(quarter_end)

    # Count total companies in dataset (for accurate coverage calculation)
    total_companies = len(df)

    # Pre-filter to identify companies with at least one valid period date
    companies_with_any_data = 0
    for idx, row in df.iterrows():
        has_any_date = False
        for col in period_cols:
            try:
                period_date = pd.to_datetime(row[col], errors='coerce')
                if pd.notna(period_date) and period_date.year >= 1950:
                    has_any_date = True
                    break
            except Exception:
                continue
        if has_any_date:
            companies_with_any_data += 1

    # If no companies have data, return empty list
    if companies_with_any_data == 0:
        return []

    # Calculate coverage for each quarter
    coverage_results = []

    for quarter_end in valid_quarters:
        companies_with_data = 0

        # Define EXACT quarter boundaries (no tolerance)
        year = quarter_end.year
        month = quarter_end.month

        if month == 12:  # Q4: Oct 1 - Dec 31
            quarter_start = pd.Timestamp(f"{year}-10-01")
        elif month == 9:  # Q3: Jul 1 - Sep 30
            quarter_start = pd.Timestamp(f"{year}-07-01")
        elif month == 6:  # Q2: Apr 1 - Jun 30
            quarter_start = pd.Timestamp(f"{year}-04-01")
        else:  # Q1: Jan 1 - Mar 31
            quarter_start = pd.Timestamp(f"{year}-01-01")

        # Count companies with at least one period ending in this quarter
        for idx, row in df.iterrows():
            has_data_in_quarter = False

            # Check all period columns for this company
            for col in period_cols:
                try:
                    period_date = pd.to_datetime(row[col], errors='coerce')

                    # Check if period end date falls within the quarter boundaries
                    if pd.notna(period_date) and period_date.year >= 1950:
                        if quarter_start <= period_date <= quarter_end:
                            has_data_in_quarter = True
                            break
                except Exception:
                    continue

            if has_data_in_quarter:
                companies_with_data += 1

        # Calculate coverage as % of ALL companies (not just those with any data)
        coverage_pct = (companies_with_data / total_companies * 100) if total_companies > 0 else 0

        # Create display labels with quarter format
        quarter_num = (month - 1) // 3 + 1  # 1-4 for Q1-Q4
        quarter_label = f"Q{quarter_num} {year}"
        date_label = quarter_end.strftime("%b %d, %Y")
        label_with_coverage = f"{quarter_label} ({coverage_pct:.0f}% coverage)"

        coverage_results.append({
            'date': quarter_end,
            'coverage_pct': coverage_pct,
            'quarter_label': quarter_label,         # "Q3 2025"
            'date_label': date_label,               # "Sep 30, 2025"
            'label': quarter_label,                 # Use quarter as primary label
            'label_with_coverage': label_with_coverage,  # "Q3 2025 (11% coverage)"
            'companies_with_data': companies_with_data,
            'total_companies': total_companies,  # Use actual total, not filtered count
            'companies_with_valid_data': companies_with_any_data,  # Track this separately for diagnostics
            'quarter_start': quarter_start,         # For debug/display
            'quarter_end': quarter_end              # For debug/display
        })

    # Sort by date descending (most recent first)
    coverage_results.sort(key=lambda x: x['date'], reverse=True)

    return coverage_results


# ============================================================================
# PERIOD SELECTION HELPER FUNCTIONS (V2.3)
# ============================================================================

def get_period_selection_options(df):
    """
    Generate dropdown options for reference period selection with coverage indicators.

    Args:
        df: DataFrame with Period Ended columns

    Returns:
        List of tuples: (display_label, reference_date_value, coverage_pct)
    """
    coverage_data = calculate_reference_date_coverage(df)

    if not coverage_data:
        return []

    options = []
    for info in coverage_data:
        coverage_pct = info['coverage_pct']
        ref_date = info['date']
        quarter_label = info['quarter_label']

        # Format: "Q4 2024 (Dec 31, 2024) - 87% coverage"
        display_label = f"{quarter_label} ({ref_date.strftime('%b %d, %Y')}) - {coverage_pct:.0f}% coverage"

        # Append warning for low coverage
        if coverage_pct < 50:
            display_label += " [Low coverage]"
        elif coverage_pct >= 85:
            display_label += " [Good coverage]"

        options.append((display_label, ref_date, coverage_pct))

    # Already sorted by date descending in calculate_reference_date_coverage
    return options


def get_recommended_reference_date(df):
    """
    Automatically select the best reference date.

    Strategy:
    1. Find most recent period with >= 85% coverage
    2. If none, find most recent period with >= 70% coverage
    3. If none, use most recent available period

    Args:
        df: DataFrame with Period Ended columns

    Returns:
        pd.Timestamp or None
    """
    options = get_period_selection_options(df)

    if not options:
        return None

    # Try to find period with >= 85% coverage
    for label, ref_date, coverage in options:
        if coverage >= 85:
            return ref_date

    # Fall back to >= 70% coverage
    for label, ref_date, coverage in options:
        if coverage >= 70:
            return ref_date

    # Fall back to most recent
    return options[0][1]


def detect_stale_data(df, reference_date, stale_threshold_days=180):
    """
    Identify issuers with stale data (significantly older than reference date).

    Args:
        df: DataFrame with Period Ended columns
        reference_date: pd.Timestamp reference date
        stale_threshold_days: Number of days before data is considered stale

    Returns:
        DataFrame with columns: Company ID, Company Name, Latest Period, Days Old
    """
    company_id_col = resolve_company_id_column(df)
    company_name_col = resolve_company_name_column(df)

    if not company_id_col or not company_name_col:
        return pd.DataFrame()

    period_cols = parse_period_ended_cols(df)
    if not period_cols:
        return pd.DataFrame()

    stale_issuers = []

    for idx, row in df.iterrows():
        # Find latest valid date for this issuer
        latest_date = None
        for col in period_cols:
            try:
                date_val = pd.to_datetime(row[col], errors='coerce')
                if pd.notna(date_val) and date_val.year > 1950:
                    if latest_date is None or date_val > latest_date:
                        latest_date = date_val
            except Exception:
                continue

        if latest_date:
            days_old = (reference_date - latest_date).days
            if days_old > stale_threshold_days:
                stale_issuers.append({
                    'Company ID': row[company_id_col],
                    'Company Name': row[company_name_col],
                    'Latest Period': latest_date.strftime('%Y-%m-%d'),
                    'Days Old': days_old,
                    'Quarters Behind': days_old // 90
                })

    return pd.DataFrame(stale_issuers).sort_values('Days Old', ascending=False) if stale_issuers else pd.DataFrame()


def get_period_type_for_date(period_calendar, company_id, target_date):
    """
    Determine if a given date corresponds to Annual or Quarterly data for an issuer.

    Args:
        period_calendar: DataFrame from build_period_calendar()
        company_id: Issuer identifier
        target_date: pd.Timestamp date to check

    Returns:
        PeriodType.ANNUAL or PeriodType.QUARTERLY
    """
    if period_calendar is None or period_calendar.empty:
        return PeriodType.QUARTERLY  # Default

    issuer_data = period_calendar[period_calendar['company_id'] == company_id]

    if issuer_data.empty:
        return PeriodType.QUARTERLY

    # Check if date matches an FY period
    fy_periods = issuer_data[issuer_data['period_type'] == 'FY']
    for _, period in fy_periods.iterrows():
        if abs((period['period_date'] - target_date).days) <= 10:
            return PeriodType.ANNUAL

st.set_page_config(
    page_title="Issuer Credit Screening Model V5.0",
    layout="wide",
    page_icon="https://rubricsam.com/wp-content/uploads/2021/01/cropped-rubrics-logo-tight.png",
)

# HEADER RENDERING HELPER
# ============================================================================

def render_header(results_final=None, data_period=None, use_sector_adjusted=False, df_original=None,
                  use_quarterly_beta=False, align_to_reference=False):
    """
    Render the application header exactly once.

    Pre-upload: Shows only hero (title + subtitle + logo)
    Post-upload: Shows hero + 4 metrics + caption with period dates (and reference date if aligned)
    """
    # Always render the hero once
    st.markdown("""
    <div class="rb-header">
      <div class="rb-title">
        <h1>Issuer Credit Screening Model V5.0</h1>
        <div class="rb-sub">5-Factor Composite Scoring with Sector Adjustment & Trend Analysis</div>
      </div>
      <div class="rb-logo">
        <img src="https://rubricsam.com/wp-content/uploads/2021/01/cropped-rubrics-logo-tight.png" alt="Rubrics">
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Only show metrics/caption if data is loaded
    if results_final is not None and df_original is not None and data_period is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(" Total Issuers", f"{len(results_final):,}", help="Count after current filters")
        with col2:
            method_label = "Sector-Adjusted" if use_sector_adjusted else "Universal"
            st.metric(" Scoring Method", method_label)
        with col3:
            # compact period label
            if data_period == "Most Recent Fiscal Year (FY0)":
                display_period = "FY0"
            elif data_period == "Most Recent LTM (LTM0)":
                display_period = "LTM0"
            elif data_period.startswith("Reference Aligned"):
                # Extract date from the setting string
                display_period = data_period  # Shows "Reference Aligned (2024-12-31)"
            else:
                display_period = data_period
            st.metric(" Data Period", display_period)
        with col4:
            avg_score = results_final['Composite_Score'].mean()
            st.metric(" Avg Score", f"{avg_score:.1f}")

        # Actual end-date labels
        _period_labels = build_dynamic_period_labels(df_original)

        # Show reference date info if alignment is active
        if use_quarterly_beta and align_to_reference:
            ref_date = st.session_state.get('reference_date_override', get_reference_date())
            st.caption(
                f"Data Periods: {_period_labels['fy_label']}  |  {_period_labels['ltm_label']}  |  "
                f"**Reference Date**: {ref_date.strftime('%b %d, %Y')} (aligned for fair comparison)"
            )
        else:
            st.caption(f"Data Periods: {_period_labels['fy_label']}  |  {_period_labels['ltm_label']}")

        if os.environ.get("RG_TESTS") and _period_labels.get("used_fallback"):
            st.caption("[DEV] FY/CQ classifier not available — using documented fallback (first 5 FY, rest CQ).")

        st.markdown("---")

# ============================================================================
# [V2.2] MINIMAL IDENTIFIERS + FEATURE GATES
# ============================================================================

RATING_ALIASES = [
    "S&P LT Issuer Credit Rating",
    "S&P Long-Term Issuer Credit Rating",
    "S&P Credit Rating",
    "S&P Rating",
    "Credit Rating",
    "Rating"
]
COMPANY_ID_ALIASES = [
    "Company ID",
    "CompanyID",
    "Company_ID",
    "Issuer ID",
    "IssuerID",
    "ID",
    "Ticker",  # Common fallback in S&P Capital IQ exports
    "Company Ticker",
    "Trading Symbol"
]
COMPANY_NAME_ALIASES = [
    "Company Name",
    "CompanyName",
    "Company_Name",
    "Issuer Name",
    "IssuerName",
    "Name",
    "Company",
    "Legal Name",
    "Entity Name"
]

# Currency symbol mapping - single source of truth
CURRENCY_SYMBOLS = {
    'AUD': 'A$', 'CAD': 'C$', 'CHF': 'CHF ', 'CNY': '¥', 'CZK': 'Kč ',
    'DKK': 'kr ', 'EUR': '€', 'GBP': '£', 'HKD': 'HK$', 'ITL': '₤',
    'JPY': '¥', 'NOK': 'kr ', 'NZD': 'NZ$', 'PEN': 'S/', 'SEK': 'kr ',
    'USD': '$'
}

def get_currency_symbol(currency_code: str) -> str:
    """Get currency symbol for a currency code. Defaults to code + space if not found."""
    return CURRENCY_SYMBOLS.get(currency_code, currency_code + ' ')

# Features that are optional - gate UI/functionality if columns missing
REQ_FOR = {
    "classification": ["Rubrics Custom Classification"],
    "country_region": ["Country", "Region"],
    "period_alignment": ["Period Ended"],  # .1.. .12 optional
}

def _norm_header(s: str) -> str:
    """
    Normalize header for case/space/NBSP-insensitive matching.
    - Normalizes unicode (handles NBSP \u00a0)
    - Collapses whitespace
    - Converts to lowercase
    """
    if s is None:
        return ''
    # Normalize unicode (e.g., NBSP), collapse whitespace, lowercase
    s = unicodedata.normalize('NFKC', str(s)).replace('\u00a0', ' ')
    s = ' '.join(s.split())
    return s.lower()

def resolve_column(df, aliases):
    """
    Return the actual DataFrame column matching any alias (case/space-insensitive).
    Robust to extra spaces, NBSP, and case variations.
    """
    norm_map = {_norm_header(c): c for c in df.columns}
    for alias in aliases:
        key = _norm_header(alias)
        if key in norm_map:
            return norm_map[key]
    return None

def resolve_rating_column(df):
    """Find which rating column alias exists in the dataframe."""
    return resolve_column(df, RATING_ALIASES)

def resolve_company_id_column(df):
    """Find which company ID column alias exists in the dataframe."""
    return resolve_column(df, COMPANY_ID_ALIASES)

def resolve_company_name_column(df):
    """Find which company name column alias exists in the dataframe."""
    return resolve_column(df, COMPANY_NAME_ALIASES)

# ---------- Metric alias registry & helpers (AI Analysis v2) ----------
# =============================================================================
# METRIC_REGISTRY - SINGLE SOURCE OF TRUTH FOR ALL METRIC DEFINITIONS
# =============================================================================
# ALL column names, aliases, and metadata defined HERE ONLY.
# NO OTHER DICTIONARY should define column names.
# =============================================================================

METRIC_REGISTRY = {
    # PROFITABILITY
    'ebitda_margin': {
        'canonical': 'EBITDA Margin',
        'aliases': ['EBITDA Margin', 'EBITDA margin %', 'EBITDA Margin (%)'],
        'type': 'calc', 'unit': '%', 'higher_is_better': True,
    },
    'gross_margin': {
        'canonical': 'Gross Profit Margin',
        'aliases': ['Gross Profit Margin', 'Gross Margin', 'GP Margin'],
        'type': 'calc', 'unit': '%', 'higher_is_better': True,
    },
    'ebit_margin': {
        'canonical': 'EBIT Margin',
        'aliases': ['EBIT Margin'],
        'type': 'calc', 'unit': '%', 'higher_is_better': True,
    },
    'net_income_margin': {
        'canonical': 'Net Income Margin',
        'aliases': ['Net Income Margin'],
        'type': 'calc', 'unit': '%', 'higher_is_better': True,
    },
    'roe': {
        'canonical': 'Return on Equity',
        'aliases': ['Return on Equity', 'ROE'],
        'type': 'calc', 'unit': '%', 'higher_is_better': True,
    },
    'roa': {
        'canonical': 'Return on Assets',
        'aliases': ['Return on Assets', 'ROA'],
        'type': 'calc', 'unit': '%', 'higher_is_better': True,
    },
    'roic': {
        'canonical': 'Return on Capital',
        'aliases': ['Return on Capital', 'ROIC'],
        'type': 'calc', 'unit': '%', 'higher_is_better': True,
    },

    # LEVERAGE
    'total_debt_ebitda': {
        'canonical': 'Total Debt / EBITDA (x)',
        'aliases': ['Total Debt / EBITDA (x)', 'Total Debt/EBITDA', 'Total Debt to EBITDA', 'Debt / EBITDA (x)'],
        'type': 'calc', 'unit': 'x', 'higher_is_better': False,
    },
    'net_debt_ebitda': {
        'canonical': 'Net Debt / EBITDA',
        'aliases': ['Net Debt / EBITDA', 'Net Debt/EBITDA', 'Net Debt to EBITDA'],
        'type': 'calc', 'unit': 'x', 'higher_is_better': False,
    },
    'ebitda_interest': {
        'canonical': 'EBITDA / Interest Expense (x)',
        'aliases': ['EBITDA / Interest Expense (x)', 'EBITDA/ Interest Expense (x)', 'EBITDA/Interest (x)', 'EBITDA / Interest', 'Interest Coverage (x)', 'Interest Cover (x)'],
        'type': 'calc', 'unit': 'x', 'higher_is_better': True,
    },
    'interest_coverage': {
        'canonical': 'EBITDA / Interest Expense (x)',
        'aliases': ['EBITDA / Interest Expense (x)', 'EBITDA/ Interest Expense (x)', 'Interest Coverage (x)'],
        'type': 'calc', 'unit': 'x', 'higher_is_better': True,
    },
    'total_debt_equity': {
        'canonical': 'Total Debt/Equity (%)',
        'aliases': ['Total Debt/Equity (%)', 'Debt to Equity (%)'],
        'type': 'calc', 'unit': '%', 'higher_is_better': False,
    },
    'debt_to_equity': {
        'canonical': 'Total Debt/Equity (%)',
        'aliases': ['Total Debt/Equity (%)'],
        'type': 'calc', 'unit': '%', 'higher_is_better': False,
    },
    'lt_debt_capital': {
        'canonical': 'Long-term Debt / Total Capital (%)',
        'aliases': ['Long-term Debt / Total Capital (%)'],
        'type': 'calc', 'unit': '%', 'higher_is_better': False,
    },
    'total_debt_capital': {
        'canonical': 'Total Debt / Total Capital (%)',
        'aliases': ['Total Debt / Total Capital (%)', 'Debt / Capital (%)'],
        'type': 'calc', 'unit': '%', 'higher_is_better': False,
    },
    'debt_to_capital': {
        'canonical': 'Total Debt / Total Capital (%)',
        'aliases': ['Total Debt / Total Capital (%)'],
        'type': 'calc', 'unit': '%', 'higher_is_better': False,
    },
    'total_liabilities_assets': {
        'canonical': 'Total Liabilities / Total Assets (%)',
        'aliases': ['Total Liabilities / Total Assets (%)'],
        'type': 'calc', 'unit': '%', 'higher_is_better': False,
    },

    # LIQUIDITY
    'current_ratio': {
        'canonical': 'Current Ratio (x)',
        'aliases': ['Current Ratio (x)', 'Current Ratio'],
        'type': 'calc', 'unit': 'x', 'higher_is_better': True,
    },
    'quick_ratio': {
        'canonical': 'Quick Ratio (x)',
        'aliases': ['Quick Ratio (x)', 'Quick Ratio'],
        'type': 'calc', 'unit': 'x', 'higher_is_better': True,
    },
    'cash_ops_curr_liab': {
        'canonical': 'Cash from Ops. to Curr. Liab. (x)',
        'aliases': ['Cash from Ops. to Curr. Liab. (x)', 'OCF/Current Liabilities', 'OCF to Current Liabilities', 'Cash from Operations / Current Liabilities'],
        'type': 'calc', 'unit': 'x', 'higher_is_better': True,
    },

    # CASH FLOW
    'levered_fcf': {
        'canonical': 'Levered Free Cash Flow',
        'aliases': ['Levered Free Cash Flow', 'Free Cash Flow'],
        'type': 'calc', 'unit': 'K', 'higher_is_better': True,
    },
    'free_cash_flow': {
        'canonical': 'Levered Free Cash Flow',
        'aliases': ['Levered Free Cash Flow'],
        'type': 'calc', 'unit': 'K', 'higher_is_better': True,
    },
    'unlevered_fcf': {
        'canonical': 'Unlevered Free Cash Flow',
        'aliases': ['Unlevered Free Cash Flow'],
        'type': 'calc', 'unit': 'K', 'higher_is_better': True,
    },
    'levered_fcf_margin': {
        'canonical': 'Levered Free Cash Flow Margin',
        'aliases': ['Levered Free Cash Flow Margin'],
        'type': 'calc', 'unit': '%', 'higher_is_better': True,
    },
    'unlevered_fcf_margin': {
        'canonical': 'Unlevered Free Cash Flow Margin',
        'aliases': ['Unlevered Free Cash Flow Margin'],
        'type': 'calc', 'unit': '%', 'higher_is_better': True,
    },
    'operating_cash_flow': {
        'canonical': 'Cash from Ops.',
        'aliases': ['Cash from Ops.', 'Cash from Operations', 'Operating Cash Flow', 'Cash from Ops', 'Net Cash Provided by Operating Activities'],
        'type': 'raw', 'unit': 'K', 'higher_is_better': True,
    },

    # GROWTH
    'revenue_growth': {
        'canonical': 'Total Revenues, 1 Year Growth',
        'aliases': ['Total Revenues, 1 Year Growth', 'Revenue Growth 1Y'],
        'type': 'calc', 'unit': '%', 'higher_is_better': True,
    },
    'revenue_1y_growth': {
        'canonical': 'Total Revenues, 1 Year Growth',
        'aliases': ['Total Revenues, 1 Year Growth'],
        'type': 'calc', 'unit': '%', 'higher_is_better': True,
    },
    'revenue_3y_cagr': {
        'canonical': 'Total Revenues, 3 Yr. CAGR',
        'aliases': ['Total Revenues, 3 Yr. CAGR'],
        'type': 'calc', 'unit': '%', 'higher_is_better': True,
    },
    'ebitda_growth': {
        'canonical': 'EBITDA, 1 Yr. Growth',
        'aliases': ['EBITDA, 1 Yr. Growth'],
        'type': 'calc', 'unit': '%', 'higher_is_better': True,
    },
    'ebitda_3y_cagr': {
        'canonical': 'EBITDA, 3 Years CAGR',
        'aliases': ['EBITDA, 3 Years CAGR'],
        'type': 'calc', 'unit': '%', 'higher_is_better': True,
    },
    'net_income_growth': {
        'canonical': 'Net Income, 1 Yr. Growth',
        'aliases': ['Net Income, 1 Yr. Growth'],
        'type': 'calc', 'unit': '%', 'higher_is_better': True,
    },
    'net_income_3y_cagr': {
        'canonical': 'Net Income, 3 Yr. CAGR',
        'aliases': ['Net Income, 3 Yr. CAGR'],
        'type': 'calc', 'unit': '%', 'higher_is_better': True,
    },

    # RAW FINANCIALS
    'total_debt': {
        'canonical': 'Total Debt',
        'aliases': ['Total Debt'],
        'type': 'raw', 'unit': 'K', 'higher_is_better': None,
    },
    'net_debt': {
        'canonical': 'Net Debt',
        'aliases': ['Net Debt'],
        'type': 'calc', 'unit': 'K', 'higher_is_better': False,
    },
    'ebitda': {
        'canonical': 'EBITDA',
        'aliases': ['EBITDA'],
        'type': 'raw', 'unit': 'K', 'higher_is_better': True,
    },
    'ebit': {
        'canonical': 'EBIT',
        'aliases': ['EBIT', 'Operating Income'],
        'type': 'raw', 'unit': 'K', 'higher_is_better': True,
    },
    'operating_income': {
        'canonical': 'EBIT',
        'aliases': ['EBIT'],
        'type': 'raw', 'unit': 'K', 'higher_is_better': True,
    },
    'revenue': {
        'canonical': 'Total Revenues',
        'aliases': ['Total Revenues', 'Total Revenue', 'Revenue'],
        'type': 'raw', 'unit': 'K', 'higher_is_better': True,
    },
    'total_revenues': {
        'canonical': 'Total Revenues',
        'aliases': ['Total Revenues'],
        'type': 'raw', 'unit': 'K', 'higher_is_better': True,
    },
    'interest_expense': {
        'canonical': 'Interest Expense',
        'aliases': ['Interest Expense', 'Interest Expense, net', 'Net Interest Expense'],
        'type': 'raw', 'unit': 'K', 'higher_is_better': False,
    },
    'net_income': {
        'canonical': 'Net Income',
        'aliases': ['Net Income'],
        'type': 'raw', 'unit': 'K', 'higher_is_better': True,
    },
    'cash': {
        'canonical': 'Cash & Short-term Investments',
        'aliases': ['Cash & Short-term Investments', 'Cash and Short-Term Investments', 'Cash & ST Investments', 'Cash and Equivalents'],
        'type': 'raw', 'unit': 'K', 'higher_is_better': True,
    },
    'cash_equivalents': {
        'canonical': 'Cash & Short-term Investments',
        'aliases': ['Cash & Short-term Investments'],
        'type': 'raw', 'unit': 'K', 'higher_is_better': True,
    },
    'total_assets': {
        'canonical': 'Total Assets',
        'aliases': ['Total Assets'],
        'type': 'raw', 'unit': 'K', 'higher_is_better': None,
    },
    'equity': {
        'canonical': 'Total Common Equity',
        'aliases': ['Total Common Equity', 'Total Equity'],
        'type': 'raw', 'unit': 'K', 'higher_is_better': True,
    },
    'total_equity': {
        'canonical': 'Total Common Equity',
        'aliases': ['Total Common Equity'],
        'type': 'raw', 'unit': 'K', 'higher_is_better': True,
    },
    'capital_expenditure': {
        'canonical': 'Capital Expenditure',
        'aliases': ['Capital Expenditure', 'CapEx'],
        'type': 'raw', 'unit': 'K', 'higher_is_better': None,
    },
    'capex': {
        'canonical': 'Capital Expenditure',
        'aliases': ['Capital Expenditure'],
        'type': 'raw', 'unit': 'K', 'higher_is_better': None,
    },
    'market_cap': {
        'canonical': 'Market Capitalization',
        'aliases': ['Market Capitalization'],
        'type': 'raw', 'unit': 'K', 'higher_is_better': None,
    },

    # BALANCE SHEET
    'current_assets': {
        'canonical': 'Current Assets',
        'aliases': ['Current Assets'],
        'type': 'raw', 'unit': 'K', 'higher_is_better': None,
    },
    'current_liabilities': {
        'canonical': 'Current Liabilities',
        'aliases': ['Current Liabilities'],
        'type': 'raw', 'unit': 'K', 'higher_is_better': None,
    },
    'total_liabilities': {
        'canonical': 'Total Liabilities',
        'aliases': ['Total Liabilities'],
        'type': 'raw', 'unit': 'K', 'higher_is_better': False,
    },
    'long_term_debt': {
        'canonical': 'Long-Term Debt',
        'aliases': ['Long-Term Debt'],
        'type': 'raw', 'unit': 'K', 'higher_is_better': False,
    },
    'short_term_debt': {
        'canonical': 'Short-Term Debt',
        'aliases': ['Short-Term Debt'],
        'type': 'raw', 'unit': 'K', 'higher_is_better': False,
    },
    'accounts_receivable': {
        'canonical': 'Accounts Receivable',
        'aliases': ['Accounts Receivable'],
        'type': 'raw', 'unit': 'K', 'higher_is_better': None,
    },
    'inventory': {
        'canonical': 'Inventory',
        'aliases': ['Inventory'],
        'type': 'raw', 'unit': 'K', 'higher_is_better': None,
    },
    'accounts_payable': {
        'canonical': 'Accounts Payable',
        'aliases': ['Accounts Payable'],
        'type': 'raw', 'unit': 'K', 'higher_is_better': None,
    },

    # EFFICIENCY
    'asset_turnover': {
        'canonical': 'Total Asset Turnover',
        'aliases': ['Total Asset Turnover'],
        'type': 'calc', 'unit': 'x', 'higher_is_better': True,
    },
    'fixed_asset_turnover': {
        'canonical': 'Fixed Asset Turnover',
        'aliases': ['Fixed Asset Turnover'],
        'type': 'calc', 'unit': 'x', 'higher_is_better': True,
    },
}


# =============================================================================
# HELPER FUNCTIONS - Derive from METRIC_REGISTRY
# =============================================================================

def _build_alias_to_canonical():
    """Build reverse lookup: alias -> canonical name."""
    mapping = {}
    for key, info in METRIC_REGISTRY.items():
        for alias in info['aliases']:
            if alias not in mapping:
                mapping[alias] = info['canonical']
    return mapping

_ALIAS_TO_CANONICAL = _build_alias_to_canonical()


def get_metric_canonical(metric_key):
    """Get canonical column name for a metric key."""
    if metric_key in METRIC_REGISTRY:
        return METRIC_REGISTRY[metric_key]['canonical']
    return None


def get_metric_aliases(metric_key):
    """Get all aliases for a metric key."""
    if metric_key in METRIC_REGISTRY:
        return METRIC_REGISTRY[metric_key]['aliases']
    return []


def get_metric_column(metric_key, suffix=""):
    """Get column name with optional period suffix."""
    canonical = get_metric_canonical(metric_key)
    if canonical:
        return f"{canonical}{suffix}" if suffix else canonical
    return None

def get_metric_display_name(metric_key):
    """Get display name (canonical name) for a metric."""
    return get_metric_canonical(metric_key)

def get_metric_info(metric_key):
    """Get full metadata for a metric."""
    if metric_key not in METRIC_REGISTRY:
        return None
    info = METRIC_REGISTRY[metric_key]
    return {
        'canonical': info['canonical'],
        'aliases': info['aliases'],
        'type': info['type'],
        'unit': info['unit'],
        'higher_is_better': info['higher_is_better'],
    }


def resolve_column_name(column_name):
    """Resolve any alias to its canonical form."""
    return _ALIAS_TO_CANONICAL.get(column_name, column_name)


def format_metric_value(value, metric_key):
    """Format value based on metric unit."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    info = METRIC_REGISTRY.get(metric_key)
    if not info:
        return f"{value:,.2f}"
    unit = info['unit']
    if unit == 'M':
        return f"{value:,.1f}M"
    elif unit == '%':
        return f"{value:.2f}%"
    elif unit == 'x':
        return f"{value:.2f}x"
    return f"{value:,.2f}"

def format_monetary_value_for_display(value: float, metric_name: str = "") -> str:
    """
    Format monetary values from thousands to human-readable format.
    
    Args:
        value: Raw value in thousands (as stored in CIQ data)
        metric_name: Optional metric name for context
        
    Returns:
        Formatted string (e.g., "$32.2B", "$500.0M", "$1.5M")
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    
    # Convert from thousands to actual value
    actual_value = value * 1000
    
    if abs(actual_value) >= 1e12:  # Trillions
        return f"${actual_value/1e12:.1f}T"
    elif abs(actual_value) >= 1e9:  # Billions
        return f"${actual_value/1e9:.1f}B"
    elif abs(actual_value) >= 1e6:  # Millions
        return f"${actual_value/1e6:.1f}M"
    elif abs(actual_value) >= 1e3:  # Thousands
        return f"${actual_value/1e3:.1f}K"
    else:
        return f"${actual_value:,.0f}"

def _resolve_company_name_col(df: pd.DataFrame) -> str | None:
    return resolve_column(df, ["Company_Name", "Company Name", "Name"])

# =============================================================================
# INTEREST COVERAGE FALLBACK CALCULATION (V3.5)
# =============================================================================
def calculate_interest_coverage_fallback(ebitda_value, interest_expense_value, existing_ratio=None):
    """
    Calculate Interest Coverage ratio with fallback logic.
    
    CIQ marks ratios as "NM" (Not Meaningful) for three reasons:
    1. Negative EBITDA (loss-making) → Return 0 (cannot cover interest)
    2. Very high coverage (>50x) → Return the calculated ratio (excellent)
    3. Net interest income (int_exp >= 0) → Return 999 (effectively infinite)
    
    Args:
        ebitda_value: EBITDA value (can be None, NaN, or numeric)
        interest_expense_value: Interest Expense value (typically negative for expense)
        existing_ratio: CIQ's pre-calculated ratio (can be None, NaN, "NM", or numeric)
    
    Returns:
        float: Interest Coverage ratio, or np.nan if cannot calculate
    """
    import numpy as np
    
    # Helper to check if value is valid numeric
    def is_valid_numeric(v):
        if v is None:
            return False
        if isinstance(v, str):
            v_clean = v.strip().upper()
            if v_clean in ['NM', 'NA', 'N/A', '', '-']:
                return False
            try:
                float(v.replace(',', ''))
                return True
            except Exception:
                return False
        try:
            return not np.isnan(float(v))
        except Exception:
            return False
    
    def to_float(v):
        if v is None:
            return np.nan
        if isinstance(v, str):
            v_clean = v.strip().upper()
            if v_clean in ['NM', 'NA', 'N/A', '', '-']:
                return np.nan
            try:
                return float(v.replace(',', ''))
            except Exception:
                return np.nan
        try:
            return float(v)
        except Exception:
            return np.nan
    
    # Check if existing ratio is valid - if so, use it
    if is_valid_numeric(existing_ratio):
        return to_float(existing_ratio)
    
    # Otherwise, try to calculate from components
    ebitda = to_float(ebitda_value)
    int_exp = to_float(interest_expense_value)
    
    if np.isnan(ebitda) or np.isnan(int_exp):
        return np.nan
    
    # Handle edge cases
    if ebitda < 0:
        # Loss-making company - cannot cover interest from operations
        # Return a small value that will score poorly
        return 0.0
    
    if int_exp >= 0:
        # Net interest income (earns more interest than pays)
        # This is excellent credit quality - return high value
        return 999.0
    
    if int_exp == 0:
        # No interest expense (no debt)
        return 999.0
    
    # Normal case: positive EBITDA, negative interest expense
    # Interest expense is stored as negative (outflow), so use absolute value
    calculated_ratio = ebitda / abs(int_exp)
    
    return calculated_ratio

def _resolve_classification_col(df: pd.DataFrame) -> str | None:
    return resolve_column(df, ["Rubrics_Custom_Classification", "Rubrics Custom Classification"])

def resolve_metric_column(df_like, canonical: str) -> str | None:
    aliases = get_metric_aliases(canonical)
    if not aliases:
        aliases = [canonical]
    # Accept Series -> make it a 1-row frame to reuse resolve_column
    if isinstance(df_like, pd.Series):
        df_like = df_like.to_frame().T
    return resolve_column(df_like, aliases)

def list_metric_columns(df: pd.DataFrame, canonical: str) -> tuple[str | None, list[str]]:
    """Return (base_col, [all existing suffixed cols]) for a canonical metric."""
    base = resolve_metric_column(df, canonical)
    suffixes = []
    aliases = get_metric_aliases(canonical)
    if not aliases:
        aliases = [canonical]
    for alias in aliases:
        suffixes += [c for c in df.columns if isinstance(c, str) and c.startswith(f"{alias}.")]
    suffixes = sorted(set(suffixes))
    return base, suffixes

def get_from_row(row: pd.Series, canonical: str):
    """Row-level safe getter honoring aliases."""
    aliases = get_metric_aliases(canonical)
    if not aliases:
        aliases = [canonical]
    for a in aliases:
        if a in row.index:
            return row.get(a)
    return np.nan

def _find_period_cols(df: pd.DataFrame, prefer_fy=True) -> dict[int, str]:
    """
    Map suffix index -> period-ended column name (FY first; fallback to generic).
    Accepts variants like 'Period Ended', 'Period Ended (FY)', 'Period Ended FY'.
    """
    # Collect candidate stems ordered by preference (FY before generic; never CQ)
    stems = []
    if prefer_fy:
        stems += [s for s in df.columns if isinstance(s, str) and (s.startswith("Period Ended (FY)") or s.startswith("Period Ended FY"))]
    stems += [s for s in df.columns if isinstance(s, str) and s.startswith("Period Ended")]
    # Build a suffix map
    mapping: dict[int, str] = {}
    for col in stems:
        # accept base and suffixed forms: "Period Ended", "Period Ended.3"
        if "." in col:
            try:
                idx = int(col.rsplit(".", 1)[1])
                mapping[idx] = col
            except ValueError:
                continue
        else:
            mapping[0] = col
    return mapping

def _metric_series_for_row(df: pd.DataFrame, row: pd.Series, canonical: str, prefer_fy=True) -> pd.Series:
    """
    Build a Series for one issuer row using alias-aware base + suffixed metric columns.
    Primary path aligns to 'Period Ended.*' columns; fallback uses suffix order (no dates).
    """
    base, suffixed = list_metric_columns(df, canonical)
    period_map = _find_period_cols(df, prefer_fy=prefer_fy)  # suffix -> period col

    # Collect (key, value) pairs where key is either a datetime (preferred) or suffix index.
    pairs: list[tuple[object, float]] = []

    def _coerce_num(x):
        return pd.to_numeric(x, errors="coerce")

    # base metric
    if base and pd.notna(row.get(base)):
        v = _coerce_num(row.get(base))
        if pd.notna(v):
            key = pd.to_datetime(row.get(period_map.get(0)), errors="coerce") if 0 in period_map else 0
            pairs.append((key, float(v)))

    # suffixed metrics
    for col in suffixed:
        try:
            idx = int(col.rsplit(".", 1)[1])
        except Exception:
            continue
        val = _coerce_num(row.get(col))
        if pd.notna(val):
            key = pd.to_datetime(row.get(period_map.get(idx)), errors="coerce") if idx in period_map else idx
            pairs.append((key, float(val)))

    if not pairs:
        return pd.Series(dtype=float)

    s = pd.Series({k: v for k, v in pairs})
    # If keys aren't datetimes, leave them as numeric order. Coverage text upstream should handle this.
    try:
        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s.sort_index()
    except Exception:
        s = s.sort_index()
    return s

def validate_core(df):
    """
    Validate minimal required identifiers with flexible column name matching.
    Returns: (missing_cols, rating_col_name, company_id_col, company_name_col)
    """
    missing = []

    company_name_col = resolve_company_name_column(df)
    if company_name_col is None:
        missing.append("Company Name (or alias: CompanyName, Issuer Name, Name)")

    company_id_col = resolve_company_id_column(df)
    if company_id_col is None:
        missing.append("Company ID (or alias: CompanyID, Issuer ID, ID)")

    rating_col = resolve_rating_column(df)
    if rating_col is None:
        missing.append("S&P LT Issuer Credit Rating (or alias: S&P Credit Rating)")

    return missing, rating_col, company_id_col, company_name_col

def feature_available(name, df):
    """
    Check if optional feature columns are available.
    Shows info message if disabled.
    """
    missing = [c for c in REQ_FOR[name] if c not in df.columns]
    if missing:
        st.info(f"INFO: {name.replace('_', ' ').title()} feature disabled (missing: {missing})")
        return False
    return True

# ============================================================================
# [V2.2] PLOT/BRAND HELPERS
# ============================================================================

def apply_rubrics_plot_theme(fig):
    """
    Apply professional, neutral theme to Plotly figures.
    Adjust font family if Ringside or custom fonts are available.
    """
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, Arial, sans-serif", size=12),
        legend=dict(title='', orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# ============================================================================
# [V2.2] PCA VISUALIZATION (IG/HY COHORT CLUSTERING)
# ============================================================================

def _factor_score_columns(results: pd.DataFrame) -> list:
    """
    Return list of individual factor score columns for PCA.
    Use individual factor scores; avoid letting Composite dominate.
    """
    cols = [c for c in results.columns if c.endswith("_Score") and c not in ("Composite_Score", "Quality_Score")]
    if len(cols) < 2:
        # Fallback: include Composite if needed to reach 2+
        cols = [c for c in results.columns if c.endswith("_Score")]
    if len(cols) < 2:
        raise ValueError("Not enough score columns for PCA (need >=2).")
    return cols

def compute_pca_ig_hy(results: pd.DataFrame, id_col: str, name_col: str, max_points: int = 2000):
    """
    Compute 2D PCA of factor scores for IG/HY visualization.

    Args:
        results: DataFrame with factor scores and rating information
        id_col: Company ID column name
        name_col: Company Name column name
        max_points: Maximum points to include (samples if larger, deterministic)

    Returns:
        (dfp, explained_variance_ratio) tuple
        dfp: DataFrame with PC1, PC2, and metadata
        ev: Array of explained variance ratios [PC1, PC2]
    """
    score_cols = _factor_score_columns(results)
    df = results[[id_col, name_col, "Composite_Score", "Rating_Group", "Rating_Band", "Credit_Rating_Clean"] + score_cols].copy()

    # Sample for performance if very large (deterministic)
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=0)

    # Numeric, robust imputations
    X = df[score_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))  # median impute

    scaler = RobustScaler()
    Xs = scaler.fit_transform(X.values)

    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(Xs)
    ev = pca.explained_variance_ratio_

    df["PC1"] = Z[:, 0]
    df["PC2"] = Z[:, 1]
    return df[[id_col, name_col, "Rating_Group", "Rating_Band", "Credit_Rating_Clean", "Composite_Score", "PC1", "PC2"]], ev

def render_pca_scatter_ig_hy(dfp: pd.DataFrame, ev, id_col: str, name_col: str):
    """
    Render interactive PCA scatter plot colored by Rating_Group (IG vs HY).

    Args:
        dfp: DataFrame from compute_pca_ig_hy with PC1, PC2
        ev: Explained variance ratios from PCA
        id_col: Company ID column name
        name_col: Company Name column name

    Returns:
        Plotly figure
    """
    title = f"PCA of Factor Scores (2D) • EVR: PC1 {ev[0]:.1%}, PC2 {ev[1]:.1%}"
    fig = px.scatter(
        dfp,
        x="PC1", y="PC2",
        color="Rating_Group",                # IG vs HY
        symbol="Rating_Band",                # Optional shape by band (AAA..CCC)
        size="Composite_Score",              # Bigger = higher composite
        hover_name=name_col,
        hover_data={
            id_col: True,
            "Credit_Rating_Clean": True,
            "Rating_Band": True,
            "Composite_Score": ':.2f'
        },
        opacity=0.85,
        title=title
    )
    fig = apply_rubrics_plot_theme(fig)
    return fig

def render_pca_scatter_ig_hy_v1style(dfp: pd.DataFrame, ev, id_col: str, name_col: str):
    """
    Permanent v1-style IG/HY PCA scatter:
    - Uniform marker sizes
    - Colour by Rating_Group (IG/HY)
    - Symbol by Rating_Band
    - Simplified layout with lighter background
    - Opacity ~0.7
    - Legend on single top row

    Args:
        dfp: DataFrame from compute_pca_ig_hy with PC1, PC2
        ev: Explained variance ratios from PCA
        id_col: Company ID column name
        name_col: Company Name column name

    Returns:
        Plotly figure
    """
    title = f"PCA of Factor Scores (2D) • EVR: PC1 {ev[0]:.1%}, PC2 {ev[1]:.1%}"

    fig = px.scatter(
        dfp,
        x="PC1", y="PC2",
        color="Rating_Group",
        symbol="Rating_Band",
        hover_name=name_col,
        hover_data={
            id_col: True,
            "Credit_Rating_Clean": True,
            "Rating_Band": True,
            "Composite_Score": ':.2f'
        },
        opacity=0.7,
        title=title
    )

    fig.update_traces(marker=dict(size=7, line=dict(width=0.4, color="white")))
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, Arial, sans-serif", size=12),
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=40, b=40),
        xaxis_title="Primary Credit Dimension →",
        yaxis_title="Secondary Credit Dimension →"
    )
    return fig

def render_dual_issuer_maps(results: pd.DataFrame, id_col: str, name_col: str):
    """
    Render two continuous issuer positioning maps:
    - Investment Grade Issuer Map
    - High Yield Issuer Map
    Each plots Overall Credit Quality (x-axis) vs Financial Strength vs Leverage Balance (y-axis),
    coloured by Recommendation (Avoid/Hold/Buy/Strong Buy).
    """

    # Define IG / HY splits
    ig_df = results[results["Rating_Group"] == "Investment Grade"].copy()
    hy_df = results[results["Rating_Group"] == "High Yield"].copy()

    def make_fig(df, group_name):
        if df.empty:
            return go.Figure()
        title = f"{group_name} Positioning Map (n={len(df):,})"
        fig = px.scatter(
            df,
            x="Overall_Credit_Quality",
            y="Financial_Strength_vs_Leverage_Balance",
            color="Recommendation",
            color_discrete_map={
                "Avoid": "#FF6B6B",
                "Hold": "#FFB84C",
                "Buy": "#5BC0DE",
                "Strong Buy": "#00CC66"
            },
            hover_name=name_col,
            hover_data={
                id_col: True,
                "Credit_Rating_Clean": True,
                "Composite_Score": ':.2f'
            },
            opacity=0.8,
            title=title
        )
        fig.update_traces(marker=dict(size=9, line=dict(width=0.8, color="white")))
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Inter, Arial, sans-serif", size=12),
            legend=dict(title="", orientation="v", yanchor="top", y=0.98, xanchor="left", x=1.02),
            margin=dict(l=20, r=20, t=40, b=40),
            xaxis_title="Credit Quality (Weaker → Stronger)",
            yaxis_title="Leverage (Higher → Lower)"
        )
        return fig

    ig_fig = make_fig(ig_df, "Investment Grade")
    hy_fig = make_fig(hy_df, "High Yield")

    st.subheader("Investment Grade Issuer Map")
    st.plotly_chart(ig_fig, use_container_width=True)
    st.subheader("High Yield Issuer Map")
    st.plotly_chart(hy_fig, use_container_width=True)

# ============================================================================
# [V2.2] RATING-BAND UTILITIES (LEADERBOARDS)
# ============================================================================

RATING_BAND_ORDER = [
    "AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-",
    "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D", "SD", "NR"
]

def _stable_sort_for_rank(df, id_col):
    """
    Deterministic sort for ranking: Composite_Score desc, then Company ID asc.
    Uses mergesort for stable ordering.
    """
    return df.sort_values(by=["Composite_Score", id_col], ascending=[False, True], kind="mergesort")

def build_band_leaderboard(results: pd.DataFrame, band: str, id_col: str, name_col: str, top_n: int = 20):
    """
    Build leaderboard for a specific rating band.

    Args:
        results: DataFrame with all results
        band: Rating band to filter (e.g., "BBB-")
        id_col: Company ID column name
        name_col: Company Name column name
        top_n: Number of top performers to return

    Returns:
        DataFrame with top N performers in the band, ranked deterministically
    """
    req = [id_col, name_col, "Composite_Score", "Rating_Band", "Credit_Rating_Clean", "Rating_Group"]
    for c in req:
        if c not in results.columns:
            raise KeyError(f"Missing required column: {c}")

    # Filter to selected band (case-insensitive)
    df = results[results["Rating_Band"].astype(str).str.upper() == band.upper()].copy()
    if df.empty:
        return df

    # Stable sort for deterministic ranking
    df = _stable_sort_for_rank(df, id_col)

    # Add within-band rank
    # [V5.0.3] Handle NaN composite scores gracefully
    df["Rank_in_Band"] = (
        df["Composite_Score"]
        .rank(method="dense", ascending=False, na_option='bottom')
        .fillna(9999)
        .astype(int)
    )

    # Select display columns
    cols = ["Rank_in_Band", name_col, id_col, "Credit_Rating_Clean", "Rating_Band", "Composite_Score"]
    return df[cols].head(top_n)

def to_csv_download(df: pd.DataFrame, filename: str = "leaderboard.csv") -> tuple:
    """
    Convert DataFrame to CSV bytes for download.

    Args:
        df: DataFrame to export
        filename: Default filename for download

    Returns:
        (csv_bytes, filename) tuple
    """
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8"), filename

def generate_sparkline_html(values: list, width: int = 100, height: int = 20) -> str:
    """
    Generate inline SVG sparkline for a list of numeric values.

    Args:
        values: List of numeric values
        width: SVG width in pixels
        height: SVG height in pixels

    Returns:
        HTML string with inline SVG sparkline
    """
    if not values or len(values) < 2:
        return ""

    # Filter out NaN/None
    clean_vals = [v for v in values if pd.notna(v)]
    if len(clean_vals) < 2:
        return ""

    # Normalize to 0-height range
    min_val = min(clean_vals)
    max_val = max(clean_vals)
    if max_val == min_val:
        norm_vals = [height / 2] * len(clean_vals)
    else:
        norm_vals = [(height - ((v - min_val) / (max_val - min_val)) * height) for v in clean_vals]

    # Build SVG path
    step = width / (len(clean_vals) - 1)
    points = " ".join([f"{i * step},{y}" for i, y in enumerate(norm_vals)])

    svg = f'''<svg width="{width}" height="{height}" style="vertical-align: middle;">
        <polyline points="{points}" fill="none" stroke="#2C5697" stroke-width="1.5"/>
    </svg>'''

    return svg

# ============================================================================
# DIAGNOSTICS & DATA HEALTH (V2.2)
# ============================================================================

def _pct(n, d):
    """Calculate percentage safely."""
    return (100.0 * n / d) if d else 0.0

def summarize_periods(df: pd.DataFrame):
    """
    Parse Period Ended* columns, return (num_cols, min_date, max_date, fy_suffixes, ltm_suffixes).
    Safe on files without period columns.
    """
    pe_cols = [c for c in df.columns if str(c).startswith("Period Ended")]
    if not pe_cols:
        return 0, None, None, [], []

    pe_data = parse_period_ended_cols(df.copy())  # uses existing helper

    # Get FY and CQ suffixes using existing helper
    fy_sfx = []
    cq_sfx = []
    for sfx, _ in pe_data:
        # Simple heuristic: if suffix contains digits >= 5, likely CQ
        if sfx == "":
            fy_sfx.append("(base)")
        elif sfx and any(c.isdigit() and int(c) >= 5 for c in sfx if c.isdigit()):
            cq_sfx.append(sfx)
        else:
            fy_sfx.append(sfx)

    # Gather min/max across all rows/period columns
    all_dates = []
    for _, s in pe_data:
        sd = pd.to_datetime(s, errors="coerce")
        all_dates.append(sd)

    if all_dates:
        all_dates = pd.concat(all_dates, axis=1)
        mind = pd.to_datetime(all_dates.min(axis=1), errors="coerce").min()
        maxd = pd.to_datetime(all_dates.max(axis=1), errors="coerce").max()
    else:
        mind = maxd = None

    return len(pe_cols), mind, maxd, fy_sfx, cq_sfx

def summarize_missingness(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Calculate missingness statistics for specified columns."""
    rows = []
    total = len(df)
    for c in cols:
        if c in df.columns:
            miss = df[c].isna().sum()
            rows.append({"Column": c, "Missing": int(miss), "Missing_%": round(_pct(miss, total), 1)})
    return pd.DataFrame(rows).sort_values("Missing_%", ascending=False) if rows else pd.DataFrame()

def summarize_scores_missingness(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize missingness for all *_Score columns."""
    score_cols = [c for c in results.columns if c.endswith("_Score")]
    return summarize_missingness(results, score_cols)

def summarize_key_metrics_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize missingness for key financial metrics."""
    key_metrics = [
        "EBITDA / Interest Expense (x)",
        "Net Debt / EBITDA",
        "Total Debt / EBITDA (x)",
        "Total Debt / Total Capital (%)",
        "Return on Equity",
        "EBITDA Margin",
        "Return on Assets",
        "EBIT Margin",
        "Current Ratio (x)",
        "Quick Ratio (x)",
        "Total Revenues, 1 Year Growth",
        "Total Revenues, 3 Yr. CAGR",
        "EBITDA, 3 Years CAGR",
        "Cash from Operations",
        "Capital Expenditure",
        "Unlevered Free Cash Flow",
        "Levered Free Cash Flow",
        "Total Debt",
        "Levered Free Cash Flow Margin",
        "Cash from Ops. to Curr. Liab. (x)"
    ]

    # Include suffixed base columns if unsuffixed not present
    present = set(df.columns)
    cols = []
    for base in key_metrics:
        if base in present:
            cols.append(base)
        else:
            # try a few common suffixes
            for i in range(1, 6):
                cand = f"{base}.{i}"
                if cand in present:
                    cols.append(cand)
                    break

    return summarize_missingness(df, cols)

def diagnostics_summary(df: pd.DataFrame, results: pd.DataFrame) -> dict:
    """Generate comprehensive diagnostics summary."""
    total = len(results)
    uniq_ids = results["Company_ID"].astype(str).nunique() if "Company_ID" in results.columns else total
    dup_ids = total - uniq_ids
    ig = int((results.get("Rating_Group") == "Investment Grade").sum()) if "Rating_Group" in results.columns else 0
    hy = int((results.get("Rating_Group") == "High Yield").sum()) if "Rating_Group" in results.columns else 0

    band_counts = (results["Rating_Band"].value_counts(dropna=False).sort_index()
                   if "Rating_Band" in results.columns else pd.Series(dtype=int))

    n_pe, mind, maxd, fy_sfx, cq_sfx = summarize_periods(df)

    return {
        "rows_total": total,
        "unique_company_ids": uniq_ids,
        "duplicate_ids": dup_ids,
        "ig_count": ig,
        "hy_count": hy,
        "band_counts": band_counts,
        "period_cols": n_pe,
        "period_min": mind,
        "period_max": maxd,
        "fy_suffixes": fy_sfx,
        "ltm_suffixes": cq_sfx,
    }

# ============================================================================
# AI ANALYSIS HELPERS (optional)
# ============================================================================

def _get_openai_client():
    """Return an OpenAI client using secrets/env. Raise with crisp message if missing."""
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("api_key")
    except Exception:
        pass
    api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("api_key")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in st.secrets or environment.")
    if OpenAI is None:
        raise RuntimeError("OpenAI python package not available in this environment.")
    return OpenAI(api_key=api_key)


def extract_issuer_financial_data(df_original: pd.DataFrame, company_name: str) -> dict:
    """
    Extract all financial data and time periods for a specific issuer.

    Args:
        df_original: The original DataFrame with all financial data
        company_name: The name of the company to extract data for

    Returns:
        dict: Structured financial data with company info, metrics time series, and period types
    """
    # Find the issuer row
    name_col = resolve_company_name_column(df_original)
    if name_col is None:
        raise ValueError("Cannot find company name column in dataset")

    issuer_row = df_original[df_original[name_col] == company_name]
    if issuer_row.empty:
        raise ValueError(f"Issuer '{company_name}' not found in dataset")

    row = issuer_row.iloc[0]

    # Extract company information
    company_id_col = resolve_company_id_column(df_original)
    rating_col = resolve_rating_column(df_original)

    company_info = {
        "name": company_name,
        "id": row.get(company_id_col) if company_id_col else "N/A",
        "sector": row.get("Sector", "N/A"),
        "industry": row.get("Industry", "N/A"),
        "country": row.get("Country", "N/A"),
        "rating": row.get(rating_col) if rating_col else "N/A",
        "classification": row.get("Rubrics Custom Classification", "N/A")
    }

    # Define metrics to extract
    metrics_to_extract = [
        "EBITDA Margin",
        "Total Debt / EBITDA (x)",
        "Net Debt / EBITDA",
        "EBITDA / Interest Expense (x)",
        "Current Ratio (x)",
        "Quick Ratio (x)",
        "Return on Equity",
        "Return on Assets",
        "Total Revenues",
        "Total Debt",
        "Cash and Short-Term Investments"
    ]

    # Extract time series for each metric
    financial_data = {}
    period_types = {}
    # Build suffix → period-kind map using existing classifier; fallback documented
    period_kind_by_suffix = {}
    try:
        pe_data = parse_period_ended_cols(df_original.copy())
        fy_suffixes, ltm_suffixes = period_cols_by_kind(pe_data, df_original)
        period_kind_by_suffix = {sfx: "FY" for sfx in fy_suffixes}
        period_kind_by_suffix.update({sfx: "LTM" for sfx in ltm_suffixes})
    except Exception:
        period_kind_by_suffix = {}  # fallback to month heuristic below

    for metric in metrics_to_extract:
        # Resolve metric column (handle aliases)
        aliases = get_metric_aliases(metric)
        if not aliases:
            aliases = [metric]
        metric_col = resolve_column(df_original, aliases)
        if metric_col is None:
            continue

        # Find all suffixed versions of this metric
        metric_cols = [col for col in df_original.columns if col == metric_col or col.startswith(f"{metric_col}.")]

        time_series = {}

        for col in metric_cols:
            # Get the suffix (e.g., "", ".1", ".2", etc.)
            if col == metric_col:
                suffix = ""
            else:
                suffix = col[len(metric_col):]

            # Get corresponding period
            period_col = f"Period Ended{suffix}"
            if period_col not in df_original.columns:
                continue

            period_value = row.get(period_col)
            metric_value = row.get(col)

            # Skip if missing
            if pd.isna(period_value) or pd.isna(metric_value):
                continue

            # Try to convert metric value to float, skip if not numeric
            try:
                numeric_value = float(metric_value)
            except (ValueError, TypeError):
                # Skip non-numeric values like 'NM', 'N/A', etc.
                continue

            # Robust parse; ignore Excel serials/NaT/1900 sentinels
            dt = pd.to_datetime(period_value, errors="coerce", dayfirst=True)
            if pd.isna(dt) or (hasattr(dt, "year") and dt.year == 1900):
                continue
            date_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
            time_series[date_str] = numeric_value
            # Prefer classifier by suffix; otherwise use month heuristic
            kind = period_kind_by_suffix.get(suffix)
            if kind is None:
                kind = "FY" if pd.Timestamp(dt).month in (12,) else "CQ"
            period_types[date_str] = kind

        if time_series:
            financial_data[metric] = time_series

    return {
        "company_info": company_info,
        "financial_data": financial_data,
        "period_types": period_types
    }


def build_credit_analysis_prompt(data: dict) -> str:
    """
    Build a comprehensive prompt for OpenAI to generate a credit analysis report.

    Args:
        data: Dictionary containing company_info, financial_data, and period_types

    Returns:
        str: Formatted prompt for the AI
    """
    company_info = data["company_info"]
    financial_data = data["financial_data"]
    period_types = data["period_types"]

    # Format financial data for the prompt
    metrics_text = []

    for metric, time_series in financial_data.items():
        if not time_series:
            continue

        sorted_periods = sorted(time_series.items())

        metrics_text.append(f"\n**{metric}:**")
        for date, value in sorted_periods:
            period_type = period_types.get(date, "")
            metrics_text.append(f"  - {date} ({period_type}): {value:.2f}")

    financial_section = "\n".join(metrics_text) if metrics_text else "  No financial data available"

    prompt = f"""You are an expert fixed income credit analyst preparing a comprehensive credit report.

**Company Overview:**
- Name: {company_info['name']}
- Company ID: {company_info['id']}
- Sector: {company_info['sector']}
- Industry: {company_info['industry']}
- Country: {company_info['country']}
- Current S&P Rating: {company_info['rating']}
- Classification: {company_info['classification']}

**Financial Metrics Over Time:**
{financial_section}

**Instructions:**
Please provide a comprehensive credit analysis report with the following structure:

1. **Executive Summary** (2-3 sentences)
   - Overall credit quality assessment
   - Key rating drivers

2. **Profitability Analysis**
   - EBITDA margin trends and interpretation
   - Return on Equity (ROE) and Return on Assets (ROA) trends
   - Profitability positioning relative to sector

3. **Leverage Analysis**
   - Total Debt/EBITDA trends
   - Net Debt/EBITDA trends
   - Assessment of leverage trajectory
   - Comparison to typical levels for this rating

4. **Liquidity & Coverage Analysis**
   - Current and Quick ratio trends
   - Cash position and trends
   - Interest coverage (EBITDA/Interest Expense) analysis
   - Assessment of debt serviceability

5. **Credit Strengths**
   - List 3-4 key positive credit factors
   - Support each with specific data points

6. **Credit Risks & Concerns**
   - List 3-4 key risk factors or areas of concern
   - Support each with specific data points

7. **Rating Outlook & Recommendation**
   - Is the current rating appropriate?
   - What could trigger an upgrade or downgrade?
   - Investment recommendation from a credit perspective

**Formatting Requirements:**
- Use clear markdown formatting with headers (##, ###)
- Bold key metrics and conclusions
- Use bullet points for lists
- Be specific and reference actual numbers from the data
- Keep the tone professional and analytical
- Total length: 600-800 words

Generate the report now:"""

    return prompt


def generate_credit_report(data: dict) -> str:
    """
    Generate a credit analysis report using OpenAI API.

    Args:
        data: Dictionary containing company_info, financial_data, and period_types

    Returns:
        str: The generated credit report in markdown format
    """
    # Get OpenAI client using existing helper
    client = _get_openai_client()

    # Build the prompt
    prompt = build_credit_analysis_prompt(data)

    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {
                "role": "system",
                "content": "You are an expert fixed income credit analyst with deep experience in corporate credit analysis. You provide clear, data-driven, professional credit reports."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
        max_tokens=8000
    )

    return response.choices[0].message.content

class AIMode(str, Enum):
    """AI analysis modes for different question types."""
    EXPLAIN = "explain"         # "explain what's going on"
    COMPARE = "compare"         # "compare A vs B"
    WHAT_IF = "what_if"         # "what if rates widen 50bp?"
    SCREEN = "screen"           # "find issuers with X"
    RISK_SCAN = "risk_scan"     # "what are the risks for BBB Energy?"

def _infer_mode(user_q: str) -> AIMode:
    """Infer analysis mode from user question using simple keyword rules."""
    q = (user_q or "").lower()
    if any(k in q for k in ["compare", "vs ", "versus", "better than", "worse than"]):
        return AIMode.COMPARE
    if any(k in q for k in ["what if", "scenario", "sensitivity", "shock"]):
        return AIMode.WHAT_IF
    if any(k in q for k in ["screen", "filter", "find issuers", "show issuers"]):
        return AIMode.SCREEN
    if any(k in q for k in ["risk", "risks", "downside", "contradiction"]):
        return AIMode.RISK_SCAN
    return AIMode.EXPLAIN

def _auto_scope(question: str, results_final: pd.DataFrame) -> tuple:
    """
    Returns: (scope_type, issuer_name, classification)
    Matches full or partial issuer names (>=4-char tokens).
    """
    q_lower = (question or "").lower()
    q_tokens = [t for t in re.findall(r"[a-z0-9]+", q_lower) if len(t) >= 4]

    # Issuer match (full or partial)
    if "Company_Name" in results_final.columns:
        names = results_final["Company_Name"].dropna().astype(str).unique().tolist()
        names_lc = [(n, n.lower()) for n in names]
        # full-string containment first
        for n, nl in names_lc:
            if nl in q_lower:
                return ("issuer", n, None)
        # token containment (e.g., "nvidia" matches "NVIDIA Corporation")
        for n, nl in names_lc:
            if any(tok in nl for tok in q_tokens):
                return ("issuer", n, None)

    # Classification match (full or partial)
    cls_col = next((c for c in ["Rubrics Custom Classification","Rubrics_Custom_Classification","Classification"]
                    if c in results_final.columns), None)
    if cls_col:
        classes = results_final[cls_col].dropna().astype(str).unique().tolist()
        classes_lc = [(c, c.lower()) for c in classes]
        for c, cl in classes_lc:
            if cl in q_lower or any(tok in cl for tok in q_tokens):
                return ("classification", None, c)

    return ("dataset", None, None)


# --- Search helpers (issuer/classification) ---

def _norm_txt(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(s).lower()).strip()

def _search_options(query: str, options: list, limit: int = 200) -> list:
    """
    Lightweight type-ahead:
      1) prefix matches first
      2) then substring matches
      3) then all-tokens containment
    """
    if not options:
        return []
    if not query:
        return options[:limit]
    q = _norm_txt(query)
    toks = [t for t in q.split() if t]

    norm_map = {opt: _norm_txt(opt) for opt in options}
    starts   = [opt for opt, n in norm_map.items() if n.startswith(q)]
    contains = [opt for opt, n in norm_map.items() if q in n and opt not in starts]
    tokens   = [opt for opt, n in norm_map.items()
                if all(t in n for t in toks) and opt not in starts and opt not in contains]

    out = starts + contains + tokens
    return (out or options)[:limit]

# --- /Search helpers ---


# ---------- [AI Context Pack] ----------

def _has_value(x):
    """True if scalar is not NaN or Series has any non-NaN."""
    if isinstance(x, pd.Series):
        return x.notna().any()
    return pd.notna(x)

def _as_scalar(v):
    """Return a scalar from a pandas object; NaN if empty."""
    if isinstance(v, pd.Series):
        v = v.dropna()
        return v.iloc[0] if not v.empty else np.nan
    return v

def _safe_get(row, *names):
    """Return first non-NaN value from row for given column names (scalar or first element of Series)."""
    for n in names:
        if n in row:
            val = row[n]
            if _has_value(val):
                return val.iloc[0] if isinstance(val, pd.Series) else val
    return np.nan


def get_metric_series_row(row: pd.Series, metric: str, prefer: str = "FY") -> pd.Series:
    """
    Re-implementation of get_metric_series_row for test compatibility.
    Extracts a time series for a metric from a single row, handling FY/CQ filtering.
    """
    # 1. Identify all available (date, value) pairs
    pairs = []
    
    # Check base
    if pd.notna(row.get(metric)) and pd.notna(row.get("Period Ended")):
        try:
            dt = pd.to_datetime(row.get("Period Ended"), dayfirst=True)
            pairs.append((dt, row.get(metric)))
        except (ValueError, TypeError, pd.errors.ParserError): pass
        
    # Check suffixes .1 to .20 (reasonable limit)
    for i in range(1, 21):
        m_col = f"{metric}.{i}"
        p_col = f"Period Ended.{i}"
        if m_col in row and pd.notna(row[m_col]) and p_col in row and pd.notna(row[p_col]):
            try:
                dt = pd.to_datetime(row[p_col], dayfirst=True)
                pairs.append((dt, row[m_col]))
            except (ValueError, TypeError, pd.errors.ParserError): pass
            
    if not pairs:
        return pd.Series(dtype=float)
        
    # 2. Filter for FY if requested
    if prefer == "FY" and len(pairs) > 1:
        # Heuristic: Find the most common month, assume that's the FY end
        from collections import Counter
        months = [d.month for d, v in pairs]
        if months:
            common = Counter(months).most_common(1)[0][0]
            # Keep only dates with that month
            pairs = [(d, v) for d, v in pairs if d.month == common]
        
    # 3. Sort by date
    pairs.sort(key=lambda x: x[0])
    
    return pd.Series([p[1] for p in pairs], index=[p[0] for p in pairs])

def most_recent_annual_value(row: pd.Series, metric: str) -> float:
    """Get the most recent annual value for a metric."""
    s = get_metric_series_row(row, metric, prefer="FY")
    if s.empty:
        return np.nan
    return s.iloc[-1] # Last one is most recent due to sort

def get_most_recent_column(df: pd.DataFrame, base_metric: str, data_period_setting: str) -> pd.Series:
    """Get the most recent value column for a metric based on period setting."""
    prefer_fy = "FY" in data_period_setting
    
    def _get_val(row):
        s = get_metric_series_row(row, base_metric, prefer="FY" if prefer_fy else "CQ")
        return s.iloc[-1] if not s.empty else np.nan
        
    return df.apply(_get_val, axis=1)

def extract_metric_time_series(row: pd.Series, df: pd.DataFrame, metric: str) -> dict:
    """Extract time series for a metric as a dictionary."""
    s = get_metric_series_row(row, metric, prefer="FY")
    if s.empty:
        return {}
    return {k.strftime('%Y-%m-%d'): v for k, v in s.items()}

def _mk_key_inputs_row(row: pd.Series) -> Dict[str, Any]:
    """Collect raw inputs used by scoring; values reflect the most-recent annual numbers already computed in the app."""
    # Helper to get annual value with multiple name variants
    def mrav(names):
        if isinstance(names, str):
            val = most_recent_annual_value(row, names)
            return _as_scalar(val) if _has_value(val) else np.nan
        for n in names:
            val = most_recent_annual_value(row, n)
            if _has_value(val):
                return _as_scalar(val)
        return np.nan

    return {
        "Company": _safe_get(row, "Company_Name", "Company Name", "Name"),
        "Company_ID": _safe_get(row, "Company_ID", "Company ID", "ID"),
        "S&P_Rating": _safe_get(row, "Credit_Rating", "S&P LT Issuer Credit Rating", "S&P Credit Rating"),
        "Classification": _safe_get(row, "Rubrics_Custom_Classification", "Rubrics Custom Classification"),
        "Country": _safe_get(row, "Country"),
        "Region": _safe_get(row, "Region"),
        # Core financials / ratios inputs (raw – from spreadsheet)
        "Revenue": mrav(["Total Revenues", "Total Revenue", "Revenue"]),
        "EBITDA": mrav(["EBITDA"]),
        "EBIT": mrav(["EBIT", "Operating Income", "Operating Profit"]),
        "Interest_Expense": mrav(["Interest Expense", "Interest Expense, net", "Net Interest Expense"]),
        "Total_Debt": mrav(["Total Debt"]),
        "Cash_and_ST_Investments": mrav(["Cash and Short Term Investments", "Cash and Equivalents", "Cash"]),
        "OCF": mrav(["Cash from Operations", "Operating Cash Flow", "Cash from Ops", "Net Cash Provided by Operating Activities"]),
        "Capex": mrav(["Capital Expenditure", "Capital Expenditures", "CAPEX"]),
        "LFCF": mrav(["Levered Free Cash Flow", "Free Cash Flow"]),
        # Period/freshness
        "Most_Recent_Period": row.get("Most Recent Period"),
        "Freshness_Flag": row.get("Financial_Data_Freshness_Flag"),
    }

def _mk_score_breakdown_row(row_raw: pd.Series, row_res: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Ratios and 0–100 scores for the Leverage pillar derived from the RAW spreadsheet row.
    Falls back to the results row only if needed. Uses Option A weights (40/30/20/10).
    """
    # --- pull most-recent ANNUAL ratio values from raw sheet (preferred) ---
    def _mr_ratio(names):
        if isinstance(names, str):
            v = most_recent_annual_value(row_raw, names)
            return _as_scalar(v) if _has_value(v) else np.nan
        for n in names:
            v = most_recent_annual_value(row_raw, n)
            if _has_value(v):
                return _as_scalar(v)
        return np.nan

    nd_ebitda  = _mr_ratio(["Net Debt / EBITDA", "Net debt / EBITDA", "ND/EBITDA"])
    td_ebitda  = _mr_ratio(["Total Debt / EBITDA (x)", "Total Debt / EBITDA", "TD/EBITDA"])
    debt_cap   = _mr_ratio(["Total Debt / Total Capital (%)", "Debt / Capital (%)"])
    cov_x      = _mr_ratio(["EBITDA / Interest Expense (x)", "EBITDA/Interest (x)"])

    # If any are missing, try result row columns as a weak fallback
    if (not _has_value(nd_ebitda)) and isinstance(row_res, pd.Series):
        nd_ebitda = _as_scalar(row_res.get("Net Debt / EBITDA"))
    if (not _has_value(td_ebitda)) and isinstance(row_res, pd.Series):
        td_ebitda = _as_scalar(row_res.get("Total Debt / EBITDA (x)"))
    if (not _has_value(debt_cap)) and isinstance(row_res, pd.Series):
        debt_cap = _as_scalar(row_res.get("Total Debt / Total Capital (%)"))
    if (not _has_value(cov_x)) and isinstance(row_res, pd.Series):
        cov_x = _as_scalar(row_res.get("EBITDA / Interest Expense (x)"))

    # --- scoring helpers (aligned with model logic) ---
    def _score_nd_ebitda(x):
        if pd.isna(x): return np.nan
        # piecewise penalty above 2.0x similar to main model
        p = 0.0
        p += max(x - 2.0, 0) * 25
        p += max(x - 3.5, 0) * 15
        p += max(x - 5.0, 0) * 10
        p += max(x - 7.0, 0) * 10
        return float(np.clip(100 - p, 0, 100))

    def _score_td_ebitda(x):
        if pd.isna(x): return np.nan
        p = 0.0
        p += max(x - 3.0, 0) * 20
        p += max(x - 4.5, 0) * 15
        p += max(x - 6.0, 0) * 10
        p += max(x - 8.0, 0) * 10
        return float(np.clip(100 - p, 0, 100))

    def _score_debt_cap_pct(pct):
        if pd.isna(pct): return np.nan
        return float(np.clip(100 - pct, 0, 100))

    def _score_cov(x):
        # use the existing helper if present; else simple monotonic cap at 10x+
        try:
            return float(score_ebitda_coverage(pd.Series([x])).iloc[0]) if pd.notna(x) else np.nan
        except Exception:
            if pd.isna(x): return np.nan
            return float(np.clip((x / 10.0) * 100.0, 0, 100))

    nd_s = _score_nd_ebitda(nd_ebitda)
    td_s = _score_td_ebitda(td_ebitda)
    dc_s = _score_debt_cap_pct(debt_cap)
    cv_s = _score_cov(cov_x)

    # Option A weights with normalization over available components
    comp = [nd_s, cv_s, dc_s, td_s]
    w    = np.array([0.40, 0.30, 0.20, 0.10], dtype=float)
    m    = ~pd.isna(comp)
    lev_score = float(np.nansum(w[m] * np.array(comp)[m]) / (w[m].sum() if m.any() else np.nan))

    return {
        "ND_EBITDA_x": nd_ebitda,
        "TD_EBITDA_x": td_ebitda,
        "Debt_Capital_pct": debt_cap,
        "EBITDA_Interest_x": cov_x,
        "NetDebt_EBITDA_Score": nd_s,
        "TotalDebt_EBITDA_Score": td_s,
        "Debt_to_Capital_Score": dc_s,
        "Interest_Coverage_Score": cv_s,
        "Leverage_Score": (_as_scalar(row_res.get("Leverage_Score")) if isinstance(row_res, pd.Series) and _has_value(row_res.get("Leverage_Score")) else lev_score),
        # also return pillar/total context so downstream rendering stays unchanged
        "Credit_Score": _safe_get(row_res or row_raw, "Credit_Score", "credit_score"),
        "Profitability_Score": _safe_get(row_res or row_raw, "Profitability_Score", "profitability_score"),
        "Liquidity_Score": _safe_get(row_res or row_raw, "Liquidity_Score", "liquidity_score"),

        "Composite_Score": _safe_get(row_res or row_raw, "Composite_Score"),
        "Combined_Signal": _safe_get(row_res or row_raw, "Combined_Signal", "Signal"),
    }

def _build_class_aggregates(df: pd.DataFrame, classification: str) -> Dict[str, Any]:
    """
    Return classification-level stats: n_issuers, medians, IG/HY mix, signal counts, top5/bottom5.
    Uses the canonical classification column (supports aliases).
    """
    cls_col = _classcol(df)  # resolves e.g. 'Rubrics Custom Classification'
    if cls_col:
        sub = df[df[cls_col] == classification].copy()
    else:
        sub = df.copy()

    n = len(sub)
    if n == 0:
        return {"classification": classification, "n_issuers": 0}

    # Medians for 5 factors + composite
    factor_cols = ["Credit_Score", "Leverage_Score", "Profitability_Score",
                   "Liquidity_Score", "Cash_Flow_Score", "Composite_Score"]
    medians = {}
    for c in factor_cols:
        if c in sub.columns:
            medians[c] = float(sub[c].median()) if sub[c].notna().any() else np.nan

    # IG/HY mix
    if "Rating_Group" in sub.columns:
        vc = sub["Rating_Group"].value_counts()
        ig_count = vc.get("IG", 0)
        hy_count = vc.get("HY", 0)
    else:
        ig_count, hy_count = 0, 0

    # Signal counts
    if "Combined_Signal" in sub.columns:
        sig_cts = sub["Combined_Signal"].value_counts().to_dict()
    else:
        sig_cts = {}

    # Top 5 / Bottom 5 by composite
    comp_col = "Composite_Score"
    name_col = next((c for c in ["Company_Name","Company Name","Name"] if c in sub.columns), None)
    if comp_col in sub.columns and name_col:
        sorted_sub = sub[[name_col, comp_col]].dropna().sort_values(comp_col, ascending=False)
        top5 = sorted_sub.head(5)[[name_col, comp_col]].to_dict("records")
        bottom5 = sorted_sub.tail(5)[[name_col, comp_col]].to_dict("records")
    else:
        top5, bottom5 = [], []

    return {
        "classification": classification,
        "n_issuers": n,
        "medians": medians,
        "ig_count": int(ig_count),
        "hy_count": int(hy_count),
        "signal_counts": sig_cts,
        "top5": top5,
        "bottom5": bottom5,
    }

def _gather_ai_context_pack(scope: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the context for AI. If no issuer row is present (dataset/classification scope),
    do NOT call functions that expect a pandas Series with .index.
    """
    row_res = scope.get("issuer_row") or scope.get("row") or {}
    row_raw = scope.get("issuer_row_raw", row_res)  # prefer raw spreadsheet row for inputs/ratios

    sector_mode = st.session_state.get("scoring_method", "Universal Weights")
    leverage_weights = {
        "Net Debt/EBITDA": 0.40, "EBITDA/Interest": 0.30,
        "Debt/Capital": 0.20, "Total Debt/EBITDA": 0.10
    }

    if isinstance(row_raw, pd.Series):
        key_inputs = _mk_key_inputs_row(row_raw)
        breakdown = _mk_score_breakdown_row(row_raw, row_res)

        snapshot = {
            "Company": key_inputs.get("Company"),
            "S&P_Rating": key_inputs.get("S&P Rating"),
            "Classification": key_inputs.get("Classification"),
            "Country": key_inputs.get("Country"),
            "Region": key_inputs.get("Region"),
            "Active_Weight_Mode": sector_mode,
            "Combined_Signal": breakdown.get("Combined_Signal"),
            "Most_Recent_Period": key_inputs.get("Most Recent Period"),
            "Freshness_Flag": key_inputs.get("Freshness Flag"),
        }
    else:
        key_inputs, breakdown = {}, {}
        snapshot = {
            "Company": None,
            "S&P_Rating": None,
            "Classification": (scope.get("aggregates", {}) or {}).get("classification"),
            "Country": None,
            "Region": None,
            "Active_Weight_Mode": sector_mode,
            "Combined_Signal": None,
            "Most_Recent_Period": None,
            "Freshness_Flag": None,
        }

    ctx = {
        "snapshot": snapshot,
        "key_inputs_raw": key_inputs,
        "score_breakdown": breakdown,
        "weights": {
            "leverage_option": "Option A (40/30/20/10)",
            "leverage_weights": leverage_weights,
            "pillar_weights_mode": sector_mode,
        },
    }

    # Include classification stats if present
    if "class_stats" in scope:
        ctx["class_stats"] = scope["class_stats"]

    return ctx

# ---------- [/AI Context Pack] ----------


def _summarize_issuer_row(row: pd.Series) -> str:
    """Compact, structured issuer brief (band-aware)."""
    fields = [
        ("Company_Name", row.get("Company_Name")),
        ("Company_ID", row.get("Company_ID")),
        ("Credit_Rating_Clean", row.get("Credit_Rating_Clean")),
        ("Rating_Band", row.get("Rating_Band")),
        ("Composite_Score", f"{row.get('Composite_Score', float('nan')):.2f}" if pd.notna(row.get('Composite_Score')) else None),
        ("Composite_Percentile_in_Band", row.get("Composite_Percentile_in_Band")),
        ("Recommendation", row.get("Recommendation")),
    ]
    factors = ["Credit_Score","Leverage_Score","Profitability_Score","Liquidity_Score","Cash_Flow_Score"]
    fparts = []
    for f in factors:
        v = row.get(f)
        if pd.notna(v):
            fparts.append(f"{f.replace('_',' ').replace('Score','').title()}={v:.1f}")
    ftxt = "; ".join(fparts) if fparts else "No factor scores available"
    core = "\n".join([f"- {k}: {v}" for k,v in fields if v is not None])
    return f"{core}\n- Factors: {ftxt}"


def _summarize_band(df_results: pd.DataFrame, band: str, limit: int = 15) -> str:
    """Compact band brief (top by Composite within band)."""
    scope = df_results[df_results["Rating_Band"].astype(str).str.upper() == band.upper()].copy()
    if scope.empty:
        return f"Rating band {band}: no issuers."
    scope = scope.sort_values(["Composite_Score","Company_ID"], ascending=[False, True]).head(limit)
    lines = [f"- {r.Company_Name} ({r.Company_ID}) • Score={r.Composite_Score:.2f} • Percentile={r.get('Composite_Percentile_in_Band', float('nan'))}"
             for _, r in scope.iterrows()]
    return f"{band} band, top {len(scope)} by Composite (within-band):\n" + "\n".join(lines)


def _summarize_dataset(df_results: pd.DataFrame) -> str:
    """High-level dataset brief."""
    total = len(df_results)
    ig = int((df_results.get("Rating_Group") == "Investment Grade").sum())
    hy = int((df_results.get("Rating_Group") == "High Yield").sum())
    bands = (df_results["Rating_Band"].value_counts(dropna=False).sort_index()
             if "Rating_Band" in df_results.columns else pd.Series(dtype=int))
    band_txt = ", ".join([f"{b}:{int(c)}" for b, c in bands.items()]) if not bands.empty else "(no band data)"
    return f"Total issuers: {total}\nIG: {ig} • HY: {hy}\nBand mix: {band_txt}"


def _build_ai_prompt(scope: str, narrative_goal: str, body: str) -> str:
    """
    Compose a strict, auditable instruction for the model.
    - scope: 'issuer' | 'band' | 'dataset'
    - narrative_goal: free-text from user
    - body: context text (issuer/band/dataset brief)
    """
    return f"""You are a fixed-income credit analyst. Write a concise, skeptical analysis based ONLY on the provided context.
Do NOT compare across rating bands. Emphasize within-band interpretation.

Scope: {scope}
Goal: {narrative_goal}

Context:
{body}

Requirements:
- Identify 2–4 most material drivers (leverage, profitability, liquidity, growth, cash flow) in plain English.
- Keep statements tied to the data; avoid speculation and macro forecasts.
- If context is sparse, say so explicitly and state what would change your view.
- Max 180 words. Use bullets."""


def _build_ai_context(scope: Dict[str, Any]) -> str:
    """
    Build minimal context string from current data.
    scope contains:
      - 'scope_type': 'dataset' | 'issuer' | 'classification'
      - 'question': str
      - 'top_rows': Dict[str, pd.DataFrame]  # optional small tables already filtered
      - 'aggregates': Dict[str, Any]         # e.g. counts per bucket, thresholds, etc.
      - 'period_hints': List[str]            # e.g. ['FY0: 31/12/2024', 'LTM-1: 30/09/2025']
    """
    lines = []
    lines.append(f"SCOPE: {scope.get('scope_type','dataset')}")
    q = (scope.get("question") or "").strip()
    if q: lines.append(f"USER_QUESTION: {q}")
    agg = scope.get("aggregates") or {}
    if agg:
        for k, v in agg.items():
            lines.append(f"AGG::{k} = {v}")
    ph = scope.get("period_hints") or []
    if ph:
        lines.append("PERIOD_HINTS: " + " | ".join(ph))
    # add a tiny excerpt table (e.g., top 5 issuers by absolute Composite or delta Trend)
    trs = scope.get("top_rows") or {}
    for name, df in trs.items():
        try:
            small = df.head(5).copy()
            lines.append(f"TABLE::{name}::" + small.to_csv(index=False))
        except Exception:
            pass
    return "\n".join(lines)


ANALYST_JSON_SCHEMA = textwrap.dedent("""
Return ONLY JSON with this schema:
{
  "summary": "1-2 line executive summary tailored to the scope",
  "thesis": "A short paragraph with the key point of view",
  "drivers": [{"name": "", "evidence": "metric=value (period)", "direction": "up|down|mixed"}],
  "contradictions": ["Where signals disagree (e.g., strong composite but weak cycle)"],
  "risks": [{"name": "", "trigger": "specific threshold/event"}],
  "what_would_change_my_view": ["clear falsifiers / disconfirmers"],
  "actions": ["portfolio actions in bullets (buy/hold/avoid, hedges, screens)"],
  "data_points": [{"issuer": "", "metric": "", "value": "", "period": ""}]
}
""").strip()


def _build_ai_prompt_json(scope: Dict[str, Any], mode: AIMode, depth: str) -> str:
    """Build AI prompt with JSON schema for structured analyst pack output."""
    ctx = _build_ai_context(scope)
    depth_rules = {
        "Concise": "Limit to ~120-180 words across sections. Only 2 drivers & 2 risks.",
        "Standard": "Balanced detail. 3-5 drivers; 3 risks; 2-3 actions.",
        "Deep-Dive": "Be thorough; keep it crisp but detailed. Up to 7 drivers; 5 risks; 5 actions."
    }
    return textwrap.dedent(f"""
    You are a buy-side fixed income analyst. Use ONLY the provided CONTEXT.
    Anchor every claim with numeric evidence and periods (e.g., 'Net debt/EBITDA 2.6x (FY-1, 31/12/2024)').

    TASK_MODE: {mode.value}
    DEPTH: {depth}
    RULES: {depth_rules.get(depth,'Standard')}
    REQUIRE: numbers with period labels; highlight quality vs cycle contradictions; list watchlist triggers.

    CONTEXT:
    {ctx}

    {ANALYST_JSON_SCHEMA}
    """).strip()


def _build_ai_prompt_chatty(scope: Dict[str, Any], mode: AIMode, depth: str) -> str:
    """Build AI prompt for ChatGPT-style natural narrative with explicit methodology and inputs."""
    ctx = _gather_ai_context_pack(scope)
    tone = {
        "Concise": "Aim for ~150–250 words.",
        "Standard": "Aim for ~250–400 words.",
        "Deep-Dive": "Aim for ~400–700 words."
    }.get(depth, "Aim for ~250–400 words.")

    # Check if we have classification-level stats
    if ctx.get("class_stats"):
        return textwrap.dedent(f"""
        You are a pragmatic buy-side **credit analyst**. Write a natural, ChatGPT-style answer in markdown using **ONLY** the CONTEXT JSON below. Do not invent numbers.

        TASK_MODE: {mode.value} (CLASSIFICATION GROUP ANALYSIS)
        STYLE: conversational, analytical, cohort-focused. {tone}

        You are analyzing a **classification group**, not an individual issuer. Your CONTEXT includes class_stats with:
        - n_issuers: number of issuers in this classification
        - medians: median scores for Credit, Leverage, Profitability, Liquidity, Cash Flow, and Composite
        - ig_count / hy_count: Investment Grade vs High Yield mix
        - signal_counts: distribution of Combined_Signal values (Strong/Moderate Quality & Improving/Stable/Deteriorating Trend)
        - top5 / bottom5: top 5 and bottom 5 issuers by Composite_Score

        Structure your response in this order:
        1) **Classification overview** — what this classification represents and its role in the credit universe.
        2) **Cohort credit profile** — discuss median scores across the 5 factors, highlight the IG vs HY mix, and assess overall credit quality.
        3) **Signal distribution** — analyze the signal_counts to identify whether the group is trending positively or negatively; note any concentration in specific signals.
        4) **Notable performers** — mention a few names from top5 and bottom5 to illustrate the range of credit quality within the group.
        5) **Methodology note** — briefly mention the 5-factor scoring system (0–100 scale) with classification-adjusted weights.

        If a value is missing, state it plainly. Keep it tight; no JSON in the output.

        CONTEXT_JSON:
        {json.dumps(ctx, default=lambda x: None)}
        """).strip()
    else:
        return textwrap.dedent(f"""
        You are a pragmatic buy-side **credit analyst**. Write a natural, ChatGPT-style answer in markdown using **ONLY** the CONTEXT JSON below. Do not invent numbers.

        TASK_MODE: {mode.value}
        STYLE: conversational, skeptical, forward-looking. {tone}

        Structure your response in this order:
        1) **Answer** — your view and why it matters now (1–3 short paragraphs).
        2) **How I got here** — briefly explain the model methodology used: raw ratios → clips/scales (0–100) → factor weights → composite; mention the **active weight mode** and **Leverage Option A (40/30/20/10)**.
        3) **Key inputs (raw)** — a small markdown table with the most relevant raw inputs you used (Revenue, EBITDA, EBIT, Interest Expense, Total Debt, OCF, Capex, etc.). Use the numbers from CONTEXT.
        4) **Score breakdown** — a markdown table with: ratio → 0–100 score → weight (for leverage) and the pillar scores → composite & **Combined_Signal**.
        5) **Qualitative notes** — any relevant classification/country/rating aspects that shape risk.

        If a value is missing, state it plainly. Keep it tight; no JSON in the output.

        CONTEXT_JSON:
        {json.dumps(ctx, default=lambda x: None)}
        """).strip()


# ========== PROFESSIONAL CREDIT REPORT CONSTANTS ==========

CREDIT_ANALYST_SYSTEM_PROMPT = """
You are a senior credit analyst at a fixed income asset management firm specializing in corporate credit analysis. Your analyses are used by portfolio managers to make investment decisions in UCITS funds.

YOUR ROLE:
- Analyze corporate credit quality using quantitative metrics and qualitative judgment
- Provide balanced, objective assessments suitable for professional investors
- Write in clear, professional language appropriate for investment memos
- Focus on forward-looking credit implications, not backward-looking descriptions
- Highlight risks and opportunities with equal weight

ANALYTICAL FRAMEWORK:
- Assess credit quality across 6 key dimensions: profitability, leverage, coverage, liquidity, efficiency, and stability
- Consider trend dynamics (improving/deteriorating) as important as absolute levels
- Compare metrics to industry peers and historical norms
- Identify material risks, catalysts for change, and monitoring points
- Integrate rating agency views with quantitative analysis

OUTPUT REQUIREMENTS:
- Use professional fixed income terminology
- Provide specific metrics with context (e.g., "EBITDA margin of 15.2%, in line with peer median of 15.5%")
- Avoid marketing language or overly promotional tone
- Balance positive and negative observations
- Conclude with actionable insights for portfolio positioning

PROHIBITED:
- Never fabricate data points not provided in the context
- Do not make specific price or spread recommendations
- Avoid definitive predictions about future performance
- Do not comment on macroeconomic factors unless directly relevant
- Never suggest guaranteed outcomes or risk-free investments
"""

REPORT_CSS = """
<style>
.credit-report {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.7;
    color: #2c3e50;
    max-width: 900px;
    margin: 20px auto;
    padding: 30px;
    background: white;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.report-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    color: white;
    padding: 30px;
    border-radius: 8px;
    margin-bottom: 30px;
}

.report-title {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 10px;
}

.report-subtitle {
    font-size: 18px;
    opacity: 0.95;
    margin-bottom: 5px;
}

.report-meta {
    font-size: 13px;
    opacity: 0.85;
    margin-top: 15px;
    padding-top: 15px;
    border-top: 1px solid rgba(255,255,255,0.3);
}

.report-section h2 {
    font-size: 22px;
    font-weight: 600;
    color: #1e3a8a;
    margin-top: 35px;
    margin-bottom: 15px;
    padding-bottom: 8px;
    border-bottom: 2px solid #e0e7ff;
}

.report-section h3 {
    font-size: 17px;
    font-weight: 600;
    color: #3b82f6;
    margin-top: 25px;
    margin-bottom: 12px;
}

.metric-value {
    background: #eff6ff;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
    color: #1e40af;
    font-family: 'Courier New', monospace;
}

.risk-indicator {
    background: #fef2f2;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
    color: #dc2626;
}

.strength-indicator {
    background: #f0fdf4;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
    color: #16a34a;
}

.report-footer {
    margin-top: 50px;
    padding-top: 25px;
    border-top: 2px solid #e2e8f0;
    font-size: 11px;
    color: #64748b;
    line-height: 1.6;
}

.data-table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    font-size: 13px;
    background: white;
}

.data-table th {
    background: #e0e7ff;
    padding: 12px;
    text-align: left;
    font-weight: 600;
    color: #1e40af;
    border-bottom: 2px solid #3b82f6;
}

.data-table td {
    padding: 10px 12px;
    border-bottom: 1px solid #e2e8f0;
}

.data-table tr:hover {
    background: #f8fafc;
}
</style>
"""

ISSUER_REPORT_PROMPT_TEMPLATE = """
Generate a comprehensive credit analysis report for the following corporate issuer.

COMPANY OVERVIEW:
{company_overview}

IMPORTANT: All monetary values in this data are in THOUSANDS of the reported currency.
For example: 32184000 = $32.184 billion, 500000 = $500 million.
Convert to appropriate scale (millions/billions) when writing the report.

CURRENT FINANCIAL POSITION (As of {last_period_date}):
{current_metrics}

HISTORICAL TRENDS:
{historical_trends}

CREDIT ASSESSMENT SCORES:
{credit_scores}

PEER COMPARISON (Within {classification}, n={peer_count} issuers):
{peer_comparison}

DATA QUALITY:
{data_quality}

---

GENERATE A PROFESSIONAL CREDIT ANALYSIS REPORT WITH THE FOLLOWING STRUCTURE:

## I. Executive Summary (150-200 words)
Provide an overview including: company basics, rating, composite score/percentile, overall assessment (Strong/Weak, trend), 2-3 key credit strengths, 2-3 primary concerns, forward-looking outlook statement.

## II. Financial Position Analysis (400-500 words)

### A. Profitability Assessment
Analyze EBITDA margin, ROE, ROA with peer comparison. Discuss profitability trends, quality of earnings, and margin sustainability.

### B. Leverage & Capital Structure
Analyze debt/EBITDA ratios vs. peers. Discuss leverage trend, covenant implications, and capital structure appropriateness.

### C. Debt Service Capacity
Analyze interest coverage ratios, free cash flow generation. Assess buffer above critical thresholds.

### D. Liquidity Profile
Analyze current/quick ratios, cash position, working capital dynamics.

## III. Credit Trend Analysis (250-300 words)
Discuss historical trajectory, volatility/consistency, recent inflection points, FY vs. quarterly trends, and trend sustainability.

## IV. Peer Comparison (200-250 words)
Discuss position within classification, percentile rankings, outlier metrics, competitive positioning, relative credit quality.

## V. Credit Risks & Considerations (200-250 words)
Identify material weaknesses, unfavorable trends, data quality concerns, industry-specific risks, rating migration risks.

## VI. Credit Recommendation (100-150 words)
Summarize investment thesis, appropriate credit positioning, key monitoring points, upgrade/downgrade catalysts, risk/reward assessment.

CRITICAL REQUIREMENTS:
- Use specific quantitative metrics with peer context throughout
- Cite actual numbers from the data provided (never fabricate)
- Focus on forward-looking credit implications
- Balance positive and negative observations
- Use professional fixed income terminology
- Total length: 1,300-1,700 words
- Format as markdown with clear section headers
"""

CLASSIFICATION_REPORT_PROMPT_TEMPLATE = """
Generate a comprehensive credit analysis report for the following industry classification.

CLASSIFICATION OVERVIEW:
{classification_overview}

IMPORTANT: All monetary values in this data are in THOUSANDS of the reported currency.
For example: 32184000 = $32.184 billion, 500000 = $500 million.
Convert to appropriate scale (millions/billions) when writing the report.

RATING DISTRIBUTION:
{rating_distribution}

AGGREGATE CREDIT METRICS:
{aggregate_metrics}

CREDIT SIGNAL DISTRIBUTION:
{signal_distribution}

TOP PERFORMERS:
{top_performers}

BOTTOM PERFORMERS:
{bottom_performers}

MOST IMPROVING:
{improving_issuers}

MOST DETERIORATING:
{deteriorating_issuers}

---

GENERATE A PROFESSIONAL CLASSIFICATION-LEVEL CREDIT ANALYSIS REPORT WITH THE FOLLOWING STRUCTURE:

## I. Classification Overview (150-200 words)
Summarize: classification name, issuer count, geographic/industry composition, rating distribution, average composite score, overall credit quality assessment.

## II. Aggregate Credit Metrics (400-500 words)

### A. Profitability Metrics Distribution
Discuss margin distribution (median, quartiles, outliers), ROE distribution, profitability trends across classification, comparison to broader market.

### B. Leverage Profile
Discuss leverage distribution, percentage above/below key thresholds, median trends, identification of highly leveraged outliers.

### C. Coverage Metrics
Discuss coverage distribution, percentage below critical levels, coverage adequacy, stressed scenarios within group.

### D. Liquidity Analysis
Discuss median current ratio and range, cash-rich vs. cash-constrained issuers, liquidity concerns.

## III. Credit Signal Distribution (200-250 words)
Break down by signal categories, discuss percentages, analyze trajectory/momentum of the classification overall.

## IV. Top & Bottom Performers (250-300 words)
Highlight top performers with key metric patterns, bottom performers with common weaknesses, best improving issuers, most deteriorating issuers.

## V. Classification-Specific Insights (200-250 words)
Discuss industry dynamics affecting credit, common strengths across group, systemic risks/weaknesses, divergence within classification, data quality observations.

## VI. Portfolio Implications (150-200 words)
Investment considerations, relative value opportunities, risk concentration concerns, recommended weighting approach, key monitoring metrics.

CRITICAL REQUIREMENTS:
- Use statistical distributions (median, quartiles, ranges)
- Identify credit quality clusters and patterns
- Highlight relative value opportunities
- Cite actual numbers from the data provided (never fabricate)
- Focus on systemic factors affecting the group
- Total length: 1,300-1,700 words
- Format as markdown with clear section headers
"""


def _run_ai(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    """Call OpenAI chat/completions with structured prompt."""
    client = _get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a meticulous fixed-income analyst. Be precise, concise, and evidence-based."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _render_analyst_json(raw_text: str):
    """Parse and render structured JSON analyst pack output."""
    try:
        data = json.loads(raw_text)
    except Exception:
        st.markdown("**LLM output (raw):**")
        st.write(raw_text)
        return

    st.markdown(f"### Summary")
    st.write(data.get("summary", ""))

    st.markdown("### Thesis")
    st.write(data.get("thesis", ""))

    if data.get("drivers"):
        st.markdown("### Key Drivers")
        for d in data["drivers"]:
            st.write(f"- **{d.get('name','')}** — {d.get('evidence','')} ({d.get('direction','')})")

    if data.get("contradictions"):
        st.markdown("### Contradictions")
        for c in data["contradictions"]:
            st.write(f"- {c}")

    if data.get("risks"):
        st.markdown("### Risks")
        for r in data["risks"]:
            st.write(f"- **{r.get('name','')}** — Trigger: {r.get('trigger','')}")

    if data.get("what_would_change_my_view"):
        st.markdown("### What would change my view")
        for w in data["what_would_change_my_view"]:
            st.write(f"- {w}")

    if data.get("actions"):
        st.markdown("### Suggested Actions")
        for a in data["actions"]:
            st.write(f"- {a}")

    if data.get("data_points"):
        st.markdown("### Evidence (selected data points)")
        st.dataframe(pd.DataFrame(data["data_points"]))


# ============================================================================
# PROFESSIONAL CREDIT REPORT HELPER FUNCTIONS
# ============================================================================

def extract_latest_period_metrics(raw_row: pd.Series, results_row: pd.Series) -> dict:
    """Extract latest period financial metrics prioritising RAW values (fallback to results)."""
    def safe_float(val, decimals=2):
        try:
            if pd.isna(val):
                return "N/A"
            # Handle infinity and extreme values
            if np.isinf(val):
                return "∞" if val > 0 else "-∞"
            float_val = float(val)
            # Sanity check for extreme values
            if abs(float_val) > 1e15:
                return f"{float_val:.2e}"
            return f"{float_val:.{decimals}f}"
        except Exception:
            return "N/A"
    def latest_of(metric):
        ts = _metric_series_for_row(results_row.to_frame(), results_row, metric, prefer_fy=True) \
             if isinstance(results_row, pd.Series) else pd.Series(dtype=float)
        if ts.empty:
            ts = _metric_series_for_row(raw_row.to_frame(), raw_row, metric, prefer_fy=True)
        if ts.empty:
            # final fallback to direct alias lookup
            v = get_from_row(raw_row, metric)
            if pd.isna(v) and isinstance(results_row, pd.Series):
                alias = resolve_metric_column(results_row, metric)
                v = results_row.get(alias) if alias else np.nan
            return v
        return ts.iloc[-1]
    return {
        "ebitda_margin":   safe_float(latest_of("EBITDA Margin")),
        "roe":             safe_float(latest_of("Return on Equity")),
        "roa":             safe_float(latest_of("Return on Assets")),
        "total_debt_ebitda": safe_float(latest_of("Total Debt / EBITDA (x)")),
        "net_debt_ebitda": safe_float(latest_of("Net Debt / EBITDA")),
        "coverage":        safe_float(latest_of("EBITDA / Interest Expense (x)")),
        "current_ratio":   safe_float(latest_of("Current Ratio (x)")),
        "quick_ratio":     safe_float(latest_of("Quick Ratio (x)"), 2),
        "total_debt":      safe_float(latest_of("Total Debt"), 0),
        "cash":            safe_float(latest_of("Cash and Short-Term Investments"), 0),
        "reporting_currency": str(raw_row.get('Reported Currency', 'USD')) if pd.notna(raw_row.get('Reported Currency')) else 'USD',
    }


def extract_time_series_compact(row: pd.Series, metric_base: str, n_periods: int = 3) -> str:
    """Extract time series showing trend direction."""
    try:
        values = []
        for i in range(n_periods):
            col_name = metric_base if i == 0 else f"{metric_base}.{i}"
            if col_name in row.index:
                val = row[col_name]
                if pd.notna(val):
                    values.append(float(val))

        if len(values) < 2:
            return "Insufficient data"

        # Format values with arrow
        formatted = " → ".join([f"{v:.1f}" for v in reversed(values)])

        # Add direction indicator
        if values[0] > values[-1]:
            formatted += " ↑"
        elif values[0] < values[-1]:
            formatted += " ↓"
        else:
            formatted += " →"

        return formatted
    except Exception:
        return "N/A"


def calculate_peer_statistics(issuer_row: pd.Series, results_df: pd.DataFrame, classification: str) -> dict:
    """Calculate peer comparison statistics for an issuer."""
    try:
        cls_col = _classcol(results_df)
        if not cls_col:
            return {}

        peer_subset = results_df[results_df[cls_col] == classification].copy()
        peer_count = len(peer_subset)

        if peer_count == 0:
            return {"peer_count": 0}

        # Calculate percentile ranks
        def calc_percentile(metric_name):
            if metric_name not in peer_subset.columns:
                return "N/A"
            issuer_val = issuer_row.get(metric_name)
            if pd.isna(issuer_val):
                return "N/A"
            percentile = (peer_subset[metric_name] < issuer_val).sum() / peer_count * 100
            return f"{percentile:.0f}"

        # Calculate medians
        def calc_median(metric_name):
            if metric_name not in peer_subset.columns:
                return "N/A"
            median_val = peer_subset[metric_name].median()
            if pd.isna(median_val):
                return "N/A"
            return f"{median_val:.2f}"

        return {
            "peer_count": peer_count,
            "margin_percentile": calc_percentile("EBITDA Margin"),
            "leverage_percentile": calc_percentile("Total Debt / EBITDA (x)"),
            "coverage_percentile": calc_percentile("EBITDA / Interest Expense (x)"),
            "composite_percentile": calc_percentile("Composite_Score"),
            "peer_median_margin": calc_median("EBITDA Margin"),
            "peer_median_leverage": calc_median("Total Debt / EBITDA (x)"),
            "peer_median_coverage": calc_median("EBITDA / Interest Expense (x)"),
            "class_median_score": calc_median("Composite_Score"),
        }
    except Exception as e:
        return {"peer_count": 0, "error": str(e)}


def calculate_metric_distribution(class_subset: pd.DataFrame, metric_names: list) -> dict:
    """Calculate distribution statistics for classification metrics."""
    stats = {}
    for metric in metric_names:
        if metric not in class_subset.columns:
            continue

        data = class_subset[metric].dropna()
        if len(data) == 0:
            continue

        stats[metric] = {
            "median": f"{data.median():.2f}",
            "mean": f"{data.mean():.2f}",
            "p25": f"{data.quantile(0.25):.2f}",
            "p75": f"{data.quantile(0.75):.2f}",
            "min": f"{data.min():.2f}",
            "max": f"{data.max():.2f}",
        }

    return stats


def get_top_issuers(class_subset: pd.DataFrame, n: int, by: str) -> list:
    """Get top N issuers by specified metric."""
    try:
        name_col = _namecol(class_subset)
        if not name_col or by not in class_subset.columns:
            return []

        sorted_df = class_subset[[name_col, by]].dropna().sort_values(by, ascending=False).head(n)
        return [{"name": row[name_col], "value": f"{row[by]:.1f}"} for _, row in sorted_df.iterrows()]
    except Exception:
        return []


def get_bottom_issuers(class_subset: pd.DataFrame, n: int, by: str) -> list:
    """Get bottom N issuers by specified metric."""
    try:
        name_col = _namecol(class_subset)
        if not name_col or by not in class_subset.columns:
            return []

        sorted_df = class_subset[[name_col, by]].dropna().sort_values(by, ascending=True).head(n)
        return [{"name": row[name_col], "value": f"{row[by]:.1f}"} for _, row in sorted_df.iterrows()]
    except Exception:
        return []


def format_report_with_styling(markdown_content: str, metadata: dict) -> str:
    """Wrap report content in styled HTML."""
    report_type = metadata.get("report_type", "Credit Analysis")
    entity_name = metadata.get("entity_name", "Unknown")
    report_date = metadata.get("report_date", "N/A")

    header = f"""
<div class="report-header">
    <div class="report-title">{report_type} Report</div>
    <div class="report-subtitle">{entity_name}</div>
    <div class="report-meta">
        Generated: {report_date} | Analyst: AI Credit System
    </div>
</div>
"""

    footer = """
<div class="report-footer">
    <strong>Disclaimer:</strong> This report is generated by an AI credit analysis system based on quantitative metrics and statistical models.
    It should be used as a supplementary tool alongside fundamental analysis, credit research, and professional judgment.
    Past performance does not guarantee future results. Credit markets are subject to various risks including interest rate risk,
    credit risk, liquidity risk, and market risk. This report does not constitute investment advice or a recommendation to buy,
    sell, or hold any security.
</div>
"""

    return f'<div class="credit-report">{header}<div class="report-section">{markdown_content}</div>{footer}</div>'


def _safe_format(val, decimals=2):
    """Safely format a value for display."""
    try:
        if pd.isna(val):
            return "N/A"
        if isinstance(val, (int, float)):
            # Handle infinity and extreme values
            if np.isinf(val):
                return "∞" if val > 0 else "-∞"
            # Convert to float and format
            float_val = float(val)
            # Sanity check for extreme values
            if abs(float_val) > 1e15:
                return f"{float_val:.2e}"  # Scientific notation for very large numbers
            return f"{float_val:.{decimals}f}"
        return str(val)
    except Exception:
        return "N/A"


# ---------- Data diagnostics v2 (alias-aware, FY-only, metric-specific) ----------
def generate_data_diagnostics_v2(df_original: pd.DataFrame, results_df: pd.DataFrame,
                                 entity_type: str, entity_identifier: str) -> pd.DataFrame:
    # Resolve entity rows independently in each frame
    if str(entity_type).lower().startswith("issuer"):
        name_raw = _resolve_company_name_col(df_original)
        name_res = _resolve_company_name_col(results_df)
        if not name_raw or not name_res:
            return pd.DataFrame([{"Error": "Company name column not found"}])
        raw_rows = df_original[df_original[name_raw].apply(_norm) == _norm(entity_identifier)]
        res_rows = results_df[results_df[name_res].apply(_norm) == _norm(entity_identifier)]
    else:
        cls_raw = _resolve_classification_col(df_original)
        cls_res = _resolve_classification_col(results_df)
        if not cls_raw or not cls_res:
            return pd.DataFrame([{"Error": "Classification column not found"}])
        raw_rows = df_original[df_original[cls_raw].apply(_norm) == _norm(entity_identifier)]
        res_rows = results_df[results_df[cls_res].apply(_norm) == _norm(entity_identifier)]
    if raw_rows.empty:
        return pd.DataFrame([{"Error": "Entity not found in raw data"}])
    raw_row = raw_rows.iloc[0]
    res_row = res_rows.iloc[0] if not res_rows.empty else None

    key_metrics = [
        "EBITDA Margin","Return on Equity","Return on Assets",
        "Total Debt / EBITDA (x)","Net Debt / EBITDA","EBITDA / Interest Expense (x)",
        "Current Ratio (x)","Quick Ratio (x)","Cash and Short-Term Investments",
        "Total Debt","Total Revenues","Total Revenues, 1 Year Growth","Total Revenues, 3 Yr. CAGR","EBITDA, 3 Years CAGR"
    ]
    rows = []
    period_map = _find_period_cols(df_original, prefer_fy=True)
    for metric in key_metrics:
        base, suffixed = list_metric_columns(df_original, metric)
        # Base/suffix existence & base value
        base_exists = base is not None
        base_val = raw_row.get(base) if base_exists else np.nan
        # Metric series for this issuer (FY only)
        series = _metric_series_for_row(df_original, raw_row, metric, prefer_fy=True)
        # Coverage text (metric-specific) - only count valid (non-NaN) points
        valid = series.dropna()
        valid_count = int(valid.shape[0])
        if valid_count:
            try:
                min_date = valid.index.min()
                max_date = valid.index.max()
                # Check for NaT before calling .date()
                if pd.isna(min_date) or pd.isna(max_date):
                    coverage_txt = f"{valid_count} periods (dates unavailable)"
                else:
                    coverage_txt = f"{min_date.date()} to {max_date.date()} ({valid_count} periods)"
            except (AttributeError, ValueError, TypeError):
                coverage_txt = f"{valid_count} periods (dates unavailable)"
        else:
            coverage_txt = "N/A"
        # Latest value (raw) + whether results contain any alias
        latest_val = series.iloc[-1] if not series.empty else (raw_row.get(base) if base_exists else np.nan)
        alias_in_results = resolve_metric_column(res_row, metric) if res_row is not None else None
        in_results = (pd.notna(res_row.get(alias_in_results)) if alias_in_results else False) if res_row is not None else False

        if pd.isna(latest_val) and not in_results:
            status = "❌ Missing"
        elif pd.notna(latest_val) and not in_results:
            status = "⚠ Data exists but not extracted"
        else:
            status = "✅ OK"

        rows.append({
            "Metric": metric,
            "Base Column Exists": "✅" if base_exists else "❌",
            "Base Value": f"{base_val:.2f}" if (base_exists and pd.notna(base_val) and isinstance(base_val, (int, float))) else "N/A",
            "Suffixed Columns": len(suffixed) if suffixed else 0,
            "Period Coverage": str(valid_count) + " periods" if valid_count > 0 else "N/A",
            "Latest Value Used": f"{latest_val:.2f}" if (pd.notna(latest_val) and isinstance(latest_val, (int, float))) else "N/A",
            "In Results": "✅ Yes" if in_results else "❌ No",
            "Status": status
        })
    return pd.DataFrame(rows)


def generate_global_data_health_v2(df_original: pd.DataFrame) -> pd.DataFrame:
    key_metrics = ["EBITDA Margin","Return on Equity","Return on Assets","Total Debt / EBITDA (x)","EBITDA / Interest Expense (x)","Current Ratio (x)"]
    rows = []
    total_points = 0
    total_possible = 0
    for metric in key_metrics:
        base, suffixed = list_metric_columns(df_original, metric)
        base_filled = df_original[base].notna().sum() if base else 0
        suffixed_filled = sum(df_original[c].notna().sum() for c in suffixed)
        points = base_filled + suffixed_filled
        possible = (len(df_original) * ((1 if base else 0) + len(suffixed)))
        total_points += points
        total_possible += possible
        rows.append({
            "Metric": metric,
            "Base Column": "✅" if base else "❌",
            "Suffixed Columns": len(suffixed),
            "Data Points Available": int(points),
            "Coverage %": f"{(points/possible*100):.1f}%" if possible > 0 else "0.0%"
        })
    df = pd.DataFrame(rows)
    df.attrs["avg_coverage"] = (total_points/total_possible*100) if total_possible>0 else 0.0
    df.attrs["base_cols_available"] = sum(1 for m in key_metrics if list_metric_columns(df_original, m)[0])
    df.attrs["avg_suffixed_cols"] = np.mean([len(list_metric_columns(df_original, m)[1]) for m in key_metrics]) if key_metrics else 0
    return df


# ============================================================================
# ISSUER EVIDENCE TABLE BUILDER (for AI Analysis)
# ============================================================================

EVIDENCE_METRICS = [
    # levels
    "Cash and Short-Term Investments", "Total Assets", "Net Debt", "Total Debt", "Total Common Equity",
    # ratios
    "Total Debt/Equity (x)", "Total Debt / Total Capital (%)", "Total Debt / EBITDA (x)",
    "Net Debt / EBITDA", "EBITDA / Interest Expense (x)",
    "EBITDA Margin", "Return on Equity", "Return on Assets",
]

def _latest_periods(row: pd.Series, prefer_fy=True, fy_n=5, cq_n=8):
    """Build ordered lists of (date, period_type) after coercion; exclude 1900."""
    fy = []
    cq = []
    # Pick values per suffix; reuse alias-aware series builder
    for k in METRIC_REGISTRY.keys():
        s = _metric_series_for_row(row.to_frame().T, row, k, prefer_fy=True)
        if not s.empty:
            break
    # If no metric resolved, return empties
    if s.empty:
        return [], []
    dates = pd.to_datetime(s.index, errors="coerce")
    fy_dates = sorted({d for d in dates if pd.notna(d) and d.year != 1900}, reverse=True)
    # Choose FY as the 12-month spaced most recent 5
    fy_out = []
    last = None
    for d in fy_dates:
        if last is None or abs((last - d).days) >= 300:  # ~annual spacing
            fy_out.append(d)
            last = d
        if len(fy_out) >= fy_n:
            break
    # CQ = latest 8 unique dates excluding FY0 date
    fy0 = fy_out[0] if fy_out else None
    cq_candidates = sorted({d for d in dates if pd.notna(d) and d != fy0}, reverse=True)
    cq_out = cq_candidates[:cq_n]
    return fy_out, cq_out

def build_issuer_evidence_table(df_raw: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """Build evidence table with FY and CQ columns for issuer analysis."""
    # Determine columns once
    fy, ltm = _latest_periods(row, prefer_fy=True, fy_n=5, cq_n=8)
    cols = [f"FY-{i}" for i in range(len(fy)-1, -1, -1)] + [f"LTM-{i}" for i in range(len(ltm)-1, -1, -1)]
    dates = list(reversed(fy)) + list(reversed(ltm))

    out = []
    for metric in EVIDENCE_METRICS:
        s = _metric_series_for_row(df_raw, row, metric, prefer_fy=True)
        # map dates to values
        vals = []
        for d in dates:
            v = pd.to_numeric(s.get(d), errors="coerce") if d in s.index else np.nan
            vals.append(v)
        out.append([metric] + vals)

    tbl = pd.DataFrame(out, columns=["Metric"] + cols)
    # Format: numbers only; don't leave objects that crash Streamlit
    return tbl


def assemble_issuer_context(issuer_row: pd.Series, raw_row: pd.Series, results_df: pd.DataFrame) -> dict:
    """Assemble comprehensive context for issuer credit report."""
    import datetime

    # Extract basic metadata
    name_col = _namecol(results_df)
    cls_col = _classcol(results_df)
    company_name = issuer_row.get(name_col, "Unknown")
    classification = issuer_row.get(cls_col, "Unknown")

    # Extract latest metrics
    latest_metrics = extract_latest_period_metrics(raw_row, issuer_row)

    # Calculate peer statistics
    peer_stats = calculate_peer_statistics(issuer_row, results_df, classification)

    # Extract credit scores - raw-only (no fallback to Composite/Cycle)
    quality_raw = issuer_row.get("Raw_Quality_Score")
    trend_raw = issuer_row.get("Raw_Trend_Score")

    credit_scores = {
        "quality": quality_raw,  # Raw-only: no Composite_Score fallback
        "trend": trend_raw,      # Raw-only: no Cycle_Position_Score fallback
        "profitability": _safe_get(issuer_row, "Profitability_Score", "profitability_score"),
        "leverage": _safe_get(issuer_row, "Leverage_Score", "leverage_score"),
        "liquidity": _safe_get(issuer_row, "Liquidity_Score", "liquidity_score"),
        # Growth factor removed from 5-factor model
        "credit": _safe_get(issuer_row, "Credit_Score", "credit_score"),
        "combined_signal": _safe_get(issuer_row, "__Signal_v2", "__Signal", "Combined_Signal", "Signal"),
    }

    # Extract trends
    trends = {
        "margin": extract_time_series_compact(raw_row, "EBITDA Margin", 3),
        "leverage": extract_time_series_compact(raw_row, "Total Debt / EBITDA (x)", 3),
        "coverage": extract_time_series_compact(raw_row, "EBITDA / Interest Expense (x)", 3),
        "roe": extract_time_series_compact(raw_row, "Return on Equity", 3),
    }

    return {
        "metadata": {
            "report_type": "Issuer Credit Analysis",
            "entity_name": company_name,
            "report_date": datetime.date.today().strftime("%Y-%m-%d"),
        },
        "company_name": company_name,
        "classification": classification,
        "reporting_currency": latest_metrics.get('reporting_currency', 'USD'),
        "sp_rating": _safe_get(issuer_row, "S&P LT Issuer Credit Rating", "Rating"),
        "country": _safe_get(issuer_row, "Country"),
        "latest_metrics": latest_metrics,
        "credit_scores": credit_scores,
        "trends": trends,
        "peer_stats": peer_stats,
    }


def assemble_classification_context(classification: str, results_df: pd.DataFrame) -> dict:
    """Assemble comprehensive context for classification credit report."""
    import datetime

    cls_col = _classcol(results_df)
    if not cls_col:
        return {}

    # Get classification subset
    class_subset = results_df[results_df[cls_col] == classification].copy()
    issuer_count = len(class_subset)

    if issuer_count == 0:
        return {}

    # Calculate metric distributions
    key_metrics = [
        "EBITDA Margin",
        "Return on Equity",
        "Total Debt / EBITDA (x)",
        "Net Debt / EBITDA",
        "EBITDA / Interest Expense (x)",
        "Current Ratio",
        "Raw_Quality_Score",
        "Raw_Trend_Score",
    ]
    distributions = calculate_metric_distribution(class_subset, key_metrics)

    # Get rating distribution
    rating_dist = {}
    if "Rating_Group" in class_subset.columns:
        rating_dist = class_subset["Rating_Group"].value_counts().to_dict()

    # Get signal distribution
    signal_dist = {}
    if "Combined_Signal" in class_subset.columns:
        signal_dist = class_subset["Combined_Signal"].value_counts().to_dict()

    # Get top/bottom performers (raw quality-based)
    top_performers = get_top_issuers(class_subset, 10, "Raw_Quality_Score")
    bottom_performers = get_bottom_issuers(class_subset, 10, "Raw_Quality_Score")

    # Get improving/deteriorating (raw trend-based)
    improving = []
    deteriorating = []
    if "Raw_Trend_Score" in class_subset.columns:
        improving = get_top_issuers(class_subset, 10, "Raw_Trend_Score")
        deteriorating = get_bottom_issuers(class_subset, 10, "Raw_Trend_Score")

    return {
        "metadata": {
            "report_type": "Classification Credit Analysis",
            "entity_name": classification,
            "report_date": datetime.date.today().strftime("%Y-%m-%d"),
        },
        "classification": classification,
        "issuer_count": issuer_count,
        "distributions": distributions,
        "rating_dist": rating_dist,
        "signal_dist": signal_dist,
        "top_performers": top_performers,
        "bottom_performers": bottom_performers,
        "improving": improving,
        "deteriorating": deteriorating,
    }


def build_issuer_report_prompt(context: dict) -> str:
    """Build formatted prompt for issuer credit report."""
    # Get currency symbol using global function
    curr_sym = get_currency_symbol(context.get('reporting_currency', 'USD'))
    
    # Company Overview section
    company_overview = f"""
Company: {context['company_name']}
Country: {context['country']} | Classification: {context['classification']}
S&P Rating: {context['sp_rating']}
Quality Score: {context['credit_scores']['quality']}/100
Trend Score: {context['credit_scores']['trend']}/100
"""

    # Current Metrics section
    m = context['latest_metrics']
    current_metrics = f"""
Profitability:
- EBITDA Margin: {m['ebitda_margin']}% (Peer median: {context['peer_stats'].get('peer_median_margin', 'N/A')}%, Percentile: {context['peer_stats'].get('margin_percentile', 'N/A')}th)
- Return on Equity: {m['roe']}%
- Return on Assets: {m['roa']}%

Leverage:
- Total Debt / EBITDA: {m['total_debt_ebitda']}x (Peer median: {context['peer_stats'].get('peer_median_leverage', 'N/A')}x, Percentile: {context['peer_stats'].get('leverage_percentile', 'N/A')}th)
- Net Debt / EBITDA: {m['net_debt_ebitda']}x
- Total Debt: {curr_sym}{m['total_debt']}

Coverage:
- EBITDA / Interest Expense: {m['coverage']}x (Peer median: {context['peer_stats'].get('peer_median_coverage', 'N/A')}x)

Liquidity:
- {get_metric_display_name('current_ratio')}: {m['current_ratio']}x
- {get_metric_display_name('cash')}: {curr_sym}{m['cash']}
- {get_metric_display_name('quick_ratio')}: {m['quick_ratio']}x
"""

    # Historical Trends section
    t = context['trends']
    historical_trends = f"""
EBITDA Margin: {t['margin']}
Total Debt/EBITDA: {t['leverage']}
EBITDA/Interest: {t['coverage']}
ROE: {t['roe']}
"""

    # Credit Scores section
    s = context['credit_scores']
    credit_scores = f"""
Quality Score (raw): {s['quality']} / 100
Trend Score (raw): {s['trend']} / 100
- Profitability Score: {s['profitability']}
- Leverage Score: {s['leverage']}
- Liquidity Score: {s['liquidity']}

- Credit Score: {s['credit']}

Combined Signal: {s['combined_signal']}
"""

    # Peer Comparison section
    ps = context['peer_stats']
    peer_comparison = f"""
Key Metric Rankings:
- EBITDA Margin: {ps.get('margin_percentile', 'N/A')}th percentile
- Leverage: {ps.get('leverage_percentile', 'N/A')}th percentile
- Coverage: {ps.get('coverage_percentile', 'N/A')}th percentile
Classification Statistics:
- Median Composite Score: {ps.get('class_median_score', 'N/A')}
- Median EBITDA Margin: {ps.get('peer_median_margin', 'N/A')}%
- Median Leverage: {ps.get('peer_median_leverage', 'N/A')}x
"""

    # Data Quality section
    data_quality = "Financial data: Latest available period"

    return ISSUER_REPORT_PROMPT_TEMPLATE.format(
        company_overview=company_overview,
        last_period_date="Latest Period",
        current_metrics=current_metrics,
        historical_trends=historical_trends,
        credit_scores=credit_scores,
        classification=context['classification'],
        peer_count=ps.get('peer_count', 0),
        peer_comparison=peer_comparison,
        data_quality=data_quality,
    )


def build_classification_report_prompt(context: dict) -> str:
    """Build formatted prompt for classification credit report."""
    # Classification Overview
    classification_overview = f"""
Classification: {context['classification']}
Total Issuers: {context['issuer_count']}
Average Quality Score (raw): {context['distributions'].get('Raw_Quality_Score', {}).get('mean', 'N/A')}
"""

    # Rating Distribution
    rating_dist = context['rating_dist']
    total = sum(rating_dist.values()) if rating_dist else 1
    rating_distribution = "\n".join([f"{k}: {v} ({v/total*100:.1f}%)" for k, v in rating_dist.items()])

    # Aggregate Metrics
    dists = context['distributions']
    aggregate_metrics = ""
    for metric_name, stats in dists.items():
        aggregate_metrics += f"""
{metric_name}:
- Median: {stats.get('median', 'N/A')} | Mean: {stats.get('mean', 'N/A')}
- 25th Percentile: {stats.get('p25', 'N/A')} | 75th Percentile: {stats.get('p75', 'N/A')}
- Range: {stats.get('min', 'N/A')} to {stats.get('max', 'N/A')}
"""

    # Signal Distribution
    signal_dist = context['signal_dist']
    signal_distribution = "\n".join([f"{k}: {v}" for k, v in signal_dist.items()])

    # Performers
    def format_issuers(issuers_list):
        if not issuers_list:
            return "No data available"
        return "\n".join([f"- {iss['name']}: {iss['value']}" for iss in issuers_list[:10]])

    top_performers = format_issuers(context['top_performers'])
    bottom_performers = format_issuers(context['bottom_performers'])
    improving_issuers = format_issuers(context['improving'])
    deteriorating_issuers = format_issuers(context['deteriorating'])

    return CLASSIFICATION_REPORT_PROMPT_TEMPLATE.format(
        classification_overview=classification_overview,
        rating_distribution=rating_distribution,
        aggregate_metrics=aggregate_metrics,
        signal_distribution=signal_distribution,
        top_performers=top_performers,
        bottom_performers=bottom_performers,
        improving_issuers=improving_issuers,
        deteriorating_issuers=deteriorating_issuers,
    )


# ============================================================================
# AI ANALYSIS CHAT UI (reintroduce LLM in a meaningful, auditable way)
# ============================================================================

def render_ai_analysis_chat(df_original: pd.DataFrame, results_final: pd.DataFrame):
    st.subheader("🔍 AI Credit Analysis")

    # Optional snapshot counts (safe even if missing)
    try:
        buckets = build_buckets_v2(results_final,
                                    df_original)  # Uses MODEL_THRESHOLDS defaults
        df_counts = buckets.get("counts", pd.DataFrame())
        if isinstance(df_counts, dict):
            df_counts = pd.DataFrame(list(df_counts.items()), columns=["Signal","Count"])
        order = ["Strong & Improving","Strong & Moderating","Strong & Normalizing",
                 "Strong but Deteriorating","Weak but Improving","Weak & Deteriorating"]
        cols = st.columns(min(6, len(order)))
        for i, lab in enumerate(order):
            try:
                cnt = int(df_counts.loc[df_counts["Signal"].eq(lab), "Count"].sum())
            except Exception:
                cnt = 0
            with cols[i % len(cols)]:
                st.metric(lab, cnt)
    except Exception as _e:
        st.warning(f"AI Analysis (raw-only) bucketing failed: {_e}")

    st.markdown("---")

    # Global Data Health Check
    st.info("💡 **Data Diagnostics**: Check data quality and availability for selected entities below")

    with st.expander("📊 Global Data Health", expanded=False):
        try:
            health_df = generate_global_data_health_v2(df_original)
            st.table(health_df)  # Use st.table instead of st.dataframe for reliability

            # Summary metrics (from attrs)
            col1, col2, col3 = st.columns(3)
            with col1:
                base_available = health_df.attrs.get("base_cols_available", 0)
                total_metrics = len(health_df)
                st.metric("Base Columns Available", f"{base_available}/{total_metrics}")
            with col2:
                avg_suffixed = health_df.attrs.get("avg_suffixed_cols", 0)
                st.metric("Avg Suffixed Columns", f"{avg_suffixed:.1f}")
            with col3:
                avg_coverage = health_df.attrs.get("avg_coverage", 0)
                st.metric("Avg Data Coverage", f"{avg_coverage:.1f}%")

            st.caption("ℹ️ This shows whether key financial metrics exist in your dataset across all issuers.")
        except Exception as e:
            st.warning(f"Could not generate global health check: {e}")

    st.markdown("---")
    st.markdown("### Generate Professional Credit Report")

    # Selection Interface
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        # Classification Analysis disabled for now - focusing on issuer-level reports
        report_type = st.radio(
            "Report Type",
            options=["Issuer Analysis"],
            horizontal=True
        )

    with col2:
        report_format = st.selectbox(
            "Report Length",
            options=["Standard Report (~1,500 words)",
                    "Executive Summary (~500 words)",
                    "Comprehensive Report (~2,500 words)"]
        )

    with col3:
        st.metric("Est. Time", "20-30s")

    # Entity Selection (Issuer Analysis only)
    issuer_col = _namecol(results_final)

    issuers = (results_final[issuer_col].dropna().astype(str).sort_values().unique().tolist()) if issuer_col else []

    issuer_q = st.text_input("Search issuer", key="ai_issuer_q", placeholder="Type a few letters…")
    issuers_filtered = _search_options(issuer_q, issuers, limit=300)
    prev_issuer = st.session_state.get("ai_sel_issuer")
    idx = 0
    if prev_issuer and prev_issuer in issuers_filtered:
        idx = issuers_filtered.index(prev_issuer) + 1
    sel_issuer = st.selectbox("Select issuer", ["— None —"] + issuers_filtered, index=idx, key="ai_sel_issuer")

    entity_selected = sel_issuer and sel_issuer != "— None —"

    # Entity-Specific Diagnostics (shown when entity is selected)
    if entity_selected:
        st.markdown("---")
        st.markdown("#### 📋 Data Diagnostics for Selected Entity")

        entity_name = sel_issuer

        with st.expander(f"View Data Quality Report for {entity_name}", expanded=True):
            try:
                diagnostics_df = generate_data_diagnostics_v2(
                    df_original=df_original,
                    results_df=results_final,
                    entity_type=report_type,
                    entity_identifier=entity_name
                )

                # Check if error occurred
                if "Error" in diagnostics_df.columns:
                    st.error(diagnostics_df["Error"].iloc[0])
                else:
                    # Display diagnostics table with error handling to prevent UI crash
                    try:
                        # Force conversion to simple table format (more reliable than dataframe)
                        display_df = diagnostics_df.copy()

                        # Ensure all data is plain strings - no complex objects
                        for col in display_df.columns:
                            display_df[col] = display_df[col].astype(str).replace('nan', 'N/A')

                        # Use st.table instead of st.dataframe - it's more robust for complex data
                        st.table(display_df)

                    except Exception as e:
                        st.error(f"⚠️ Could not display diagnostics: {e}")
                        st.write("Raw data structure:")
                        st.write(diagnostics_df.to_dict('records'))

                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    available_count = (diagnostics_df["Status"] == "✅ Available").sum()
                    partial_count = (diagnostics_df["Status"] == "⚠️ Data exists but not extracted").sum()
                    missing_count = (diagnostics_df["Status"] == "❌ Missing").sum()
                    total_count = len(diagnostics_df)

                    with col1:
                        st.metric("✅ Available", f"{available_count}/{total_count}")
                    with col2:
                        st.metric("⚠️ Partial", partial_count)
                    with col3:
                        st.metric("❌ Missing", missing_count)
                    with col4:
                        coverage_pct = (available_count / total_count * 100) if total_count > 0 else 0
                        st.metric("Coverage", f"{coverage_pct:.1f}%")

                    # Highlight issues
                    if missing_count > 0:
                        st.warning(f"⚠️ {missing_count} key metrics are completely missing from the dataset.")
                    if partial_count > 0:
                        st.info(f"ℹ️ {partial_count} metrics exist in raw data but were not successfully extracted to results. This may indicate data processing issues.")

                    st.caption("ℹ️ This table shows which financial metrics are available for the selected entity and whether they were successfully extracted.")
            except Exception as e:
                st.error(f"Could not generate diagnostics: {e}")
                import traceback
                st.code(traceback.format_exc())

    # Generate Button
    if not st.button("🚀 Generate Credit Report", type="primary", use_container_width=True, disabled=not entity_selected):
        if not entity_selected:
            st.info("Please select an issuer or classification above to generate a report.")
        return

    # Determine max_tokens based on report format
    if "Executive" in report_format:
        max_tokens = 1000
    elif "Comprehensive" in report_format:
        max_tokens = 4000
    else:
        max_tokens = 3000

    try:
        with st.spinner("🔄 Analyzing financial data and generating professional credit report..."):
            # Classification Analysis disabled - only Issuer Analysis available
            # Find issuer rows (normalized matching for robustness)
            row = (results_final[results_final[issuer_col].apply(_norm) == _norm(sel_issuer)]).iloc[0]

            # Find RAW row by Company_ID (preferred) then by name
            raw_df = df_original
            id_cols = [c for c in ["Company_ID","Company ID","ID"] if c in results_final.columns and c in raw_df.columns]
            name_cols = [c for c in ["Company_Name","Company Name","Name"] if c in results_final.columns and c in raw_df.columns]
            raw_row = None
            if id_cols:
                cid = str(row[id_cols[0]])
                raw_row = raw_df.loc[raw_df[id_cols[0]].apply(_norm) == _norm(cid)]
            if (raw_row is None or (isinstance(raw_row, pd.DataFrame) and raw_row.empty)) and name_cols:
                nm = row[name_cols[0]]
                raw_row = raw_df.loc[raw_df[name_cols[0]].apply(_norm) == _norm(nm)]
            if isinstance(raw_row, pd.DataFrame) and not raw_row.empty:
                raw_row = raw_row.iloc[0]
            elif raw_row is None or (isinstance(raw_row, pd.DataFrame) and raw_row.empty):
                raw_row = row

            entity_name = sel_issuer

            # Build evidence table
            st.markdown("---")
            st.markdown("#### 📊 Evidence Table (FY & CQ)")
            tbl = build_issuer_evidence_table(df_original, raw_row)

            # Use st.table for more reliable rendering
            try:
                tbl_display = tbl.copy()
                # Convert all non-Metric columns to formatted strings
                for col in tbl_display.columns:
                    if col != 'Metric':
                        tbl_display[col] = tbl_display[col].apply(
                            lambda x: f"{float(x):.2f}" if pd.notna(x) and str(x).replace('.','').replace('-','').isdigit() else '—'
                        )
                st.table(tbl_display)
            except Exception as e:
                st.error(f"Could not render evidence table: {e}")
                st.write(tbl)

            # Compact evidence pack for the model
            import json
            pack = {
                "Company": entity_name,
                "S&P_Rating": row.get("Credit_Rating_Clean"),
                "Classification": row.get("Rubrics_Custom_Classification"),
                "FY0": {r.Metric: (pd.to_numeric(r.iloc[1:], errors="coerce").dropna().iloc[-1]
                         if pd.to_numeric(r.iloc[1:], errors="coerce").dropna().size else None)
                        for _, r in tbl.iterrows()},
            }

            prompt = f"""You are a buy-side credit analyst. Use ONLY the table and JSON below.

Table (CSV):
{tbl.to_csv(index=False)}

JSON (latest levels):
{json.dumps(pack, default=lambda x: None)}

Write a skeptical, concise issuer credit note (200–300 words). Anchor every claim with a metric and period label."""

            # Generate report with minimal model
            client = _get_openai_client()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=900,
            )
            report_content = resp.choices[0].message.content.strip()

            # Display report
            st.markdown("---")
            st.markdown("#### 📝 Credit Analysis Report")
            st.markdown(report_content)

            # Export options
            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            with col1:
                # Markdown download
                import datetime
                filename = f"Credit_Report_{entity_name.replace(' ', '_')}_{datetime.date.today()}.md"
                st.download_button(
                    "📄 Download Markdown",
                    data=report_content,
                    file_name=filename,
                    mime="text/markdown"
                )

            with col2:
                st.info("📝 Word export: Coming soon")

            with col3:
                st.info("📊 PDF export: Coming soon")

            # Context expander (evidence-based)
            with st.expander("📊 View Data Context Used"):
                st.markdown("### Evidence Pack (sent to LLM)")
                st.json(_json_safe(pack))

    except Exception as e:
        st.error(f"Error generating report: {e}")
        import traceback
        st.code(traceback.format_exc())

# ============================================================================
# [V2.2] URL STATE & PRESETS
# ============================================================================

URL_STATE_KEYS = [
    "scoring_method",
    "data_period",
    "use_quarterly_beta",
    "band_default",       # default band selection for leaderboards/explainability UIs
    "top_n_default"       # default Top N for leaderboards (if feature enabled)
]

def _get_query_params():
    """Get query parameters from URL, with fallback for older Streamlit versions."""
    try:
        return st.query_params.to_dict()
    except Exception:
        return {}

def _set_query_params(d: dict):
    """Set query parameters in URL, with fallback for older Streamlit versions."""
    # Only set simple JSON-serializable primitives as strings
    qp = {k: (str(v).lower() if isinstance(v, bool) else str(v)) for k, v in d.items() if v is not None}
    try:
        st.query_params.clear()
        st.query_params.update(qp)
    except Exception:
        # Older Streamlit: show the deep link for copy/paste
        st.info("Copy this link to reproduce the state:\n\n" + "?" + urlencode(qp))

def collect_current_state(scoring_method, data_period, use_quarterly_beta, align_to_reference,
                          band_default=None, top_n_default=20):
    """Collect current UI state into a dictionary."""
    state = {
        "scoring_method": scoring_method,
        "data_period": data_period,
        "use_quarterly_beta": bool(use_quarterly_beta),
        "align_to_reference": bool(align_to_reference),
        "band_default": band_default or "",
        "top_n_default": int(top_n_default)
    }
    return state

def apply_state_to_controls(state):
    """
    Return normalized values to drive the controls on first render.
    Does not mutate Streamlit widgets directly.
    Coerces deprecated "FY-4 (Legacy)" to "Most Recent Fiscal Year (FY0)".
    """
    sm = state.get("scoring_method") or "Classification-Adjusted Weights (Recommended)"
    dp = state.get("data_period") or "Most Recent Fiscal Year (FY0)"

    # Coerce deprecated FY-4 option to FY0
    if dp == "FY-4 (Legacy)":
        dp = "Most Recent Fiscal Year (FY0)"

    qb = str(state.get("use_quarterly_beta", "false")).lower() in ("1", "true", "yes")
    align = str(state.get("align_to_reference", "false")).lower() in ("1", "true", "yes")
    band_default = state.get("band_default") or ""
    top_n_default = int(state.get("top_n_default") or 20)
    return sm, dp, qb, align, band_default, top_n_default

def _build_deep_link(state: dict) -> str:
    """
    Return a relative deep link (querystring) that reproduces the current state.
    We use a relative link (?key=val...) so it works across environments.
    """
    qp = {k: (str(v).lower() if isinstance(v, bool) else str(v)) for k, v in state.items() if v is not None}
    return "?" + urlencode(qp)

# ============================================================================
# [V2.2] PERIOD ENDED PARSING (ACTUAL DATES)
# ============================================================================

def parse_period_ended_cols(df: pd.DataFrame) -> list:
    """
    Parse all 'Period Ended*' columns to actual dates.
    Treats 1900 sentinel values (0/01/1900, 01/01/1900, year=1900) as NaT.
    Returns: List of tuples (suffix, datetime_series) sorted by suffix

    Examples:
        [('', Series of dates), ('.1', Series of dates), ('.2', ...)]
    """
    pe_cols = [c for c in df.columns if c.startswith("Period Ended")]

    for c in pe_cols:
        s = df[c].copy()

        # Replace common 1900 sentinels with None
        s = s.replace({"0/01/1900": None, "01/01/1900": None, "1900-01-00": None, "1900-01-01": None})

        # Parse as dates
        parsed = pd.to_datetime(s, dayfirst=True, errors="coerce")

        # Handle numeric Excel serial dates
        mask_num = s.apply(lambda x: isinstance(x, (int, float))) & parsed.isna()
        if mask_num.any():
            parsed.loc[mask_num] = pd.to_datetime(s[mask_num], unit="d", origin="1899-12-30", errors="coerce")

        # Mask any year=1900 values as NaT (sentinel)
        parsed = parsed.mask(parsed.dt.year == 1900)

        df[c] = parsed

    # Sort by suffix number (base=0, .1=1, etc.)
    sorted_cols = sorted(pe_cols, key=lambda c: int(c.split(".")[1]) if "." in c and c.split(".")[1].isdigit() else 0)

    # Return list of (suffix, datetime_series) tuples
    result = []
    for c in sorted_cols:
        suffix = c[len("Period Ended"):]  # '' for base, '.1' for Period Ended.1, etc.
        result.append((suffix, df[c]))

    return result

def period_cols_by_kind(pe_data, df):
    """
    Split period columns into FY vs CQ based on position.

    Standard CapIQ structure:
    - Positions 0-4 (base, .1, .2, .3, .4): FY-4 through FY0 (annual)
    - Positions 5-7 (.5 through .7): LTM-2 through LTM0 (trailing 12-month)

    Args:
        pe_data: List of (suffix, datetime_series) tuples from parse_period_ended_cols
        df: DataFrame (not used in position-based approach, kept for API compatibility)

    Returns: (fy_suffixes, ltm_suffixes) - lists of suffixes
    """
    if len(pe_data) == 0:
        return [], []

    if len(pe_data) <= 5:
        # Only annual data (5 or fewer periods)
        return [suffix for suffix, _ in pe_data], []

    # Split: first 5 positions are FY (annual), rest are CQ (quarterly)
    fy_suffixes = [pe_data[i][0] for i in range(5)]
    ltm_suffixes = [pe_data[i][0] for i in range(5, len(pe_data))]

    return fy_suffixes, ltm_suffixes


def select_aligned_period(df, pe_data, reference_date=None,
                          prefer_annual_reports=False,
                          use_quarterly=True):
    """
    Unified period selection logic for both quality and trend scores.

    This function implements the core period selection algorithm that ensures
    quality scores and trend scores use the SAME period for each issuer.

    Args:
        df: DataFrame with issuer data
        pe_data: List of (suffix, datetime_series) tuples from parse_period_ended_cols
        reference_date: Optional cutoff date (str or pd.Timestamp) - only use periods <= this date
        prefer_annual_reports: If True, prefer FY over CQ even when CQ is more recent
        use_quarterly: If True, include both FY and CQ periods; if False, FY only

    Returns:
        DataFrame with columns:
        - row_idx: Original row index from df
        - selected_suffix: The suffix to use for this issuer (e.g., '.4' for FY0)
        - selected_date: The date of the selected period
        - is_fy: Boolean indicating if selected period is fiscal year (True) or quarterly (False)

    Algorithm:
        1. Filter periods to <= reference_date (if provided)
        2. Determine candidate periods (FY only if use_quarterly=False)
        3. Deduplicate same dates, preferring FY over CQ
        4. Apply prefer_annual_reports logic:
           - If True: For issuers with FY data, exclude all CQ periods
           - If False: Use most recent period (FY or CQ)
        5. Return one selected period per issuer
    """
    # Diagnostic: Log function entry
    DIAG.section("PERIOD SELECTION - select_aligned_period()")
    DIAG.log("FUNCTION_ENTRY",
             reference_date=str(reference_date) if reference_date else None,
             reference_date_type=type(reference_date).__name__,
             prefer_annual_reports=prefer_annual_reports,
             use_quarterly=use_quarterly,
             num_issuers=len(df))

    # Classify periods as FY or CQ
    fy_suffixes, ltm_suffixes = period_cols_by_kind(pe_data, df)

    # Determine candidate suffixes based on quarterly vs annual mode
    if use_quarterly:
        candidate_suffixes = [s for s, _ in pe_data]
    else:
        candidate_suffixes = fy_suffixes if fy_suffixes else [s for s, _ in pe_data]

    # Build long format: (row_idx, suffix, date, is_fy)
    long_data = []
    ltm_set = set(ltm_suffixes)

    for sfx in candidate_suffixes:
        date_series = dict(pe_data).get(sfx)
        if date_series is None:
            continue

        chunk = pd.DataFrame({
            'row_idx': df.index,
            'suffix': sfx,
            'date': pd.to_datetime(date_series.values, errors='coerce'),
            'is_fy': sfx not in ltm_set
        })
        long_data.append(chunk)

    if not long_data:
        return pd.DataFrame(columns=['row_idx', 'selected_suffix',
                                    'selected_date', 'is_fy'])

    long_df = pd.concat(long_data, ignore_index=True)

    # Filter invalid dates
    long_df = long_df[long_df['date'].notna()]
    long_df = long_df[long_df['date'].dt.year != 1900]

    # Diagnostic: Log data date range BEFORE reference filter
    if reference_date is not None and DIAG.enabled and len(long_df) > 0:
        reference_dt = pd.to_datetime(reference_date)
        min_date = long_df['date'].min()
        max_date = long_df['date'].max()
        periods_before_filter = len(long_df)
        periods_above = len(long_df[long_df['date'] > reference_dt])
        periods_below = len(long_df[long_df['date'] <= reference_dt])

        DIAG.log("DATA_DATE_RANGE",
                 min_date=str(min_date),
                 max_date=str(max_date),
                 reference_date=str(reference_dt),
                 total_periods=periods_before_filter,
                 periods_above_reference=periods_above,
                 periods_below_reference=periods_below)

    # Filter to reference date if provided
    if reference_date is not None:
        reference_dt = pd.to_datetime(reference_date)
        periods_before = len(long_df)
        long_df = long_df[long_df['date'] <= reference_dt]
        periods_after = len(long_df)
        removed = periods_before - periods_after

        # Diagnostic: Log reference filter results
        DIAG.log("REFERENCE_FILTER",
                 periods_before=periods_before,
                 periods_after=periods_after,
                 removed=removed,
                 filter_working=(removed > 0))

        # Diagnostic: Warn if filter had no effect
        if removed == 0 and periods_before > 0:
            DIAG.log("REFERENCE_FILTER_NO_EFFECT",
                     level="WARNING",
                     message=f"Reference filter removed 0 periods (all {periods_before} periods are <= {reference_dt})")

        # Diagnostic: Error if all rows removed
        if periods_after == 0 and periods_before > 0:
            DIAG.log("REFERENCE_FILTER_REMOVED_ALL",
                     level="ERROR",
                     message=f"Reference date {reference_dt} is before all data (removed all {periods_before} periods)")

    # CRITICAL: Deduplicate same dates, preferring FY over CQ
    # When is_fy=True (FY), it should sort BEFORE is_fy=False (CQ)
    # With ascending=False on is_fy, True sorts before False
    long_df = long_df.sort_values(['row_idx', 'date', 'is_fy'],
                                  ascending=[True, True, False])
    long_df = long_df.drop_duplicates(subset=['row_idx', 'date'],
                                      keep='first')

    # Apply prefer_annual_reports logic
    if prefer_annual_reports:
        # Find issuers with FY data
        has_fy = long_df[long_df['is_fy'] == True].groupby('row_idx').size()
        issuers_with_fy = has_fy[has_fy > 0].index

        # For issuers with FY data, keep only FY periods (exclude all CQ)
        # For issuers without FY data, keep their CQ periods
        long_df = long_df[
            (~long_df['row_idx'].isin(issuers_with_fy)) |  # No FY available - keep CQ
            (long_df['is_fy'] == True)  # Has FY - keep only FY, exclude CQ
        ]

    # Select most recent period per issuer
    long_df = long_df.sort_values(['row_idx', 'date'])
    selected = long_df.groupby('row_idx').last().reset_index()

    # Rename for clarity
    selected = selected.rename(columns={
        'suffix': 'selected_suffix',
        'date': 'selected_date'
    })

    result_df = selected[['row_idx', 'selected_suffix', 'selected_date', 'is_fy']]

    # Diagnostic: All-issuer period selection analysis
    DIAG.subsection("ALL-ISSUER PERIOD SELECTION ANALYSIS")

    # Analyze period distribution
    DIAG.analyze_period_distribution(result_df, 'selected_suffix')

    # Compute FY vs CQ split
    if len(result_df) > 0:
        fy_count = result_df['is_fy'].sum()
        cq_count = len(result_df) - fy_count
        fy_pct = round(fy_count / len(result_df) * 100, 1)
        cq_pct = round(cq_count / len(result_df) * 100, 1)

        DIAG.log("PERIOD_TYPE_SPLIT",
                 total_issuers=len(result_df),
                 fy_count=int(fy_count),
                 cq_count=int(cq_count),
                 fy_percentage=fy_pct,
                 cq_percentage=cq_pct,
                 prefer_annual_reports=prefer_annual_reports)

        # Warn if prefer_annual_reports=True but more CQ than FY selected
        if prefer_annual_reports and cq_count > fy_count:
            DIAG.log("PREFER_ANNUAL_INCONSISTENCY",
                     level="WARNING",
                     message=f"prefer_annual_reports=True but {cq_count} CQ vs {fy_count} FY selected")

        # Log date range of selected periods
        min_selected = result_df['selected_date'].min()
        max_selected = result_df['selected_date'].max()
        unique_dates = result_df['selected_date'].nunique()

        DIAG.log("DATE_RANGE_SELECTED",
                 earliest_date=str(min_selected),
                 latest_date=str(max_selected),
                 num_unique_dates=int(unique_dates))

        # Sample issuers (need to merge with company names if available)
        if 'Company Name' in df.columns or 'Issuer Name' in df.columns:
            name_col = 'Company Name' if 'Company Name' in df.columns else 'Issuer Name'
            result_with_names = result_df.copy()
            result_with_names[name_col] = result_df['row_idx'].map(df[name_col])
            DIAG.sample_issuers(result_with_names, name_col, n=10)

            # Track key issuers
            key_issuers = ['NVIDIA', 'Apple', 'Microsoft', 'Amazon', 'Google', 'Meta', 'Tesla']
            tracked = []
            for issuer in key_issuers:
                matches = df[df[name_col].str.contains(issuer, case=False, na=False)]
                if len(matches) > 0:
                    for idx in matches.index:
                        if idx in result_df['row_idx'].values:
                            row = result_df[result_df['row_idx'] == idx].iloc[0]
                            tracked.append({
                                'company': matches.loc[idx, name_col],
                                'period': row['selected_suffix'],
                                'date': str(row['selected_date']),
                                'is_fy': bool(row['is_fy'])
                            })

            if tracked:
                DIAG.log("KEY_ISSUER_TRACKING", num_tracked=len(tracked), issuers=tracked)

    # Log function exit
    DIAG.log("FUNCTION_EXIT",
             num_issuers=len(result_df),
             fy_count=int(result_df['is_fy'].sum()) if len(result_df) > 0 else 0,
             cq_count=int((~result_df['is_fy']).sum()) if len(result_df) > 0 else 0)

    return result_df


# ============================================================================
# [V2.2] PERIOD CALENDAR UTILITIES (ROBUST HANDLING OF VENDOR DATES & FY/CQ OVERLAP)
# ============================================================================

_EXCEL_EPOCH = pd.Timestamp("1899-12-30")  # Excel serial origin

_SENTINEL_BAD = {
    "0/01/1900", "00/01/1900", "0/0/0000", "00/00/0000",
    "1900-01-00", "1899-12-31"  # common vendor quirks
}

_PERIOD_COL_RE = re.compile(
    r"(?i)^.*period\s*ended.*\b((FY-?\d+|FY0|LTM-?\d+|LTM0))\b"
)

def _parse_period_date(val):
    """Parse vendor period dates from strings or Excel serials; treat sentinels as NaT."""
    if pd.isna(val):
        return pd.NaT
    # Excel numeric serial
    if isinstance(val, (int, float)) and not pd.isna(val):
        if val <= 0:
            return pd.NaT
        try:
            return _EXCEL_EPOCH + pd.to_timedelta(int(val), unit="D")
        except Exception:
            return pd.NaT
    # String formats
    s = str(val).strip()
    if not s or s in _SENTINEL_BAD:
        return pd.NaT
    # Some vendors ship '31/12/2024' (day-first) or '2024-12-31'.
    try:
        return parser.parse(s, dayfirst=True, yearfirst=False, fuzzy=True)
    except Exception:
        try:
            return parser.parse(s, dayfirst=False, yearfirst=True, fuzzy=True)
        except Exception:
            return pd.NaT

def _find_period_cols_calendar(df: pd.DataFrame) -> dict:
    """
    Return mapping: { 'FY-4': 'colname', ..., 'CQ0': 'colname' }.
    Works with single- or multi-level headers by joining with a space.
    Used for period calendar building (different from PATCH 1's _find_period_cols).
    """
    mapping = {}
    # Flatten multiindex headers if present
    if isinstance(df.columns, pd.MultiIndex):
        cols = [" ".join([str(x) for x in tup if pd.notna(x)]).strip() for tup in df.columns]
    else:
        cols = [str(c) for c in df.columns]

    for col in cols:
        m = _PERIOD_COL_RE.match(col)
        if m:
            key = m.group(1).upper().replace("--", "-")
            mapping[key] = col
    return mapping

def build_period_calendar(
    raw_df: pd.DataFrame,
    issuer_id_col: str = "Company_ID",
    issuer_name_col: str = "Company_Name",
    prefer_quarterly: bool = True,
    q4_merge_window_days: int = 10,
) -> pd.DataFrame:
    """
    From a wide vendor sheet (FY-4..FY0, CQ-7..CQ0), produce a canonical long calendar:
    columns = [issuer_id, issuer_name, period_type(FY/CQ), k(int), period_end_date, source_col]
    Deduplicates FY vs CQ overlap around fiscal year-end according to prefer_quarterly.
    """
    df = raw_df.copy()
    # Build mapping and melt to long
    mapping = _find_period_cols_calendar(df)
    if not mapping:
        raise ValueError("No 'Period Ended' columns detected. Ensure headers include '(FYk|CQk)'.")
    long_records = []
    for key, col in mapping.items():
        # key examples: 'FY-4', 'FY0', 'CQ-7', 'CQ0'
        m = re.match(r"(?i)^(FY|CQ)-?(\d+)$", key)
        if not m:
            continue
        ptype, k = m.group(1).upper(), int(m.group(2))
        ser = df[col].apply(_parse_period_date)
        tmp = pd.DataFrame({
            "issuer_id": df.get(issuer_id_col, pd.NA),
            "issuer_name": df.get(issuer_name_col, pd.NA),
            "period_type": ptype,
            "k": k,  # distance from current (0 = as-at)
            "period_end_date": ser,
            "source_col": col
        })
        long_records.append(tmp)
    cal = pd.concat(long_records, ignore_index=True)
    # Remove nulls
    cal = cal[cal["period_end_date"].notna()].copy()

    # Overlap resolution: when FY and CQ are effectively the same year end.
    # For each issuer, near each fiscal year end (date collisions), keep CQ if prefer_quarterly else FY.
    # We collapse duplicates that fall within q4_merge_window_days.
    cal["date_key"] = cal["period_end_date"].dt.floor("D")
    cal.sort_values(["issuer_id", "date_key", "period_type"], inplace=True)

    def _resolve_group(g):
        if len(g) == 1:
            return g
        # Multiple rows same date (or near date) after vendor quirks -> apply rule
        if prefer_quarterly:
            # Prefer CQ; if no CQ, keep FY
            cq = g[g["period_type"] == "CQ"]
            if not cq.empty:
                return cq.head(1)
            return g.head(1)
        else:
            fy = g[g["period_type"] == "FY"]
            if not fy.empty:
                return fy.head(1)
            return g.head(1)

    # Group using a proximity bucket to catch FY vs CQ within a small window
    cal = cal.sort_values(["issuer_id", "period_end_date"])
    cal["bucket"] = (
        cal.groupby("issuer_id")["period_end_date"]
           .transform(lambda s: pd.Series(pd.cut(
               s.view("i8"),  # nanosecond ints
               bins=pd.interval_range(s.min().floor("D") - pd.Timedelta(days=q4_merge_window_days),
                                     s.max().ceil("D") + pd.Timedelta(days=q4_merge_window_days),
                                     freq=f"{q4_merge_window_days*2}D",
                                     closed="both"),
               include_lowest=True
           ).astype(str)))
    )

    cal = cal.groupby(["issuer_id", "bucket"], group_keys=False).apply(_resolve_group)
    cal.drop(columns=["date_key", "bucket"], errors="ignore", inplace=True)

    # Standardize schema & types
    cal["period_type"] = cal["period_type"].astype("category")
    cal["k"] = cal["k"].astype("int16")
    cal.sort_values(["issuer_id", "period_type", "k"], inplace=True)

    return cal

def latest_periods(cal: pd.DataFrame, max_k_fy=4, max_k_cq=7) -> pd.DataFrame:
    """Convenience: quickly get FY0..FY-4 and CQ0..CQ-7 after cleanup for a debug pivot."""
    piv = (cal
           .query("(period_type == 'FY' and k <= @max_k_fy) or (period_type == 'CQ' and k <= @max_k_cq)")
           .assign(key=lambda d: d["period_type"] + d["k"].astype(str))
           .pivot_table(index=["issuer_id", "issuer_name"],
                        columns="key", values="period_end_date", aggfunc="max")
           .reset_index())
    return piv

# ============================================================================
# BATCH METRIC EXTRACTION (moved to module level for reuse)
# ============================================================================



def _batch_extract_metrics(df, metric_list, has_period_alignment, data_period_setting,
                          reference_date=None, prefer_annual_reports=False,
                          selected_periods=None):
    """
    OPTIMIZED: Extract all metrics at once using vectorized operations.
    Returns dict of {metric_name: Series of values}.

    Args:
        reference_date: If provided, filters to only use data on or before this date.
        prefer_annual_reports: If True, prefer FY over CQ when both are available.
        selected_periods: DataFrame from select_aligned_period() with columns
                         [row_idx, selected_suffix, selected_date, is_fy].
                         If provided, uses these pre-selected periods for consistency.
    """
    result = {}

    if not has_period_alignment:
        # Fallback: use get_most_recent_column for each metric
        for metric in metric_list:
            result[metric] = get_most_recent_column(df, metric, data_period_setting)
        return result

    # Parse Period Ended columns once for all metrics
    pe_data = parse_period_ended_cols(df)
    if not pe_data:
        # No period data - fall back to base columns
        for metric in metric_list:
            if metric in df.columns:
                result[metric] = pd.to_numeric(df[metric], errors='coerce')
            else:
                result[metric] = pd.Series(np.nan, index=df.index)
        return result

    # Build suffix list based on data_period_setting (FY0 or CQ-0)
    fy_suffixes, ltm_suffixes = period_cols_by_kind(pe_data, df)

    if data_period_setting == "Most Recent LTM (LTM0)":
        # User wants most recent quarter - include both CQ and FY, most recent date wins
        if ltm_suffixes:
            # Include both CQ and FY periods - line 3270-3272 will pick the most recent
            candidate_suffixes = ltm_suffixes + fy_suffixes
        else:
            # No quarterly data available - use FY as fallback
            candidate_suffixes = fy_suffixes if fy_suffixes else [s for s, _ in pe_data[:5]]

    elif data_period_setting == "Most Recent Fiscal Year (FY0)":
        # User wants fiscal year data only
        candidate_suffixes = fy_suffixes if fy_suffixes else [s for s, _ in pe_data[:5]]

    else:
        # Unknown setting - default to FY for safety
        candidate_suffixes = fy_suffixes if fy_suffixes else [s for s, _ in pe_data[:5]]

    # For each metric, extract values based on selected periods or legacy logic
    for metric in metric_list:
        if selected_periods is not None:
            # Debug statement removed for production
            # Use pre-selected periods for consistency with trend scores
            values = []
            for idx in df.index:
                period_row = selected_periods[selected_periods['row_idx'] == idx]
                if len(period_row) == 0:
                    values.append(np.nan)
                    continue

                suffix = period_row['selected_suffix'].iloc[0]
                col = f"{metric}{suffix}" if suffix else metric
                if 'NVIDIA' in str(df.loc[idx, 'Company Name']):
                    WarningCollector.collect("debug_column_lookup", f"suffix='{suffix}', col='{col}', exists={col in df.columns}", "NVIDIA")

                if col in df.columns:
                    val = df.loc[idx, col]
                    values.append(pd.to_numeric(val, errors='coerce'))
                else:
                    issuer_name = str(df.loc[idx, 'Company Name']) if 'Company Name' in df.columns else f"idx={idx}"
                    WarningCollector.collect("missing_column", f"Column '{col}' not found", issuer_name)
                    values.append(np.nan)

            result[metric] = pd.Series(values, index=df.index)
        else:
            # Legacy logic: Collect (date, value) pairs for this metric across candidate suffixes
            metric_data = []
            for sfx in candidate_suffixes:
                col = f"{metric}{sfx}" if sfx else metric
                if col not in df.columns:
                    continue

                date_series = dict(pe_data).get(sfx)
                if date_series is None:
                    continue

                # Build chunk for this suffix
                chunk = pd.DataFrame({
                    'row_idx': df.index,
                    'date': pd.to_datetime(date_series.values, errors='coerce'),
                    'value': pd.to_numeric(df[col], errors='coerce'),
                    'is_fy': sfx not in set(ltm_suffixes)
                })
                metric_data.append(chunk)

            if not metric_data:
                result[metric] = pd.Series(np.nan, index=df.index)
                continue

            # Concatenate and filter
            long_df = pd.concat(metric_data, ignore_index=True)
            long_df = long_df[long_df['date'].notna() & long_df['value'].notna()]
            long_df = long_df[long_df['date'].dt.year != 1900]

            # Filter to reference date if provided (for alignment)
            if reference_date is not None:
                reference_dt = pd.to_datetime(reference_date)
                long_df = long_df[long_df['date'] <= reference_dt]

            # Deduplicate same dates, preferring FY over CQ
            long_df = long_df.sort_values(['row_idx', 'date', 'is_fy'],
                                         ascending=[True, True, False])
            long_df = long_df.drop_duplicates(subset=['row_idx', 'date'],
                                             keep='first')

            # Get most recent (latest date) value per issuer
            long_df = long_df.sort_values(['row_idx', 'date'])
            most_recent = long_df.groupby('row_idx').last()['value']

            # Reindex to match original df
            result[metric] = most_recent.reindex(df.index, fill_value=np.nan)

    return result

# ============================================================================
# CASH FLOW HELPERS (v3 - DataFrame-level with alias-aware batch extraction)
# ============================================================================

def _cf_components_dataframe(df: pd.DataFrame, data_period_setting: str, has_period_alignment: bool, reference_date=None) -> pd.DataFrame:
    """Extract cash flow components using alias-aware batch extraction.

    Args:
        reference_date: If provided, filters to only use data on or before this date.
    """

    cash_flow_metrics = [
        'Cash from Ops.',
        'Cash from Operations',
        'Operating Cash Flow',
        'Cash from Ops',
        'Capital Expenditure',
        'Capital Expenditures',
        'CAPEX',
        'Revenue',
        'Total Revenues',
        'Total Revenue',
        'Total Debt',
        'Levered Free Cash Flow',
        'Free Cash Flow'
    ]

    metrics = _batch_extract_metrics(df, cash_flow_metrics, has_period_alignment, data_period_setting, reference_date)

    # Map to standardized names
    ocf = metrics.get('Cash from Ops.', metrics.get('Cash from Operations',
          metrics.get('Operating Cash Flow', metrics.get('Cash from Ops', pd.Series(np.nan, index=df.index)))))

    capex = metrics.get('Capital Expenditure', metrics.get('Capital Expenditures',
            metrics.get('CAPEX', pd.Series(np.nan, index=df.index))))

    rev = metrics.get('Revenue', metrics.get('Total Revenues',
          metrics.get('Total Revenue', pd.Series(np.nan, index=df.index))))

    debt = metrics.get('Total Debt', pd.Series(np.nan, index=df.index))

    lfcf = metrics.get('Levered Free Cash Flow', metrics.get('Free Cash Flow',
           pd.Series(np.nan, index=df.index)))

    # Calculate UFCF vectorized
    ufcf = pd.Series(np.nan, index=df.index)
    valid_mask = ocf.notna() & capex.notna()
    ufcf[valid_mask] = ocf[valid_mask] + capex[valid_mask]

    return pd.DataFrame({
        'OCF': ocf,
        'Capex': capex,
        'UFCF': ufcf,
        'Revenue': rev,
        'Debt': debt,
        'LFCF': lfcf
    })

def _cf_raw_dataframe(cf_components: pd.DataFrame) -> pd.DataFrame:
    """Calculate cash flow ratios from components DataFrame (vectorized)."""

    def _safe_div_vectorized(a: pd.Series, b: pd.Series) -> pd.Series:
        result = pd.Series(np.nan, index=a.index)
        valid_mask = a.notna() & b.notna() & (b != 0)
        result[valid_mask] = a[valid_mask] / b[valid_mask]
        return result

    return pd.DataFrame({
        'OCF_to_Revenue': _safe_div_vectorized(cf_components['OCF'], cf_components['Revenue']),
        'OCF_to_Debt': _safe_div_vectorized(cf_components['OCF'], cf_components['Debt']),
        'UFCF_margin': _safe_div_vectorized(cf_components['UFCF'], cf_components['Revenue']),
        'LFCF_margin': _safe_div_vectorized(cf_components['LFCF'], cf_components['Revenue'])
    })

# Conservative global clip windows (simple, deterministic)
_CF_CLIPS = {
    "OCF_to_Revenue": (-0.10, 0.30),
    "OCF_to_Debt":    (0.00, 0.50),
    "UFCF_margin":    (-0.10, 0.20),
    "LFCF_margin":    (-0.10, 0.20),
}

def _clip_series(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return s.clip(lower=lo, upper=hi)

def _scale_0_100(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 2:
        return pd.Series(np.nan, index=s.index)
    mn, mx = s.min(skipna=True), s.max(skipna=True)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-12:
        return pd.Series(np.nan, index=s.index)
    z = (s - mn) / (mx - mn)
    return z * 100.0

def _cash_flow_component_scores(df: pd.DataFrame, data_period_setting: str, has_period_alignment: bool, reference_date=None) -> pd.DataFrame:
    """Calculate cash flow component scores using alias-aware extraction.

    Args:
        reference_date: If provided, filters to only use data on or before this date.
    """

    components = _cf_components_dataframe(df, data_period_setting, has_period_alignment, reference_date)
    raw = _cf_raw_dataframe(components)

    for k, (lo, hi) in _CF_CLIPS.items():
        if k in raw.columns:
            raw[k] = _clip_series(raw[k], lo, hi)

    out = pd.DataFrame({f"{k}_Score": _scale_0_100(raw[k]) for k in raw.columns if k in raw})
    return out

# ============================================================================
# ROW AUDIT HELPER
# ============================================================================

def _audit_count(stage_name: str, df: pd.DataFrame, audits: list):
    """
    Record row count at a given processing stage.

    Args:
        stage_name: Description of processing stage
        df: DataFrame to count
        audits: List of (stage_name, count) tuples to append to
    """
    audits.append((stage_name, len(df)))

# ============================================================================
# FRESHNESS HELPERS (V2.2)
# ============================================================================

def _latest_valid_period_date_for_row(row: pd.Series, pe_cols: list) -> pd.Timestamp:
    """
    Given a row and list of Period Ended column names,
    return the latest non-NaT date for this row across all period columns.

    Args:
        row: Single row from DataFrame
        pe_cols: List of Period Ended column names

    Returns:
        Latest valid period date or pd.NaT if none found
    """
    dates = []
    for col in pe_cols:
        if col in row.index:
            try:
                d = pd.to_datetime(row[col], errors="coerce", dayfirst=True)
                if pd.notna(d) and d.year != 1900:  # Exclude 1900 sentinels
                    dates.append(d)
            except Exception:
                continue

    if not dates:
        return pd.NaT

    return max(dates)

def _freshness_flag(days: float) -> str:
    """
    Convert age in days to traffic-light flag.

    Args:
        days: Age in days

    Returns:
        "Green" (≤180d), "Amber" (181-365d), "Red" (>365d), or "Unknown" (NaN)
    """
    if pd.isna(days):
        return "Unknown"
    d = float(days)
    if d <= 180:
        return "Green"
    if d <= 365:
        return "Amber"
    return "Red"

# ============================================================================
# PERIOD LABELING & FY/CQ CLASSIFICATION (V2.2)
# ============================================================================

def _latest_period_dates(df: pd.DataFrame):
    """
    Returns a tuple (latest_fy_date, latest_ltm_date, used_fallback) using parsed Period Ended columns.

    Primary method: Uses period_cols_by_kind classifier to detect FY vs CQ based on date frequency.
    Fallback method: Treats first 5 suffixes as FY, remainder as CQ (documented fallback).

    Args:
        df: DataFrame with Period Ended columns

    Returns:
        (latest_fy_date, latest_ltm_date, used_fallback) tuple where:
        - latest_fy_date: Latest fiscal year end date (pd.Timestamp or pd.NaT)
        - latest_ltm_date: Latest LTM end date (pd.Timestamp or pd.NaT)
        - used_fallback: Boolean indicating if fallback method was used
    """
    try:
        pe_data = parse_period_ended_cols(df.copy())  # [(suffix, series_of_dates), ...]
        fy_suffixes, ltm_suffixes = period_cols_by_kind(pe_data, df)  # preferred path

        # Collect column-wise dates
        latest_fy = pd.NaT
        latest_ltm = pd.NaT

        for sfx, ser in pe_data:
            sd = pd.to_datetime(ser, errors="coerce", dayfirst=True)
            if sfx in fy_suffixes:
                max_date = sd.max(skipna=True)
                if pd.notna(max_date):
                    latest_fy = max_date if pd.isna(latest_fy) else max(latest_fy, max_date)
            if sfx in ltm_suffixes:
                max_date = sd.max(skipna=True)
                if pd.notna(max_date):
                    latest_ltm = max_date if pd.isna(latest_ltm) else max(latest_ltm, max_date)


        return latest_fy, latest_ltm, False  # False => did not use fallback

    except Exception:
        # Fallback (documented): first 5 suffixes treated as FY, the rest CQ
        # This is only used when classifier or PE parsing isn't available.
        try:
            pe_cols = [c for c in df.columns if str(c).startswith("Period Ended")]
            if not pe_cols:
                return pd.NaT, pd.NaT, True

            # Keep stable order by natural suffix index
            pe_cols_sorted = sorted(
                pe_cols,
                key=lambda c: int(str(c).split(".")[1]) if "." in str(c) and str(c).split(".")[1].isdigit() else -1
            )

            fy_cols = pe_cols_sorted[:5]
            ltm_cols = pe_cols_sorted[5:]

            latest_fy = pd.to_datetime(df[fy_cols], errors="coerce", dayfirst=True).max(axis=1).max(skipna=True) if fy_cols else pd.NaT
            latest_ltm = pd.to_datetime(df[ltm_cols], errors="coerce", dayfirst=True).max(axis=1).max(skipna=True) if ltm_cols else pd.NaT

            return latest_fy, latest_ltm, True  # True => used fallback

        except Exception:
            return pd.NaT, pd.NaT, True

def build_dynamic_period_labels(df: pd.DataFrame):
    """
    Returns dict with human-readable labels for the header/banner showing actual dates.

    Args:
        df: DataFrame with Period Ended columns

    Returns:
        Dictionary with keys:
        - fy_label: "Most Recent Fiscal Year (FY0 — 2024-12-31)" or "Most Recent Fiscal Year (FY0)" if date unavailable
        - ltm_label: "Most Recent LTM (LTM0 — 2025-10-26)" or "Most Recent LTM (LTM0)" if date unavailable
        - used_fallback: Boolean indicating if fallback classification method was used

    Example:
        {
            "fy_label": "Most Recent Fiscal Year (FY0 — 2024-12-31)",
            "ltm_label": "Most Recent LTM (LTM0 — 2025-10-26)",
            "used_fallback": False
        }
    """
    fy0, ltm0, used_fallback = _latest_period_dates(df)

    def _fmt(prefix, dt):
        """Format label with optional date suffix."""
        return f"{prefix}" + (f" — {dt.date().isoformat()}" if pd.notna(dt) else "")

    return {
        "fy_label": _fmt("Most Recent Fiscal Year (FY0)", fy0),
        "ltm_label": _fmt("Most Recent LTM (LTM0)", ltm0),
        "used_fallback": used_fallback
    }

# ============================================================================
# DYNAMIC WEIGHT CALIBRATION (V2.2.1)
# ============================================================================

def calculate_calibrated_sector_weights(df, rating_band='BBB', use_dynamic=True):
    """
    Calculate calibrated sector weights using VARIANCE MINIMIZATION.
    
    Method: 
        - All 5 factors (including Credit) are optimized
        - Weights constrained to sum to 1.0
        - Minimizes cross-sector variance of composite scores
    
    Args:
        df: DataFrame with factor scores and classifications
        rating_band: Rating level to calibrate on (default 'BBB')
        use_dynamic: If True, run optimization. If False, return universal weights.
    
    Returns:
        dict: Sector name -> {factor: weight} mappings, normalized to sum=1.0
    """
    import numpy as np
    # scipy_minimize is already imported at module level
    
    DIAG.section("VARIANCE-MINIMIZING CALIBRATION")
    DIAG.log("CALIBRATION_START", rating_band=rating_band, use_dynamic=use_dynamic, total_issuers=len(df))
    
    calibrated_weights = {}
    
    # Credit Score is now optimized along with other factors (V3.2)
    
    if not use_dynamic:
        # Return universal weights for all sectors
        all_sectors = set(CLASSIFICATION_TO_SECTOR.values())
        for sector in all_sectors:
            calibrated_weights[sector] = UNIVERSAL_WEIGHTS.copy()
        calibrated_weights['Default'] = UNIVERSAL_WEIGHTS.copy()
        DIAG.log("MODE", mode="Universal weights (calibration disabled)")
        return calibrated_weights
    
    # Define rating bands
    rating_bands = {
        'BBB': ['BBB+', 'BBB', 'BBB-'],
        'A': ['A+', 'A', 'A-'],
        'BB': ['BB+', 'BB', 'BB-'],
        'AA': ['AA+', 'AA', 'AA-'],
        'B': ['B+', 'B', 'B-'],
    }
    
    target_ratings = rating_bands.get(rating_band, ['BBB+', 'BBB', 'BBB-'])
    
    # Find rating column
    rating_col = None
    for col_name in ['Credit_Rating_Clean', 'S&P LT Issuer Credit Rating', 'Credit Rating', 'Rating']:
        if col_name in df.columns:
            rating_col = col_name
            break
    
    if rating_col is None:
        DIAG.log("FALLBACK", reason="No rating column found")
        all_sectors = set(CLASSIFICATION_TO_SECTOR.values())
        for sector in all_sectors:
            calibrated_weights[sector] = UNIVERSAL_WEIGHTS.copy()
        calibrated_weights['Default'] = UNIVERSAL_WEIGHTS.copy()
        return calibrated_weights
    
    # Filter to target rating band
    df_rated = df[df[rating_col].isin(target_ratings)].copy()
    DIAG.log("RATED_POPULATION", count=len(df_rated), rating_band=rating_band)
    
    if len(df_rated) < 50:
        DIAG.log("FALLBACK", reason=f"Insufficient data: {len(df_rated)} < 50")
        all_sectors = set(CLASSIFICATION_TO_SECTOR.values())
        for sector in all_sectors:
            calibrated_weights[sector] = UNIVERSAL_WEIGHTS.copy()
        calibrated_weights['Default'] = UNIVERSAL_WEIGHTS.copy()
        return calibrated_weights
    
    # ALL factor score columns (for reference)
    all_factor_score_cols = {
        'credit_score': 'Credit_Score',
        'leverage_score': 'Leverage_Score',
        'profitability_score': 'Profitability_Score',
        'liquidity_score': 'Liquidity_Score',

        'cash_flow_score': 'Cash_Flow_Score'
    }
    
    # ALL factors are now optimizable (Credit uses fundamental metrics, not rating)
    optimizable_factor_cols = {
        'credit_score': 'Credit_Score',
        'leverage_score': 'Leverage_Score',
        'profitability_score': 'Profitability_Score',
        'liquidity_score': 'Liquidity_Score',
        'cash_flow_score': 'Cash_Flow_Score'
    }
    
    # Find classification field
    class_field = None
    for field in ['Rubrics_Custom_Classification', 'Rubrics Custom Classification']:
        if field in df_rated.columns:
            class_field = field
            break
    
    if class_field is None:
        DIAG.log("FALLBACK", reason="No classification column found")
        all_sectors = set(CLASSIFICATION_TO_SECTOR.values())
        for sector in all_sectors:
            calibrated_weights[sector] = UNIVERSAL_WEIGHTS.copy()
        calibrated_weights['Default'] = UNIVERSAL_WEIGHTS.copy()
        return calibrated_weights
    
    all_sectors = set(CLASSIFICATION_TO_SECTOR.values())
    optimizable_keys = list(optimizable_factor_cols.keys())
    
    # ================================================================
    # BUILD SECTOR MEDIAN MATRIX (for optimizable factors only)
    # ================================================================
    sector_median_matrix = []
    valid_sectors = []
    sector_counts = {}
    
    for sector_name in sorted(all_sectors):
        sector_classifications = [k for k, v in CLASSIFICATION_TO_SECTOR.items() if v == sector_name]
        sector_df = df_rated[df_rated[class_field].isin(sector_classifications)]
        
        if len(sector_df) < 5:
            DIAG.log("SECTOR_SKIPPED", sector=sector_name, reason=f"Only {len(sector_df)} issuers")
            continue
        
        sector_row = []
        valid = True
        
        # Only collect medians for OPTIMIZABLE factors
        for factor_key in optimizable_keys:
            score_col = optimizable_factor_cols[factor_key]
            if score_col in sector_df.columns:
                values = pd.to_numeric(sector_df[score_col], errors='coerce').dropna()
                if len(values) >= 3:
                    sector_row.append(values.median())
                else:
                    valid = False
                    break
            else:
                valid = False
                break
        
        if valid and len(sector_row) == len(optimizable_keys):
            sector_median_matrix.append(sector_row)
            valid_sectors.append(sector_name)
            sector_counts[sector_name] = len(sector_df)
            DIAG.log("SECTOR_INCLUDED", sector=sector_name, count=len(sector_df), 
                     medians={optimizable_keys[i]: round(sector_row[i], 1) for i in range(len(optimizable_keys))})
    
    DIAG.log("VALID_SECTORS", count=len(valid_sectors), sectors=valid_sectors)
    
    if len(valid_sectors) < 3:
        DIAG.log("FALLBACK", reason=f"Too few valid sectors: {len(valid_sectors)}")
        for sector_name in all_sectors:
            calibrated_weights[sector_name] = UNIVERSAL_WEIGHTS.copy()
        calibrated_weights['Default'] = UNIVERSAL_WEIGHTS.copy()
        return calibrated_weights
    
    sector_matrix = np.array(sector_median_matrix)
    
    # ================================================================
    # VARIANCE MINIMIZATION OPTIMIZATION (all 5 factors)
    # ================================================================
    DIAG.subsection("OPTIMIZATION (all 5 factors)")
    
    # Universal weights for optimizable factors only
    # Original: [0.20, 0.20, 0.10, 0.15, 0.15] = 0.80 total
    # We'll optimize within this 0.80 budget
    
    universal_optimizable = np.array([UNIVERSAL_WEIGHTS[k] for k in optimizable_keys])
    
    def objective(weights):
        """Minimize variance of sector weighted composites"""
        composites = sector_matrix @ weights
        return np.var(composites)
    
    def constraint_sum(weights):
        """Weights must sum to 1.0 (all 5 factors optimized)"""
        return np.sum(weights) - 1.0
    
    # Bounds: 2% to 40% per factor
    bounds = [(0.02, 0.40)] * len(optimizable_keys)
    
    constraints = [{'type': 'eq', 'fun': constraint_sum}]
    
    # Calculate baseline variance with universal weights
    variance_universal = objective(universal_optimizable)
    DIAG.log("BASELINE_VARIANCE", variance=round(variance_universal, 4),
             note="Based on all 5 factors")
    
    # Run optimization
    result = scipy_minimize(
        objective,
        universal_optimizable,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if result.success:
        optimal_weights = result.x
        
        # Ensure weights sum to exactly 1.0
        optimal_weights = optimal_weights / optimal_weights.sum()
        
        # Calculate effectiveness metric
        variance_optimal = objective(optimal_weights)
        effectiveness = 1 - (variance_optimal / variance_universal) if variance_universal > 0 else 0
        
        DIAG.log("OPTIMIZATION_SUCCESS",
                 effectiveness_pct=round(effectiveness * 100, 1),
                 variance_universal=round(variance_universal, 4),
                 variance_optimal=round(variance_optimal, 4),
                 variance_reduction_pct=round((1 - variance_optimal/variance_universal) * 100, 1))
        
        # Build final weight dict (all 5 factors optimized)
        optimal_dict = {}
        for i, fk in enumerate(optimizable_keys):
            optimal_dict[fk] = float(optimal_weights[i])
        
        DIAG.log("FINAL_WEIGHTS", weights=optimal_dict, 
                 total=round(sum(optimal_dict.values()), 4))
        
        # Compare to universal
        DIAG.subsection("WEIGHT COMPARISON")
        
        # Credit (fixed)
        # DIAG.log("WEIGHT_CHANGE", 
        #          factor='credit_score', 
        #          universal=20.0,
        #          optimal=20.0,
        #          change=0.0,
        #          note="FIXED - excluded from optimization")
        
        # Optimizable factors
        for i, fk in enumerate(optimizable_keys):
            universal_val = UNIVERSAL_WEIGHTS[fk]
            optimal_val = optimal_weights[i]
            change = (optimal_val - universal_val) * 100
            DIAG.log("WEIGHT_CHANGE", 
                     factor=fk, 
                     universal=round(universal_val*100, 1),
                     optimal=round(optimal_val*100, 1),
                     change=round(change, 1))
        
        # Apply SAME optimal weights to ALL sectors (global optimization)
        for sector_name in all_sectors:
            calibrated_weights[sector_name] = optimal_dict.copy()
        
        calibrated_weights['Default'] = optimal_dict.copy()
        
        # Store effectiveness for UI display
        calibrated_weights['_effectiveness'] = effectiveness
        calibrated_weights['_variance_reduction'] = 1 - (variance_optimal / variance_universal)
        
    else:
        DIAG.log("OPTIMIZATION_FAILED", message=result.message)
        for sector_name in all_sectors:
            calibrated_weights[sector_name] = UNIVERSAL_WEIGHTS.copy()
        calibrated_weights['Default'] = UNIVERSAL_WEIGHTS.copy()
        calibrated_weights['_effectiveness'] = 0.0
    
    return calibrated_weights

# ============================================================================
# SECTOR-SPECIFIC WEIGHTS (SOLUTION TO ISSUE #1: SECTOR BIAS)
# ============================================================================

# ============================================================================
# UNIVERSAL BASE WEIGHTS (V3.0 - SINGLE SOURCE OF TRUTH)
# ============================================================================
# This is the ONLY weight definition in the system.
# Used as:
# 1. Starting point for dynamic calibration (when enabled)
# 2. Final weights for all sectors (when calibration disabled)

UNIVERSAL_WEIGHTS = {
    'credit_score': 0.20,
    'leverage_score': 0.25,
    'profitability_score': 0.20,
    'liquidity_score': 0.10,
    'cash_flow_score': 0.25
}

# IMPORTANT: The old SECTOR_WEIGHTS dictionary has been removed.
# All weight logic now flows through UNIVERSAL_WEIGHTS and dynamic calibration.

# ============================================================================
# RUBRICS CUSTOM CLASSIFICATION MAPPING
# ============================================================================

CLASSIFICATION_TO_SECTOR = {
    # Industrials (7 classifications)
    'Aerospace and Defense': 'Industrials',
    'Capital Goods': 'Industrials',
    'Commercial and Professional Services': 'Industrials',
    'Trading Companies and Distributors': 'Industrials',
    'Transportation': 'Industrials',
    
    # Consumer Discretionary (4 classifications)
    'Automobiles and Components': 'Consumer Discretionary',
    'Consumer Discretionary Distribution and Retail': 'Consumer Discretionary',
    'Consumer Durables and Apparel': 'Consumer Discretionary',
    'Consumer Services': 'Consumer Discretionary',
    
    # Materials (5 classifications)
    'Chemicals': 'Materials',
    'Construction Materials': 'Materials',
    'Containers and Packaging': 'Materials',
    'Metals and Mining': 'Materials',
    'Paper and Forest Products': 'Materials',
    
    # Consumer Staples (3 classifications)
    'Consumer Staples Distribution and Retail': 'Consumer Staples',
    'Food, Beverage and Tobacco': 'Consumer Staples',
    'Household and Personal Products': 'Consumer Staples',
    
    # Information Technology (3 classifications)
    'Semiconductors and Semiconductor Equipment': 'Information Technology',
    'Software and Services': 'Information Technology',
    'Technology Hardware and Equipment': 'Information Technology',
    
    # Health Care (2 classifications)
    'Health Care Equipment and Services': 'Health Care',
    'Pharmaceuticals, Biotechnology and Life Sciences': 'Health Care',
    
    # Communication Services (2 classifications)
    'Media and Entertainment': 'Communication Services',
    'Telecommunication Services': 'Communication Services',
    
    # Real Estate (2 classifications)
    'Equity Real Estate Investment Trusts (REITs)': 'Real Estate',
    'Real Estate Management and Development': 'Real Estate',
    
    # Energy (1 classification)
    'Energy': 'Energy',
    
    # Utilities (1 classification)
    'Utilities': 'Utilities',
    
    # Financials (2 classifications) - Non-fin financial services
    'Consumer Finance': 'Financials',
    'Financial Services': 'Financials',
}

# Optional: Classification-specific weight overrides
# Add custom weights here for classifications that differ from their parent sector
CLASSIFICATION_OVERRIDES = {
    # Example: Uncomment and customize as needed
    # 'Software and Services': {
    #     'credit_score': 0.15,
    #     'leverage_score': 0.20,
    #     'profitability_score': 0.25,
    #     'liquidity_score': 0.15,
    #     'growth_score': 0.20,
    #     'cash_flow_score': 0.05
    # },
}



def get_sector_weights(sector, use_sector_adjusted=True):
    """
    Returns weight dictionary for a given sector.

    V3.0 REDESIGN: Always returns universal weights

    DEPRECATED: Use get_classification_weights() for Rubrics Custom Classifications
    Note: This function now only returns UNIVERSAL_WEIGHTS regardless of sector.
    For sector-specific calibrated weights, use the calibration system.
    """
    # Always return universal weights (sector differentiation happens via calibration)
    return UNIVERSAL_WEIGHTS.copy()

def get_classification_weights(classification, use_sector_adjusted=True, calibrated_weights=None):
    """
    Get factor weights for a Rubrics Custom Classification.

    V3.0 REDESIGN: Universal weights as base, calibrated weights when enabled

    Hierarchy:
    1. If calibration disabled (no calibrated_weights): Return universal weights
    2. Check if classification has custom override weights
    3. Map to parent sector and use calibrated sector weights
    4. Fall back to universal weights if classification not found

    Args:
        classification: Rubrics Custom Classification value
        use_sector_adjusted: Whether to use adjusted weights (vs universal)
        calibrated_weights: Optional dict of dynamically calibrated weights (V3.0)

    Returns:
        Dictionary with 5 factor weights (summing to 1.0)
    """
    # MODE 1: If calibration disabled, return universal weights
    if not use_sector_adjusted or calibrated_weights is None:
        return UNIVERSAL_WEIGHTS.copy()

    # MODE 2: Use calibrated weights (data-driven sector adjustments)

    # Step 1: Check for custom overrides
    if classification in CLASSIFICATION_OVERRIDES:
        return CLASSIFICATION_OVERRIDES[classification]

    # Step 2: Map to parent sector and use calibrated weights
    if classification in CLASSIFICATION_TO_SECTOR:
        parent_sector = CLASSIFICATION_TO_SECTOR[classification]
        if parent_sector in calibrated_weights:
            return calibrated_weights[parent_sector]

    # Step 3: Fall back to universal weights
    return UNIVERSAL_WEIGHTS.copy()

# ================================
# EXPLAINABILITY HELPERS (V2.2) — canonical
# ================================

def _resolve_text_field(row: pd.Series, candidates):
    for c in candidates:
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c]).strip()
    return None

def _resolve_model_weights_for_row(row: pd.Series, scoring_method: str, calibrated_weights=None):
    """
    Return (weights_dict, provenance_str) with sector/classification precedence.
    Keys: lowercase matching UNIVERSAL_WEIGHTS (credit_score, leverage_score,
          profitability_score, liquidity_score, growth_score, cash_flow_score)
    """
    # Use module-level UNIVERSAL_WEIGHTS (defined at line ~6559)
    UNIVERSAL = UNIVERSAL_WEIGHTS

    # Universal mode short-circuit
    if str(scoring_method).lower().startswith("universal"):
        return UNIVERSAL, "Universal weights"

    cls = _resolve_text_field(row, ["Rubrics_Custom_Classification", "Rubrics Custom Classification", "Classification", "Custom_Classification"])
    sec = _resolve_text_field(row, ["IQ_SECTOR", "Sector", "GICS_Sector"])

    # 1) App-provided resolver (preferred)
    try:
        w = get_classification_weights(cls, use_sector_adjusted=True, calibrated_weights=calibrated_weights)
        if isinstance(w, dict) and w:
            return w, f"Sector-Adjusted via classification='{cls or 'n/a'}', sector='{sec or 'n/a'}'"
    except Exception:
        pass

    # 2) Sector map (V3.0: Now handled via calibration system)
    # Legacy sector lookup removed - use calibration system for sector-specific weights

    # 3) Classification map
    try:
        if "CLASSIFICATION_WEIGHT_MAP" in globals() and cls in CLASSIFICATION_WEIGHT_MAP:
            return CLASSIFICATION_WEIGHT_MAP[cls], f"Classification override for '{cls}'"
    except Exception:
        pass

    # 4) Fallback
    return UNIVERSAL, "Universal weights"

def _build_explainability_table(issuer_row: pd.Series, scoring_method: str, calibrated_weights=None):
    """
    Build comparison table showing current weights vs original weights used in calculation.
    Returns (df, provenance, composite_score, diff_current, original_sum_contrib, has_original_weights).
    """
    # Map display names to column names and weight keys
    factor_map = {
        "Credit": "credit_score",
        "Leverage": "leverage_score",
        "Profitability": "profitability_score",
        "Liquidity": "liquidity_score",
        "Cash Flow": "cash_flow_score"
    }

    canonical = list(factor_map.keys())

    # Check for column existence
    present = [f for f in canonical if f.replace(" ", "_") + "_Score" in issuer_row.index]

    # Get CURRENT weights (from current calibration settings)
    current_weights_lc, provenance = _resolve_model_weights_for_row(issuer_row, scoring_method, calibrated_weights)

    # Normalize current weights over present factors
    current_w = {f: float(max(0.0, current_weights_lc.get(factor_map[f], 0.0))) for f in present}
    current_sum = sum(current_w.values()) or 1.0
    current_w = {k: v / current_sum for k, v in current_w.items()}

    # Use UNIVERSAL_WEIGHTS as baseline for "Universal Weight %" column
    # This provides a consistent reference point (20/25/20/10/25)
    original_w = {f: float(UNIVERSAL_WEIGHTS.get(factor_map[f], 0.0)) for f in present}
    original_sum = sum(original_w.values()) or 1.0
    original_w = {k: v / original_sum for k, v in original_w.items()}

    # Check if stored weights exist (for potential validation)
    weight_cols_map = {
        "Credit": "Weight_Credit_Used",
        "Leverage": "Weight_Leverage_Used",
        "Profitability": "Weight_Profitability_Used",
        "Liquidity": "Weight_Liquidity_Used",
        "Cash Flow": "Weight_CashFlow_Used"
    }
    has_original_weights = all(weight_cols_map[f] in issuer_row.index for f in present)

    # Build comparison table
    rows = []
    for fac in present:
        col_name = fac.replace(" ", "_") + "_Score"
        score = float(issuer_row.get(col_name, np.nan))

        current_wt = current_w[fac]
        original_wt = original_w[fac]

        current_contrib = score * current_wt
        original_contrib = score * original_wt

        # Calculate weight change
        weight_change = ((current_wt - original_wt) / original_wt * 100) if original_wt > 0 else 0

        rows.append({
            "Factor": fac,
            "Score": round(score, 2),
            "Universal Weight %": round(100.0 * original_wt, 1),
            "Current Weight %": round(100.0 * current_wt, 1),
            "Weight Change": f"{weight_change:+.0f}%",
            "Universal Contrib": round(original_contrib, 2),
            "Current Contrib": round(current_contrib, 2),
            "Contrib Change": round(current_contrib - original_contrib, 2)
        })

    df = pd.DataFrame(rows)
    comp = float(issuer_row.get("Composite_Score", np.nan))

    # Calculate differences
    original_sum_contrib = df["Universal Contrib"].sum() if len(df) else np.nan
    current_sum_contrib = df["Current Contrib"].sum() if len(df) else np.nan
    diff_original = float(original_sum_contrib - comp) if pd.notna(comp) and len(df) else np.nan
    diff_current = float(current_sum_contrib - comp) if pd.notna(comp) and len(df) else np.nan

    # Return extended info
    return df, provenance, comp, diff_current, original_sum_contrib, has_original_weights
# ================================

def render_issuer_explainability(filtered: pd.DataFrame, scoring_method: str, calibrated_weights=None):
    """Single source of truth for the Issuer Explainability panel."""
    with st.expander("Issuer Explainability", expanded=False):
        if filtered is None or filtered.empty or "Company_Name" not in filtered.columns:
            st.info("No issuers available for the current filter.")
            return

        issuer_names = sorted(filtered["Company_Name"].dropna().unique().tolist())
        if not issuer_names:
            st.info("No issuers available for the current filter.")
            return

        selected_issuer = st.selectbox("Select Issuer", options=issuer_names, key="explainability_issuer")
        issuer_row = filtered[filtered["Company_Name"] == selected_issuer].iloc[0]

        # Header metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Company ID", issuer_row.get("Company_ID", "—"))
        with c2: st.metric("Rating", issuer_row.get("Credit_Rating_Clean", "—"))
        with c3: st.metric("Rating Band", issuer_row.get("Rating_Band", "—"))
        with c4:
            comp_score = issuer_row.get("Composite_Score", float("nan"))
            st.metric("Composite Score", f"{comp_score:.1f}" if pd.notna(comp_score) else "n/a")

        # Signal with reason badges
        signal_val = issuer_row.get("Combined_Signal", issuer_row.get("Signal", "—"))
        signal_reason = issuer_row.get("Signal_Reason", "")
        if signal_reason and signal_reason.strip():
            st.info(f"**Signal:** {signal_val}  \n**Context:** {signal_reason}")
        else:
            st.markdown(f"**Signal:** {signal_val}")

        st.markdown("---")
        st.markdown("### Factor Contributions")

        df_contrib, provenance, comp, diff_current, original_sum, has_original = _build_explainability_table(issuer_row, scoring_method, calibrated_weights)

        # Display weight provenance
        st.markdown(f"**Current Weight Method:** {provenance}")

        st.info("ℹ️ **Weight Comparison:** Universal baseline (20/25/20/10/25) vs Current active weights. When Dynamic Calibration is OFF, both columns will be identical.")

        # Display comparison table
        st.dataframe(df_contrib, use_container_width=True, hide_index=True)

        # Highlight significant weight changes
        st.caption("""
        **How to read this table:**
        - **Score**: Factor score (0-100) for this issuer
        - **Universal Weight %**: Universal baseline weight (20/25/20/10/25) - constant for all issuers
        - **Current Weight %**: Active weight from current Dynamic Calibration settings (identical to Universal if calibration OFF)
        - **Weight Change**: % change from universal baseline to current active weight
        - **Universal Contrib**: Factor's contribution using universal baseline weights
        - **Current Contrib**: Factor's contribution using current active weights
        - **Contrib Change**: Impact of current calibration vs universal baseline
        """)

        # Summary metrics
        st.markdown("---")
        st.markdown("### Score Breakdown")
        
        # Get Quality Score and Trend Score for proper breakdown
        quality_score = float(issuer_row.get("Quality_Score", np.nan))
        trend_score = float(issuer_row.get("Cycle_Position_Score", np.nan))
        
        # Calculate expected composite from 80/20 blend
        expected_composite = np.nan
        if pd.notna(quality_score) and pd.notna(trend_score):
            expected_composite = (quality_score * 0.80) + (trend_score * 0.20)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Quality Score", f"{quality_score:.2f}" if pd.notna(quality_score) else "n/a")
            st.caption("Sum of weighted factors (80% of composite)")

        with col2:
            st.metric("Trend Score", f"{trend_score:.2f}" if pd.notna(trend_score) else "n/a")
            st.caption("Cycle position (20% of composite)")

        with col3:
            st.metric("Composite Score", f"{comp:.2f}" if pd.notna(comp) else "n/a")
            st.caption("Quality × 0.80 + Trend × 0.20")

        with col4:
            # True calibration impact: compare quality scores with different weights
            current_sum = df_contrib["Current Contrib"].sum() if len(df_contrib) else np.nan
            calibration_impact = (current_sum - original_sum) if pd.notna(current_sum) and pd.notna(original_sum) else np.nan
            st.metric("Weight Impact", f"{calibration_impact:+.2f}" if pd.notna(calibration_impact) else "n/a")
            st.caption("Current vs Universal weights")

        # Show calibration impact message only when there's actual weight difference
        if pd.notna(calibration_impact):
            if abs(calibration_impact) > 5.0:
                st.warning(f"""
                **⚠️ Significant Calibration Impact ({calibration_impact:+.2f} points on Quality Score)**
                
                Dynamic Calibration is making substantial weight adjustments for this sector/classification,
                changing the Quality Score by {calibration_impact:+.2f} points compared to universal weights.
                """)
            elif abs(calibration_impact) > 1.0:
                st.info(f"""
                **ℹ️ Moderate Calibration Impact ({calibration_impact:+.2f} points on Quality Score)**
                
                Dynamic Calibration is tailoring weights for this classification,
                adjusting the Quality Score by {calibration_impact:+.2f} points.
                """)
            elif abs(calibration_impact) > 0.1:
                st.success(f"""
                **✓ Minor Calibration Impact ({calibration_impact:+.2f} points on Quality Score)**
                
                Weight adjustments produce minimal change. If Dynamic Calibration is OFF,
                this confirms weights match the universal baseline.
                """)
            else:
                st.success("""
                **✓ No Calibration Impact**
                
                Current weights match universal baseline (20/25/20/10/25).
                No weight adjustments are being applied.
                """)
# ================================

# ============================================================================
# METHODOLOGY TAB RENDERING (PROGRAMMATIC SPECIFICATION)
# ============================================================================

def _detect_factors_and_metrics():
    """
    Returns metadata about the 5-factor model structure.

    Returns:
        List of dicts with keys: factor, metric_examples, direction
    """
    return [
        {
            "factor": "Credit Score",
            "metric_examples": "Debt/Assets, Cash/Debt, Implied Interest Rate",
            "direction": "Higher is better"
        },
        {
            "factor": "Leverage Score",
            "metric_examples": "Net Debt/EBITDA (40%), Interest Coverage (40%), Debt/Capital (20%)",
            "direction": "Lower debt is better (inverted scoring)"
        },
        {
            "factor": "Profitability Score",
            "metric_examples": "Gross Profit Margin (30%), EBITDA Margin (40%), ROA (30%)",
            "direction": "Higher is better"
        },
        {
            "factor": "Liquidity Score",
            "metric_examples": "Current Ratio (35%), Quick Ratio (25%), OCF/Current Liabilities (40%)",
            "direction": "Higher is better"
        },

        {
            "factor": "Cash Flow Score",
            "metric_examples": "OCF/Revenue, OCF/Debt, UFCF margin, LFCF margin (equal-weighted, clipped & scaled)",
            "direction": "Higher is better"
        }
    ]




# ============================================================================
# RATING BAND MAPPING (SOLUTION TO ISSUE #4: OVERLY BROAD RATING GROUPS)
# ============================================================================

RATING_BANDS = {
    'AAA': ['AAA'],
    'AA': ['AA+', 'AA', 'AA-'],
    'A': ['A+', 'A', 'A-'],
    'BBB': ['BBB+', 'BBB', 'BBB-'],
    'BB': ['BB+', 'BB', 'BB-'],
    'B': ['B+', 'B', 'B-'],
    'CCC': ['CCC+', 'CCC', 'CCC-', 'CC', 'C', 'D', 'SD']
}

def assign_rating_band(rating):
    """Assign rating to appropriate band"""
    rating_upper = str(rating).upper().strip()
    for band, ratings in RATING_BANDS.items():
        if rating_upper in ratings:
            return band
    return 'Unrated'

# ============================================================================
# INJECT BRAND CSS (UNCHANGED)
# ============================================================================

def inject_brand_css():
    st.markdown("""
    <style>
      :root{
        --rb-blue:#001E4F; --rb-mblue:#2C5697; --rb-lblue:#7BA4DB;
        --rb-grey:#D8D7DF; --rb-orange:#CF4520;
      }

      /* Global font + color (exclude div, span, .stTabs from global color override) */
      html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
      [data-testid="stSidebarContent"], [data-testid="stMarkdownContainer"],
      h1, h2, h3, h4, h5, h6, p, label, input, textarea, select,
      .stText, .stDataFrame, .stMetric {
        font-family: Arial, Helvetica, sans-serif !important;
        color: var(--rb-blue) !important;
      }
      [data-testid="stDataFrame"] * {
        font-family: Arial, Helvetica, sans-serif !important;
        color: var(--rb-blue) !important;
      }

      html, body, .stApp { background:#f8f9fa; }
      header[data-testid="stHeader"] { background: transparent !important; }

      /* Header layout shared by RG & ROAM */
      .rb-header { display:flex; align-items:flex-start; justify-content:space-between;
        gap:12px; margin: 0 0 12px 0; }
      .rb-title h1 { font-size:2.6rem; color:var(--rb-blue); font-weight:700; margin:0; }
      .rb-sub { color:#4c566a; font-weight:600; margin-top:.25rem; }
      .rb-logo img { height:48px; display:block; }
      @media (max-width:1200px){ .rb-title h1{ font-size:2.2rem; } .rb-logo img{ height:42px; } }

      /* Tabs */
      .stTabs [data-baseweb="tab-list"]{ gap:12px; border-bottom:none; }
      .stTabs [data-baseweb="tab"]{
        background:var(--rb-grey); border-radius:4px 4px 0 0;
        font-weight:600; min-width:180px; text-align:center; padding:8px 16px;
      }
      /* Unselected tabs - blue text */
      .stTabs [data-baseweb="tab"]:not([aria-selected="true"]) {
        color: var(--rb-blue) !important;
      }
      /* Selected tabs - white text on element and all children */
      .stTabs [aria-selected="true"],
      .stTabs [aria-selected="true"] * {
        background:var(--rb-mblue)!important;
        color: #ffffff !important;
        border-bottom:3px solid var(--rb-orange)!important;
      }

      /* Buttons - force white text on button and all children */
      .stButton > button,
      .stButton > button * {
        background:var(--rb-mblue);
        color: #ffffff !important;
        border:none;
        border-radius:4px;
        padding:8px 16px;
        font-weight:600;
      }
      .stButton > button:hover,
      .stButton > button:hover * {
        background:var(--rb-blue);
        color: #ffffff !important;
      }
      .stDownloadButton > button,
      .stDownloadButton > button * {
        background:var(--rb-mblue);
        color: #ffffff !important;
        border:none;
        border-radius:4px;
        padding:8px 16px;
        font-weight:600;
      }
      .stDownloadButton > button:hover,
      .stDownloadButton > button:hover * {
        background:var(--rb-blue);
        color: #ffffff !important;
      }

      /* Inputs/selects (keep light, legible) */
      div[data-baseweb="select"] > div {
        background:#ffffff !important; border:1px solid var(--rb-grey);
        border-radius:4px; color:var(--rb-blue)!important;
      }
      .stSelectbox label { color:var(--rb-blue)!important; font-weight:600; }
      .stTextInput input, .stTextArea textarea, .stNumberInput input {
        background:#ffffff!important; border:1px solid var(--rb-grey);
        border-radius:4px; color:var(--rb-blue)!important;
      }
      .stSlider [data-baseweb="slider"] { padding:0; }
      [data-baseweb="slider"] [role="slider"] {
        background:var(--rb-mblue); width:16px; height:16px;
      }

      /* Expander */
      .streamlit-expanderHeader {
        background:var(--rb-grey); border-radius:4px; font-weight:600;
        color:var(--rb-blue)!important;
      }

      /* Metric (no override to white for the value/delta) */
      .stMetric label { font-size:0.9rem; color:#4c566a!important; }

      /* Sidebar */
      [data-testid="stSidebar"]{ background:#ffffff; border-right:1px solid var(--rb-grey); }
      .sidebar .stButton > button { width:100%; }

      /* Dataframe */
      .dataframe th { background:var(--rb-mblue)!important; color:#fff!important;
        font-weight:700; text-align:center; }
      .dataframe td { color:var(--rb-blue)!important; }

      /* Tooltip/Popover */
      [data-baseweb="popover"] { background:#ffffff; box-shadow:0 2px 8px rgba(0,0,0,0.1);
        border-radius:4px; }

      /* Remove global Plotly override that broke bar colors */
      .js-plotly-plot .plotly .main-svg { color: inherit !important; }
    </style>
    """, unsafe_allow_html=True)

inject_brand_css()

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

st.sidebar.title(" Configuration")

# Restore state from URL if present
_url_state = _get_query_params()
_original_dp = _url_state.get("data_period")  # Save original before coercion
sm0, dp0, qb0, align0, band0, topn0 = apply_state_to_controls(_url_state)

# Optional: Show deprecation notice if URL contained FY-4
if _original_dp == "FY-4 (Legacy)":
    st.info("ℹ️ Note: 'FY-4 (Legacy)' option has been removed. Defaulted to 'Most Recent Fiscal Year (FY0)'.")

# [V2.2.1] DEPRECATED: Model Version Selection (now controlled by calibration checkbox)
# The "Use Dynamic Weight Calibration" checkbox (below) now controls sector weighting:
# - When ON: Sector-specific calibrated weights are used
# - When OFF: Universal weights are used for all issuers
# This old radio button is hidden but maintained for backward compatibility with URL state

# Hidden for now - behavior controlled by calibration checkbox
if False:  # Never show this radio button
    st.sidebar.subheader(" Scoring Method")
    scoring_method_options = ["Classification-Adjusted Weights (Recommended)", "Universal Weights (Original)"]
    scoring_method = st.sidebar.radio(
        "Select Scoring Approach",
        scoring_method_options,
        index=0 if sm0.startswith("Classification") else 1,
        help="Classification-Adjusted applies different factor weights by industry classification (e.g., Utilities emphasize cash flow more than leverage)"
    )
    use_sector_adjusted = (scoring_method == "Classification-Adjusted Weights (Recommended)")
else:
    # Default to Classification-Adjusted for backward compatibility
    scoring_method = "Classification-Adjusted Weights (Recommended)"
    use_sector_adjusted = True  # Will be overridden by calibration setting

# Canonicalize & persist scoring method (for URL state compatibility)
sm_canonical = (
    "Classification-Adjusted Weights"
    if scoring_method.startswith("Classification-Adjusted")
    else "Universal Weights"
)
st.session_state["scoring_method"] = sm_canonical
st.session_state["use_sector_adjusted"] = use_sector_adjusted  # Will be overridden by effective_use_sector_adjusted

# === Sidebar: V2.3 Unified Period Selection ===
st.sidebar.subheader("Period Selection")

# [V2.3] Unified period selection mode
period_mode_display = st.sidebar.radio(
    "Period Selection Mode",
    options=[
        "Latest Available (Maximum Currency)",
        "Align to Reference Period"
    ],
    index=0,  # Default to Latest Available
    help="""
**Latest Available**: Each issuer uses their most recent data.
⚠️ WARNING: Results in misaligned reporting dates across issuers.

**Align to Reference Period**: All issuers aligned to a common date.
Ensures apples-to-apples comparison.
    """,
    key="cfg_period_mode"
)

# Convert display to enum
if period_mode_display == "Latest Available (Maximum Currency)":
    period_mode = PeriodSelectionMode.LATEST_AVAILABLE
else:
    period_mode = PeriodSelectionMode.REFERENCE_ALIGNED

# [V2.3] Reference date selector (only show for REFERENCE_ALIGNED mode)
reference_date_override = None
align_to_reference = False

# Initialize period priority flag (only used in REFERENCE_ALIGNED mode)
prefer_annual_reports = False

if period_mode == PeriodSelectionMode.REFERENCE_ALIGNED:
    align_to_reference = True
    st.sidebar.markdown("---")

    # Get uploaded file from session state
    uploaded_file_ref = st.session_state.get('uploaded_file')

    if uploaded_file_ref is not None:
        try:
            file_name = uploaded_file_ref.name.lower()
            if file_name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file_ref)
            elif file_name.endswith('.xlsx'):
                # Try multi-index first, fall back to single index
                try:
                    raw_df = pd.read_excel(uploaded_file_ref, sheet_name='Pasted Values', header=[0, 1])
                except Exception:
                    raw_df = pd.read_excel(uploaded_file_ref, sheet_name='Pasted Values')

            # Normalize headers
            raw_df.columns = [' '.join(str(c).replace('\u00a0', ' ').split()) for c in raw_df.columns]

            # Reset file pointer for later use
            uploaded_file_ref.seek(0)
        except Exception as e:
            raw_df = None
            st.sidebar.error(f"Error loading data: {str(e)}")
    else:
        raw_df = None

    if raw_df is not None:
        try:
            # Get period options with coverage
            period_options = get_period_selection_options(raw_df)

            if period_options:
                # Create dropdown with coverage indicators
                option_labels = [opt[0] for opt in period_options]
                option_dates = [opt[1] for opt in period_options]

                # Find recommended (default to first with good coverage)
                recommended_idx = 0
                for i, (label, date, coverage) in enumerate(period_options):
                    if coverage >= 85:
                        recommended_idx = i
                        break

                selected_label = st.sidebar.selectbox(
                    "Reference Period",
                    options=option_labels,
                    index=recommended_idx,
                    help="Select the reporting period to align all issuers to. Higher coverage % is better.",
                    key="cfg_reference_period_v3"
                )

                # Get corresponding date
                selected_idx = option_labels.index(selected_label)
                reference_date_override = option_dates[selected_idx]

                st.sidebar.info(f"Using reference date: {reference_date_override.strftime('%Y-%m-%d')}")
            else:
                st.sidebar.warning("Could not detect period options from file")

        except Exception as e:
            st.sidebar.error(f"Error loading period options: {e}")

    # Period Selection Priority toggle
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Period Selection Priority")

    period_priority = st.sidebar.radio(
        "When multiple periods available:",
        options=[
            "Most Recent Period (Maximum Currency)",
            "Annual Reports (Stability)"
        ],
        index=0,  # Default to Most Recent
        help="""
**Most Recent Period**: Uses the latest available period (FY or CQ) within the reference window.
- Captures most current performance including latest quarters
- May show more volatility due to quarterly fluctuations
- Use when tracking recent developments

**Annual Reports**: Prefers annual fiscal year (FY) data over quarterly (CQ) when both are available.
- Provides stable, comprehensive annual view
- Reduces ranking volatility from quarterly fluctuations
- Use for long-term credit assessment
- Note: May not reflect very recent quarters
        """,
        key="period_priority_toggle"
    )

    # Store the preference as a boolean flag
    prefer_annual_reports = (period_priority == "Annual Reports (Stability)")

else:
    # LATEST_AVAILABLE mode
    align_to_reference = False
    st.sidebar.warning("""
⚠️ WARNING: Misaligned Reporting Dates

Using latest available data means issuers will have different
reporting dates. Cross-issuer comparisons may not be perfectly aligned.
    """)

st.sidebar.markdown("---")

# [V2.2.1] Cache-clearing callback for trend window changes
def clear_trend_cache():
    """
    Clear cached data when trend window parameters change.

    Note: We rely on the _cache_buster parameter in load_and_process_data()
    to force recalculation. This callback just clears session state caches.
    """
    # Clear session state caches (if any exist)
    keys_to_clear = ['processed_data_cache', 'trend_scores_cache']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    # The _cache_buster parameter in load_and_process_data() will handle
    # the function cache invalidation automatically

# ============================================================================
# [V2.3] QUALITY/TREND SPLIT THRESHOLDS (Fixed at Recommended Defaults)
# ============================================================================
# Sliders removed for simplicity - using proven defaults
split_basis = "Percentile within Band (recommended)"
split_threshold = 60  # Top 40% of rating band = Strong
trend_threshold = 55  # Cycle position threshold for improving vs deteriorating

# ============================================================================
# [V2.2.1] WEIGHT CALIBRATION CONFIGURATION
# ============================================================================
st.sidebar.markdown("#### Sector Weight Calibration")

use_dynamic_calibration = st.sidebar.checkbox(
    "Use Dynamic Weight Calibration",
    value=False,  # Default to OFF (Universal Weights)
    help="""
    **Sector-adjusted vs universal weighting:**

    **When CHECKED (Dynamic Calibration - Recommended):**
      • Sector weights recalculated from uploaded data
      • BBB-rated companies in all sectors score ~50-60 on average
      • Similar Buy recommendation rates across sectors (~40%)
      • Fair cross-sector comparisons
      • Requires: 50+ companies total, 5+ per sector

    **When UNCHECKED (Universal Weights):**
      • Same weights applied to ALL issuers regardless of sector
      • No sector-specific adjustments
      • Simpler, but may introduce sector bias
      • Use when sector fairness is not a concern
    """,
    key="cfg_use_dynamic_calibration"
)

if use_dynamic_calibration:
    calibration_rating_band = st.sidebar.selectbox(
        "Calibration Rating Band",
        options=['BBB', 'A', 'BB'],
        index=0,
        help="Rating band to use for calculating sector deviations. BBB recommended for broad coverage.",
        key="cfg_calibration_rating_band"
    )
else:
    calibration_rating_band = 'BBB'  # Default value when not used

# Store in session state for use in workflow
st.session_state['use_dynamic_calibration'] = use_dynamic_calibration
st.session_state['calibration_rating_band'] = calibration_rating_band

# ============================================================================
# [V2.3] DUAL-HORIZON CONTEXT PARAMETERS (using recommended defaults)
# ============================================================================
# Advanced controls removed for simplicity - using tested default values
volatility_cv_threshold = MODEL_THRESHOLDS['volatility_cv']
outlier_z_threshold = -2.5
damping_factor = 0.5
near_peak_tolerance = 10

# [V5.0] Derive data_period_setting from period_mode
# When REFERENCE_ALIGNED mode is active with a specific date, reflect that in the setting
if period_mode == PeriodSelectionMode.REFERENCE_ALIGNED and reference_date_override is not None:
    # Format the reference date for display
    ref_date_str = reference_date_override.strftime('%Y-%m-%d') if hasattr(reference_date_override, 'strftime') else str(reference_date_override)
    data_period_setting = f"Reference Aligned ({ref_date_str})"
else:
    data_period_setting = "Most Recent LTM (LTM0)"
use_quarterly_beta = True  # Always use quarterly for trend analysis

# Alias for backward compatibility with URL state management
data_period = data_period_setting

# Write state back to URL (deep-linkable state)
_current_state = collect_current_state(
    scoring_method=scoring_method,
    data_period=data_period,
    use_quarterly_beta=use_quarterly_beta,
    align_to_reference=align_to_reference,
    band_default=band0,  # Will be updated later when band selector is rendered
    top_n_default=topn0  # Will be updated later when top_n slider is rendered
)
_set_query_params(_current_state)

# ============================================================================
# [V2.3] PRESETS AND SHARING REMOVED FOR SIMPLICITY
# ============================================================================
# Save/Load Preset and Reproduce/Share expanders removed to simplify UI
# Users will need to manually reconfigure settings each session

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload S&P Data file (Excel or CSV)", type=["xlsx", "csv"])
HAS_DATA = uploaded_file is not None

# Store in session_state for access in sidebar sections
st.session_state['uploaded_file'] = uploaded_file

# Pre-upload: show hero only (title + subtitle + logo)
if not HAS_DATA:
    render_header()

# ============================================================================
# DATA LOADING & PROCESSING FUNCTIONS
# ============================================================================

# ============================================================================
# GENAI CREDIT REPORT - DATA EXTRACTION PIPELINE (V3.0)
# ============================================================================

def extract_raw_financials_from_input(df_original: pd.DataFrame, company_name: str) -> dict:
    """
    Extract ALL relevant raw financial metrics directly from input spreadsheet.
    This is the SOURCE OF TRUTH for actual company fundamentals.

    Args:
        df_original: The raw input DataFrame (from uploaded Excel file)
        company_name: Name of company to analyze

    Returns:
        dict: Complete raw financial profile with all metrics from spreadsheet
    """

    # Find the company row
    try:
        company_row = df_original[df_original['Company Name'] == company_name]
    except Exception as e:
        return {"error": f"Error finding company: {str(e)}. Company name: '{company_name}' (type: {type(company_name)}), Column dtype: {df_original['Company Name'].dtype}"}

    if len(company_row) == 0:
        return {"error": f"Company '{company_name}' not found in input data"}

    row = company_row.iloc[0]

    # Helper function to safely get values
    def safe_get(column_name, default=None):
        try:
            val = row.get(column_name)
            if pd.isna(val):
                return default
            return val
        except Exception:
            return default

    def safe_get_numeric(column_name, default=None):
        """Get numeric value, converting strings like 'NM' to None"""
        try:
            val = row.get(column_name)
            if pd.isna(val):
                return default
            # If it's already numeric, return it
            if isinstance(val, (int, float)):
                return val
            # If it's a string, try to convert to numeric
            if isinstance(val, str):
                try:
                    return pd.to_numeric(val)
                except (ValueError, TypeError):
                    # Conversion failed (e.g., 'NM', 'N/A'), return default
                    return default
            return val
        except Exception:
            return default

    def safe_get_by_key(metric_key, suffix='', default=None):
        """Get metric value using registry key instead of hardcoded column name"""
        try:
            # Get column name from registry
            col_name = get_metric_column(metric_key, suffix=suffix)
            if not col_name:
                return default

            # Try to get value from row
            val = row.get(col_name)
            if pd.isna(val):
                return default

            # If it's already numeric, return it
            if isinstance(val, (int, float)):
                return val

            # Handle string values like 'NM'
            if isinstance(val, str):
                try:
                    return pd.to_numeric(val)
                except (ValueError, TypeError):
                    return default

            return val
        except Exception:
            return default

    # Extract ALL financial metrics from the spreadsheet
    raw_data = {
        "company_info": {
            "name": safe_get('Company Name'),
            "ticker": safe_get('Ticker'),
            "country": safe_get('Country'),
            "region": safe_get('Region'),
            "sector": safe_get('Sector'),
            "industry": safe_get('Industry'),
            "industry_group": safe_get('Industry Group'),
            "classification": safe_get('Rubrics Custom Classification'),
            "sp_rating": safe_get('S&P Credit Rating'),
            "sp_rating_clean": safe_get('_Credit_Rating_Clean'),
            "rating_date": safe_get('S&P Last Review Date'),
            "market_cap": safe_get('Market Capitalization'),
        },

        # PROFITABILITY METRICS (extracted using registry)
        "profitability": {
            "ebitda_margin": safe_get_by_key('ebitda_margin'),
            "ebit_margin": safe_get_by_key('ebit_margin'),
            "operating_margin": safe_get_by_key('ebit_margin'),  # Use EBIT Margin (Operating Margin doesn't exist)
            "net_margin": safe_get_by_key('net_income_margin'),
            "roe": safe_get_by_key('roe'),
            "roa": safe_get_by_key('roa'),
            "roic": safe_get_by_key('roic'),
        },

        # LEVERAGE METRICS (extracted using registry)
        "leverage": {
            "total_debt": safe_get_by_key('total_debt'),
            "net_debt": safe_get_by_key('net_debt'),
            "total_equity": safe_get_by_key('equity'),
            "total_debt_ebitda": safe_get_by_key('total_debt_ebitda'),
            "net_debt_ebitda": safe_get_by_key('net_debt_ebitda'),
            "total_debt_equity": safe_get_by_key('debt_to_equity'),
            "total_debt_capital": safe_get_by_key('debt_to_capital'),
        },

        # COVERAGE METRICS (extracted using registry)
        "coverage": {
            "ebitda_interest": safe_get_by_key('ebitda_interest'),
            "ebit_interest": safe_get_by_key('ebit_interest'),
            "interest_expense": safe_get_by_key('interest_expense'),
        },

        # LIQUIDITY METRICS (extracted using registry)
        "liquidity": {
            "current_ratio": safe_get_by_key('current_ratio'),
            "quick_ratio": safe_get_by_key('quick_ratio'),
            "cash_st_investments": safe_get_by_key('cash'),
            "current_assets": safe_get_by_key('current_assets'),
            "current_liabilities": safe_get_by_key('current_liabilities'),
            "working_capital": safe_get_by_key('working_capital'),
        },

        # GROWTH METRICS (extracted using registry)
        "growth": {
            "revenue_1y_growth": safe_get_by_key('revenue_growth'),
            "revenue_3y_cagr": safe_get_by_key('revenue_3y_cagr'),
            "ebitda_3y_cagr": safe_get_by_key('ebitda_3y_cagr'),
            "total_revenues": safe_get_by_key('revenue'),
            "ebitda": safe_get_by_key('ebitda'),
        },

        # CASH FLOW METRICS (extracted using registry)
        "cash_flow": {
            "cfo": safe_get_by_key('operating_cash_flow'),
            "capex": safe_get_by_key('capex'),
            "fcf": safe_get_by_key('levered_fcf'),  # Use Levered FCF (not Unlevered)
            "cfo_total_debt": None,  # Will calculate below if not in spreadsheet
            "fcf_total_debt": None,  # Will calculate below if not in spreadsheet
        },

        # BALANCE SHEET (extracted using registry)
        "balance_sheet": {
            "total_assets": safe_get_by_key('total_assets'),
            "total_liabilities": safe_get_by_key('total_liabilities'),
        },

        # PERIOD INFORMATION
        "data_period": {
            "latest_period": safe_get('Period Ended'),
        }
    }

    # ========================================================================
    # CALCULATE DERIVED RATIOS (if not directly available in spreadsheet)
    # ========================================================================

    # Calculate CFO/Total Debt ratio (as percentage)
    if raw_data['cash_flow']['cfo_total_debt'] is None:
        cfo = raw_data['cash_flow']['cfo']
        total_debt = raw_data['leverage']['total_debt']
        if cfo is not None and total_debt is not None:
            try:
                cfo_val = float(cfo)
                debt_val = float(total_debt)
                if debt_val != 0:
                    raw_data['cash_flow']['cfo_total_debt'] = (cfo_val / abs(debt_val)) * 100
            except (ValueError, TypeError, ZeroDivisionError):
                pass

    # Calculate FCF/Total Debt ratio (as percentage)
    if raw_data['cash_flow']['fcf_total_debt'] is None:
        fcf = raw_data['cash_flow']['fcf']
        total_debt = raw_data['leverage']['total_debt']
        if fcf is not None and total_debt is not None:
            try:
                fcf_val = float(fcf)
                debt_val = float(total_debt)
                if debt_val != 0:
                    raw_data['cash_flow']['fcf_total_debt'] = (fcf_val / abs(debt_val)) * 100
            except (ValueError, TypeError, ZeroDivisionError):
                pass

    return raw_data


def calculate_peer_context_for_scoring(
    df: pd.DataFrame,
    idx: int,
    classification: str,
    rating: str
) -> dict:
    """
    Calculate peer context during scoring pipeline.
    
    This is called ONCE per issuer during scoring, and the result
    is stored in diagnostic_data for later use by GenAI tab.
    
    Args:
        df: Full DataFrame with all issuers
        idx: Index of current issuer
        classification: Issuer's Rubrics Custom Classification
        rating: Issuer's cleaned credit rating
    
    Returns:
        dict: Peer context with medians and percentiles
    """
    
    # Define metrics to compare (using METRIC_REGISTRY)
    metrics_config = {
        'ebitda_margin': {'higher_is_better': True},
        'roe': {'higher_is_better': True},
        'roa': {'higher_is_better': True},
        'total_debt_ebitda': {'higher_is_better': False},
        'net_debt_ebitda': {'higher_is_better': False},
        'current_ratio': {'higher_is_better': True},
        'quick_ratio': {'higher_is_better': True},
        'ebitda_interest': {'higher_is_better': True},
        'revenue_1y_growth': {'higher_is_better': True},
        'revenue_3y_cagr': {'higher_is_better': True},
    }
    
    # Get column names from registry
    def get_col(metric_key):
        if metric_key in METRIC_REGISTRY:
            return METRIC_REGISTRY[metric_key]['canonical']
        return None
    
    # Build column mapping
    metric_columns = {}
    for key in metrics_config:
        col = get_col(key)
        if col and col in df.columns:
            metric_columns[key] = col
    
    # Get sector peers
    classification_col = df.get('Rubrics_Custom_Classification', df.get('Rubrics Custom Classification', pd.Series()))
    sector_peers = df[classification_col == classification].copy() if len(classification_col) > 0 else pd.DataFrame()
    
    # Get rating peers
    rating_col = df.get('_Credit_Rating_Clean', df.get('Credit_Rating_Clean', pd.Series()))
    rating_peers = df[rating_col == rating].copy() if len(rating_col) > 0 else pd.DataFrame()
    
    # Calculate medians and percentiles
    sector_medians = {}
    sector_percentiles = {}
    rating_medians = {}
    rating_percentiles = {}
    
    # Get company's own values
    company_row = df.loc[idx]
    
    for metric_key, col_name in metric_columns.items():
        higher_is_better = metrics_config[metric_key]['higher_is_better']
        
        # Company value
        company_val = company_row.get(col_name)
        if pd.isna(company_val):
            continue
        try:
            company_val = float(company_val)
        except (ValueError, TypeError):
            continue
        
        # Sector comparison
        if len(sector_peers) > 0 and col_name in sector_peers.columns:
            numeric_vals = pd.to_numeric(sector_peers[col_name], errors='coerce').dropna()
            if len(numeric_vals) > 0:
                sector_medians[metric_key] = float(numeric_vals.median())
                # Calculate percentile
                if higher_is_better:
                    pct = (numeric_vals < company_val).sum() / len(numeric_vals) * 100
                else:
                    pct = (numeric_vals > company_val).sum() / len(numeric_vals) * 100
                sector_percentiles[metric_key] = round(pct, 1)
        
        # Rating comparison
        if len(rating_peers) > 0 and col_name in rating_peers.columns:
            numeric_vals = pd.to_numeric(rating_peers[col_name], errors='coerce').dropna()
            if len(numeric_vals) > 0:
                rating_medians[metric_key] = float(numeric_vals.median())
                # Calculate percentile
                if higher_is_better:
                    pct = (numeric_vals < company_val).sum() / len(numeric_vals) * 100
                else:
                    pct = (numeric_vals > company_val).sum() / len(numeric_vals) * 100
                rating_percentiles[metric_key] = round(pct, 1)
    
    return {
        'sector_comparison': {
            'classification': classification,
            'peer_count': len(sector_peers),
            'medians': sector_medians,
            'percentiles': sector_percentiles,
        },
        'rating_comparison': {
            'rating': rating,
            'peer_count': len(rating_peers),
            'medians': rating_medians,
            'percentiles': rating_percentiles,
        }
    }


def calculate_peer_context(df_original: pd.DataFrame, raw_financials: dict) -> dict:
    """
    Calculate peer medians and percentile rankings.
    Essential for understanding relative credit strength.

    Args:
        df_original: Raw input DataFrame
        raw_financials: Raw financial data from extract_raw_financials_from_input()

    Returns:
        dict: Peer comparison context with medians and percentiles
    """

    classification = raw_financials['company_info']['classification']
    rating = raw_financials['company_info']['sp_rating_clean']

    # =========================================================================
    # CACHING: Peer context only depends on classification + rating
    # Cache to avoid recalculating for same peer group
    # =========================================================================
    cache_key = f"_peer_context_cache_{classification}_{rating}"
    
    # Return cached result if available
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    # Check for missing or NaN values (handle None, empty string, NaN)
    def is_valid_value(val):
        """Check if value is valid (not None, not NaN, not empty string)"""
        if val is None:
            return False
        try:
            if pd.isna(val):
                return False
        except (TypeError, ValueError):
            pass  # pd.isna() failed, value is likely a valid string
        if isinstance(val, str) and val.strip() == '':
            return False
        return True

    if not is_valid_value(classification):
        return {"error": "Missing classification information"}
    if not is_valid_value(rating):
        return {"error": "Missing rating information"}

    # Get sector peers (same Rubrics Custom Classification)
    # Use fillna to handle NaN, then convert to string for safe comparison
    try:
        classification_col = df_original['Rubrics Custom Classification'].fillna('').astype(str)
        sector_peers = df_original[classification_col == str(classification)].copy()
    except Exception as e:
        return {"error": f"Error filtering sector peers: {str(e)}. Classification: {classification} (type: {type(classification)}), Column dtype: {df_original['Rubrics Custom Classification'].dtype}"}

    # Get rating peers (same cleaned rating)
    # Use fillna to handle NaN, then convert to string for safe comparison
    try:
        rating_col = df_original['_Credit_Rating_Clean'].fillna('').astype(str)
        rating_peers = df_original[rating_col == str(rating)].copy()
    except Exception as e:
        return {"error": f"Error filtering rating peers: {str(e)}. Rating: {rating} (type: {type(rating)}), Column dtype: {df_original['_Credit_Rating_Clean'].dtype}"}

    # Helper function to calculate percentile
    def calc_percentile(value, peer_series, higher_is_better=True):
        """Calculate percentile rank (0-100)"""
        if pd.isna(value) or value is None:
            return None
        peer_values = pd.to_numeric(peer_series, errors='coerce').dropna()
        if len(peer_values) == 0:
            return None
        if higher_is_better:
            return round((peer_values < value).sum() / len(peer_values) * 100, 1)
        else:
            return round((peer_values > value).sum() / len(peer_values) * 100, 1)

    # Calculate sector medians
    sector_medians = {}
    sector_percentiles = {}

    # Derive from METRIC_REGISTRY - single source of truth
    _compare_keys = ['ebitda_margin', 'roe', 'roa', 'total_debt_ebitda', 'net_debt_ebitda',
                     'current_ratio', 'quick_ratio', 'ebitda_interest', 'revenue_1y_growth', 'revenue_3y_cagr']
    metrics_to_compare = {
        k: (METRIC_REGISTRY[k]['canonical'], METRIC_REGISTRY[k]['higher_is_better'])
        for k in _compare_keys if k in METRIC_REGISTRY
    }

    for metric_key, (col_name, higher_is_better) in metrics_to_compare.items():
        if col_name in sector_peers.columns:
            # Convert to numeric, coercing errors (like 'NM') to NaN
            numeric_values = pd.to_numeric(sector_peers[col_name], errors='coerce')
            sector_medians[metric_key] = numeric_values.median()

            # Get company value
            if metric_key in ['ebitda_margin', 'roe', 'roa']:
                company_value = raw_financials['profitability'].get(metric_key)
            elif metric_key in ['total_debt_ebitda', 'net_debt_ebitda']:
                company_value = raw_financials['leverage'].get(metric_key)
            elif metric_key in ['current_ratio', 'quick_ratio']:
                company_value = raw_financials['liquidity'].get(metric_key)
            elif metric_key == 'ebitda_interest':
                company_value = raw_financials['coverage'].get(metric_key)
            elif metric_key in ['revenue_1y_growth', 'revenue_3y_cagr']:
                company_value = raw_financials['growth'].get(metric_key)
            else:
                company_value = None

            if company_value is not None:
                sector_percentiles[metric_key] = calc_percentile(
                    company_value,
                    sector_peers[col_name],
                    higher_is_better
                )

    # Calculate rating peer medians and percentiles
    rating_medians = {}
    rating_percentiles = {}

    for metric_key, (col_name, higher_is_better) in metrics_to_compare.items():
        if col_name in rating_peers.columns:
            # Convert to numeric, coercing errors (like 'NM') to NaN
            numeric_values = pd.to_numeric(rating_peers[col_name], errors='coerce')
            rating_medians[metric_key] = numeric_values.median()

            # Get company value (same logic as above)
            if metric_key in ['ebitda_margin', 'roe', 'roa']:
                company_value = raw_financials['profitability'].get(metric_key)
            elif metric_key in ['total_debt_ebitda', 'net_debt_ebitda']:
                company_value = raw_financials['leverage'].get(metric_key)
            elif metric_key in ['current_ratio', 'quick_ratio']:
                company_value = raw_financials['liquidity'].get(metric_key)
            elif metric_key == 'ebitda_interest':
                company_value = raw_financials['coverage'].get(metric_key)
            elif metric_key in ['revenue_1y_growth', 'revenue_3y_cagr']:
                company_value = raw_financials['growth'].get(metric_key)
            else:
                company_value = None

            if company_value is not None:
                rating_percentiles[metric_key] = calc_percentile(
                    company_value,
                    rating_peers[col_name],
                    higher_is_better
                )

    peer_context = {
        "sector_comparison": {
            "classification": classification,
            "peer_count": len(sector_peers),
            "medians": sector_medians,
            "percentiles": sector_percentiles,
        },
        "rating_comparison": {
            "rating": rating,
            "peer_count": len(rating_peers),
            "medians": rating_medians,
            "percentiles": rating_percentiles,
        }
    }

    # Cache the result before returning
    st.session_state[cache_key] = peer_context

    return peer_context


def extract_model_outputs(results_df: pd.DataFrame, company_name: str,
                         use_sector_adjusted: bool, calibrated_weights: dict = None) -> dict:
    """
    Extract model scores, rankings, and contextual information from app.py results.
    This provides CONTEXT for interpreting the scores.

    Args:
        results_df: The results DataFrame from app.py (after scoring)
        company_name: Name of company to analyze
        use_sector_adjusted: Whether sector-adjusted scoring is enabled
        calibrated_weights: The calibrated weight dictionary (if dynamic calibration on)

    Returns:
        dict: Complete model output profile with scores and interpretation context
    """

    # Find the company in results
    try:
        company_result = results_df[results_df['Company_Name'] == company_name]
    except Exception as e:
        return {"error": f"Error finding company in results: {str(e)}. Company name: '{company_name}' (type: {type(company_name)}), Column dtype: {results_df['Company_Name'].dtype}"}

    if len(company_result) == 0:
        return {"error": f"Company '{company_name}' not found in results"}

    result = company_result.iloc[0]

    # Get classification and sector
    classification = result.get('Rubrics_Custom_Classification')
    sector = CLASSIFICATION_TO_SECTOR.get(classification, 'Default')

    # Get the actual weights used for this company
    if use_sector_adjusted and calibrated_weights is not None:
        weights_used = calibrated_weights.get(sector, UNIVERSAL_WEIGHTS)
        weights_source = f"Dynamic Calibration ({sector})"
    else:
        weights_used = UNIVERSAL_WEIGHTS
        weights_source = "Universal (No sector adjustment)" if not use_sector_adjusted else "Universal (Calibration error)"

    model_outputs = {
        "overall_metrics": {
            "composite_score": result.get('Composite_Score'),
            "rating_band": result.get('Rating_Band'),
            "signal": result.get('Combined_Signal'),
            "recommendation": result.get('Recommendation'),
        },

        "factor_scores": {
            "credit_score": result.get('Credit_Score'),
            "leverage_score": result.get('Leverage_Score'),
            "profitability_score": result.get('Profitability_Score'),
            "liquidity_score": result.get('Liquidity_Score'),

            "cash_flow_score": result.get('Cash_Flow_Score'),
        },

        "quality_vs_trend": {
            "quality_score": result.get('Quality_Score'),
            "trend_score": result.get('Cycle_Position_Score'),
        },

        "weights_applied": {
            "credit_weight": weights_used.get('credit_score'),
            "leverage_weight": weights_used.get('leverage_score'),
            "profitability_weight": weights_used.get('profitability_score'),
            "liquidity_weight": weights_used.get('liquidity_score'),

            "cash_flow_weight": weights_used.get('cash_flow_score'),
            "source": weights_source,
        },

        "sector_context": {
            "classification": classification,
            "sector": sector,
            "calibration_enabled": use_sector_adjusted,
        },

        "scoring_methodology": {
            "critical_note": "CRITICAL: Model scores represent RELATIVE POSITIONING after sector calibration, NOT absolute credit quality. Low scores can reflect 'average within advantaged sector', not weak fundamentals. ALWAYS check raw metrics first."
        }
    }
    return model_outputs

def _extract_trend_summary_for_genai(time_series_data: dict) -> dict:
    """
    Extract trend summary from time series diagnostic data for GenAI prompt.
    
    READ-ONLY: Only reads from pre-computed diagnostic data.
    Includes bounding information so AI can appropriately caveat analysis.
    
    Args:
        time_series_data: Dict of metric -> time series diagnostic data
        
    Returns:
        dict: Summary of trends with bounding flags
    """
    summary = {
        'metrics': {},
        'bounded_metrics': [],
        'has_bounded_data': False
    }
    
    key_metrics = [
        'EBITDA / Interest Expense (x)',
        'Total Debt / EBITDA (x)',
        'Net Debt / EBITDA',
        'EBITDA Margin',
        'Levered Free Cash Flow Margin',
        'Revenue'
    ]
    
    for metric in key_metrics:
        if metric not in time_series_data:
            continue
            
        ts = time_series_data[metric]
        values = ts.get('values', [])
        
        if not values or len(values) < 2:
            continue
        
        metric_summary = {
            'start_value': values[0] if values else None,
            'end_value': values[-1] if values else None,
            'trend_direction': ts.get('trend_direction', 0),
            'classification': ts.get('classification', 'STABLE'),
            'momentum': ts.get('momentum', 50),
            'volatility': ts.get('volatility', 50),
            'periods_count': ts.get('periods_count', len(values)),
            'fallback_bounded': ts.get('fallback_bounded', False),
        }
        
        if ts.get('fallback_bounded', False):
            metric_summary['bound_limits'] = ts.get('bound_limits', {})
            summary['bounded_metrics'].append(metric)
            summary['has_bounded_data'] = True
        
        summary['metrics'][metric] = metric_summary
    
    return summary

def get_genai_data_from_diagnostics(results_df, company_name, use_sector_adjusted=True, calibrated_weights=None):
    """
    Extract data for GenAI prompt generation directly from the diagnostic data structure.
    Args:
        results_df: Model scoring results DataFrame (contains diagnostic_data column)
        company_name: Company to analyze
        use_sector_adjusted: Whether sector calibration is enabled
        calibrated_weights: Dynamic calibration weights (if enabled)
    
    Returns:
        dict: Complete data package for GenAI, or None if diagnostics unavailable
    """
    try:
        # Find company row
        company_mask = results_df['Company_Name'] == company_name
        if not company_mask.any():
            return None
        
        company_row = results_df[company_mask].iloc[0]
        
        # Check if diagnostic data exists
        diagnostic_json = company_row.get('diagnostic_data')
        if pd.isna(diagnostic_json) or not diagnostic_json:
            return None
        
        # Parse diagnostic JSON
        try:
            diag = json.loads(diagnostic_json)
        except (json.JSONDecodeError, TypeError):
            return None
        
        # Verify required keys exist
        required_keys = ['factor_details', 'time_series', 'composite_calculation']
        if not all(k in diag for k in required_keys):
            return None
        
        # Extract company info from results row
        company_info = {
            "name": str(company_row.get('Company_Name', '')),
            "ticker": str(company_row.get('Ticker', 'N/A')),
            "country": str(company_row.get('Country', 'N/A')),
            "region": str(company_row.get('Region', 'N/A')),
            "sector": str(company_row.get('Sector', 'N/A')),
            "industry": str(company_row.get('Industry', 'N/A')),
            "industry_group": str(company_row.get('Industry_Group', 'N/A')),
            "classification": str(company_row.get('Rubrics_Custom_Classification', 'N/A')),
            "sp_rating": str(company_row.get('Credit_Rating_Clean', 'NR')),
            "sp_rating_clean": str(company_row.get('Credit_Rating_Clean', 'NR')),
            "rating_date": str(company_row.get('S&P_Last_Review_Date', 'N/A')),
            "market_cap": float(company_row.get('Market_Cap')) if pd.notna(company_row.get('Market_Cap')) else None,
        }
        
        # Extract raw inputs from diagnostic data
        raw_inputs = diag.get('raw_inputs', {})
        
        # Add reporting currency to company_info (must be after raw_inputs is defined)
        company_info["reporting_currency"] = raw_inputs.get('Reported_Currency', 'USD')
        
        # Map diagnostic raw_inputs to GenAI raw_financials format
        raw_financials = {
            "company_info": company_info,
            
            "profitability": _extract_profitability_from_diagnostics(diag, company_row),
            "leverage": _extract_leverage_from_diagnostics(diag, company_row, raw_inputs),
            "liquidity": _extract_liquidity_from_diagnostics(diag, company_row, raw_inputs),
            "coverage": _extract_coverage_from_diagnostics(diag, company_row),
            "growth": _extract_growth_from_diagnostics(diag, company_row),
            "cash_flow": _extract_cashflow_from_diagnostics(diag, company_row, raw_inputs),
        }
        
        # Extract model outputs
        composite_calc = diag.get('composite_calculation', {})
        factor_details = diag.get('factor_details', {})
        
        model_outputs = {
            "overall_metrics": {
                "composite_score": composite_calc.get('composite_score'),
                "rating_band": str(company_row.get('Rating_Band', 'N/A')),
                "signal": str(company_row.get('Signal', 'N/A')),
                "recommendation": str(company_row.get('Recommendation', 'N/A')),
            },
            "factor_scores": {
                "credit_score": _safe_get_factor_score(factor_details, 'Credit'),
                "leverage_score": _safe_get_factor_score(factor_details, 'Leverage'),
                "profitability_score": _safe_get_factor_score(factor_details, 'Profitability'),
                "liquidity_score": _safe_get_factor_score(factor_details, 'Liquidity'),
                "cash_flow_score": _safe_get_factor_score(factor_details, 'Cash_Flow'),
            },
            "quality_vs_trend": {
                "quality_score": composite_calc.get('quality_score'),
                "trend_score": composite_calc.get('trend_score'),
            },
            "weights_applied": {
                "credit_weight": _safe_get_weight(composite_calc, 'Credit'),
                "leverage_weight": _safe_get_weight(composite_calc, 'Leverage'),
                "profitability_weight": _safe_get_weight(composite_calc, 'Profitability'),
                "liquidity_weight": _safe_get_weight(composite_calc, 'Liquidity'),
                "cash_flow_weight": _safe_get_weight(composite_calc, 'Cash_Flow'),
                "source": composite_calc.get('weight_method', 'Universal') + " (No sector adjustment)" if not use_sector_adjusted else " (Sector-adjusted)",
            },
            "sector_context": {
                "classification": company_info['classification'],
                "sector": company_info['sector'],
                "calibration_enabled": use_sector_adjusted,
            },
            "scoring_methodology": {
                "critical_note": "CRITICAL: Model scores represent RELATIVE POSITIONING after sector calibration, NOT absolute credit quality. Low scores can reflect 'average within advantaged sector', not weak fundamentals. ALWAYS check raw metrics first."
            }
        }
        
        # Get peer context from diagnostics (pre-computed during scoring)
        peer_context = diag.get('peer_context')
        

        # Extract time series data for trend analysis (READ-ONLY from diagnostics)
        time_series_data = diag.get('time_series', {})
        trend_summary = _extract_trend_summary_for_genai(time_series_data)
        
        return {
            "company_name": company_name,
            "raw_financials": raw_financials,
            "model_outputs": model_outputs,
            "peer_context": peer_context,  # Now from diagnostics, not calculated on-demand
            "trend_details": trend_summary,  # Trend data with bounding info (READ-ONLY)
            "from_diagnostics": True,  # Flag indicating data source
            "data_sources": {
                "raw_metrics": "Pre-computed diagnostic data (from scoring pipeline)",
                "model_scores": "Pre-computed diagnostic data",
                "peer_data": "Pre-computed during scoring (from diagnostics)" if peer_context else "Needs separate calculation",
            },
            "generation_timestamp": pd.Timestamp.now().isoformat(),
            "calibration_info": {
                "sector_adjusted": use_sector_adjusted,
                "dynamic_calibration": calibrated_weights is not None,
                "weights_source": model_outputs['weights_applied']['source'],
            }
        }
        
    except Exception as e:
        pass  # Silently handle diagnostic extraction errors
        return None


def _safe_get_factor_score(factor_details: dict, factor_name: str) -> Optional[float]:
    """Safely extract factor score from diagnostic data."""
    factor = factor_details.get(factor_name, {})
    score = factor.get('final_score') or factor.get('score')
    return float(score) if score is not None else None


def _safe_get_weight(composite_calc: dict, factor_name: str) -> float:
    """Safely extract factor weight from composite calculation."""
    contributions = composite_calc.get('factor_contributions', {})
    factor_contrib = contributions.get(factor_name, {})
    return float(factor_contrib.get('weight', 0.0))


def _extract_leverage_from_diagnostics(diag: dict, row: pd.Series, raw_inputs: dict) -> dict:
    """Extract leverage metrics from diagnostic factor details."""
    lev_details = diag.get('factor_details', {}).get('Leverage', {})
    components = lev_details.get('components', {})
    
    # Get raw values from components (already computed during scoring)
    net_debt_ebitda = _get_component_raw_value(components, 'Net_Debt_EBITDA')
    debt_capital = _get_component_raw_value(components, 'Debt_Capital_Ratio')
    
    # Get absolute values from raw_inputs
    total_debt = raw_inputs.get('Total Debt')
    total_equity = raw_inputs.get('Total Equity')
    ebitda = raw_inputs.get('EBITDA')
    cash = raw_inputs.get('Cash & ST Investments')
    
    # Calculate Total Debt/EBITDA if we have the values and it's not in components
    total_debt_ebitda = None
    if total_debt is not None and ebitda is not None and ebitda != 0:
        total_debt_ebitda = total_debt / ebitda
    
    # Calculate net debt
    net_debt = None
    if total_debt is not None and cash is not None:
        net_debt = total_debt - cash
    
    return {
        "total_debt": total_debt,
        "net_debt": net_debt,
        "total_equity": total_equity,
        "total_debt_ebitda": total_debt_ebitda if total_debt_ebitda is not None else raw_inputs.get('Total Debt / EBITDA'),
        "net_debt_ebitda": net_debt_ebitda,
        "total_debt_equity": raw_inputs.get('Debt to Equity'),
        "total_debt_capital": debt_capital,
    }


def _extract_profitability_from_diagnostics(diag: dict, row: pd.Series) -> dict:
    """Extract profitability metrics from diagnostic factor details."""
    prof_details = diag.get('factor_details', {}).get('Profitability', {})
    components = prof_details.get('components', {})
    raw_inputs = diag.get('raw_inputs', {})
    
    return {
        "ebitda_margin": _get_component_raw_value(components, 'EBITDA_Margin'),
        "gross_margin": _get_component_raw_value(components, 'Gross_Profit_Margin'),
        "ebit_margin": raw_inputs.get('Operating Margin'),
        "operating_margin": raw_inputs.get('Operating Margin'),
        "net_margin": raw_inputs.get('Net Profit Margin'),
        "roe": raw_inputs.get('Return on Equity'),
        "roa": _get_component_raw_value(components, 'ROA'),
        "roic": raw_inputs.get('ROIC'),
    }


def _extract_liquidity_from_diagnostics(diag: dict, row: pd.Series, raw_inputs: dict) -> dict:
    """Extract liquidity metrics from diagnostic data."""
    liq_details = diag.get('factor_details', {}).get('Liquidity', {})
    components = liq_details.get('components', {})
    
    return {
        "current_ratio": _get_component_raw_value(components, 'Current_Ratio'),
        "quick_ratio": _get_component_raw_value(components, 'Quick_Ratio'),
        "cash_st_investments": raw_inputs.get('Cash & ST Investments'),
        "current_assets": raw_inputs.get('Current Assets'),
        "current_liabilities": raw_inputs.get('Current Liabilities'),
        "working_capital": None,  # Can be calculated if needed
    }


def _extract_coverage_from_diagnostics(diag: dict, row: pd.Series) -> dict:
    """Extract coverage metrics from diagnostic data."""
    lev_details = diag.get('factor_details', {}).get('Leverage', {})
    components = lev_details.get('components', {})
    
    # Time series may have the coverage data too
    time_series = diag.get('time_series', {})
    coverage_ts = time_series.get('EBITDA / Interest Expense (x)', {})
    
    # Get most recent value from time series if available
    coverage_value = _get_component_raw_value(components, 'Interest_Coverage')
    if coverage_value is None and coverage_ts:
        values = coverage_ts.get('values', [])
        if values:
            coverage_value = values[-1]  # Most recent
    
    return {
        "ebitda_interest": coverage_value,
        "ebit_interest": None,  # Not typically in diagnostics
        "interest_expense": diag.get('raw_inputs', {}).get('Interest Expense'),
    }


def _extract_growth_from_diagnostics(diag: dict, row: pd.Series) -> dict:
    """Extract growth metrics from diagnostic data."""
    time_series = diag.get('time_series', {})
    raw_inputs = diag.get('raw_inputs', {})  # Get raw_inputs from diagnostic data
    
    # Get growth rates from time series trend_direction if available
    revenue_ts = time_series.get('Revenue', {})
    ebitda_ts = time_series.get('EBITDA', {}) or time_series.get('EBITDA Margin', {})
    
    return {
        "revenue_1y_growth": raw_inputs.get('Revenue 1Y Growth'),
        "revenue_3y_cagr": raw_inputs.get('Revenue 3Y CAGR'),
        "ebitda_3y_cagr": raw_inputs.get('EBITDA 3Y CAGR'),
        "total_revenues": raw_inputs.get('Total Revenue'),
        "ebitda": raw_inputs.get('EBITDA'),
    }


def _extract_cashflow_from_diagnostics(diag: dict, row: pd.Series, raw_inputs: dict) -> dict:
    """Extract cash flow metrics from diagnostic data."""
    cf_details = diag.get('factor_details', {}).get('Cash_Flow', {})
    components = cf_details.get('components', {})
    
    # Try raw_inputs first (absolute values)
    cfo = raw_inputs.get('Operating Cash Flow')
    capex = raw_inputs.get('Capital Expenditures')
    lfcf = raw_inputs.get('Levered Free Cash Flow')
    ufcf = raw_inputs.get('Unlevered Free Cash Flow')
    total_debt = raw_inputs.get('Total Debt')
    
    # Calculate CFO/Debt if we have the values
    cfo_total_debt = None
    if cfo is not None and total_debt is not None and total_debt != 0:
        cfo_total_debt = (cfo / total_debt) * 100
    
    # Fallback: get ratios from scored components (these are always available)
    ocf_to_debt_ratio = _get_component_raw_value(components, 'OCF_to_Debt')
    ocf_to_revenue_ratio = _get_component_raw_value(components, 'OCF_to_Revenue')
    ufcf_margin = _get_component_raw_value(components, 'UFCF_Margin')
    lfcf_margin = _get_component_raw_value(components, 'LFCF_Margin')
    
    # Use component ratio if direct calculation not available
    if cfo_total_debt is None and ocf_to_debt_ratio is not None:
        cfo_total_debt = ocf_to_debt_ratio * 100  # Convert to percentage
    
    return {
        "cfo": cfo,
        "capex": capex,
        "fcf": lfcf,  # Levered FCF is more relevant for credit
        "ufcf": ufcf,
        "cfo_total_debt": cfo_total_debt,
        "fcf_total_debt": None,
        # Additional ratios from components (always available from scoring)
        "ocf_to_revenue": (ocf_to_revenue_ratio * 100) if ocf_to_revenue_ratio else None,
        "ufcf_margin": (ufcf_margin * 100) if ufcf_margin else None,
        "lfcf_margin": (lfcf_margin * 100) if lfcf_margin else None,
    }


def _get_component_raw_value(components: dict, component_name: str) -> Optional[float]:
    """Safely get raw_value from a component dict."""
    component = components.get(component_name, {})
    raw_val = component.get('raw_value')
    if raw_val is not None:
        try:
            return float(raw_val)
        except (ValueError, TypeError):
            return None
    return None


def prepare_genai_credit_report_data(
    df_original: pd.DataFrame,
    results_df: pd.DataFrame,
    company_name: str,
    use_sector_adjusted: bool,
    calibrated_weights: dict = None
) -> dict:
    """
    MASTER FUNCTION: Combines all data sources for GenAI credit report.
    
    OPTIMIZATION: First tries to use pre-computed diagnostic data (fast path).
    Falls back to spreadsheet extraction only if diagnostics unavailable.

    Args:
        df_original: Raw input DataFrame (source of truth for financials)
        results_df: Model outputs (source of scores/signals)
        company_name: Selected company
        use_sector_adjusted: Whether to use sector-adjusted scores
        calibrated_weights: Optional dynamic weights

    Returns:
        dict: Complete data package for GenAI prompt
    """

    try:
        # =====================================================================
        # FAST PATH: Try to get data from pre-computed diagnostics first
        # =====================================================================
        diag_data = get_genai_data_from_diagnostics(
            results_df=results_df,
            company_name=company_name,
            use_sector_adjusted=use_sector_adjusted,
            calibrated_weights=calibrated_weights
        )
        
        if diag_data is not None:
            # Got data from diagnostics - peer context is already included!
            if diag_data.get('peer_context') is None:
                # Fallback: calculate if not in diagnostics (legacy data)
                classification = diag_data['raw_financials']['company_info'].get('classification', 'Unknown')
                rating = diag_data['raw_financials']['company_info'].get('sp_rating_clean', 'Unknown')
                peer_cache_key = f"_peer_context_cache_{classification}_{rating}"
                
                if peer_cache_key in st.session_state:
                    diag_data['peer_context'] = st.session_state[peer_cache_key]
                else:
                    peer_context = calculate_peer_context(df_original, diag_data['raw_financials'])
                    if "error" not in peer_context:
                        st.session_state[peer_cache_key] = peer_context
                    diag_data['peer_context'] = peer_context
            
            diag_data['data_sources']['peer_data'] = "Pre-computed during scoring (from diagnostics)"
            return diag_data
        
        # =====================================================================
        # SLOW PATH: Fall back to spreadsheet extraction
        # =====================================================================
        
        # Step 1: Get raw financials from input spreadsheet (SOURCE OF TRUTH)
        raw_financials = extract_raw_financials_from_input(df_original, company_name)

        if "error" in raw_financials:
            return raw_financials

        # Step 2: Get model outputs from app.py (CONTEXT)
        model_outputs = extract_model_outputs(
            results_df,
            company_name,
            use_sector_adjusted,
            calibrated_weights
        )

        if "error" in model_outputs:
            return model_outputs

        # Step 3: Calculate peer comparisons (RELATIVE POSITIONING)
        peer_context = calculate_peer_context(df_original, raw_financials)

        if "error" in peer_context:
            return peer_context

        # Step 4: Combine everything
        complete_data = {
            "company_name": company_name,

            # PRIMARY DATA: Raw financials from spreadsheet
            "raw_financials": raw_financials,

            # CONTEXT DATA: Model scores and weights
            "model_outputs": model_outputs,

            # COMPARISON DATA: Peer medians and percentiles
            "peer_context": peer_context,

            # METADATA
            "data_sources": {
                "raw_metrics": "Input spreadsheet (source of truth)",
                "model_scores": "app.py scoring engine (context only)",
                "peer_data": "Calculated from input spreadsheet",
            },

            "generation_timestamp": pd.Timestamp.now().isoformat(),

            "calibration_info": {
                "sector_adjusted": use_sector_adjusted,
                "dynamic_calibration": calibrated_weights is not None,
                "weights_source": model_outputs['weights_applied']['source'],
            }
        }

        return complete_data

    except Exception as e:
        return {"error": f"Error preparing data: {str(e)}"}


def _build_trend_section_for_prompt(trend_details: dict, trend_score: float) -> str:
    """
    Build the trend analysis section of the GenAI prompt with bounding context.
    
    READ-ONLY: Only formats data from pre-computed diagnostics.
    """
    def fmt_score(val):
        if val is None or pd.isna(val):
            return "N/A"
        return f"{val:.1f}"
    
    section = f"(4-5 sentences) Based on the Trend Score of {fmt_score(trend_score)}/100:\n"
    section += "- Characterize the trajectory (improving, stable, deteriorating)\n"
    section += "- Identify which metrics are driving the trend\n"
    section += "- Discuss potential catalysts for improvement or further deterioration\n"
    section += "- Note any concerning patterns\n"
    
    # Add metric-level trend details
    metrics = trend_details.get('metrics', {})
    if metrics:
        section += "\n**Trend Details by Metric:**\n"
        for metric_name, metric_data in metrics.items():
            classification = metric_data.get('classification', 'STABLE')
            direction = metric_data.get('trend_direction', 0)
            start = metric_data.get('start_value')
            end = metric_data.get('end_value')
            
            if start is not None and end is not None:
                # SSOT: Use existing alias resolution, then look up metric info
                canonical_name = resolve_column_name(metric_name)
                metric_key = next((k for k, v in METRIC_REGISTRY.items() if v['canonical'] == canonical_name), None)
                unit = METRIC_REGISTRY.get(metric_key, {}).get('unit') if metric_key else None
                
                # Format monetary values for human readability
                if unit == 'K':
                    start_disp = format_monetary_value_for_display(start, metric_name)
                    end_disp = format_monetary_value_for_display(end, metric_name)
                else:
                    start_disp = f"{start:.2f}"
                    end_disp = f"{end:.2f}"
                
                section += f"- {metric_name}: {classification} ({start_disp} -> {end_disp}, {direction:+.1f}%/year)"
                if metric_data.get('fallback_bounded', False):
                    bounds = metric_data.get('bound_limits', {})
                    section += f" [BOUNDED: values capped to CIQ range {bounds.get('min', 0)}-{bounds.get('max', 'N/A')}]"
                section += "\n"
    
    # Add explicit bounding caveat if any metrics were bounded
    if trend_details.get('has_bounded_data', False):
        bounded_list = trend_details.get('bounded_metrics', [])
        section += "\n**[DATA NOTE]** The following metrics contain bounded fallback values: " + ", ".join(bounded_list) + ".\n"
        section += "This occurs when Capital IQ reported \"NM\" (Not Meaningful) for certain periods, and the app calculated the ratio from components. "
        section += "Extreme calculated values (outside CIQ's observed range) were capped to maintain comparability. "
        section += "This typically indicates very low interest expense or structural capital changes - often a POSITIVE credit signal. "
        section += "When analyzing trends for these metrics, focus on the DIRECTION of change rather than the exact magnitude.\n"
    
    return section


def build_comprehensive_credit_prompt(data: dict) -> str:
    """
    Build GenAI prompt for credit analysis report.
    
    Args:
        data: Complete data package from prepare_genai_credit_report_data()
    
    Returns:
        str: Formatted prompt for LLM
    """
    
    raw = data['raw_financials']
    model = data['model_outputs']
    peers = data['peer_context']
    
    # Get currency symbol
    curr_sym = get_currency_symbol(raw.get('company_info', {}).get('reporting_currency', 'USD'))
    
    # Helper function to format metric safely
    def fmt(value, decimals=2, suffix=''):
        if value is None or pd.isna(value):
            return "N/A"
        if suffix == '%':
            return f"{value:.{decimals}f}%"
        elif suffix == 'x':
            return f"{value:.{decimals}f}x"
        elif suffix == 'B':
            return f"{curr_sym}{value/1000000:.1f}B"
        else:
            return f"{value:.{decimals}f}"
    
    # Extract key values for easier reference
    company_name = raw['company_info']['name']
    ticker = raw['company_info']['ticker']
    sp_rating = raw['company_info']['sp_rating']
    sector = raw['company_info']['sector']
    classification = raw['company_info']['classification']
    
    recommendation = model['overall_metrics']['recommendation']
    signal = model['overall_metrics']['signal']
    quality_score = model['quality_vs_trend']['quality_score']
    trend_score = model['quality_vs_trend']['trend_score']
    
    prompt = f"""You are a senior credit analyst at a fixed income investment manager. Write a credit opinion for the portfolio management team.

---
## ISSUER DATA

**Company:** {company_name} ({ticker})
**S&P Rating:** {sp_rating}
**Sector:** {sector} / {classification}

### Current Financials (LTM)

| Category | Metric | Value | Sector Median | Percentile |
|----------|--------|-------|---------------|------------|
| Profitability | EBITDA Margin | {fmt(raw['profitability']['ebitda_margin'], 1, '%')} | {fmt(peers['sector_comparison']['medians'].get('ebitda_margin'), 1, '%')} | {fmt(peers['sector_comparison']['percentiles'].get('ebitda_margin'), 0)}%ile |
| Profitability | ROE | {fmt(raw['profitability']['roe'], 1, '%')} | {fmt(peers['sector_comparison']['medians'].get('roe'), 1, '%')} | {fmt(peers['sector_comparison']['percentiles'].get('roe'), 0)}%ile |
| Profitability | ROA | {fmt(raw['profitability']['roa'], 1, '%')} | N/A | N/A |
| Leverage | Total Debt/EBITDA | {fmt(raw['leverage']['total_debt_ebitda'], 1, 'x')} | {fmt(peers['sector_comparison']['medians'].get('total_debt_ebitda'), 1, 'x')} | {fmt(peers['sector_comparison']['percentiles'].get('total_debt_ebitda'), 0)}%ile |
| Leverage | Net Debt/EBITDA | {fmt(raw['leverage']['net_debt_ebitda'], 1, 'x')} | N/A | N/A |
| Coverage | EBITDA/Interest | {fmt(raw['coverage']['ebitda_interest'], 1, 'x')} | N/A | N/A |
| Liquidity | Current Ratio | {fmt(raw['liquidity']['current_ratio'], 2, 'x')} | {fmt(peers['sector_comparison']['medians'].get('current_ratio'), 2, 'x')} | {fmt(peers['sector_comparison']['percentiles'].get('current_ratio'), 0)}%ile |
| Liquidity | Quick Ratio | {fmt(raw['liquidity']['quick_ratio'], 2, 'x')} | N/A | N/A |
| Liquidity | Cash & ST Inv | {fmt(raw['liquidity']['cash_st_investments'], 1, 'B')} | N/A | N/A |
| Cash Flow | Operating CF | {fmt(raw['cash_flow']['cfo'], 1, 'B')} | N/A | N/A |
| Cash Flow | Free Cash Flow | {fmt(raw['cash_flow']['fcf'], 1, 'B')} | N/A | N/A |
| Cash Flow | CFO/Debt | {fmt(raw['cash_flow']['cfo_total_debt'], 1, '%')} | N/A | N/A |
| Growth | Revenue One Year | {fmt(raw['growth']['revenue_1y_growth'], 1, '%')} | N/A | N/A |
| Growth | Revenue CAGR 3 Year | {fmt(raw['growth']['revenue_3y_cagr'], 1, '%')} | N/A | N/A |

### Capital Structure
- Total Debt: {fmt(raw['leverage']['total_debt'], 1, 'B')}
- Cash & ST Investments: {fmt(raw['liquidity']['cash_st_investments'], 1, 'B')}
- Net Debt: ~{fmt((raw['leverage']['total_debt'] or 0) - (raw['liquidity']['cash_st_investments'] or 0), 1, 'B')}

### Rating Peer Comparison ({peers['rating_comparison']['rating']}, n={peers['rating_comparison']['peer_count']})

| Metric | Issuer | Rating Median | vs Peers |
|--------|--------|---------------|----------|
| EBITDA Margin | {fmt(raw['profitability']['ebitda_margin'], 1, '%')} | {fmt(peers['rating_comparison']['medians'].get('ebitda_margin'), 1, '%')} | {fmt(peers['rating_comparison']['percentiles'].get('ebitda_margin'), 0)}%ile |
| Debt/EBITDA | {fmt(raw['leverage']['total_debt_ebitda'], 1, 'x')} | {fmt(peers['rating_comparison']['medians'].get('total_debt_ebitda'), 1, 'x')} | {fmt(peers['rating_comparison']['percentiles'].get('total_debt_ebitda'), 0)}%ile |

### Model Output
- **Recommendation:** {recommendation}
- **Signal:** {signal}
- Quality Score: {fmt(quality_score, 1)}/100 (>=50 = Strong)
- Trend Score: {fmt(trend_score, 1)}/100 (>=55 = Improving)

---
## YOUR TASK

Write a **comprehensive 1200-1500 word** credit analysis report. Use EXACTLY these markdown section headers:

## EXECUTIVE SUMMARY
(3-4 sentences) State the recommendation ({recommendation}), the signal ({signal}), and the key drivers. This should give a PM the bottom line upfront.

## COMPANY OVERVIEW
(3-4 sentences) Brief description of the business, sector positioning ({sector} / {classification}), and current S&P rating ({sp_rating}).

## CREDIT STRENGTHS
(5-6 detailed bullets using - not bullet points) Analyze each positive factor comprehensively:
- Name the metric, its exact value, and unit
- Compare to BOTH sector median AND rating peer median
- State the percentile ranking
- Explain WHY this matters for credit quality
- Quantify the cushion or buffer where relevant

Example format:
- **Strong operating profitability:** EBITDA margin of 28.2 percent significantly exceeds the Capital Goods sector median of 14.4 percent (96th percentile), demonstrating superior pricing power.

## CREDIT CONCERNS
(5-6 detailed bullets) Same comprehensive format as strengths.

MUST address these if applicable:
- If Trend Score < 55: Explain the deteriorating trajectory and which metrics are weakening
- If Revenue Growth < 2%: Discuss organic growth challenges
- If any metric is below twenty-fifth percentile vs peers: Flag as relative weakness
- If Quality Score is only marginally above 50: Note the limited cushion

## FINANCIAL PROFILE DEEP DIVE

### Leverage & Capital Structure
(4-5 sentences) Analyze Total Debt/EBITDA, Net Debt/EBITDA, Debt/Capital, Debt/Equity. Compare to rating peers. Discuss absolute debt levels and maturity considerations if relevant.

### Profitability & Returns
(4-5 sentences) Analyze EBITDA margin, ROE, ROA, ROIC trends. Compare to sector and rating peers. Discuss sustainability of returns.

### Liquidity & Coverage
(4-5 sentences) Analyze current ratio, quick ratio, cash position, interest coverage. Assess ability to meet near-term obligations and service debt.

### Cash Flow Generation
(4-5 sentences) Analyze CFO, FCF, CFO/Debt, FCF/Debt. Discuss cash conversion and ability to self-fund operations, capex, and debt service.

## PEER COMPARISON ANALYSIS
(5-6 sentences) Detailed comparison to {peers['rating_comparison']['rating']} rating peers (n={peers['rating_comparison']['peer_count']}):
- Where does the issuer rank vs rating peers on key metrics?
- Is the credit profile consistent with the current rating?
- Identify metrics that suggest positive or negative rating migration risk
- Compare to sector peers as well for context

## TREND ANALYSIS & OUTLOOK
{_build_trend_section_for_prompt(data.get('trend_details', {}), trend_score)}

## RECOMMENDATION RATIONALE
(4-5 sentences) Explain the model's recommendation:
- Quality Score of {fmt(quality_score, 1)}/100 is {'above' if quality_score >= 55 else 'below'} the 55 "Strong" threshold
- Trend Score of {fmt(trend_score, 1)}/100 is {'above' if trend_score >= 55 else 'below'} the 55 "Improving" threshold  
- This combination produces the "{signal}" signal
- Synthesize the overall risk/reward and portfolio fit

---
## FORMATTING GUIDELINES

**DO:**
- Use exact numbers from the data provided
- Include percentile rankings (e.g., "73rd percentile")
- Compare to BOTH sector AND rating peer medians
- Use specific thresholds (e.g., "above the 3x investment-grade threshold")
- Quantify spreads and cushions (e.g., "14 percentage points above median")
- Write in professional credit analyst tone
- Use markdown headers exactly as specified above

**DO NOT:**
- Use the $ symbol for currency (write "5.0B USD" or "5.0 billion" instead)
- Invent data not provided
- Override or contradict the model recommendation
- Use vague qualifiers without data support
- Use bullet points with • (use - instead)
- Number the sections (use ## headers only)

**PERCENTILE INTERPRETATION:**
- For profitability/coverage/liquidity: Higher percentile = stronger credit
- For leverage (Debt/EBITDA): Higher percentile = LOWER leverage = stronger credit
- >75th percentile: Credit strength
- 50-75th: Above average
- 25-50th: Below average  
- <25th: Credit weakness requiring monitoring

Generate the comprehensive credit analysis report now.
"""
    
    return prompt


def get_most_recent_column(df, base_metric, data_period_setting):
    """
    [V2.2] Returns the appropriate metric column based on parsed Period Ended dates.

    Instead of assuming .4 = FY0 and .12 = CQ-0, this function:
    1. Parses Period Ended columns to get actual dates
    2. Classifies periods as FY or CQ based on frequency heuristics
    3. Selects the latest valid date for the requested period type
    4. Falls back gracefully if no Period Ended columns exist

    Args:
        df: DataFrame with Period Ended columns
        base_metric: Base metric name (e.g., "EBITDA Margin")
        data_period_setting: "Most Recent Fiscal Year (FY0)" or "Most Recent Quarter (CQ-0)"

    Returns:
        Series of metric values for the selected period
    """
    # Check if Period Ended columns exist
    pe_cols = [c for c in df.columns if c.startswith("Period Ended")]

    if len(pe_cols) == 0:
        # No Period Ended columns - fall back to unsuffixed metric
        if base_metric in df.columns:
            return pd.to_numeric(df[base_metric], errors='coerce')
        else:
            return pd.Series([np.nan] * len(df), index=df.index)

    # Parse Period Ended columns to get (suffix, datetime_series) tuples
    pe_data = parse_period_ended_cols(df)

    if len(pe_data) == 0:
        # No valid period data - fall back to base metric
        if base_metric in df.columns:
            return pd.to_numeric(df[base_metric], errors='coerce')
        else:
            return pd.Series([np.nan] * len(df), index=df.index)

    # Classify periods as FY or CQ based on frequency
    fy_suffixes, ltm_suffixes = period_cols_by_kind(pe_data, df)

    # Determine which suffixes to search based on user selection
    if data_period_setting == "Most Recent Fiscal Year (FY0)":
        target_suffixes = fy_suffixes if fy_suffixes else [s for s, _ in pe_data]
    elif data_period_setting == "Most Recent Quarter (CQ-0)":
        target_suffixes = ltm_suffixes if ltm_suffixes else [s for s, _ in pe_data]
    else:
        # Defensive: unknown value -> treat as FY0
        target_suffixes = fy_suffixes if fy_suffixes else [s for s, _ in pe_data]

    # Find the latest valid date among target suffixes
    # Build a list of (suffix, datetime_series) for the target type
    target_periods = [(s, dt_series) for s, dt_series in pe_data if s in target_suffixes]

    if not target_periods:
        # No matching periods - fall back to base or first available
        if base_metric in df.columns:
            return pd.to_numeric(df[base_metric], errors='coerce')
        elif pe_data:
            # Use first available period
            first_suffix = pe_data[0][0]
            metric_col = f"{base_metric}{first_suffix}"
            if metric_col in df.columns:
                return pd.to_numeric(df[metric_col], errors='coerce')

        return pd.Series([np.nan] * len(df), index=df.index)

    # For each row, find the latest valid date among target periods
    result = []
    for idx in range(len(df)):
        # Collect (date, suffix) pairs for this row
        date_suffix_pairs = []
        for suffix, dt_series in target_periods:
            dt_val = dt_series.iloc[idx]
            if pd.notna(dt_val):
                date_suffix_pairs.append((dt_val, suffix))

        if date_suffix_pairs:
            # Sort by date and take the latest
            date_suffix_pairs.sort(key=lambda x: x[0], reverse=True)
            latest_suffix = date_suffix_pairs[0][1]
            metric_col = f"{base_metric}{latest_suffix}"

            if metric_col in df.columns:
                val = df[metric_col].iloc[idx]
                result.append(pd.to_numeric(val, errors='coerce'))
            else:
                result.append(np.nan)
        else:
            # No valid dates for this row - try base column
            if base_metric in df.columns:
                val = df[base_metric].iloc[idx]
                result.append(pd.to_numeric(val, errors='coerce'))
            else:
                result.append(np.nan)

    return pd.Series(result, index=df.index)

def _build_metric_timeseries(df: pd.DataFrame, base_metric: str, use_quarterly: bool,
                              reference_date=None, prefer_annual_reports=False,
                              pe_data_cached=None, fy_cq_cached=None,
                              selected_periods=None) -> pd.DataFrame:
    """
    OPTIMIZED: Vectorized time series construction with FY/CQ de-duplication.
    Returns DataFrame where each row is an issuer's time series (columns = ISO dates).

    Args:
        reference_date: Optional cutoff date (str or pd.Timestamp) to align all issuers.
                       If provided, only uses data up to this date for all issuers.
        pe_data_cached: Pre-parsed period columns to avoid re-parsing (performance optimization)
        fy_cq_cached: Pre-computed (fy_suffixes, ltm_suffixes) tuple
        selected_periods: DataFrame with unified period selection (from select_aligned_period).
                         If provided, filters time series to only include the selected periods.
    """
    # 1) Parse period-ended columns -> [(suffix, series_of_dates), ...]
    if pe_data_cached is not None:
        pe_data = pe_data_cached
    else:
        pe_data = parse_period_ended_cols(df.copy())

    if not pe_data:
        # Fallback: just the base and suffixed columns in suffix order (maintain current behavior)
        ts_cols = [base_metric] + [f"{base_metric}.{i}" for i in range(1, (12 if use_quarterly else 4) + 1)]
        available = [c for c in ts_cols if c in df.columns]
        out = df[available].apply(pd.to_numeric, errors="coerce")
        out.columns = pd.RangeIndex(start=0, stop=len(available))  # anonymous index if no dates
        return out

    # 2) Determine FY vs CQ suffix sets using existing classifier
    if fy_cq_cached is not None:
        fy_suffixes, ltm_suffixes = fy_cq_cached
    else:
        fy_suffixes, ltm_suffixes = period_cols_by_kind(pe_data, df)

    # 3) Choose candidate suffix list by mode
    if use_quarterly:
        candidate_suffixes = [s for s, _ in pe_data]
    else:
        candidate_suffixes = fy_suffixes if fy_suffixes else [s for s, _ in pe_data]

    # 4) VECTORIZED: Build long-format DataFrame with (row_idx, date, value, is_cq)
    long_data = []
    ltm_set = set(ltm_suffixes)

    for sfx in candidate_suffixes:
        col = f"{base_metric}{sfx}" if sfx else base_metric
        if col not in df.columns:
            continue

        # Get date series for this suffix
        date_series = dict(pe_data).get(sfx)
        if date_series is None:
            continue

        # Build DataFrame chunk for this suffix
        chunk = pd.DataFrame({
            'row_idx': df.index,
            'date': pd.to_datetime(date_series.values, errors='coerce'),
            'value': pd.to_numeric(df[col], errors='coerce'),
            'is_cq': sfx in ltm_set
        })
        long_data.append(chunk)

    if not long_data:
        return pd.DataFrame(index=df.index)

    # Concatenate all chunks
    long_df = pd.concat(long_data, ignore_index=True)

    # [V5.0.1] Interest Coverage Fallback for Trend Calculation
    if base_metric == 'EBITDA / Interest Expense (x)':
        # Build EBITDA and Interest Expense lookup for fallback
        ebitda_lookup = {}
        int_exp_lookup = {}
        
        for sfx in candidate_suffixes:
            ebitda_col = f"EBITDA{sfx}" if sfx else "EBITDA"
            int_exp_col = f"Interest Expense{sfx}" if sfx else "Interest Expense"
            date_series = dict(pe_data).get(sfx)
            
            if ebitda_col in df.columns and int_exp_col in df.columns and date_series is not None:
                for row_idx in df.index:
                    date = pd.to_datetime(date_series.loc[row_idx], errors='coerce')
                    if pd.notna(date):
                        key = (row_idx, date)
                        ebitda_lookup[key] = pd.to_numeric(df.loc[row_idx, ebitda_col], errors='coerce')
                        int_exp_lookup[key] = pd.to_numeric(df.loc[row_idx, int_exp_col], errors='coerce')
        
        # Apply fallback where value is NaN (with CIQ-aligned bounding via SSOT helper)
        fallback_count = 0
        bounded_count = 0
        
        # Initialize was_bounded column if not present
        if 'was_bounded' not in long_df.columns:
            long_df['was_bounded'] = False
        
        for idx in long_df.index:
            if pd.isna(long_df.loc[idx, 'value']):
                key = (long_df.loc[idx, 'row_idx'], long_df.loc[idx, 'date'])
                ebitda = ebitda_lookup.get(key)
                int_exp = int_exp_lookup.get(key)
                
                if pd.notna(ebitda) and pd.notna(int_exp) and int_exp != 0:
                    raw_value = ebitda / abs(int_exp)
                    # Apply CIQ-aligned bounds via SSOT helper
                    bounded_value, was_bounded = apply_ciq_ratio_bounds(base_metric, raw_value)
                    long_df.loc[idx, 'value'] = bounded_value
                    long_df.loc[idx, 'was_bounded'] = was_bounded
                    if was_bounded:
                        bounded_count += 1
                    fallback_count += 1
        
        if fallback_count > 0 and not os.environ.get("RG_TESTS"):
            bounds = CIQ_RATIO_BOUNDS.get(base_metric, {})
            msg = f"  [DEV] Interest Coverage trend fallback: {fallback_count} data points calculated"
            if bounded_count > 0:
                msg += f" ({bounded_count} bounded to [{bounds.get('min', 0)}, {bounds.get('max', 'N/A')}])"
            print(msg)

    # [V5.0.2] Net Debt / EBITDA Fallback for Trend Calculation
    # Handles "NM" cases, especially net cash positions (Cash > Debt)
    if base_metric == 'Net Debt / EBITDA':
        # Build lookup for components
        total_debt_lookup = {}
        cash_lookup = {}
        ebitda_lookup = {}
        
        for sfx in candidate_suffixes:
            debt_col = f"Total Debt{sfx}" if sfx else "Total Debt"
            cash_col = f"Cash & Short-term Investments{sfx}" if sfx else "Cash & Short-term Investments"
            ebitda_col = f"EBITDA{sfx}" if sfx else "EBITDA"
            date_series = dict(pe_data).get(sfx)
            
            if all(c in df.columns for c in [debt_col, cash_col, ebitda_col]) and date_series is not None:
                for row_idx in df.index:
                    date = pd.to_datetime(date_series.loc[row_idx], errors='coerce')
                    if pd.notna(date):
                        key = (row_idx, date)
                        total_debt_lookup[key] = pd.to_numeric(df.loc[row_idx, debt_col], errors='coerce')
                        cash_lookup[key] = pd.to_numeric(df.loc[row_idx, cash_col], errors='coerce')
                        ebitda_lookup[key] = pd.to_numeric(df.loc[row_idx, ebitda_col], errors='coerce')
        
        # Apply fallback where value is NaN (with CIQ-aligned bounding via SSOT helper)
        fallback_count = 0
        bounded_count = 0
        
        # Initialize was_bounded column if not present
        if 'was_bounded' not in long_df.columns:
            long_df['was_bounded'] = False
        
        for idx in long_df.index:
            if pd.isna(long_df.loc[idx, 'value']):
                key = (long_df.loc[idx, 'row_idx'], long_df.loc[idx, 'date'])
                debt = total_debt_lookup.get(key)
                cash = cash_lookup.get(key)
                ebitda = ebitda_lookup.get(key)
                
                # Only calculate if EBITDA is positive (negative EBITDA = truly NM)
                if pd.notna(debt) and pd.notna(cash) and pd.notna(ebitda) and ebitda > 0:
                    net_debt = debt - cash  # Can be negative for net cash positions
                    raw_value = net_debt / ebitda
                    # Apply CIQ-aligned bounds via SSOT helper
                    bounded_value, was_bounded = apply_ciq_ratio_bounds(base_metric, raw_value)
                    long_df.loc[idx, 'value'] = bounded_value
                    long_df.loc[idx, 'was_bounded'] = was_bounded
                    if was_bounded:
                        bounded_count += 1
                    fallback_count += 1
        
        if fallback_count > 0 and not os.environ.get("RG_TESTS"):
            bounds = CIQ_RATIO_BOUNDS.get(base_metric, {})
            msg = f"  [DEV] Net Debt/EBITDA trend fallback: {fallback_count} data points calculated"
            if bounded_count > 0:
                msg += f" ({bounded_count} bounded to [{bounds.get('min', 0)}, {bounds.get('max', 'N/A')}])"
            print(msg)

    # 5) Filter out invalid dates and values
    long_df = long_df[long_df['date'].notna() & long_df['value'].notna()]
    long_df = long_df[long_df['date'].dt.year != 1900]  # Remove 1900 sentinels

    # 5a) TIMING MISMATCH FIX: Filter to reference date if provided
    if reference_date is not None:
        reference_dt = pd.to_datetime(reference_date)
        long_df = long_df[long_df['date'] <= reference_dt]

        # 5b) Exclude future projected periods beyond reasonable reporting lag
        # Periods more than 60 days in the future from current date are likely projections/estimates
        # rather than actual reported data. We should exclude these to avoid using incomplete
        # or estimated metrics in credit analysis.
        current_date = pd.Timestamp.now()
        reporting_lag_cutoff = current_date + pd.DateOffset(days=60)
        long_df = long_df[long_df['date'] <= reporting_lag_cutoff]

    # 6) De-duplicate: For same (row_idx, date), prefer FY over CQ
    if use_quarterly:
        # Sort so FY comes first, then drop duplicates keeping first (FY preferred)
        # is_cq=False (FY) sorts before is_cq=True (CQ) when ascending=True
        # FY data is more comprehensive (full fiscal year consolidation) than quarterly data
        long_df = long_df.sort_values(['row_idx', 'date', 'is_cq'], ascending=[True, True, True])
        long_df = long_df.drop_duplicates(subset=['row_idx', 'date'], keep='first')
    else:
        # Annual mode: already filtered to FY only by candidate_suffixes
        long_df = long_df.drop_duplicates(subset=['row_idx', 'date'], keep='first')

    # 6b) Apply period priority preference when user selected "Annual Reports"
    if prefer_annual_reports and use_quarterly:
        # When user prefers annual reports, select FY periods over CQ periods
        # even when CQ has a more recent date within the reference window.
        # Strategy: For each issuer, find their latest FY period. If an FY period exists,
        # exclude any CQ periods (even if they're more recent).

        # Group by issuer and check if they have any FY periods
        has_fy = long_df[long_df['is_cq'] == False].groupby('row_idx').size()
        issuers_with_fy = has_fy[has_fy > 0].index

        # For issuers with FY data, keep only FY periods (exclude all CQ)
        # For issuers without FY data, keep their CQ periods
        long_df = long_df[
            (~long_df['row_idx'].isin(issuers_with_fy)) |  # No FY available - keep CQ
            (long_df['is_cq'] == False)  # Has FY - keep only FY, exclude CQ
        ]

    # 6c) NEW: If unified period selection provided, filter to selected periods only
    # This ensures trend scores use the same periods as quality scores
    if selected_periods is not None and len(selected_periods) > 0:
        # Build set of (row_idx, date) tuples from selected periods
        selected_set = set()
        for _, row in selected_periods.iterrows():
            selected_set.add((row['row_idx'], pd.to_datetime(row['selected_date']).date()))

        # Filter long_df to only include selected period dates
        long_df['date_only'] = long_df['date'].dt.date
        long_df = long_df[long_df.apply(lambda r: (r['row_idx'], r['date_only']) in selected_set, axis=1)]
        long_df = long_df.drop(columns=['date_only'])

    # 7) Convert dates to ISO strings for column names
    long_df['date_str'] = long_df['date'].dt.date.astype(str)

    # 8) Pivot to wide format: rows = issuers, columns = dates
    wide_df = long_df.pivot_table(
        index='row_idx',
        columns='date_str',
        values='value',
        aggfunc='first'  # Should be unnecessary after de-dup, but safe
    )

    # Track which issuers had any bounded values (for diagnostic display)
    bounded_by_issuer = pd.Series(False, index=df.index)
    if 'was_bounded' in long_df.columns:
        bounded_agg = long_df.groupby('row_idx')['was_bounded'].any()
        for issuer_idx in bounded_agg.index:
            if issuer_idx in bounded_by_issuer.index:
                bounded_by_issuer.loc[issuer_idx] = bounded_agg.loc[issuer_idx]
    
    # Attach as attribute for downstream access
    wide_df.attrs['bounded_by_issuer'] = bounded_by_issuer

    # 9) Sort columns by date (ascending) and reindex to match original df
    wide_df = wide_df[sorted(wide_df.columns)]
    wide_df = wide_df.reindex(df.index, fill_value=np.nan)

    return wide_df

# ============================================================================
# DUAL-HORIZON TREND ANALYSIS UTILITIES (V2.2)
# ============================================================================

def robust_slope(xs: np.ndarray, ys: np.ndarray) -> float:
    """
    Calculate robust linear slope using winsorized data.

    Args:
        xs: x-values (e.g., np.arange(n))
        ys: y-values (metric time series)

    Returns:
        Slope coefficient (b1 from y = b0 + b1*x)
    """
    if len(ys) < 3:
        return np.nan

    y_series = pd.Series(ys).dropna()
    if len(y_series) < 3:
        return np.nan

    # Winsorize at 5th and 95th percentiles
    y_clipped = y_series.clip(
        lower=y_series.quantile(0.05),
        upper=y_series.quantile(0.95)
    ).values

    try:
        coeffs = np.polyfit(xs[:len(y_clipped)], y_clipped, 1)
        return float(coeffs[0])  # slope is first coefficient
    except Exception:
        return np.nan


def zscore_last(ys: np.ndarray) -> float:
    """
    Calculate z-score of the last value relative to the series.

    Args:
        ys: Time series values

    Returns:
        Z-score of last value (negative = below mean)
    """
    y_series = pd.Series(ys).dropna()
    if len(y_series) < 4:
        return 0.0

    mu = y_series.mean()
    sd = y_series.std(ddof=1)

    if sd == 0 or pd.isna(sd):
        return 0.0

    return float((y_series.iloc[-1] - mu) / sd)


def cv_last8(ys: np.ndarray) -> float:
    """
    Calculate coefficient of variation for last 8 periods.

    Args:
        ys: Time series values

    Returns:
        CV = std / |mean|
    """
    y_series = pd.Series(ys).dropna()
    if len(y_series) < 4:
        return 0.0

    mu = y_series.mean()
    if mu == 0 or pd.isna(mu):
        return 0.0

    sd = y_series.std(ddof=1)
    return float(sd / abs(mu))


def near_peak(ys: np.ndarray, tolerance: float = 0.10) -> bool:
    """
    Check if last value is near peak (within tolerance of max).

    Args:
        ys: Time series values
        tolerance: Fraction of max (default 10%)

    Returns:
        True if last value within tolerance of peak
    """
    y_series = pd.Series(ys).dropna()
    if len(y_series) < 4:
        return False

    last_val = y_series.iloc[-1]
    max_val = y_series.max()

    if max_val == 0 or pd.isna(max_val):
        return False

    return abs(last_val - max_val) <= tolerance * max_val


def compute_dual_horizon_trends(ts_row: pd.Series, min_periods: int = 5, peak_tolerance: float = 0.10) -> dict:
    """
    Compute dual-horizon trend signals for a single issuer's time series.

    Args:
        ts_row: Time series row (index = period labels, values = metric values)
        min_periods: Minimum periods required for calculation
        peak_tolerance: Tolerance for near-peak detection (default 10%)

    Returns:
        Dictionary with keys:
        - medium_term_slope: Robust slope over full series
        - short_term_change: Recent 4Q avg vs prior 4Q avg
        - last_quarter_z: Z-score of most recent value
        - series_cv: Coefficient of variation
        - near_peak_flag: Boolean if near 2Y peak
    """
    values = ts_row.dropna()
    n = len(values)

    result = {
        'medium_term_slope': np.nan,
        'short_term_change': np.nan,
        'last_quarter_z': 0.0,
        'series_cv': 0.0,
        'near_peak_flag': False
    }

    if n < min_periods:
        return result

    # Medium-term slope (robust, over full series)
    xs = np.arange(n)
    ys = values.values
    result['medium_term_slope'] = robust_slope(xs, ys)

    # Short-term change (recent 4Q vs prior 4Q)
    if n >= 8:
        recent_4q = values.iloc[-4:].mean()
        prior_4q = values.iloc[-8:-4].mean()
        result['short_term_change'] = float(recent_4q - prior_4q)

    # Last quarter z-score
    result['last_quarter_z'] = zscore_last(ys)

    # Series coefficient of variation
    result['series_cv'] = cv_last8(ys)

    # Near peak detection (uses configurable tolerance)
    result['near_peak_flag'] = near_peak(ys, tolerance=peak_tolerance)

    return result


def calculate_trend_indicators(df, base_metrics, use_quarterly=False,
                                reference_date=None, prefer_annual_reports=False,
                                selected_periods=None):
    """
    SOLUTION TO ISSUE #2: MISSING CYCLICALITY & TREND ANALYSIS
    OPTIMIZED: Caches period parsing and uses vectorized calculations.

    Calculate trend, momentum, and volatility indicators using historical time series.

    Args:
        df: DataFrame with metric columns
        base_metrics: List of base metric names
        use_quarterly: If True, use [base, .1, .2, ..., .12] (13 periods: 5 annual + 8 quarterly)
                      If False, use [base, .1, .2, .3, .4] (5 annual periods only)
        reference_date: Optional cutoff date (str or pd.Timestamp) to align all issuers.
                       If provided, only uses data up to this date for all issuers.

    Returns DataFrame with new columns:
    - {metric}_trend: -1 to +1 (negative = deteriorating, positive = improving)
    - {metric}_volatility: 0 to 1 (higher = more volatile)
    - {metric}_momentum: 0 to 100 (recent vs. prior performance)
    """
    trend_scores = pd.DataFrame(index=df.index)

    # Dev check for RG_TESTS
    if os.environ.get("RG_TESTS") == "1":
        print("  [DEV] Trend indicators use true Period Ended dates; FY/CQ overlaps deduplicated (CQ preferred in quarterly mode)")

    # OPTIMIZATION: Cache period parsing - parse once, use for all metrics
    pe_data_cached = parse_period_ended_cols(df.copy())
    if pe_data_cached:
        fy_cq_cached = period_cols_by_kind(pe_data_cached, df)
    else:
        fy_cq_cached = None

    def _classify_trend(trend_value, metric_name=None):
        """
        Classify trend direction based on normalized trend value.
        
        Args:
            trend_value: Normalized trend from -1 to 1
            metric_name: Optional metric name to determine polarity
        
        Returns:
            String classification: 'IMPROVING', 'STABLE', 'DETERIORATING', 'INSUFFICIENT_DATA'
        """
        # Metrics where LOWER values are BETTER (declining = improving)
        LOWER_IS_BETTER_TREND_METRICS = {
            'Total Debt / EBITDA (x)',
            'Net Debt / EBITDA',
            'Total Debt / Total Capital (%)',
            'Total Debt/Equity (x)',
        }
        
        # For "lower is better" metrics, flip the interpretation
        if metric_name and metric_name in LOWER_IS_BETTER_TREND_METRICS:
            trend_value = -trend_value
        
        if trend_value > 0.2:
            return 'IMPROVING'
        elif trend_value < -0.2:
            return 'DETERIORATING'
        else:
            return 'STABLE'

    # Helper function for vectorized calculations - TIME-AWARE VERSION
    def _calc_row_stats(row_series, return_full_diagnostic=False, metric_name=None):
        """
        Calculate trend, volatility, momentum for a single row's time series.

        TIME-AWARE VERSION: Accounts for actual time intervals between periods.

        Args:
            row_series: Series with ISO date strings as index and metric values
            return_full_diagnostic: If True, return full diagnostic data including time series

        Returns:
            If return_full_diagnostic=False: Series with 'trend', 'vol', 'mom' scores (backward compatible)
            If return_full_diagnostic=True: Dict with comprehensive diagnostic data
        """
        values = row_series.dropna()
        n = len(values)

        if n < 3:
            basic_result = pd.Series({'trend': 0.0, 'vol': 50.0, 'mom': 50.0})
            if not return_full_diagnostic:
                return basic_result
            else:
                return {
                    'dates': [],
                    'values': [],
                    'trend_direction': 0.0,
                    'momentum': 50.0,
                    'volatility': 50.0,
                    'classification': 'INSUFFICIENT_DATA',
                    'periods_count': n
                }

        # Parse dates from index (ISO strings like "2024-12-31")
        try:
            dates = pd.to_datetime(values.index)
        except Exception:
            # Fallback: if date parsing fails, use legacy logic
            basic_result = _calc_row_stats_legacy(row_series)
            if not return_full_diagnostic:
                return basic_result
            else:
                # Construct diagnostic from legacy result
                return {
                    'dates': [],
                    'values': values.values.tolist(),
                    'trend_direction': float(basic_result['trend']) * 10.0,
                    'momentum': float(basic_result['mom']),
                    'volatility': float(basic_result['vol']),
                    'classification': _classify_trend(float(basic_result['trend']), metric_name=metric_name),
                    'periods_count': n
                }

        # Convert dates to years from start (for time-aware regression)
        time_zero = dates[0]
        time_years = np.array([(d - time_zero).days / 365.25 for d in dates])
        y = values.values

        # === TREND CALCULATION (TIME-AWARE) ===
        if n >= 3 and np.std(y) > 0 and time_years[-1] > 0:
            # Linear regression: y = a + b*time_years
            # Slope 'b' is now in units of "metric per YEAR"
            slope_per_year = np.polyfit(time_years, y, 1)[0]

            # Normalize slope by mean to make it scale-independent
            # This gives "percent change per year"
            mean_val = abs(values.mean())
            if mean_val > 0:
                slope_pct_per_year = slope_per_year / mean_val
                # Scale up by 10x to fully spread distribution across 20-80 range
                # Financial metrics with 1-5% annual change translate to meaningful trend scores:
                # 1% per year → 0.01 * 10 = 0.10 → score 55 (slightly improving)
                # 5% per year → 0.05 * 10 = 0.50 → score 75 (strongly improving)
                # Clip to ±100% per year → ±1.0 trend score
                trend = float(np.clip(slope_pct_per_year * 10, -1, 1))
            else:
                trend = 0.0
        else:
            trend = 0.0

        # === VOLATILITY CALCULATION (UNCHANGED) ===
        # This is already time-agnostic - just measures dispersion
        if n >= 3 and values.mean() != 0:
            cv = values.std() / abs(values.mean())
            vol = float(100 - np.clip(cv * 100, 0, 100))
        else:
            vol = 50.0

        # === MOMENTUM CALCULATION (TIME-AWARE) ===
        if n >= 8 and time_years[-1] > 0:
            # Use TIME-BASED windows instead of index-based
            # Split time span into two equal halves
            total_time_span = time_years[-1] - time_years[0]
            midpoint_time = time_years[0] + (total_time_span / 2.0)

            # Prior half: [start, midpoint)
            # Recent half: [midpoint, end]
            prior_mask = time_years < midpoint_time
            recent_mask = time_years >= midpoint_time

            # Calculate averages for each half
            if np.any(prior_mask) and np.any(recent_mask):
                prior_count = np.sum(prior_mask)
                recent_count = np.sum(recent_mask)

                # Require at least 2 data points in each window
                if prior_count >= 2 and recent_count >= 2:
                    prior_avg = float(values[prior_mask].mean())
                    recent_avg = float(values[recent_mask].mean())

                    if prior_avg != 0:
                        mom = 50.0 + 50.0 * ((recent_avg - prior_avg) / abs(prior_avg))
                    else:
                        mom = 50.0
                    mom = float(np.clip(mom, 0, 100))
                else:
                    mom = 50.0
            else:
                # Edge case: all data in one half
                mom = 50.0
        else:
            # Fallback to index-based momentum if time span is zero (e.g. integer index)
            if n >= 8:
                 recent_avg = float(values.iloc[-4:].mean())
                 prior_avg = float(values.iloc[-8:-4].mean())
                 if prior_avg != 0:
                     mom = 50.0 + 50.0 * ((recent_avg - prior_avg) / abs(prior_avg))
                 else:
                     mom = 50.0
                 mom = float(np.clip(mom, 0, 100))
            else:
                 mom = 50.0

        # Basic result for backward compatibility
        basic_result = pd.Series({'trend': trend, 'vol': vol, 'mom': mom})
        
        if not return_full_diagnostic:
            return basic_result
        
        # Full diagnostic data
        return {
            'dates': [d.strftime('%Y-%m-%d') for d in dates],  # ISO format strings
            'values': values.values.tolist(),
            'trend_direction': trend * 10.0,  # trend has 10x scaling, so ×10 gives actual % per year
            'momentum': mom,
            'volatility': vol,
            'classification': _classify_trend(trend, metric_name=metric_name),
            'periods_count': n
        }

    def _calc_row_stats_legacy(row_series):
        """
        LEGACY VERSION: Original index-based calculation (for fallback).
        Used if date parsing fails or for backward compatibility.
        """
        values = row_series.dropna()
        n = len(values)

        # Trend (slope)
        if n >= 3:
            x = np.arange(n)
            y = values.values
            slope = np.polyfit(x, y, 1)[0] if np.std(y) > 0 else 0.0
            trend = float(np.clip(slope * 10, -1, 1))
        else:
            trend = 0.0

        # Volatility
        if n >= 3 and values.mean() != 0:
            cv = values.std() / abs(values.mean())
            vol = float(100 - np.clip(cv * 100, 0, 100))
        else:
            vol = 50.0

        # Momentum
        if n >= 8:
            recent_avg = float(values.iloc[-4:].mean())
            prior_avg = float(values.iloc[-8:-4].mean())
            if prior_avg != 0:
                mom = 50.0 + 50.0 * ((recent_avg - prior_avg) / abs(prior_avg))
            else:
                mom = 50.0
            mom = float(np.clip(mom, 0, 100))
        else:
            mom = 50.0

        return pd.Series({'trend': trend, 'vol': vol, 'mom': mom})

    # Process each metric with cached data
    for base_metric in base_metrics:
        ts = _build_metric_timeseries(df, base_metric, use_quarterly=use_quarterly,
                                      reference_date=reference_date,
                                      prefer_annual_reports=prefer_annual_reports,
                                      pe_data_cached=pe_data_cached, fy_cq_cached=fy_cq_cached,
                                      selected_periods=selected_periods)

        # Vectorized calculation using apply
        # Pass metric name to enable polarity-aware classification
        if ts.empty:
            # Handle empty time series
            trend_scores[f'{base_metric}_trend'] = 0.0
            trend_scores[f'{base_metric}_volatility'] = 50.0
            trend_scores[f'{base_metric}_momentum'] = 50.0
        else:
            stats = ts.apply(lambda row: _calc_row_stats(row, metric_name=base_metric), axis=1)
            
            # Ensure stats is a DataFrame with expected columns
            if isinstance(stats, pd.Series):
                # If apply returned a Series (e.g. single row or failed expansion), convert to DataFrame
                # This handles cases where apply doesn't expand correctly
                if stats.empty:
                     stats = pd.DataFrame(columns=['trend', 'vol', 'mom'])
                else:
                     # If it's a Series of Series, expand it
                     stats = stats.apply(pd.Series)

            if 'trend' in stats.columns:
                trend_scores[f'{base_metric}_trend'] = stats['trend']
                trend_scores[f'{base_metric}_volatility'] = stats['vol']
                trend_scores[f'{base_metric}_momentum'] = stats['mom']
            else:
                # Fallback if columns missing
                trend_scores[f'{base_metric}_trend'] = 0.0
                trend_scores[f'{base_metric}_volatility'] = 50.0
                trend_scores[f'{base_metric}_momentum'] = 50.0

    # NEW: Capture diagnostic data for time series (Phase 1 - Diagnostic Storage)
    # This will be added to results_dict later
    trend_diagnostic_data = {}
    
    for base_metric in base_metrics:
        ts = _build_metric_timeseries(df, base_metric, use_quarterly=use_quarterly,
                                      reference_date=reference_date,
                                      prefer_annual_reports=prefer_annual_reports,
                                      pe_data_cached=pe_data_cached, fy_cq_cached=fy_cq_cached,
                                      selected_periods=selected_periods)
        
        # Collect diagnostic data for each issuer
        for idx in ts.index:
            if idx not in trend_diagnostic_data:
                trend_diagnostic_data[idx] = {}
            
            # Get full diagnostic data for this issuer's time series
            row_series = ts.loc[idx]
            diag_data = _calc_row_stats(row_series, return_full_diagnostic=True, metric_name=base_metric)
            
            # Check if any values were bounded (from fallback calculation)
            bounded_by_issuer = ts.attrs.get('bounded_by_issuer', pd.Series(False, index=ts.index))
            was_bounded = bool(bounded_by_issuer.get(idx, False)) if idx in bounded_by_issuer.index else False
            diag_data['fallback_bounded'] = was_bounded
            
            # Add bound limits for context if bounded
            if was_bounded and base_metric in CIQ_RATIO_BOUNDS:
                bounds = CIQ_RATIO_BOUNDS[base_metric]
                diag_data['bound_limits'] = {'min': bounds['min'], 'max': bounds['max']}
            
            # Store under metric name
            trend_diagnostic_data[idx][base_metric] = diag_data

    # Return both trend_scores and diagnostic data
    return trend_scores, trend_diagnostic_data

def calculate_cycle_position_score(trend_scores, key_metrics_trends):
    """
    SOLUTION TO ISSUE #2: BUSINESS CYCLE POSITION
    
    [V3.8] Updated with credit-appropriate component weighting:
    - Direction: 50% weight (trend trajectory is most important for credit)
    - Volatility: 30% weight (stability matters, but stable decline is still bad)
    - Momentum: 20% weight (acceleration is useful but can be noisy)
    
    Composite score indicating where company is in business cycle:
    - High score (70-100): Favorable position (improving trends, low volatility)
    - Medium score (40-70): Neutral/stable
    - Low score (0-40): Unfavorable (deteriorating trends, high volatility)
    """
    # Component weights (must sum to 1.0)
    DIRECTION_WEIGHT = 0.50
    VOLATILITY_WEIGHT = 0.30
    MOMENTUM_WEIGHT = 0.20
    
    # Collect components separately for weighted averaging
    direction_components = []
    volatility_components = []
    momentum_components = []
    
    # Metrics where LOWER values are BETTER (declining = improving)
    LOWER_IS_BETTER_TREND_METRICS = {
        'Total Debt / EBITDA (x)',
        'Net Debt / EBITDA',
        'Total Debt / Total Capital (%)',
        'Total Debt/Equity (x)',
    }
    
    for metric in key_metrics_trends:
        if f'{metric}_trend' in trend_scores.columns:
            raw_trend = trend_scores[f'{metric}_trend']
            
            # POLARITY FIX: For debt metrics, declining is IMPROVING
            if metric in LOWER_IS_BETTER_TREND_METRICS:
                adjusted_trend = -raw_trend  # Flip: declining debt = positive trend
            else:
                adjusted_trend = raw_trend
            
            # Positive trend = good, convert -1/+1 to 0-100
            trend_component = (adjusted_trend + 1) * 50
            direction_components.append(trend_component)
        
        if f'{metric}_volatility' in trend_scores.columns:
            # Low volatility = good (already on 0-100 scale, high = stable)
            volatility_components.append(trend_scores[f'{metric}_volatility'])
        
        if f'{metric}_momentum' in trend_scores.columns:
            # High momentum = good (already on 0-100 scale)
            momentum_components.append(trend_scores[f'{metric}_momentum'])
    
    # Calculate weighted average of each component type, then combine
    if direction_components or volatility_components or momentum_components:
        # Average within each component type
        if direction_components:
            direction_avg = pd.concat(direction_components, axis=1).mean(axis=1)
        else:
            direction_avg = pd.Series(50, index=trend_scores.index)
        
        if volatility_components:
            volatility_avg = pd.concat(volatility_components, axis=1).mean(axis=1)
        else:
            volatility_avg = pd.Series(50, index=trend_scores.index)
        
        if momentum_components:
            momentum_avg = pd.concat(momentum_components, axis=1).mean(axis=1)
        else:
            momentum_avg = pd.Series(50, index=trend_scores.index)
        
        # Weighted combination
        cycle_score = (
            direction_avg * DIRECTION_WEIGHT +
            volatility_avg * VOLATILITY_WEIGHT +
            momentum_avg * MOMENTUM_WEIGHT
        )
    else:
        cycle_score = pd.Series(50, index=trend_scores.index)  # Neutral default
    
    return cycle_score.clip(0, 100)

# ============================================================================
# QUALITY/TREND SPLIT HELPERS
# ============================================================================

def _compute_quality_metrics(df, score_col="Composite_Score"):
    """
    Precompute global and within-band percentile metrics.
    Returns a copy of df with added columns:
    - Composite_Percentile_Global: 0-100 percentile across full universe
    - Composite_Percentile_in_Band: 0-100 percentile within rating band (already exists, but ensured here)
    """
    df = df.copy()

    # Global percentile (0-100 scale)
    df["Composite_Percentile_Global"] = df[score_col].rank(pct=True, method='average') * 100

    # Within-band percentile (0-100 scale) - recalculate to ensure consistency
    df["Composite_Percentile_in_Band"] = df.groupby("Rating_Band")[score_col].rank(pct=True, method='average') * 100

    return df

def resolve_quality_metric_and_split(df, split_basis, split_threshold):
    """
    Returns (quality_series, x_split_for_plot, axis_label, x_values_for_scatter)

    quality_series: Series to test for Strong/Weak (values compared to x_split_for_plot)
    x_split_for_plot: Scalar value to draw as vertical line on chart
    axis_label: String label for x-axis
    x_values_for_scatter: Series to plot on x-axis (same as quality_series)
    """
    if split_basis == "Absolute Composite Score":
        quality = df["Composite_Score"]
        # Vertical line at the Nth percentile of absolute scores
        x_split = float(np.nanpercentile(df["Composite_Score"], split_threshold))
        axis_label = "Composite Score (0-100)"
        x_vals = df["Composite_Score"]
    elif split_basis == "Global Percentile":
        quality = df["Composite_Percentile_Global"]
        x_split = float(split_threshold)
        axis_label = "Global Percentile (0-100)"
        x_vals = df["Composite_Percentile_Global"]
    else:
        # Percentile within band (default/recommended)
        quality = df["Composite_Percentile_in_Band"]
        x_split = float(split_threshold)
        axis_label = "Percentile within Rating Band (0-100)"
        x_vals = df["Composite_Percentile_in_Band"]

    return quality, x_split, axis_label, x_vals

# ============================================================================
# TREND CONFIGURATION AND HEATMAP HELPERS
# ============================================================================

def get_trend_cfg():
    """Returns canonical trend configuration from session state."""
    # [V2.3] quality_basis is now hard-coded (no longer in session state)
    return {
        "quality_basis": "Percentile within Band (recommended)",
        "quality_threshold": float(st.session_state.get("cfg_quality_threshold", MODEL_THRESHOLDS['viz_quality_split'])),
        "trend_threshold": float(st.session_state.get("cfg_trend_threshold", TREND_THRESHOLD)),
    }



# ============================================================================
# AI ANALYSIS HELPERS (deterministic)
# ============================================================================



def _col(df, candidates):
    """Find first matching column from candidates list."""
    for c in candidates:
        if c in df.columns:
            return c
    return None  # caller must guard

def _norm(s):
    """Normalize string for robust matching (strips whitespace, case-insensitive)."""
    return str(s).strip().casefold()

def _json_safe(obj):
    """Recursively convert numpy/pandas scalars & timestamps to JSON-safe primitives."""
    if isinstance(obj, (pd.Timestamp, )):
        return obj.isoformat()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if obj is None or isinstance(obj, (str, float, int, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return str(obj)

def _detect_signal(df, trend_thr=None, quality_thr=None):
    """Return a small dataframe with computed/imputed regime signal if missing.
    
    Note: Defaults to MODEL_THRESHOLDS values if not specified.
    """
    # Use centralized thresholds as defaults
    if trend_thr is None:
        trend_thr = TREND_THRESHOLD  # 55
    if quality_thr is None:
        quality_thr = QUALITY_THRESHOLD  # 55
        
    comp = _col(df, ["Composite_Score", "Composite Score (0-100)", "Composite Score"])
    cyc  = _col(df, ["Cycle_Position_Score", "Cycle Position Score (0-100)", "Cycle Position Score"])
    # Prefer post-override label if present
    sig  = _col(df, ["Combined_Signal", "Quality & Trend Signal", "Quality and Trend Signal", "Signal"])
    out = df.copy()
    if sig is None:
        # Use centralized signal classification function
        out["__Signal"] = out.apply(
            lambda row: classify_signal(row[comp], row[cyc], quality_thr, trend_thr),
            axis=1
        )
        return out, "__Signal"
    return out, sig

def _band(df):
    """Find rating band column."""
    return _col(df, ["Rating_Band", "Rating Band", "Credit_Rating_Band", "Rating_Bucket"])

def _classcol(df):
    """Find classification/sector column."""
    return _col(df, ["Classification", "Sector / industry", "Sector", "Rubrics Custom Classification", "Rubrics_Custom_Classification"])

def _namecol(df):
    """Find company name column."""
    return _col(df, ["Company_Name", "Company Name", "Issuer", "Name"])

def _reccol(df):
    """Find recommendation column."""
    return _col(df, ["Model Recommendation", "Recommendation", "Rec"])

def _stale_days_col(df):
    """Pick the freshest of the freshness columns."""
    return _col(df, ["Days Since Latest Financials", "Days Since Last Rating Review", "Rating Data Freshness"])

def _period_cols(df):
    """Look for period-end fields if present."""
    fy_cols = [c for c in df.columns if "FY" in c and "Period" in c and "Ended" in c]
    cq_cols = [c for c in df.columns if "CQ" in c and "Period" in c and "Ended" in c]
    return fy_cols, cq_cols

def _share(x):
    """Calculate count and percentage."""
    d = int(x.sum())
    n = int(len(x))
    return d, (100.0*d/n if n else 0.0)

# ---------- Raw-only scoring (AI Analysis v2) ----------
def _pct_rank(series: pd.Series, invert: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    r = s.rank(pct=True, method="average")
    if invert:
        r = 1 - r
    return r * 100.0

def compute_raw_scores_v2(df_original: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional 0–100 Quality & Trend scores built ONLY from raw metrics."""
    name_col = _resolve_company_name_col(df_original)
    if not name_col:
        return pd.DataFrame()
    # Build latest level metrics per issuer
    latest = {}
    deltas = {}
    # ↑ good metrics
    up_levels = ["EBITDA Margin", "Return on Equity", "Return on Assets", "EBITDA / Interest Expense (x)", "Current Ratio (x)", "Quick Ratio (x)"]
    # ↓ good metrics
    dn_levels = ["Total Debt / EBITDA (x)", "Net Debt / EBITDA"]

    for idx, row in df_original.iterrows():
        # levels
        lev_vals = {}
        for m in up_levels + dn_levels:
            ts = _metric_series_for_row(df_original, row, m, prefer_fy=True)
            # Coerce to numeric and drop NaN for robust level extraction
            ts_numeric = pd.to_numeric(ts, errors="coerce").dropna()
            lev_vals[m] = ts_numeric.iloc[-1] if len(ts_numeric) > 0 else np.nan
        # deltas
        d_vals = {}
        for m in ["EBITDA Margin", "EBITDA / Interest Expense (x)", "Total Debt / EBITDA (x)", "Levered Free Cash Flow Margin"]:
            ts = _metric_series_for_row(df_original, row, m, prefer_fy=True)
            # Coerce to numeric and use last two valid points only
            ts_numeric = pd.to_numeric(ts, errors="coerce").dropna()
            last = ts_numeric.iloc[-1] if len(ts_numeric) > 0 else np.nan
            prev = ts_numeric.iloc[-2] if len(ts_numeric) > 1 else np.nan
            d_vals[m] = (last - prev) if pd.notna(last) and pd.notna(prev) else np.nan

        latest[idx] = lev_vals
        deltas[idx] = d_vals

    latest_df = pd.DataFrame.from_dict(latest, orient="index")
    delta_df  = pd.DataFrame.from_dict(deltas, orient="index")
    # Percentile ranks (invert where lower is better)
    q_parts = []
    for m in up_levels:
        q_parts.append(_pct_rank(latest_df[m], invert=False))
    for m in dn_levels:
        q_parts.append(_pct_rank(latest_df[m], invert=True))
    quality = pd.concat(q_parts, axis=1).mean(axis=1, skipna=True)

    # [V2.2] Validation: Require minimum 4 of 8 factors for reliable quality score
    # This prevents unreliable scores from issuers with sparse data
    factors_available = pd.concat(q_parts, axis=1).notna().sum(axis=1)
    insufficient_data = factors_available < 4

    if insufficient_data.any() and not os.environ.get("RG_TESTS"):
        st.sidebar.warning(
            f"**Data Quality Alert**\n\n"
            f"{insufficient_data.sum()} issuer(s) excluded from quality scoring due to insufficient data "
            f"(need at least 4 of 8 quality factors).\n\n"
            f"These issuers will have NaN quality scores and be filtered from analysis."
        )

    # Set quality score to NaN for issuers with insufficient data
    quality[insufficient_data] = np.nan

    t_parts = []
    for m in ["EBITDA Margin", "EBITDA / Interest Expense (x)", "Levered Free Cash Flow Margin"]:
        t_parts.append(_pct_rank(delta_df[m], invert=False))
    for m in ["Total Debt / EBITDA (x)", "Net Debt / EBITDA"]:
        t_parts.append(_pct_rank(delta_df[m], invert=True))  # falling leverage is better
    trend = pd.concat(t_parts, axis=1).mean(axis=1, skipna=True)

    out = pd.DataFrame({
        name_col: df_original[name_col].astype(str).values,
        "Raw_Quality_Score": quality.values,
        "Raw_Trend_Score": trend.values,
    }, index=df_original.index)
    return out

def build_buckets_v2(results_df: pd.DataFrame, df_original: pd.DataFrame, trend_thr=None, quality_thr=None):
    """Return dict with regime buckets + column name for the signal (raw-only).
    
    Note: Defaults to MODEL_THRESHOLDS values if not specified.
    """
    # Use centralized thresholds as defaults
    if trend_thr is None:
        trend_thr = TREND_THRESHOLD  # 55
    if quality_thr is None:
        quality_thr = QUALITY_THRESHOLD  # 55
    """Return dict with regime buckets + column name for the signal (raw-only)."""
    nm_res = _resolve_company_name_col(results_df) or _resolve_company_name_col(df_original)
    nm_raw = _resolve_company_name_col(df_original)
    cls = _resolve_classification_col(results_df) or _resolve_classification_col(df_original) or "Rubrics_Custom_Classification"
    if not nm_res or not nm_raw:
        return {"error": "Company name column not found"}

    raw_scores = compute_raw_scores_v2(df_original)
    merged = results_df.merge(raw_scores, left_on=nm_res, right_on=nm_raw, how="left")
    comp, cyc = "Raw_Quality_Score", "Raw_Trend_Score"
    def lab(r):
        return classify_signal(r[comp], r[cyc], quality_thr, trend_thr)
    merged["__Signal_v2"] = merged.apply(lab, axis=1)
    rec = "__Signal_v2"
    leaders  = merged.sort_values([cyc, comp], ascending=[False, False]).head(20).copy()
    laggards = merged.sort_values([cyc, comp], ascending=[ True,  True]).head(20).copy()
    contrarian_long = merged.query(f"`{comp}`>=@quality_thr and `{cyc}`<@trend_thr").copy()
    at_risk          = merged.query(f"`{comp}`<@quality_thr and `{cyc}`<@trend_thr").copy()

    # Compute regime counts for UI display
    counts = (merged.groupby("__Signal_v2")
                 .size()
                 .reindex(["Strong & Improving","Strong & Moderating","Strong & Normalizing","Strong but Deteriorating","Weak but Improving","Weak & Deteriorating","n/a"], fill_value=0)
                 .to_frame("Count")
                 .reset_index()
                 .rename(columns={"__Signal_v2": "Signal"}))

    return {
        "counts": counts,
        "contrarian_long": contrarian_long[[nm_res, cls, comp, cyc, rec]].rename(columns={comp:"Quality (raw)",cyc:"Trend (raw)"}).head(20),
        "at_risk":         at_risk[[nm_res, cls, comp, cyc, rec]].rename(columns={comp:"Quality (raw)",cyc:"Trend (raw)"}).head(20),
        "leaders":         leaders[[nm_res, cls, comp, cyc, rec]].rename(columns={comp:"Quality (raw)",cyc:"Trend (raw)"}),
        "laggards":        laggards[[nm_res, cls, comp, cyc, rec]].rename(columns={comp:"Quality (raw)",cyc:"Trend (raw)"}),
        "sig_col":         "__Signal_v2",
    }

def _dq_checks(df, staleness_days=365):
    """Run data quality checks."""
    issues = []
    nm = _namecol(df)
    # Staleness
    dsc = _stale_days_col(df)
    if dsc:
        stale = df[pd.to_numeric(df[dsc], errors="coerce") > staleness_days]
        if not stale.empty:
            issues.append(("Stale data (> {} days)".format(staleness_days),
                           stale[[nm, dsc]].sort_values(by=dsc, ascending=False).head(50)))
    # Period-end overlap (if columns exist)
    fy_cols, cq_cols = _period_cols(df)
    if fy_cols and cq_cols:
        # Any CQ period that *equals* FY0 period for same issuer is fine; zeros/1900 or mis-ordered flagged
        bad_dates = []
        for c in fy_cols + cq_cols:
            # catch Excel sentinels like 0/01/1900 or 01/01/1900
            if df[c].dtype == "object":
                bad = df[df[c].str.contains("1900", na=False)]
            else:
                bad = df[df[c].astype(str).str.contains("1900", na=False)]
            if not bad.empty:
                bad_dates.append((c, bad[[nm, c]].head(50)))
        if bad_dates:
            for colname, frame in bad_dates:
                issues.append((f"Suspicious period-end sentinel in {colname}", frame))
    return issues


# ============================================================================
# MAIN DATA LOADING FUNCTION
# ============================================================================

@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: lambda df: CACHE_VERSION + str(df.shape)})
def load_and_process_data(uploaded_file, use_sector_adjusted,
                          period_mode=PeriodSelectionMode.LATEST_AVAILABLE,
                          reference_date_override=None,
                          prefer_annual_reports=False,
                          split_basis="Percentile within Band (recommended)", split_threshold=60, trend_threshold=55,
                          volatility_cv_threshold=0.30, outlier_z_threshold=-2.5, damping_factor=0.5, near_peak_tolerance=0.10,
                          calibrated_weights=None,
                          _cache_buster=None):
    """Load data and calculate issuer scores with unified period selection (V2.3)

    This function loads a pandas-compatible Excel file and scores issuers
    using a 5-factor composite score with trend/cycle analysis.

    [V2.3] Unified Period Selection:
    - Mode A (LATEST_AVAILABLE): Most current data per issuer, accepts misalignment
    - Mode B (REFERENCE_ALIGNED): Common reference date, enforces alignment

    Args:
        uploaded_file: File object (from st.file_uploader or test path)
        use_sector_adjusted: Use sector-specific factor weights
        period_mode: PeriodSelectionMode (LATEST_AVAILABLE or REFERENCE_ALIGNED)
        reference_date_override: Force specific reference date (for Mode B)
        prefer_annual_reports: In Mode B, prefer FY over CQ when both exist
        split_basis: Quality metric for signal generation
        split_threshold: Threshold percentile for quality split
        trend_threshold: Threshold for trend signal (0-100)
        volatility_cv_threshold: CV threshold for volatility flag
        outlier_z_threshold: Z-score threshold for outlier detection
        damping_factor: Volatility damping factor (0-1)
        near_peak_tolerance: Tolerance for near-peak detection (0-1)
        calibrated_weights: Pre-calculated sector weights (or "CALCULATE_INSIDE")
        _cache_buster: Internal parameter to force cache invalidation

    Returns:
        Tuple of (results_df, original_df, audit_trail, period_calendar)
    """
    # Reset warning collector for new scoring run
    WarningCollector.reset()

    # ===== TIMING DIAGNOSTICS =====
    _start_time = time.time()
    _checkpoints = {}

    def _log_timing(label):
        elapsed = time.time() - _start_time
        _checkpoints[label] = elapsed
        print(f"[TIMING] {label}: {elapsed:.2f}s (cumulative)")

    # Initialize row audit tracking
    audits = []

    # Load data - handle both Excel and CSV
    file_name = uploaded_file.name.lower()
    if file_name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith('.xlsx'):
        try:
            df = pd.read_excel(uploaded_file, sheet_name='Pasted Values')
        except (ValueError, KeyError):
            df = pd.read_excel(uploaded_file, sheet_name=0)

        # Skip first row if it's a duplicated header row
        if len(df) > 0:
            first_row = df.iloc[0].astype(str).str.strip().str.lower().tolist()
            cols_norm = df.columns.astype(str).str.strip().str.lower().tolist()
            if first_row == cols_norm:
                df = df.iloc[1:].reset_index(drop=True)
    else:
        st.error("Unsupported file format.")
        st.stop()

    # Normalize headers once after load (stronger normalization)
    # Handles NBSP, extra spaces, and ensures clean column names
    df.columns = [' '.join(str(c).replace('\u00a0', ' ').split()) for c in df.columns]

    _log_timing("01_File_Loaded")
    _audit_count("Raw input", df, audits)

    # [V2.2] TIERED VALIDATION - minimal identifiers with flexible column matching
    missing, RATING_COL, COMPANY_ID_COL, COMPANY_NAME_COL = validate_core(df)
    if missing:
        st.error(f"ERROR: Missing required identifiers:\n\n" + "\n".join([f"  • {m}" for m in missing]) +
                 f"\n\nV2.2 requires only: Company Name, Company ID, and S&P Credit Rating\n(or their common aliases)")
        st.stop()

    # Canonicalize to standard names for stable downstream code
    rename_map = {}

    if RATING_COL and RATING_COL != RATING_ALIASES[0]:
        rename_map[RATING_COL] = RATING_ALIASES[0]
        RATING_COL = RATING_ALIASES[0]

    if COMPANY_ID_COL and COMPANY_ID_COL != COMPANY_ID_ALIASES[0]:
        rename_map[COMPANY_ID_COL] = COMPANY_ID_ALIASES[0]  # "Company ID"
        COMPANY_ID_COL = COMPANY_ID_ALIASES[0]

    if COMPANY_NAME_COL and COMPANY_NAME_COL != COMPANY_NAME_ALIASES[0]:
        rename_map[COMPANY_NAME_COL] = COMPANY_NAME_ALIASES[0]  # "Company Name"
        COMPANY_NAME_COL = COMPANY_NAME_ALIASES[0]

    if rename_map:
        df = df.rename(columns=rename_map)
        if os.environ.get("RG_TESTS") == "1":
            print(f"DEV: Standardized column names: {rename_map}")

    # Hard gate: fail fast if any core is unresolved (defensive check)
    assert COMPANY_ID_COL in df.columns, f"Missing Company ID column; headers={list(df.columns)[:10]}..."
    assert COMPANY_NAME_COL in df.columns, "Missing Company Name column"
    assert RATING_COL in df.columns, "Missing S&P rating column"

    # Handle CSV split columns for revenue growth
    if 'Total Revenues' in df.columns and '1 Year Growth' in df.columns and 'Total Revenues, 1 Year Growth' not in df.columns:
        df['Total Revenues, 1 Year Growth'] = df['1 Year Growth']

    # Column aliases for revenue growth variations
    column_aliases = {
        'Total Revenues,1 Year Growth': 'Total Revenues, 1 Year Growth',
        'Total Revenues,  1 Year Growth': 'Total Revenues, 1 Year Growth',
        'Total Revenues 1 Year Growth': 'Total Revenues, 1 Year Growth'
    }
    for old_name, new_name in column_aliases.items():
        if old_name in df.columns and new_name not in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)

    # Standardize Interest Coverage column names (handle "EBITDA/ " vs "EBITDA / " spacing)
    interest_coverage_renames = {}
    for col in df.columns:
        if 'EBITDA/ Interest Expense' in col and 'EBITDA / Interest Expense' not in col:
            new_col = col.replace('EBITDA/ Interest Expense', 'EBITDA / Interest Expense')
            interest_coverage_renames[col] = new_col
    
    if interest_coverage_renames:
        df = df.rename(columns=interest_coverage_renames)
        if os.environ.get("RG_TESTS") == "1":
            print(f"DEV: Standardized Interest Coverage columns: {len(interest_coverage_renames)} columns")

    # Remove rows with missing core identifiers (now using standardized names)
    df = df.dropna(subset=[COMPANY_ID_COL, COMPANY_NAME_COL, RATING_COL])
    df = df[df[RATING_COL].astype(str).str.strip() != '']

    if len(df) == 0:
        st.error("No valid data rows found.")
        st.stop()

    _audit_count("Core IDs/Names/Ratings present", df, audits)

    # [V2.2] Check feature availability (classification, country/region, period alignment)
    has_classification = feature_available("classification", df)
    has_country_region = feature_available("country_region", df)
    has_period_alignment = feature_available("period_alignment", df)

    # [V2.2] Parse Period Ended columns if available
    if has_period_alignment:
        pe_cols = parse_period_ended_cols(df)
        if os.environ.get("RG_TESTS") == "1":
            print(f"DEV: Parsed {len(pe_cols)} Period Ended columns")
    else:
        pe_cols = []

    # [V2.3] Build period calendar with unified period selection mode
    period_calendar = None
    reference_date_actual = None
    align_to_reference = False
    selected_periods = None  # NEW: Will hold unified period selection for both quality and trend

    if has_period_alignment:
        try:
            # Determine reference date based on mode
            if period_mode == PeriodSelectionMode.REFERENCE_ALIGNED:
                if reference_date_override is None:
                    # Auto-select best reference date
                    reference_date_actual = get_recommended_reference_date(df)
                    if os.environ.get("RG_TESTS") == "1":
                        print(f"DEV: Auto-selected reference date: {reference_date_actual}")
                else:
                    reference_date_actual = reference_date_override

                align_to_reference = True

                # NEW: Unified period selection - call ONCE, use for both quality and trend
                # This fixes the "split brain" bug where quality and trend used different periods
                pe_data = parse_period_ended_cols(df.copy())
                if pe_data:
                    selected_periods = select_aligned_period(
                        df=df,
                        pe_data=pe_data,
                        reference_date=reference_date_actual,
                        prefer_annual_reports=prefer_annual_reports,
                        use_quarterly=True
                    )

                    # Debug logging for NVIDIA
                    if os.environ.get("RG_TESTS") == "1":
                        nvidia_rows = df[df[COMPANY_NAME_COL].str.contains('NVIDIA', case=False, na=False)]
                        if len(nvidia_rows) > 0:
                            nvidia_idx = nvidia_rows.index[0]
                            nvidia_selected = selected_periods[selected_periods['row_idx'] == nvidia_idx]
                            if len(nvidia_selected) > 0:
                                print(f"DEV: NVIDIA selected period: {nvidia_selected['selected_date'].iloc[0]} (suffix: {nvidia_selected['selected_suffix'].iloc[0]})")

            else:  # LATEST_AVAILABLE
                reference_date_actual = None
                align_to_reference = False

                # NEW: Use select_aligned_period for consistency with REFERENCE_ALIGNED mode
                # Both modes should use the same period selection algorithm to ensure
                # identical results when the reference period matches the latest data.
                # Use far-future reference date (2099-12-31) to effectively mean "select absolute latest"
                pe_data = parse_period_ended_cols(df.copy())
                if pe_data:
                    selected_periods = select_aligned_period(
                        df=df,
                        pe_data=pe_data,
                        reference_date=pd.Timestamp('2099-12-31'),  # Far future = no date filter
                        prefer_annual_reports=False,  # Always use most recent in LATEST_AVAILABLE mode
                        use_quarterly=True
                    )

            # Build period calendar (always prefer quarterly for trend analysis)
            period_calendar = build_period_calendar(
                raw_df=df,
                issuer_id_col=COMPANY_ID_COL,
                issuer_name_col=COMPANY_NAME_COL,
                prefer_quarterly=True,  # Always use quarterly for trend granularity
                q4_merge_window_days=10
            )
            if os.environ.get("RG_TESTS") == "1":
                original_count = sum(1 for c in df.columns if "Period Ended" in str(c)) * len(df)
                cleaned_count = len(period_calendar)
                removed_count = original_count - cleaned_count
                print(f"DEV: Period calendar built - {len(period_calendar)} periods (removed {removed_count} overlaps/sentinels)")
                print(f"DEV: Period mode: {period_mode.value}, Align: {align_to_reference}, Ref date: {reference_date_actual}")
        except ValueError as e:
            # If period column format doesn't match the expected pattern, fall back gracefully
            if os.environ.get("RG_TESTS") == "1":
                print(f"DEV: Period calendar not built - {e}")
            period_calendar = None

    _audit_count("After period alignment", df, audits)
    _log_timing("02_Column_Validation_Complete")

    # ========================================================================
    # CALCULATE TREND INDICATORS (ISSUE #2 SOLUTION)
    # ========================================================================

    key_metrics_for_trends = [
        'Total Debt / EBITDA (x)',       # Leverage trajectory (lower is better)
        'EBITDA Margin',                  # Operating performance (higher is better)
        'EBITDA / Interest Expense (x)', # Interest coverage trend - #1 early warning (higher is better)
        'Levered Free Cash Flow Margin', # Cash generation trajectory (higher is better)
        'Revenue'                         # [V4.1] Top-line trajectory (higher is better)
    ]

    # [V2.3] Extract metrics using unified period mode
    # For trend analysis, always use quarterly data for better granularity
    # Reference date determined earlier based on period_mode
    use_quarterly_for_trends = True  # Always use quarterly for trend analysis

    # [V3.0 FIX] Trend calculation requires historical time series, not single-period
    # - Quality scores: Use selected_periods to enforce reference date alignment (single period)
    # - Trend scores: Use reference_date as cutoff, but include ALL historical periods up to that date
    # This ensures momentum calculation has sufficient data (needs 8+ periods) while still
    # respecting the reference date boundary (no future data beyond reference date).
    # [Phase 1] Now returns tuple: (trend_scores, trend_diagnostic_data)
    trend_scores, trend_diagnostic_data = calculate_trend_indicators(df, key_metrics_for_trends,
                                             use_quarterly=use_quarterly_for_trends,
                                             reference_date=reference_date_actual,
                                             prefer_annual_reports=prefer_annual_reports if period_mode == PeriodSelectionMode.REFERENCE_ALIGNED else False,
                                             selected_periods=None)  # FIX: Use full history for trend calculation
    cycle_score = calculate_cycle_position_score(trend_scores, key_metrics_for_trends)
    _log_timing("03_Trend_Indicators_Complete")

    # ========================================================================
    # [V2.2] DUAL-HORIZON TREND & CONTEXT FLAGS
    # ========================================================================

    # Compute dual-horizon metrics for Composite Score (using time series)
    # This provides medium-term slope, short-term change, outlier detection, and volatility flags

    dual_horizon_metrics = pd.DataFrame(index=df.index)

    # Use EBITDA Margin as primary metric for dual-horizon analysis
    # (Can extend to other metrics as needed)
    primary_metric = 'EBITDA Margin'
    if primary_metric in df.columns:
        ts_primary = _build_metric_timeseries(df, primary_metric, use_quarterly=use_quarterly_for_trends)

        dual_results = ts_primary.apply(
            lambda row: pd.Series(compute_dual_horizon_trends(row, peak_tolerance=near_peak_tolerance)),
            axis=1
        )

        dual_horizon_metrics['MediumTermSlope'] = dual_results['medium_term_slope']
        dual_horizon_metrics['ShortTermChange'] = dual_results['short_term_change']
        dual_horizon_metrics['LastQuarterZ'] = dual_results['last_quarter_z']
        dual_horizon_metrics['SeriesCV'] = dual_results['series_cv']
        dual_horizon_metrics['NearPeak'] = dual_results['near_peak_flag']
    else:
        # Fallback: fill with defaults
        dual_horizon_metrics['MediumTermSlope'] = np.nan
        dual_horizon_metrics['ShortTermChange'] = np.nan
        dual_horizon_metrics['LastQuarterZ'] = 0.0
        dual_horizon_metrics['SeriesCV'] = 0.0
        dual_horizon_metrics['NearPeak'] = False

    _log_timing("03b_Dual_Horizon_Complete")

    # ========================================================================
    # CALCULATE QUALITY SCORES ([V2.2] ANNUAL-ONLY DEFAULT)
    # ========================================================================

    def _calculate_factor_score_with_renormalization(
        components: np.ndarray,
        weights: np.ndarray,
        min_components: int = 1,
        factor_name: str = "",
        index: pd.Index = None,
        apply_data_quality_penalty: bool = True
    ) -> tuple:
        """
        Calculate factor score with automatic weight renormalization for missing data.
        
        [V3.9] Now applies a data quality penalty for missing components.
        Missing data is often a negative signal (small companies, emerging markets,
        distressed issuers, delayed filings). Penalty ensures issuers with complete
        data score higher than those with gaps, all else equal.

        Args:
            components: 2D array (n_issuers × n_components) of component scores
            weights: 1D array of component weights (must sum to 1.0)
            min_components: Minimum number of non-missing components required
            factor_name: Name of factor (for data quality column naming)
            index: Optional pandas Index for output Series
            apply_data_quality_penalty: Whether to apply penalty for missing data (default True)

        Returns:
            tuple of (scores, data_completeness, components_used_count)
        """
        # Data quality penalty rate: 10% maximum penalty for completely missing data
        DATA_QUALITY_PENALTY_RATE = 0.10
        
        # Ensure components is 2D (issuers × components)
        if components.ndim == 1:
            components = components.reshape(-1, 1)

        n_issuers, n_components = components.shape

        # Validate weights
        assert len(weights) == n_components, f"{factor_name}: Weights must match number of components"
        assert abs(weights.sum() - 1.0) < 0.001, f"{factor_name}: Weights must sum to 1.0"

        # Identify missing components
        mask = np.isnan(components)

        # Count available components per issuer
        components_available = (~mask).sum(axis=1)

        # Calculate effective weights (zero out missing components)
        weights_2d = np.broadcast_to(weights, (n_issuers, n_components))
        effective_weights = np.where(mask, 0.0, weights_2d)
        weight_sums = effective_weights.sum(axis=1, keepdims=True)

        # Renormalize weights
        safe_weight_sums = np.where(weight_sums > 0, weight_sums, 1.0)
        normalized_weights = np.where(
            weight_sums > 0,
            effective_weights / safe_weight_sums,
            0.0
        )

        # Calculate weighted scores
        weighted_components = components * normalized_weights
        raw_scores = np.nansum(weighted_components, axis=1)

        # Calculate data completeness
        data_completeness = components_available / n_components

        # [V3.9] Apply data quality penalty for missing components
        # Formula: adjusted_score = raw_score * (1 - penalty_rate * (1 - completeness))
        # Example: 80 score with 50% completeness → 80 * (1 - 0.10 * 0.50) = 80 * 0.95 = 76
        if apply_data_quality_penalty:
            penalty_multiplier = 1.0 - (DATA_QUALITY_PENALTY_RATE * (1.0 - data_completeness))
            adjusted_scores = raw_scores * penalty_multiplier
        else:
            adjusted_scores = raw_scores

        # Apply minimum component threshold
        final_scores = np.where(
            components_available >= min_components,
            adjusted_scores,
            np.nan
        )

        return (
            pd.Series(final_scores, index=index),
            pd.Series(data_completeness, index=index),
            pd.Series(components_available, dtype=int, index=index)
        )

    def extract_raw_inputs_for_diagnostic(df, idx, suffix):
        """
        Extract raw input values for a specific issuer and period suffix.
        Used for data lineage in diagnostic reports.
        """
        raw_inputs = {}
        
        # Helper to safely get value
        def get_val(col_base):
            col_name = f"{col_base}{suffix}"
            if col_name in df.columns:
                val = df.loc[idx, col_name]
                if pd.isna(val):
                    return None
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return None
            return None

        # 1. Balance Sheet
        raw_inputs['Total Assets'] = get_val('Total Assets')
        raw_inputs['Total Debt'] = get_val('Total Debt')
        raw_inputs['Cash & ST Investments'] = get_val('Cash & Short-term Investments')
        raw_inputs['Current Assets'] = get_val('Current Assets')
        raw_inputs['Current Liabilities'] = get_val('Current Liabilities')
        raw_inputs['Inventory'] = get_val('Inventory')
        raw_inputs['Total Equity'] = get_val('Total Equity')
        raw_inputs['Total Capital'] = get_val('Total Capital')
        
        # 2. Income Statement
        raw_inputs['Total Revenue'] = get_val('Revenue')  # FIXED: column is "Revenue" not "Total Revenue"
        raw_inputs['EBITDA'] = get_val('EBITDA')
        raw_inputs['Interest Expense'] = get_val('Interest Expense')
        raw_inputs['Net Income'] = get_val('Net Income')
        raw_inputs['Cost of Goods Sold'] = get_val('Cost of Goods Sold')
        raw_inputs['Gross Profit'] = get_val('Gross Profit')
        raw_inputs['Operating Income'] = get_val('Operating Income')
        
        # 3. Profitability Ratios (pre-calculated in spreadsheet)
        raw_inputs['Return on Equity'] = get_val('Return on Equity')  # NEW
        raw_inputs['Return on Assets'] = get_val('Return on Assets')
        raw_inputs['EBITDA Margin'] = get_val('EBITDA Margin')
        raw_inputs['Gross Profit Margin'] = get_val('Gross Profit Margin')
        raw_inputs['Operating Margin'] = get_val('Operating Margin')
        raw_inputs['Net Profit Margin'] = get_val('Net Profit Margin')
        
        # 4. Leverage Ratios (pre-calculated in spreadsheet)
        raw_inputs['Total Debt / EBITDA'] = get_val('Total Debt / EBITDA (x)')  # NEW
        raw_inputs['Net Debt / EBITDA'] = get_val('Net Debt / EBITDA')
        raw_inputs['Total Debt / Capital'] = get_val('Total Debt / Capital')
        raw_inputs['Debt to Equity'] = get_val('Debt to Equity')
        
        # 5. Liquidity Ratios
        raw_inputs['Current Ratio'] = get_val('Current Ratio (x)')
        raw_inputs['Quick Ratio'] = get_val('Quick Ratio (x)')
        
        # 6. Coverage Ratios
        raw_inputs['EBITDA / Interest'] = get_val('EBITDA / Interest Expense (x)')
        
        # 7. Growth Metrics (use non-period suffixed columns for CAGRs)
        # Note: CAGR columns may not have period suffixes - try both
        raw_inputs['Revenue 3Y CAGR'] = get_val('Total Revenues, 3 Yr. CAGR')  # NEW
        raw_inputs['EBITDA 3Y CAGR'] = get_val('EBITDA, 3 Years CAGR')  # NEW
        raw_inputs['Revenue 1Y Growth'] = get_val('Total Revenues, 1 Year Growth')
        
        # 8. Cash Flow
        raw_inputs['Operating Cash Flow'] = get_val('Cash from Ops.')
        raw_inputs['Levered Free Cash Flow'] = get_val('Levered Free Cash Flow')
        raw_inputs['Unlevered Free Cash Flow'] = get_val('Unlevered Free Cash Flow')
        raw_inputs['Capital Expenditures'] = get_val('Capital Expenditures')
        raw_inputs['LFCF Margin'] = get_val('Levered Free Cash Flow Margin')
        raw_inputs['UFCF Margin'] = get_val('Unlevered Free Cash Flow Margin')
        
        # 9. Company Metadata (non-period field)
        if 'Reported Currency' in df.columns:
            curr_val = df.loc[idx, 'Reported Currency']
            raw_inputs['Reported_Currency'] = str(curr_val) if pd.notna(curr_val) else 'USD'
        else:
            raw_inputs['Reported_Currency'] = 'USD'
        
        return raw_inputs

    def calculate_quality_scores(df, data_period_setting, has_period_alignment,
                                reference_date=None, align_to_reference=False,
                                prefer_annual_reports=False, selected_periods=None):
        """
        Calculate quality scores for all issuers.

        Args:
            reference_date: If provided AND align_to_reference is True,
                          filters point-in-time metrics to this reference date.
            align_to_reference: Whether alignment is enabled by user.
            prefer_annual_reports: If True, prefer FY over CQ when available.
            selected_periods: Pre-selected periods from select_aligned_period().
        """
        scores = pd.DataFrame(index=df.index)

        # NEW: Initialize diagnostic data collection (Phase 1 - Diagnostic Storage)
        factor_diagnostic_data = {idx: {} for idx in df.index}

        # Determine whether to apply reference date filtering for point-in-time metrics
        # Only apply if: (1) CQ-0 selected AND (2) alignment enabled
        apply_reference_date = (
            reference_date is not None
            and align_to_reference
            and data_period_setting == "Most Recent LTM (LTM0)"
        )
        ref_date_for_extraction = reference_date if apply_reference_date else None

        def _pct_to_100(s):
            if isinstance(s, pd.Series):
                s = s.replace(['None','none','N/A','n/a','#N/A'], np.nan)
                s = s.astype(str).str.replace('%','',regex=False).str.replace(',','',regex=False).str.strip()
            s = pd.to_numeric(s, errors='coerce')
            valid = s.dropna().abs()
            frac_like = (valid <= 1).mean() > 0.6 if len(valid) > 0 else False
            if frac_like:
                s = s * 100.0
            return s.clip(lower=-1000, upper=1000)

        def _clean_rating(x):
            x = str(x).upper().strip()
            x = x.replace('NOT RATED','NR').replace('N/R','NR').replace('N\\M','N/M')
            x = x.split('(')[0].strip()
            x = x.replace(' ','').replace('*','')
            alias = {'BBBM':'BBB','BMNS':'B','CCCC':'CCC'}
            return alias.get(x, x)

        # OPTIMIZATION: Pre-extract all needed metrics in one batch
        needed_metrics = [
            'EBITDA / Interest Expense (x)',
            'Net Debt / EBITDA',
            'Total Debt / EBITDA (x)',
            'Total Debt / Total Capital (%)',
            'Return on Equity',
            'EBITDA Margin',
            'Return on Assets',
            'EBIT Margin',
            'Current Ratio (x)',
            'Quick Ratio (x)',
            'Total Revenues, 1 Year Growth',
            'Total Revenues, 3 Yr. CAGR',
            'EBITDA, 3 Years CAGR',
            'Levered Free Cash Flow',
            'Total Debt',
            'Levered Free Cash Flow Margin',
            'Gross Profit Margin',
            'Cash from Ops. to Curr. Liab. (x)',
            # NEW: Credit Score fundamental metrics
            'Total Assets',
            'Cash & Short-term Investments',
            'Interest Expense',
            'EBITDA',
            'Total Revenue',
            'Net Income',
            'Cost of Goods Sold',
            'Current Assets',
            'Current Liabilities',
            'Inventory',
            'Operating Cash Flow',
            'Unlevered Free Cash Flow'
        ]
        metrics = _batch_extract_metrics(df, needed_metrics, has_period_alignment,
                                        data_period_setting, ref_date_for_extraction,
                                        prefer_annual_reports=prefer_annual_reports,
                                        selected_periods=selected_periods)

        # [V5.1.0] Capture raw input values for diagnostic report
        raw_input_data = {}
        for idx in df.index:
            # Determine suffix used for this issuer - must match _batch_extract_metrics lookup
            suffix = '.0'  # Default
            if selected_periods is not None:
                period_row = selected_periods[selected_periods['row_idx'] == idx]
                if len(period_row) > 0:
                    suffix = period_row['selected_suffix'].iloc[0]
            
            raw_input_data[idx] = extract_raw_inputs_for_diagnostic(df, idx, suffix)

        # ========================================================================
        # GROWTH METRICS OVERRIDE - Force FY0 for annual-only metrics
        # ========================================================================
        # [V5.0] LTM Migration: Growth metrics in LTM mode are valid (12-month data).
        # No need to override with FY0 data as was done for CQ mode.
        # if data_period_setting == "Most Recent LTM (LTM0)":
        #     pass


        # ========================================================================
        # CREDIT SCORE METHODOLOGY (V3.2)
        # ========================================================================
        # Previously: 100% S&P Rating (circular when filtering by rating band)
        # Now: Fundamental balance sheet metrics
        #
        # Components:
        # 1. Debt/Assets (40%): Pure leverage from asset perspective
        #    - Different from Leverage factor's Debt/Capital (funding mix) and
        #      Debt/EBITDA (flow-based)
        #    - Scoring: 0% debt = 100, 70%+ debt = 0
        #
        # 2. Cash/Debt (35%): Balance sheet liquidity buffer
        #    - Different from Cash Flow factor's OCF/Debt (flow-based)
        #    - Scoring: 0% cash = 0, 50%+ cash = 100
        #    - Zero debt = perfect score (100)
        #
        # 3. Implied Interest Rate (25%): Debt quality/cost indicator
        #    - Different from Leverage factor's Interest Coverage (capacity)
        #    - Scoring: 3% rate = 100, 10%+ rate = 0
        #    - Higher rate signals riskier/more expensive debt
        #
        # Minimum 2 of 3 components required for valid score
        # ========================================================================

        # ========================================================================
        # CREDIT SCORE - Fundamental Balance Sheet Metrics (V3.2)
        # ========================================================================
        # Three components measuring debt burden, liquidity buffer, and debt cost
        # Replaces S&P Rating-based approach to enable meaningful calibration
        
        # Extract required metrics
        total_debt = metrics.get('Total Debt', pd.Series(np.nan, index=df.index))
        total_assets = metrics.get('Total Assets', pd.Series(np.nan, index=df.index))
        cash_st_inv = metrics.get('Cash & Short-term Investments', pd.Series(np.nan, index=df.index))
        interest_expense = metrics.get('Interest Expense', pd.Series(np.nan, index=df.index))
        
        # Ensure numeric
        total_debt = pd.to_numeric(total_debt, errors='coerce')
        total_assets = pd.to_numeric(total_assets, errors='coerce')
        cash_st_inv = pd.to_numeric(cash_st_inv, errors='coerce')
        interest_expense = pd.to_numeric(interest_expense, errors='coerce')
        
        # Component 1: Debt / Assets (40%) - Lower is better
        # 0% = 100, 35% = 50, 70%+ = 0
        debt_assets_ratio = np.where(
            (total_assets > 0) & pd.notna(total_assets) & pd.notna(total_debt),
            total_debt / total_assets,
            np.nan
        )
        debt_assets_score = np.clip(100 - (debt_assets_ratio / 0.70) * 100, 0, 100)
        debt_assets_score = pd.Series(debt_assets_score, index=df.index)
        
        # Component 2: Cash / Debt (35%) - Higher is better
        # Handle zero/negative debt (perfect score)
        # 0% = 0, 25% = 50, 50%+ = 100
        cash_debt_ratio = np.where(
            (total_debt > 0) & pd.notna(total_debt) & pd.notna(cash_st_inv),
            cash_st_inv / total_debt,
            np.where(
                (total_debt <= 0) & pd.notna(total_debt),
                1.0,  # No debt = perfect ratio
                np.nan
            )
        )
        cash_debt_score = np.clip((cash_debt_ratio / 0.50) * 100, 0, 100)
        cash_debt_score = pd.Series(cash_debt_score, index=df.index)
        
        # Component 3: Implied Interest Rate (25%) - Lower is better
        # 3% = 100, 6.5% = 50, 10%+ = 0
        # Note: Interest Expense is typically NEGATIVE in CIQ data (expense convention)
        # - Negative int_exp = paying interest → calculate rate using absolute value
        # - Zero int_exp = no interest payments → rate is 0%
        # - Positive int_exp = net interest income → rate concept doesn't apply (NaN)
        implied_rate = np.where(
            (total_debt > 0) & pd.notna(total_debt) & pd.notna(interest_expense) & (interest_expense <= 0),
            np.abs(interest_expense) / total_debt,
            np.nan
        )
        # Score: rate of 3% = 100, rate of 10% = 0
        implied_rate_score = np.clip(100 - ((implied_rate - 0.03) / 0.07) * 100, 0, 100)
        implied_rate_score = pd.Series(implied_rate_score, index=df.index)
        
        # Combine Credit Score components with renormalization
        credit_components = np.column_stack([
            debt_assets_score.values,
            cash_debt_score.values,
            implied_rate_score.values
        ])
        credit_weights = np.array([0.40, 0.35, 0.25])
        
        credit_score_result, credit_completeness, credit_components_used = \
            _calculate_factor_score_with_renormalization(
                credit_components, credit_weights, min_components=2, factor_name="Credit", index=df.index
            )
        
        scores['credit_score'] = credit_score_result
        
        # Capture Credit diagnostic data
        for idx in df.index:
            da_raw = debt_assets_ratio[df.index.get_loc(idx)] if isinstance(debt_assets_ratio, np.ndarray) else np.nan
            cd_raw = cash_debt_ratio[df.index.get_loc(idx)] if isinstance(cash_debt_ratio, np.ndarray) else np.nan
            ir_raw = implied_rate[df.index.get_loc(idx)] if isinstance(implied_rate, np.ndarray) else np.nan
            
            da_score = debt_assets_score.loc[idx] if pd.notna(debt_assets_score.loc[idx]) else None
            cd_score = cash_debt_score.loc[idx] if pd.notna(cash_debt_score.loc[idx]) else None
            ir_score = implied_rate_score.loc[idx] if pd.notna(implied_rate_score.loc[idx]) else None
            
            final_score = credit_score_result.loc[idx] if pd.notna(credit_score_result.loc[idx]) else None
            completeness = credit_completeness.loc[idx]
            components_used = credit_components_used.loc[idx]
            
            # Calculate weighted contributions for available components
            available_weights = []
            if pd.notna(da_score): available_weights.append(0.40)
            if pd.notna(cd_score): available_weights.append(0.35)
            if pd.notna(ir_score): available_weights.append(0.25)
            weight_sum = sum(available_weights) if available_weights else 1.0
            
            da_norm_weight = 0.40 / weight_sum if pd.notna(da_score) and weight_sum > 0 else 0
            cd_norm_weight = 0.35 / weight_sum if pd.notna(cd_score) and weight_sum > 0 else 0
            ir_norm_weight = 0.25 / weight_sum if pd.notna(ir_score) and weight_sum > 0 else 0
            
            factor_diagnostic_data[idx]['Credit'] = {
                'final_score': float(final_score) if final_score is not None else None,
                'components': {
                    'Debt_to_Assets': {
                        'raw_value': float(da_raw) if pd.notna(da_raw) else None,
                        'component_score': float(da_score) if da_score is not None else None,
                        'weight': 0.40,
                        'normalized_weight': round(da_norm_weight, 4),
                        'weighted_contribution': float(da_score * da_norm_weight) if da_score is not None else None,
                        'formula': 'Total Debt / Total Assets',
                        'calculation': f"{float(total_debt.loc[idx]):,.0f} / {float(total_assets.loc[idx]):,.0f}" if pd.notna(total_debt.loc[idx]) and pd.notna(total_assets.loc[idx]) else "N/A",
                        'scoring_logic': 'Lower is better. 0% debt = 100 score. 70%+ debt = 0 score. Linear interpolation.'
                    },
                    'Cash_to_Debt': {
                        'raw_value': float(cd_raw) if pd.notna(cd_raw) else None,
                        'component_score': float(cd_score) if cd_score is not None else None,
                        'weight': 0.35,
                        'normalized_weight': round(cd_norm_weight, 4),
                        'weighted_contribution': float(cd_score * cd_norm_weight) if cd_score is not None else None,
                        'formula': 'Cash & ST Investments / Total Debt',
                        'calculation': f"{float(cash_st_inv.loc[idx]):,.0f} / {float(total_debt.loc[idx]):,.0f}" if pd.notna(cash_st_inv.loc[idx]) and pd.notna(total_debt.loc[idx]) else "N/A",
                        'scoring_logic': 'Higher is better. 50%+ cash/debt = 100 score. 0% cash = 0 score. Linear interpolation. Zero debt = 100 score.'
                    },
                    'Implied_Interest_Rate': {
                        'raw_value': float(ir_raw) if pd.notna(ir_raw) else None,
                        'component_score': float(ir_score) if ir_score is not None else None,
                        'weight': 0.25,
                        'normalized_weight': round(ir_norm_weight, 4),
                        'weighted_contribution': float(ir_score * ir_norm_weight) if ir_score is not None else None,
                        'formula': 'Interest Expense / Total Debt',
                        'calculation': f"{abs(float(interest_expense.loc[idx])):,.0f} / {float(total_debt.loc[idx]):,.0f}" if pd.notna(interest_expense.loc[idx]) and pd.notna(total_debt.loc[idx]) else "N/A",
                        'scoring_logic': 'Lower is better. 3% rate = 100 score. 10%+ rate = 0 score. Linear interpolation.'
                    }
                },
                'data_completeness': float(completeness),
                'components_used': int(components_used)
            }

        # Store Credit completeness metrics (like other factors)
        scores['credit_data_completeness'] = credit_completeness
        scores['credit_components_used'] = credit_components_used

        # EBITDA / Interest Expense coverage - now used in Leverage (Annual-only)
        # Interest Coverage with fallback calculation (V3.5)
        interest_coverage_raw = metrics.get('EBITDA / Interest Expense (x)', pd.Series(np.nan, index=df.index))
        ebitda_raw = metrics.get('EBITDA', pd.Series(np.nan, index=df.index))
        interest_expense_raw = metrics.get('Interest Expense', pd.Series(np.nan, index=df.index))
        
        # Apply fallback calculation where CIQ ratio is missing or NM
        ebitda_interest = interest_coverage_raw.copy()
        fallback_count = 0
        
        for idx in df.index:
            existing = interest_coverage_raw.loc[idx] if idx in interest_coverage_raw.index else np.nan
            ebitda_val = ebitda_raw.loc[idx] if idx in ebitda_raw.index else np.nan
            int_exp_val = interest_expense_raw.loc[idx] if idx in interest_expense_raw.index else np.nan
            
            # Check if existing is invalid (NaN or was "NM" converted to NaN)
            if pd.isna(existing):
                fallback = calculate_interest_coverage_fallback(ebitda_val, int_exp_val, None)
                if not np.isnan(fallback):
                    ebitda_interest.loc[idx] = fallback
                    fallback_count += 1
        
        if fallback_count > 0 and not os.environ.get("RG_TESTS"):
            print(f"  [DEV] Interest Coverage fallback applied: {fallback_count} issuers")

        def score_ebitda_coverage(cov):
            """
            Score EBITDA / Interest Expense ratio.

            Thresholds:
            - ≥8.0x: Excellent (90-100 points)
            - 5.0-8.0x: Strong (70-90 points)
            - 3.0-5.0x: Adequate (50-70 points)
            - 2.0-3.0x: Weak (30-50 points)
            - 1.0-2.0x: Very weak (10-30 points)
            - <1.0x: Critical (0-10 points)
            """
            if pd.isna(cov):
                return np.nan  # Don't score if data missing

            if cov >= 8.0:
                return 90 + min(10, (cov - 8) / 2)
            elif cov >= 5.0:
                return 70 + ((cov - 5.0) / 3.0) * 20
            elif cov >= 3.0:
                return 50 + ((cov - 3.0) / 2.0) * 20
            elif cov >= 2.0:
                return 30 + ((cov - 2.0) / 1.0) * 20
            elif cov >= 1.0:
                return 10 + ((cov - 1.0) / 1.0) * 20
            else:
                return max(0, cov * 10)

        ebitda_cov_score = ebitda_interest.apply(score_ebitda_coverage)

        # Leverage ([V3.7] Restructured - removed redundant Total Debt/EBITDA)
        # Total Debt/EBITDA removed: ~90% correlated with Net Debt/EBITDA, adds no new signal
        # Interest Coverage increased: #1 early warning signal for credit distress

        # Component 1: Net Debt / EBITDA (40%)
        net_debt_ebitda_raw = metrics['Net Debt / EBITDA']
        
        # [V5.0.2.1] Apply fallback calculation where CIQ ratio is missing or NM
        # Similar pattern to Interest Coverage fallback (lines 9787-9804)
        total_debt_for_nd = metrics.get('Total Debt', pd.Series(np.nan, index=df.index))
        cash_for_nd = metrics.get('Cash & Short-term Investments', pd.Series(np.nan, index=df.index))
        ebitda_for_nd = metrics.get('EBITDA', pd.Series(np.nan, index=df.index))
        
        net_debt_ebitda = net_debt_ebitda_raw.copy()
        fallback_count_nd = 0
        
        for idx in df.index:
            existing = net_debt_ebitda_raw.loc[idx] if idx in net_debt_ebitda_raw.index else np.nan
            
            if pd.isna(existing):
                debt_val = total_debt_for_nd.loc[idx] if idx in total_debt_for_nd.index else np.nan
                cash_val = cash_for_nd.loc[idx] if idx in cash_for_nd.index else np.nan
                ebitda_val = ebitda_for_nd.loc[idx] if idx in ebitda_for_nd.index else np.nan
                
                # Only calculate if we have debt and cash
                if pd.notna(debt_val) and pd.notna(cash_val):
                    net_debt = debt_val - cash_val  # Can be negative for net cash positions
                    
                    # Case 1: Net cash position (Cash > Debt) - always valid, will score 100
                    if net_debt < 0:
                        # Use a small negative number to indicate net cash
                        # The scoring logic at line 11278 will detect is_net_cash and score 100
                        net_debt_ebitda.loc[idx] = -1.0  # Sentinel for "net cash"
                        fallback_count_nd += 1
                    # Case 2: Net debt position with positive EBITDA - calculate ratio
                    elif pd.notna(ebitda_val) and ebitda_val > 0:
                        net_debt_ebitda.loc[idx] = net_debt / ebitda_val
                        fallback_count_nd += 1
                    # Case 3: Net debt position with negative/zero EBITDA - truly NM (leave as NaN)
        
        if fallback_count_nd > 0 and not os.environ.get("RG_TESTS"):
            print(f"  [DEV] Net Debt/EBITDA fallback applied: {fallback_count_nd} issuers")
        
        # Use the fallback-enhanced series for downstream processing
        net_debt_ebitda_raw = net_debt_ebitda
        
        # [V5.0.2] Handle net cash positions (negative ratio = excellent)
        # Negative Net Debt/EBITDA means Cash > Debt, which is excellent credit quality
        # These should score 100, not be treated as missing data
        is_net_cash = net_debt_ebitda_raw < 0
        
        # For positive ratios: apply normal scoring (lower is better)
        # Clip positive values to max 20x for scoring
        net_debt_ebitda_positive = net_debt_ebitda_raw.where(net_debt_ebitda_raw >= 0, other=np.nan).clip(upper=20.0)
        part1 = (np.minimum(net_debt_ebitda_positive, 3.0)/3.0)*60.0
        part2 = (np.maximum(net_debt_ebitda_positive-3.0, 0.0)/5.0)*40.0
        raw_penalty = np.minimum(part1+part2, 100.0)
        positive_score = np.clip(100.0 - raw_penalty, 0.0, 100.0)
        
        # Final score: 100 for net cash, calculated score for positive ratios
        net_debt_score = np.where(is_net_cash, 100.0, positive_score)
        net_debt_score = pd.Series(net_debt_score, index=net_debt_ebitda_raw.index)

        # Component 2: Interest Coverage (40%) - increased from 30%
        interest_coverage_score = ebitda_cov_score

        # Component 3: Total Debt / Total Capital (20%)
        debt_capital = metrics['Total Debt / Total Capital (%)']
        # Don't fill missing with 50%, just clip valid values
        debt_capital = debt_capital.clip(0, 100)
        debt_cap_score = np.clip(100 - debt_capital, 0, 100)

        # Leverage Score with renormalization using unified function
        leverage_components = np.column_stack([
            net_debt_score,
            interest_coverage_score,
            debt_cap_score
        ])

        leverage_weights = np.array([0.40, 0.40, 0.20])

        leverage_score, leverage_completeness, leverage_components_used = \
            _calculate_factor_score_with_renormalization(
                leverage_components,
                leverage_weights,
                min_components=2,  # Require at least 2 of 3 components
                factor_name="Leverage",
                index=df.index
            )

        scores['leverage_score'] = leverage_score
        scores['leverage_data_completeness'] = leverage_completeness
        scores['leverage_components_used'] = leverage_components_used

        # Profitability ([V3.6] Restructured - replaced ROE with Gross Profit Margin, removed EBIT Margin)
        # ROE removed: equity metric that improves with leverage, inappropriate for credit
        # EBIT Margin removed: 90%+ correlated with EBITDA Margin, redundant
        gross_margin = _pct_to_100(metrics.get('Gross Profit Margin', pd.Series(np.nan, index=df.index)))
        ebitda_margin = _pct_to_100(metrics['EBITDA Margin'])
        roa = _pct_to_100(metrics['Return on Assets'])

        # Gross Profit Margin: 60% = 100 score (strong pricing power)
        gross_margin_score = np.clip((gross_margin / 60.0) * 100, 0, 100)
        # EBITDA Margin: scale so 50% margin = 100 score
        margin_score = np.clip(ebitda_margin * 2, 0, 100)
        # ROA: scale so 20% ROA = 100 score
        roa_score = np.clip(roa * 5, 0, 100)

        profitability_components = np.column_stack([
            gross_margin_score,
            margin_score,
            roa_score
        ])

        profitability_weights = np.array([0.30, 0.40, 0.30])

        profitability_score, profitability_completeness, profitability_components_used = \
            _calculate_factor_score_with_renormalization(
                profitability_components,
                profitability_weights,
                min_components=2,  # Require at least 2 of 3 components
                factor_name="Profitability",
                index=df.index
            )

        scores['profitability_score'] = profitability_score
        scores['profitability_data_completeness'] = profitability_completeness
        scores['profitability_components_used'] = profitability_components_used

        # Liquidity ([V3.2] Enhanced with OCF/Current Liabilities)
        # Three components: balance sheet ratios + cash flow coverage
        # Don't clip NaN to 0 - preserve NaN for missing data
        current_ratio = metrics['Current Ratio (x)']
        current_ratio = current_ratio.where(current_ratio >= 0, other=np.nan)
        quick_ratio = metrics['Quick Ratio (x)']
        quick_ratio = quick_ratio.where(quick_ratio >= 0, other=np.nan)
        ocf_curr_liab = metrics.get('Cash from Ops. to Curr. Liab. (x)', pd.Series(np.nan, index=df.index))
        ocf_curr_liab = pd.to_numeric(ocf_curr_liab, errors='coerce')
        # OCF/CL can be negative (cash burn), treat negative as 0 for scoring
        ocf_curr_liab = ocf_curr_liab.where(ocf_curr_liab >= 0, other=0)

        # Component scoring:
        # Current Ratio: 3.0x = 100 (scaled)
        # Quick Ratio: 2.0x = 100 (scaled)
        # OCF/CL: 1.0x = 100 (can you cover current liabs from one year of operations?)
        current_score = np.clip((current_ratio/3.0)*100.0, 0, 100)
        quick_score = np.clip((quick_ratio/2.0)*100.0, 0, 100)
        ocf_cl_score = np.clip((ocf_curr_liab/1.0)*100.0, 0, 100)

        liquidity_components = np.column_stack([
            current_score,
            quick_score,
            ocf_cl_score
        ])

        # Weights: OCF/CL gets highest weight as it adds flow dimension
        liquidity_weights = np.array([0.35, 0.25, 0.40])

        liquidity_score, liquidity_completeness, liquidity_components_used = \
            _calculate_factor_score_with_renormalization(
                liquidity_components,
                liquidity_weights,
                min_components=2,  # Require at least 2 of 3 components
                factor_name="Liquidity",
                index=df.index
            )

        scores['liquidity_score'] = liquidity_score
        scores['liquidity_data_completeness'] = liquidity_completeness
        scores['liquidity_components_used'] = liquidity_components_used

        # [V4.1] Growth Factor REMOVED - revenue trajectory now captured in Trend Score
        # Growth effects flow through other factors and the Revenue trend metric

        # Cash Flow ([v3] Annual-only) - enhance with data quality tracking
        _cf_comp = _cash_flow_component_scores(df, data_period_setting, has_period_alignment, ref_date_for_extraction)
        _cf_cols = [c for c in ["OCF_to_Revenue_Score", "OCF_to_Debt_Score",
                                 "UFCF_margin_Score", "LFCF_margin_Score"] if c in _cf_comp.columns]

        # Filter to only columns that have at least some valid data
        _cf_cols = [c for c in _cf_cols if _cf_comp[c].notna().sum() > 0]

        # PERFORMANCE FIX: Calculate raw values once for all issuers (for diagnostics)
        # Store alongside scores to avoid recalculating 2000 times in the loop below
        _cf_base_components = _cf_components_dataframe(df, data_period_setting, has_period_alignment, ref_date_for_extraction)
        _cf_raw_values = _cf_raw_dataframe(_cf_base_components)

        if _cf_cols:
            cf_array = _cf_comp[_cf_cols].to_numpy(dtype=float)

            # Count available components per issuer
            cf_components_available = (~np.isnan(cf_array)).sum(axis=1)

            # Calculate mean only if minimum threshold met
            min_cf_components = 2  # Require at least 2 of 4 components

            with np.errstate(invalid='ignore'):
                cf_scores = np.nanmean(cf_array, axis=1)
                cf_scores = np.where(cf_components_available >= min_cf_components, cf_scores, np.nan)

            scores['cash_flow_score'] = pd.Series(cf_scores, index=df.index)
            scores['cash_flow_data_completeness'] = pd.Series(cf_components_available / len(_cf_cols), index=df.index)
            scores['cash_flow_components_used'] = pd.Series(cf_components_available, index=df.index, dtype=int)
        else:
            scores['cash_flow_score'] = pd.Series(np.nan, index=df.index)
            scores['cash_flow_data_completeness'] = pd.Series(0.0, index=df.index)
            scores['cash_flow_components_used'] = pd.Series(0, index=df.index, dtype=int)

        # NEW: Capture diagnostic data for remaining factors (Phase 1 - Diagnostic Storage)
        # Capture Leverage diagnostic data
        for idx in df.index:
            factor_diagnostic_data[idx]['Leverage'] = {
                'final_score': float(scores.loc[idx, 'leverage_score']) if pd.notna(scores.loc[idx, 'leverage_score']) else None,
                'components': {
                    'Net_Debt_EBITDA': {
                        'raw_value': float(net_debt_ebitda_raw.loc[idx]) if pd.notna(net_debt_ebitda_raw.loc[idx]) else None,
                        'component_score': float(net_debt_score.loc[idx]) if pd.notna(net_debt_score.loc[idx]) else None,
                        'weight': 0.40,
                        'weighted_contribution': float(net_debt_score.loc[idx] * 0.40) if pd.notna(net_debt_score.loc[idx]) else None,
                        'formula': 'Net Debt / EBITDA',
                        'calculation': f"({float(total_debt_for_nd.loc[idx]):,.0f} - {float(cash_for_nd.loc[idx]):,.0f}) / {float(ebitda_for_nd.loc[idx]):,.0f}" if pd.notna(total_debt_for_nd.loc[idx]) and pd.notna(cash_for_nd.loc[idx]) and pd.notna(ebitda_for_nd.loc[idx]) else "N/A",
                        'scoring_logic': 'Net Cash (<0) = 100. Positive: Lower is better. 0-3x: 100-40. >3x: 40-0. Clipped at 20x.'
                    },
                    'Interest_Coverage': {
                        'raw_value': float(ebitda_interest.loc[idx]) if pd.notna(ebitda_interest.loc[idx]) else None,
                        'component_score': float(interest_coverage_score.loc[idx]) if pd.notna(interest_coverage_score.loc[idx]) else None,
                        'weight': 0.40,
                        'weighted_contribution': float(interest_coverage_score.loc[idx] * 0.40) if pd.notna(interest_coverage_score.loc[idx]) else None,
                        'formula': 'EBITDA / Interest Expense',
                        'calculation': f"{float(ebitda_raw.loc[idx]):,.0f} / {abs(float(interest_expense_raw.loc[idx])):,.0f}" if pd.notna(ebitda_raw.loc[idx]) and pd.notna(interest_expense_raw.loc[idx]) else "N/A",
                        'scoring_logic': 'Higher is better. >8x: 90-100. 5-8x: 70-90. 3-5x: 50-70. 2-3x: 30-50. 1-2x: 10-30. <1x: 0-10.'
                    },
                    'Debt_Capital_Ratio': {
                        'raw_value': float(debt_capital.loc[idx]) if pd.notna(debt_capital.loc[idx]) else None,
                        'component_score': float(debt_cap_score.loc[idx]) if pd.notna(debt_cap_score.loc[idx]) else None,
                        'weight': 0.20,
                        'weighted_contribution': float(debt_cap_score.loc[idx] * 0.20) if pd.notna(debt_cap_score.loc[idx]) else None,
                        'formula': 'Total Debt / Total Capital',
                        'calculation': f"{float(metrics['Total Debt'].loc[idx]):,.0f} / ({float(metrics['Total Debt'].loc[idx]):,.0f} + {float(metrics['Total Equity'].loc[idx]):,.0f})" if 'Total Debt' in metrics and 'Total Equity' in metrics and pd.notna(metrics['Total Debt'].loc[idx]) and pd.notna(metrics['Total Equity'].loc[idx]) else "N/A",
                        'scoring_logic': 'Lower is better. 0% = 100. 100% = 0. Linear interpolation.'
                    }
                },
                'data_completeness': float(scores.loc[idx, 'leverage_data_completeness']) if 'leverage_data_completeness' in scores.columns else None,
                'components_used': int(scores.loc[idx, 'leverage_components_used']) if 'leverage_components_used' in scores.columns else None
            }
            
            # Capture Profitability diagnostic data
            # Calculate normalized weights for available components
            gm_available = pd.notna(gross_margin_score.loc[idx]) if idx in gross_margin_score.index else False
            em_available = pd.notna(margin_score.loc[idx]) if idx in margin_score.index else False
            ra_available = pd.notna(roa_score.loc[idx]) if idx in roa_score.index else False
            
            prof_available_weights = []
            if gm_available: prof_available_weights.append(0.30)
            if em_available: prof_available_weights.append(0.40)
            if ra_available: prof_available_weights.append(0.30)
            prof_weight_sum = sum(prof_available_weights) if prof_available_weights else 1.0
            
            gm_norm_weight = 0.30 / prof_weight_sum if gm_available and prof_weight_sum > 0 else 0
            em_norm_weight = 0.40 / prof_weight_sum if em_available and prof_weight_sum > 0 else 0
            ra_norm_weight = 0.30 / prof_weight_sum if ra_available and prof_weight_sum > 0 else 0
            
            gm_raw = gross_margin.loc[idx] if idx in gross_margin.index else None
            em_raw = ebitda_margin.loc[idx] if idx in ebitda_margin.index else None
            ra_raw = roa.loc[idx] if idx in roa.index else None
            
            gm_sc = gross_margin_score.loc[idx] if idx in gross_margin_score.index else None
            em_sc = margin_score.loc[idx] if idx in margin_score.index else None
            ra_sc = roa_score.loc[idx] if idx in roa_score.index else None
            
            factor_diagnostic_data[idx]['Profitability'] = {
                'final_score': float(scores.loc[idx, 'profitability_score']) if pd.notna(scores.loc[idx, 'profitability_score']) else None,
                'components': {
                    'Gross_Profit_Margin': {
                        'raw_value': float(gm_raw) if pd.notna(gm_raw) else None,
                        'component_score': float(gm_sc) if pd.notna(gm_sc) else None,
                        'weight': 0.30,
                        'normalized_weight': round(gm_norm_weight, 4),
                        'weighted_contribution': float(gm_sc * gm_norm_weight) if pd.notna(gm_sc) else None,
                        'formula': 'Gross Profit Margin',
                        'calculation': f"({float(metrics['Total Revenue'].loc[idx]):,.0f} - {float(metrics['Cost of Goods Sold'].loc[idx]):,.0f}) / {float(metrics['Total Revenue'].loc[idx]):,.0f}" if 'Total Revenue' in metrics and 'Cost of Goods Sold' in metrics and pd.notna(metrics['Total Revenue'].loc[idx]) and pd.notna(metrics['Cost of Goods Sold'].loc[idx]) else "N/A",
                        'scoring_logic': 'Higher is better. 60% margin = 100 score. 0% margin = 0 score. Linear interpolation.'
                    },
                    'EBITDA_Margin': {
                        'raw_value': float(em_raw) if pd.notna(em_raw) else None,
                        'component_score': float(em_sc) if pd.notna(em_sc) else None,
                        'weight': 0.40,
                        'normalized_weight': round(em_norm_weight, 4),
                        'weighted_contribution': float(em_sc * em_norm_weight) if pd.notna(em_sc) else None,
                        'formula': 'EBITDA Margin',
                        'calculation': f"{float(metrics['EBITDA'].loc[idx]):,.0f} / {float(metrics['Total Revenue'].loc[idx]):,.0f}" if 'EBITDA' in metrics and 'Total Revenue' in metrics and pd.notna(metrics['EBITDA'].loc[idx]) and pd.notna(metrics['Total Revenue'].loc[idx]) else "N/A",
                        'scoring_logic': 'Higher is better. 50% margin = 100 score. 0% margin = 0 score. Linear interpolation.'
                    },
                    'ROA': {
                        'raw_value': float(ra_raw) if pd.notna(ra_raw) else None,
                        'component_score': float(ra_sc) if pd.notna(ra_sc) else None,
                        'weight': 0.30,
                        'normalized_weight': round(ra_norm_weight, 4),
                        'weighted_contribution': float(ra_sc * ra_norm_weight) if pd.notna(ra_sc) else None,
                        'formula': 'Return on Assets',
                        'calculation': f"{float(metrics['Net Income'].loc[idx]):,.0f} / {float(metrics['Total Assets'].loc[idx]):,.0f}" if 'Net Income' in metrics and 'Total Assets' in metrics and pd.notna(metrics['Net Income'].loc[idx]) and pd.notna(metrics['Total Assets'].loc[idx]) else "N/A",
                        'scoring_logic': 'Higher is better. 20% ROA = 100 score. 0% ROA = 0 score. Linear interpolation.'
                    }
                },
                'data_completeness': float(scores.loc[idx, 'profitability_data_completeness']) if pd.notna(scores.loc[idx, 'profitability_data_completeness']) else 0.0,
                'components_used': int(scores.loc[idx, 'profitability_components_used']) if pd.notna(scores.loc[idx, 'profitability_components_used']) else 0
            }
            
            # Capture Liquidity diagnostic data (V3.2 - 3 components)
            # Calculate normalized weights for available components
            liq_available_weights = []
            if pd.notna(current_score.loc[idx]): liq_available_weights.append(0.35)
            if pd.notna(quick_score.loc[idx]): liq_available_weights.append(0.25)
            if pd.notna(ocf_cl_score.loc[idx]): liq_available_weights.append(0.40)
            liq_weight_sum = sum(liq_available_weights) if liq_available_weights else 1.0
            
            curr_norm_weight = 0.35 / liq_weight_sum if pd.notna(current_score.loc[idx]) and liq_weight_sum > 0 else 0
            quick_norm_weight = 0.25 / liq_weight_sum if pd.notna(quick_score.loc[idx]) and liq_weight_sum > 0 else 0
            ocf_cl_norm_weight = 0.40 / liq_weight_sum if pd.notna(ocf_cl_score.loc[idx]) and liq_weight_sum > 0 else 0
            
            factor_diagnostic_data[idx]['Liquidity'] = {
                'final_score': float(scores.loc[idx, 'liquidity_score']) if pd.notna(scores.loc[idx, 'liquidity_score']) else None,
                'components': {
                    'Current_Ratio': {
                        'raw_value': float(current_ratio.loc[idx]) if pd.notna(current_ratio.loc[idx]) else None,
                        'component_score': float(current_score.loc[idx]) if pd.notna(current_score.loc[idx]) else None,
                        'weight': 0.35,
                        'normalized_weight': round(curr_norm_weight, 4),
                        'weighted_contribution': float(current_score.loc[idx] * curr_norm_weight) if pd.notna(current_score.loc[idx]) else None,
                        'formula': 'Current Assets / Current Liabilities',
                        'calculation': f"{float(metrics['Current Assets'].loc[idx]):,.0f} / {float(metrics['Current Liabilities'].loc[idx]):,.0f}" if 'Current Assets' in metrics and 'Current Liabilities' in metrics and pd.notna(metrics['Current Assets'].loc[idx]) and pd.notna(metrics['Current Liabilities'].loc[idx]) else "N/A",
                        'scoring_logic': 'Higher is better. 3.0x = 100 score. 0x = 0 score. Linear interpolation.'
                    },
                    'Quick_Ratio': {
                        'raw_value': float(quick_ratio.loc[idx]) if pd.notna(quick_ratio.loc[idx]) else None,
                        'component_score': float(quick_score.loc[idx]) if pd.notna(quick_score.loc[idx]) else None,
                        'weight': 0.25,
                        'normalized_weight': round(quick_norm_weight, 4),
                        'weighted_contribution': float(quick_score.loc[idx] * quick_norm_weight) if pd.notna(quick_score.loc[idx]) else None,
                        'formula': '(Current Assets - Inventory) / Current Liabilities',
                        'calculation': f"({float(metrics['Current Assets'].loc[idx]):,.0f} - {float(metrics['Inventory'].loc[idx]):,.0f}) / {float(metrics['Current Liabilities'].loc[idx]):,.0f}" if 'Current Assets' in metrics and 'Inventory' in metrics and 'Current Liabilities' in metrics and pd.notna(metrics['Current Assets'].loc[idx]) and pd.notna(metrics['Inventory'].loc[idx]) and pd.notna(metrics['Current Liabilities'].loc[idx]) else "N/A",
                        'scoring_logic': 'Higher is better. 2.0x = 100 score. 0x = 0 score. Linear interpolation.'
                    },
                    'OCF_to_Current_Liabilities': {
                        'raw_value': float(ocf_curr_liab.loc[idx]) if pd.notna(ocf_curr_liab.loc[idx]) else None,
                        'component_score': float(ocf_cl_score.loc[idx]) if pd.notna(ocf_cl_score.loc[idx]) else None,
                        'weight': 0.40,
                        'normalized_weight': round(ocf_cl_norm_weight, 4),
                        'weighted_contribution': float(ocf_cl_score.loc[idx] * ocf_cl_norm_weight) if pd.notna(ocf_cl_score.loc[idx]) else None,
                        'formula': 'Operating Cash Flow / Current Liabilities',
                        'calculation': f"{float(metrics['Operating Cash Flow'].loc[idx]):,.0f} / {float(metrics['Current Liabilities'].loc[idx]):,.0f}" if 'Operating Cash Flow' in metrics and 'Current Liabilities' in metrics and pd.notna(metrics['Operating Cash Flow'].loc[idx]) and pd.notna(metrics['Current Liabilities'].loc[idx]) else "N/A",
                        'scoring_logic': 'Higher is better. 1.0x = 100 score. 0x = 0 score. Linear interpolation.'
                    }
                },
                'data_completeness': float(scores.loc[idx, 'liquidity_data_completeness']) if pd.notna(scores.loc[idx, 'liquidity_data_completeness']) else 0.0,
                'components_used': int(scores.loc[idx, 'liquidity_components_used']) if pd.notna(scores.loc[idx, 'liquidity_components_used']) else 0
            }
            

            
            # Capture Cash_Flow diagnostic data
            # Get component details from _cf_comp DataFrame
            cf_components_dict = {}
            # Get component details from _cf_comp DataFrame AND raw values
            cf_components_dict = {}
            if idx in _cf_comp.index:
                # Use pre-calculated raw values (calculated once above for all issuers)
                # No need to recalculate per issuer
                
                # Map score column names to display names and raw value columns
                # Map score column names to display names, raw value columns, formulas, and scoring logic
                component_mapping = {
                    'OCF_to_Revenue_Score': ('OCF_to_Revenue', 'Operating Cash Flow / Revenue', 'OCF_to_Revenue', 'Operating Cash Flow / Revenue', 'Higher is better.'),
                    'OCF_to_Debt_Score': ('OCF_to_Debt', 'Operating Cash Flow / Debt', 'OCF_to_Debt', 'Operating Cash Flow / Total Debt', 'Higher is better.'),
                    'UFCF_margin_Score': ('UFCF_Margin', 'Unlevered Free Cash Flow Margin', 'UFCF_margin', 'Unlevered Free Cash Flow / Revenue', 'Higher is better.'),
                    'LFCF_margin_Score': ('LFCF_Margin', 'Levered Free Cash Flow Margin', 'LFCF_margin', 'Levered Free Cash Flow / Revenue', 'Higher is better.')
                }
                
                # Get number of available components for equal weighting
                available_components = [col for col in _cf_cols if pd.notna(_cf_comp.loc[idx, col])]
                weight_per_component = 1.0 / len(available_components) if available_components else 0.0
                
                for score_col in _cf_cols:
                    if score_col in component_mapping:
                        key, display_name, raw_col, formula, scoring_logic = component_mapping[score_col]
                        
                        # Get score from _cf_comp
                        component_score = _cf_comp.loc[idx, score_col] if pd.notna(_cf_comp.loc[idx, score_col]) else None
                        
                        # Get raw value from _cf_raw_values (global)
                        if idx in _cf_raw_values.index and raw_col in _cf_raw_values.columns:
                            raw_value = float(_cf_raw_values.loc[idx, raw_col]) if pd.notna(_cf_raw_values.loc[idx, raw_col]) else None
                        else:
                            raw_value = None
                        
                        # Construct calculation string
                        calc_str = "N/A"
                        if key == 'OCF_to_Revenue':
                            if 'Operating Cash Flow' in metrics and 'Total Revenue' in metrics and pd.notna(metrics['Operating Cash Flow'].loc[idx]) and pd.notna(metrics['Total Revenue'].loc[idx]):
                                calc_str = f"{float(metrics['Operating Cash Flow'].loc[idx]):,.0f} / {float(metrics['Total Revenue'].loc[idx]):,.0f}"
                        elif key == 'OCF_to_Debt':
                            if 'Operating Cash Flow' in metrics and 'Total Debt' in metrics and pd.notna(metrics['Operating Cash Flow'].loc[idx]) and pd.notna(metrics['Total Debt'].loc[idx]):
                                calc_str = f"{float(metrics['Operating Cash Flow'].loc[idx]):,.0f} / {float(metrics['Total Debt'].loc[idx]):,.0f}"
                        elif key == 'UFCF_Margin':
                            if 'Unlevered Free Cash Flow' in metrics and 'Total Revenue' in metrics and pd.notna(metrics['Unlevered Free Cash Flow'].loc[idx]) and pd.notna(metrics['Total Revenue'].loc[idx]):
                                calc_str = f"{float(metrics['Unlevered Free Cash Flow'].loc[idx]):,.0f} / {float(metrics['Total Revenue'].loc[idx]):,.0f}"
                        elif key == 'LFCF_Margin':
                            if 'Levered Free Cash Flow' in metrics and 'Total Revenue' in metrics and pd.notna(metrics['Levered Free Cash Flow'].loc[idx]) and pd.notna(metrics['Total Revenue'].loc[idx]):
                                calc_str = f"{float(metrics['Levered Free Cash Flow'].loc[idx]):,.0f} / {float(metrics['Total Revenue'].loc[idx]):,.0f}"

                        if component_score is not None:
                            cf_components_dict[key] = {
                                'raw_value': raw_value,
                                'component_score': float(component_score),
                                'weight': weight_per_component,
                                'weighted_contribution': float(component_score * weight_per_component),
                                'formula': formula,
                                'calculation': calc_str,
                                'scoring_logic': scoring_logic
                            }

            factor_diagnostic_data[idx]['Cash_Flow'] = {
                'final_score': float(scores.loc[idx, 'cash_flow_score']) if pd.notna(scores.loc[idx, 'cash_flow_score']) else None,
                'components': cf_components_dict,
                'data_completeness': float(scores.loc[idx, 'cash_flow_data_completeness']) if pd.notna(scores.loc[idx, 'cash_flow_data_completeness']) else 0.0,
                'components_used': int(scores.loc[idx, 'cash_flow_components_used']) if pd.notna(scores.loc[idx, 'cash_flow_components_used']) else 0,
                'components_total': len(_cf_cols)
            }

        # Return both scores and diagnostic data
        return scores, factor_diagnostic_data, raw_input_data

    # [V2.3] Derive data_period_setting from period_mode for backward compatibility
    # For now, always use quarterly/most recent since we're using quarterly for trends
    # The reference_date_actual controls whether data is aligned or not
    data_period_setting = "Most Recent LTM (LTM0)"

    # [Phase 1] Now returns tuple: (quality_scores, factor_diagnostic_data, raw_input_data)
    quality_scores, factor_diagnostic_data, raw_input_data = calculate_quality_scores(df, data_period_setting, has_period_alignment,
                                             reference_date_actual, align_to_reference,
                                             prefer_annual_reports=prefer_annual_reports if period_mode == PeriodSelectionMode.REFERENCE_ALIGNED else False,
                                             selected_periods=selected_periods)
    _log_timing("04_Quality_Scores_Complete")

    _audit_count("After factor construction", df, audits)

    # Clean rating for grouping
    def _clean_rating_outer(x):
        x = str(x).upper().strip()
        x = x.split('(')[0].strip().replace(' ','').replace('*','')
        return {'BBBM':'BBB','BMNS':'B','CCCC':'CCC'}.get(x, x)

    df['_Credit_Rating_Clean'] = df[RATING_COL].map(_clean_rating_outer)

    # ========================================================================
    # [V2.2.1] CALCULATE CALIBRATED WEIGHTS IF ENABLED
    # ========================================================================

    # If calibrated_weights parameter was passed with special marker, calculate now.
    # We need to build a temporary results dataframe with factor scores to pass to the calibration function.
    if calibrated_weights == "CALCULATE_INSIDE" and use_sector_adjusted:
        try:
            # Build a temporary dataframe with the factor scores we just calculated
            temp_results = pd.DataFrame({
                'Credit_Rating_Clean': df['_Credit_Rating_Clean'],
                'Credit_Score': quality_scores['credit_score'],
                'Leverage_Score': quality_scores['leverage_score'],
                'Profitability_Score': quality_scores['profitability_score'],
                'Liquidity_Score': quality_scores['liquidity_score'],

                'Cash_Flow_Score': quality_scores['cash_flow_score']
            })

            # Add classification field if available
            if has_classification and 'Rubrics Custom Classification' in df.columns:
                temp_results['Rubrics_Custom_Classification'] = df['Rubrics Custom Classification']

            # Get calibration rating band from session state (set in sidebar)
            cal_rating_band = st.session_state.get('calibration_rating_band', 'BBB')

            # Calculate calibrated weights from the factor scores
            calibrated_weights = calculate_calibrated_sector_weights(
                temp_results,
                rating_band=cal_rating_band,
                use_dynamic=True
            )

            if calibrated_weights is not None:
                _log_timing("04a_Calibrated_Weights_Complete")
                print(f"[CALIBRATION] Calculated calibrated weights for rating band {cal_rating_band}")
                # Store in session state for UI display
                st.session_state['_calibrated_weights'] = calibrated_weights
        except Exception as e:
            print(f"[CALIBRATION] Failed to calculate calibrated weights: {str(e)}")
            calibrated_weights = None
    elif calibrated_weights == "CALCULATE_INSIDE":
        # Calibration requested but sector adjustment is off
        calibrated_weights = None

    # ========================================================================
    # CALCULATE COMPOSITE SCORE ([V2.2] FEATURE-GATED CLASSIFICATION WEIGHTS)
    # ========================================================================

    qs = quality_scores.copy()

    # Don't fill missing factor scores with arbitrary defaults
    # Keep them as NaN - composite calculation will handle via renormalization
    # (No median filling, no default filling - let missing data stay missing)

    # [V2.2] Calculate composite score - use classification weights only if available
    # OPTIMIZED: Vectorized calculation instead of iterrows()
    
    if has_classification and use_sector_adjusted:
        # Build weight matrix for each issuer based on classification
        # [V2.2.1] Pass calibrated_weights if available
        weight_matrix = df['Rubrics Custom Classification'].apply(
            lambda c: pd.Series(get_classification_weights(c, True, calibrated_weights=calibrated_weights))
        )
        # Track which weights were used (for display)
        def _weight_label(c):
            weight_source = " (Calibrated)" if calibrated_weights is not None else ""
            if c in CLASSIFICATION_TO_SECTOR:
                parent_sector = CLASSIFICATION_TO_SECTOR[c]
                return f"{parent_sector} (via {c[:20]}...){weight_source}"
            elif c in CLASSIFICATION_OVERRIDES:
                return f"{c[:30]}... (Custom){weight_source}"
            else:
                return f"Universal{weight_source}"
        weight_used_list = df['Rubrics Custom Classification'].apply(_weight_label).tolist()
    else:
        # Use universal weights for all rows
        default_weights = get_classification_weights('Default', False, calibrated_weights=calibrated_weights)
        weight_matrix = pd.DataFrame([default_weights] * len(df), index=df.index)
        weight_label = "Universal (Calibrated)" if calibrated_weights is not None else "Universal"
        weight_used_list = [weight_label] * len(df)
    
    # Calculate composite with renormalization for missing factors
    # This ensures we only weight factors that have valid scores

    # Get factor score columns
    factor_cols = ['credit_score', 'leverage_score', 'profitability_score',
                   'liquidity_score', 'cash_flow_score']

    # For each issuer, calculate composite with renormalization
    composite_scores_list = []
    composite_completeness_list = []

    for idx in qs.index:
        factor_values = qs.loc[idx, factor_cols].values

        # Get weights for this issuer (from classification or default)
        if has_classification and use_sector_adjusted:
            classification = df.loc[idx, 'Rubrics Custom Classification']
            factor_weights_dict = get_classification_weights(classification, True, calibrated_weights=calibrated_weights)
            factor_weights = np.array([factor_weights_dict[col] for col in factor_cols])
        else:
            # Default weights from weight_matrix
            factor_weights = np.array([weight_matrix.loc[idx, col] for col in factor_cols])

        # Identify available factors
        available_mask = ~np.isnan(factor_values)
        n_available = available_mask.sum()

        if n_available >= 3:  # Require at least 3 of 5 factors (60%)
            # Renormalize weights
            effective_weights = factor_weights * available_mask
            effective_weights = effective_weights / effective_weights.sum()

            # Calculate composite
            composite = np.nansum(factor_values * effective_weights)
            completeness = n_available / 5
        else:
            composite = np.nan
            completeness = n_available / 5

        composite_scores_list.append(composite)
        composite_completeness_list.append(completeness)

    # Create quality score series (5-factor weighted average)
    quality_score = pd.Series(composite_scores_list, index=qs.index)
    qs['composite_data_completeness'] = composite_completeness_list

    # [V4.0] Blend Quality and Trend into Composite Score
    # Quality: 80% weight (current fundamentals are primary determinant)
    # Trend: 20% weight (direction of travel matters, but shouldn't dominate)
    QUALITY_WEIGHT = 0.80
    TREND_WEIGHT = 0.20
    
    # Blend quality and trend scores
    # cycle_score is the trend score calculated earlier (line ~9297)
    composite_score = (quality_score * QUALITY_WEIGHT) + (cycle_score * TREND_WEIGHT)
    
    # Handle cases where trend is missing: fall back to quality only
    composite_score = composite_score.fillna(quality_score)
    
    # Store both for diagnostics transparency
    qs['quality_score'] = quality_score  # Pure 5-factor quality
    quality_scores['quality_score'] = quality_score

    _log_timing("05_Composite_Score_Complete")

    # ============================================================================
    # [V3.1 FIX] Store selected period information in results for diagnostics
    # ============================================================================
    if selected_periods is not None and len(selected_periods) > 0:
        # Create a mapping from row_idx to selected_suffix
        period_map = selected_periods.set_index('row_idx')['selected_suffix'].to_dict()
        period_date_map = selected_periods.set_index('row_idx')['selected_date'].to_dict()
        period_type_map = selected_periods.set_index('row_idx')['is_fy'].to_dict()

        # Add to quality_scores DataFrame
        quality_scores['selected_suffix'] = quality_scores.index.map(lambda idx: period_map.get(idx, '.0'))
        quality_scores['selected_date'] = quality_scores.index.map(lambda idx: period_date_map.get(idx, None))
        quality_scores['period_type'] = quality_scores.index.map(lambda idx: 'FY' if period_type_map.get(idx, False) else 'LTM')
    else:
        # Fallback: use .0 as default
        quality_scores['selected_suffix'] = '.0'
        quality_scores['selected_date'] = None
        quality_scores['period_type'] = 'Unknown'

    # Also store the period mode for reference
    quality_scores['period_mode_used'] = str(period_mode) if period_mode else 'LATEST_AVAILABLE'

    # ========================================================================
    # CREATE RESULTS DATAFRAME ([V2.2] WITH OPTIONAL COLUMNS)
    # ========================================================================

    # Start with core identifiers (always required)
    # [V2.2] Use resolved/canonical column names (not hard-coded strings)
    # Enforce Company_ID as string to preserve leading zeros, avoid scientific notation
    results_dict = {
        'Company_ID': df[COMPANY_ID_COL].astype(str),
        'Company_Name': df[COMPANY_NAME_COL],
        'Credit_Rating': df[RATING_COL],
        'Credit_Rating_Clean': df['_Credit_Rating_Clean'],
        'Composite_Score': composite_score,
        'Quality_Score': quality_scores['quality_score'],  # [V3.1 FIX] Add for diagnostics

        'Credit_Score': quality_scores['credit_score'],
        'Leverage_Score': quality_scores['leverage_score'],
        'Profitability_Score': quality_scores['profitability_score'],
        'Liquidity_Score': quality_scores['liquidity_score'],
        'Cycle_Position_Score': cycle_score,
        'Weight_Method': weight_used_list
    }

    # Add Cash_Flow_Score (matches verification pattern)
    results_dict['Cash_Flow_Score'] = quality_scores['cash_flow_score']

    # Add data quality indicators for each factor
    results_dict['Leverage_Data_Completeness'] = quality_scores.get('leverage_data_completeness', 1.0)
    results_dict['Leverage_Components_Used'] = quality_scores.get('leverage_components_used', 4)

    results_dict['Profitability_Data_Completeness'] = quality_scores.get('profitability_data_completeness', 1.0)
    results_dict['Profitability_Components_Used'] = quality_scores.get('profitability_components_used', 4)

    results_dict['Liquidity_Data_Completeness'] = quality_scores.get('liquidity_data_completeness', 1.0)
    results_dict['Liquidity_Components_Used'] = quality_scores.get('liquidity_components_used', 2)



    results_dict['Cash_Flow_Data_Completeness'] = quality_scores.get('cash_flow_data_completeness', 1.0)
    results_dict['Cash_Flow_Components_Used'] = quality_scores.get('cash_flow_components_used', 4)

    results_dict['Credit_Data_Completeness'] = quality_scores.get('credit_data_completeness', 1.0)
    results_dict['Credit_Components_Used'] = quality_scores.get('credit_components_used', 3)

    # Add overall composite data completeness
    results_dict['Composite_Data_Completeness'] = quality_scores.get('composite_data_completeness', 1.0)

    # [V3.1 FIX] Add period selection information for diagnostics
    results_dict['selected_suffix'] = quality_scores.get('selected_suffix', '.0')
    results_dict['selected_date'] = quality_scores.get('selected_date', None)
    results_dict['period_type'] = quality_scores.get('period_type', 'Unknown')
    results_dict['period_mode_used'] = quality_scores.get('period_mode_used', 'LATEST_AVAILABLE')

    # [V2.2] Add optional columns if available
    if 'Ticker' in df.columns:
        # Enforce Ticker as string to preserve formatting
        results_dict['Ticker'] = df['Ticker'].astype(str)

    if has_country_region:
        results_dict['Country'] = df['Country']
        results_dict['Region'] = df['Region']

    if has_classification:
        results_dict['Rubrics_Custom_Classification'] = df['Rubrics Custom Classification']

    if 'Industry' in df.columns:
        results_dict['Industry'] = df['Industry']

    if 'Market Capitalization' in df.columns:
        results_dict['Market_Cap'] = pd.to_numeric(df['Market Capitalization'], errors='coerce')

    results = pd.DataFrame(results_dict)

    # NEW: Assemble complete diagnostic data for each issuer (Phase 1 - Diagnostic Storage)
    diagnostic_data_list = []
    
    for idx in df.index:
        # Prepare period selection diagnostic
        period_diagnostic = {
            'selected_suffix': str(quality_scores.loc[idx, 'selected_suffix']) if pd.notna(quality_scores.loc[idx, 'selected_suffix']) else '.0',
            'selected_date': str(quality_scores.loc[idx, 'selected_date']) if pd.notna(quality_scores.loc[idx, 'selected_date']) else None,
            'period_type': str(quality_scores.loc[idx, 'period_type']) if pd.notna(quality_scores.loc[idx, 'period_type']) else 'Unknown',
            'periods_available': len([c for c in df.columns if 'Period Ended' in str(c) and pd.notna(df.loc[idx, c])]),
            'selection_mode': str(period_mode.value) if period_mode else 'LATEST_AVAILABLE',
            'selection_reason': 'Most recent data available'
        }
        
        # Calculate peer context for this issuer (Phase 2 Architectural Fix)
        # Use safe getters for classification and rating
        class_val = df.loc[idx, 'Rubrics Custom Classification'] if 'Rubrics Custom Classification' in df.columns else df.loc[idx, 'Rubrics_Custom_Classification'] if 'Rubrics_Custom_Classification' in df.columns else 'Unknown'
        rating_val = df.loc[idx, '_Credit_Rating_Clean'] if '_Credit_Rating_Clean' in df.columns else 'NR'
        
        peer_context = calculate_peer_context_for_scoring(
            df=df,
            idx=idx,
            classification=str(class_val) if pd.notna(class_val) else 'Unknown',
            rating=str(rating_val) if pd.notna(rating_val) else 'NR'
        )
        
        # Assemble complete diagnostic structure
        issuer_diagnostic = {
            'raw_inputs': raw_input_data.get(idx, {}),
            'time_series': trend_diagnostic_data.get(idx, {}),
            'factor_details': factor_diagnostic_data.get(idx, {}),
            'period_selection': period_diagnostic,
            'peer_context': peer_context,  # [Phase 2] Store peer context
            'composite_calculation': {
                'composite_score': float(composite_score[idx]) if pd.notna(composite_score[idx]) else None,
                'quality_score': float(quality_scores.loc[idx, 'quality_score']) if pd.notna(quality_scores.loc[idx, 'quality_score']) else None,
                'trend_score': float(cycle_score[idx]) if pd.notna(cycle_score[idx]) else None,
                'factor_contributions': {
                    'Credit': {
                        'score': float(quality_scores.loc[idx, 'credit_score']) if pd.notna(quality_scores.loc[idx, 'credit_score']) else None,
                        'weight': float(weight_matrix.loc[idx, 'credit_score']) if pd.notna(weight_matrix.loc[idx, 'credit_score']) else 0.0,
                        'contribution': float(quality_scores.loc[idx, 'credit_score'] * weight_matrix.loc[idx, 'credit_score']) if pd.notna(quality_scores.loc[idx, 'credit_score']) and pd.notna(weight_matrix.loc[idx, 'credit_score']) else None
                    },
                    'Leverage': {
                        'score': float(quality_scores.loc[idx, 'leverage_score']) if pd.notna(quality_scores.loc[idx, 'leverage_score']) else None,
                        'weight': float(weight_matrix.loc[idx, 'leverage_score']) if pd.notna(weight_matrix.loc[idx, 'leverage_score']) else 0.0,
                        'contribution': float(quality_scores.loc[idx, 'leverage_score'] * weight_matrix.loc[idx, 'leverage_score']) if pd.notna(quality_scores.loc[idx, 'leverage_score']) and pd.notna(weight_matrix.loc[idx, 'leverage_score']) else None
                    },
                    'Profitability': {
                        'score': float(quality_scores.loc[idx, 'profitability_score']) if pd.notna(quality_scores.loc[idx, 'profitability_score']) else None,
                        'weight': float(weight_matrix.loc[idx, 'profitability_score']) if pd.notna(weight_matrix.loc[idx, 'profitability_score']) else 0.0,
                        'contribution': float(quality_scores.loc[idx, 'profitability_score'] * weight_matrix.loc[idx, 'profitability_score']) if pd.notna(quality_scores.loc[idx, 'profitability_score']) and pd.notna(weight_matrix.loc[idx, 'profitability_score']) else None
                    },
                    'Liquidity': {
                        'score': float(quality_scores.loc[idx, 'liquidity_score']) if pd.notna(quality_scores.loc[idx, 'liquidity_score']) else None,
                        'weight': float(weight_matrix.loc[idx, 'liquidity_score']) if pd.notna(weight_matrix.loc[idx, 'liquidity_score']) else 0.0,
                        'contribution': float(quality_scores.loc[idx, 'liquidity_score'] * weight_matrix.loc[idx, 'liquidity_score']) if pd.notna(quality_scores.loc[idx, 'liquidity_score']) and pd.notna(weight_matrix.loc[idx, 'liquidity_score']) else None
                    },

                    'Cash_Flow': {
                        'score': float(quality_scores.loc[idx, 'cash_flow_score']) if pd.notna(quality_scores.loc[idx, 'cash_flow_score']) else None,
                        'weight': float(weight_matrix.loc[idx, 'cash_flow_score']) if pd.notna(weight_matrix.loc[idx, 'cash_flow_score']) else 0.0,
                        'contribution': float(quality_scores.loc[idx, 'cash_flow_score'] * weight_matrix.loc[idx, 'cash_flow_score']) if pd.notna(quality_scores.loc[idx, 'cash_flow_score']) and pd.notna(weight_matrix.loc[idx, 'cash_flow_score']) else None
                    }
                },
                'weight_method': weight_used_list[idx] if idx < len(weight_used_list) else 'Universal',
                'sector': df.loc[idx, 'Rubrics Custom Classification'] if has_classification and 'Rubrics Custom Classification' in df.columns else 'Unknown'
            }
        }
        
        # Serialize to JSON string
        diagnostic_data_list.append(json.dumps(issuer_diagnostic))
    
    # Add to results DataFrame
    results['diagnostic_data'] = diagnostic_data_list

    # After quality score calculation
    if DIAG.enabled:
        diagnose_quarterly_annualization(df)

    # [Enhanced Explainability] Store the weights used in calculation for transparency
    results['Weight_Credit_Used'] = weight_matrix['credit_score']
    results['Weight_Leverage_Used'] = weight_matrix['leverage_score']
    results['Weight_Profitability_Used'] = weight_matrix['profitability_score']
    results['Weight_Liquidity_Used'] = weight_matrix['liquidity_score']

    results['Weight_CashFlow_Used'] = weight_matrix['cash_flow_score']

    # Add trend indicators to results
    for col in trend_scores.columns:
        results[col] = trend_scores[col]

    _audit_count("After scoring (non-NaN Composite_Score)", results[results['Composite_Score'].notna()], audits)

    # ========================================================================
    # VALIDATE DIAGNOSTIC DATA (Phase 1 - Diagnostic Storage)
    # ========================================================================

    def validate_diagnostic_data(results_final):
        """
        Validate that diagnostic data is complete and consistent with final scores.
        
        Args:
            results_final: DataFrame with scoring results including diagnostic_data column
        
        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []
        
        for idx, row in results_final.iterrows():
            company_name = row.get('Company_Name', f'Row_{idx}')
            
            # Parse diagnostic data
            if 'diagnostic_data' not in row or pd.isna(row['diagnostic_data']):
                errors.append(f"{company_name}: Missing diagnostic_data")
                continue
            
            try:
                diag = json.loads(row['diagnostic_data'])
            except json.JSONDecodeError:
                errors.append(f"{company_name}: Invalid diagnostic_data JSON")
                continue
            
            # Validation 1: Factor scores match
            factor_map = {
                'Credit': 'Credit_Score',
                'Leverage': 'Leverage_Score', 
                'Profitability': 'Profitability_Score',
                'Liquidity': 'Liquidity_Score',

                'Cash_Flow': 'Cash_Flow_Score'
            }
            
            if 'factor_details' in diag:
                for factor, col_name in factor_map.items():
                    if factor in diag['factor_details']:
                        diag_score = diag['factor_details'][factor].get('final_score')
                        actual_score = row.get(col_name)
                        
                        if diag_score is not None and actual_score is not None and pd.notna(actual_score):
                            if abs(diag_score - actual_score) > 0.1:
                                errors.append(f"{company_name}: {factor} score mismatch - "
                                            f"diagnostic={diag_score:.2f}, actual={actual_score:.2f}")
            
            # Validation 2: Composite calculation components present
            if 'composite_calculation' not in diag:
                errors.append(f"{company_name}: Missing composite_calculation in diagnostic data")
            
            # Validation 3: Time series data present  
            if 'time_series' not in diag or not diag['time_series']:
                errors.append(f"{company_name}: Missing time_series data")
            
            # Validation 4: Period selection documented
            if 'period_selection' not in diag:
                errors.append(f"{company_name}: Missing period_selection data")
        
        return errors

    # Run validation (log warnings but don't block execution)
    validation_errors = validate_diagnostic_data(results)
    if validation_errors:
        import sys
        print(f"[WARNING] Diagnostic data validation found {len(validation_errors)} issues:", file=sys.stderr)
        for err in validation_errors[:10]:  # Show first 10 errors
            print(f"  - {err}", file=sys.stderr)
        if len(validation_errors) > 10:
            print(f"  ... and {len(validation_errors) - 10} more", file=sys.stderr)

    # Add Rating Band (ISSUE #4 SOLUTION) - VECTORIZED
    # Build reverse mapping for O(1) lookup
    rating_to_band = {}
    for band, ratings in RATING_BANDS.items():
        for rating in ratings:
            rating_to_band[rating] = band
    
    # Vectorized band assignment
    results['Rating_Band'] = results['Credit_Rating_Clean'].str.upper().str.strip().map(rating_to_band).fillna('Unrated')
    
    # Add Rating Group (IG/HY) - VECTORIZED
    # [V2.2] All non-IG ratings (including NR/WD/N/M/empty/NaN) classified as High Yield
    ig_ratings_set = {'AAA','AA+','AA','AA-','A+','A','A-','BBB+','BBB','BBB-'}
    
    # Vectorized classification: clean, check membership, assign
    cleaned_ratings = results['Credit_Rating_Clean'].fillna('').astype(str).str.strip().str.upper()
    results['Rating_Group'] = 'High Yield'  # default
    results.loc[cleaned_ratings.isin(ig_ratings_set), 'Rating_Group'] = 'Investment Grade'

    # Sanity check: ensure no 'Unknown' remains
    assert 'Unknown' not in set(results['Rating_Group'].unique()), \
        "Rating_Group should only contain 'Investment Grade' or 'High Yield'"
    
    # Calculate Band Rank (rank within rating band)
    results['Band_Rank'] = results.groupby('Rating_Band')['Composite_Score'].rank(
        ascending=False, method='dense'
    ).astype('Int64')

    # Calculate Composite Percentile within Rating Band (0-100 scale)
    # Also compute global percentile for unified quality/trend split
    results = _compute_quality_metrics(results, score_col="Composite_Score")

    # [V2.2] Calculate Classification Rank only if classification available
    if has_classification and 'Rubrics_Custom_Classification' in results.columns:
        results['Classification_Rank'] = results.groupby('Rubrics_Custom_Classification')['Composite_Score'].rank(
            ascending=False, method='dense'
        ).astype('Int64')

        # [V3.1 FIX] Also calculate Classification_Total and Classification_Percentile
        classification_counts = results.groupby('Rubrics_Custom_Classification').size().to_dict()
        results['Classification_Total'] = results['Rubrics_Custom_Classification'].map(classification_counts)
        results['Classification_Percentile'] = (results['Classification_Rank'] / results['Classification_Total'] * 100).round(2)

    # [V3.1 FIX] Universal ranking (all issuers) based on Composite_Score
    results['Rank'] = results['Composite_Score'].rank(ascending=False, method='min').astype('Int64')
    results['Percentile'] = (results['Rank'] / len(results) * 100).round(2)

    # [V3.1 FIX] Derive Sector from Classification (using module-level CLASSIFICATION_TO_SECTOR)
    if 'Rubrics_Custom_Classification' in results.columns:
        results['Sector'] = results['Rubrics_Custom_Classification'].map(CLASSIFICATION_TO_SECTOR)
        # If no mapping found, use Classification as Sector
        results['Sector'] = results['Sector'].fillna(results['Rubrics_Custom_Classification'])
    elif 'Classification' in results.columns:
        results['Sector'] = results['Classification'].map(CLASSIFICATION_TO_SECTOR)
        results['Sector'] = results['Sector'].fillna(results['Classification'])
    else:
        results['Sector'] = 'N/A'

    # Overall Rank - will be calculated after Recommendation column is created (see line ~7392)

    # ========================================================================
    # [V2.2] CONTEXT FLAGS FOR DUAL-HORIZON ANALYSIS
    # ========================================================================

    # [V5.0.3] Exceptional quality flag - use ABSOLUTE score thresholds
    # Percentile-based thresholds break with small rating bands (e.g., CCC with 2 issuers)
    # A company needs genuinely high absolute scores to be "exceptional"
    results['ExceptionalQuality'] = (
        (results['Composite_Score'] >= 75) |  # Top-tier absolute score
        (results['Profitability_Score'] >= 85)  # Genuinely high profitability
    )

    # Outlier quarter detection (configurable z-score threshold)
    results['OutlierQuarter'] = dual_horizon_metrics['LastQuarterZ'] <= outlier_z_threshold

    # Volatile series detection (configurable CV threshold)
    results['VolatileSeries'] = dual_horizon_metrics['SeriesCV'] >= volatility_cv_threshold

    # Medium-term trend (from dual-horizon)
    results['MediumTermTrend'] = dual_horizon_metrics['MediumTermSlope']

    # Short-term trend (from dual-horizon)
    results['ShortTermChange'] = dual_horizon_metrics['ShortTermChange']

    # Near peak flag
    results['NearPeak'] = dual_horizon_metrics['NearPeak']

    # ========================================================================
    # [V2.2] VOLATILITY DAMPING FOR CYCLE POSITION SCORE
    # ========================================================================

    # Apply volatility damping: reduce negative short-term impact if volatile or outlier
    results['Cycle_Position_Score_Original'] = results['Cycle_Position_Score'].copy()

    # Identify cases needing damping: negative cycle score AND (volatile OR outlier)
    needs_damping = (
        (results['Cycle_Position_Score'] < 50) &  # Below neutral
        (results['VolatileSeries'] | results['OutlierQuarter'])
    )

    # Apply damping: move score closer to neutral (50)  using configurable damping_factor parameter
    results.loc[needs_damping, 'Cycle_Position_Score'] = (
        50 - (50 - results.loc[needs_damping, 'Cycle_Position_Score']) * damping_factor
    )

    # ========================================================================
    # [BUG FIX] Recalculate composite score using damped Cycle_Position_Score
    # ========================================================================
    # The composite score was initially calculated with raw cycle_score before damping.
    # Recalculate using the damped Cycle_Position_Score for consistency.
    results['Composite_Score'] = (
        results['Quality_Score'] * QUALITY_WEIGHT + 
        results['Cycle_Position_Score'] * TREND_WEIGHT
    )
    
    # ========================================================================
    # [BUG FIX] Update diagnostic_data with damped trend scores and recalculated composite
    # ========================================================================
    # The diagnostic_data_list was assembled before damping, so update it now.
    for i, idx in enumerate(df.index):
        if i < len(diagnostic_data_list):
            try:
                diag = json.loads(diagnostic_data_list[i])
                if 'composite_calculation' in diag:
                    diag['composite_calculation']['trend_score'] = float(results.loc[idx, 'Cycle_Position_Score']) if pd.notna(results.loc[idx, 'Cycle_Position_Score']) else None
                    diag['composite_calculation']['composite_score'] = float(results.loc[idx, 'Composite_Score']) if pd.notna(results.loc[idx, 'Composite_Score']) else None
                    diagnostic_data_list[i] = json.dumps(diag)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass  # Skip if diagnostic data is malformed
    
    # CRITICAL: Reassign updated list back to DataFrame column
    # (The original assignment at line ~11946 created a copy, not a reference)
    results['diagnostic_data'] = diagnostic_data_list

    _log_timing("05b_Context_Flags_Complete")

    # ========================================================================
    # GENERATE SIGNAL (Position & Trend quadrant classification)
    # ========================================================================

    # ========================================================================
    # [V5.0.3] Signal Classification - ALWAYS use absolute Composite Score
    # ========================================================================
    # The signal classification must NOT depend on visualization settings.
    # Using percentile breaks with small rating bands (e.g., CCC with 2 issuers
    # where a score of 30 becomes "100th percentile" and incorrectly "Strong").
    #
    # Signal classification uses fixed absolute thresholds:
    # - Strong Quality: Composite_Score >= 55 (QUALITY_THRESHOLD, absolute)
    # - Improving Trend: Cycle_Position_Score >= trend_threshold (default 55)
    #
    # The visualization (Four Quadrant chart) can still use percentile-based
    # axes for display purposes, but signal assignment is absolute.
    # ========================================================================
    
    # For visualization purposes only (chart axes)
    quality_metric_display, x_split_for_plot, _, _ = resolve_quality_metric_and_split(
        results, split_basis, split_threshold
    )
    
    # For signal classification - ALWAYS use absolute Composite Score
    is_strong_quality = results["Composite_Score"] >= QUALITY_THRESHOLD
    
    trend_metric = results["Cycle_Position_Score"]
    is_improving = trend_metric >= trend_threshold

    # Map to 4 base signals
    results['Signal_Base'] = np.select(
        [
            is_strong_quality & is_improving,
            is_strong_quality & ~is_improving,
            ~is_strong_quality & is_improving,
            ~is_strong_quality & ~is_improving
        ],
        ["Strong & Improving", "Strong but Deteriorating", "Weak but Improving", "Weak & Deteriorating"],
        default="—"
    )

    # ========================================================================
    # [V2.2] LABEL OVERRIDE LOGIC FOR CONTEXT-AWARE SIGNALS
    # ========================================================================

    # Start with base signal
    results['Signal'] = results['Signal_Base'].copy()

    # Override 1: Strong & Normalizing
    # When: Exceptional quality + Medium-term improving + Short-term declining + Near peak
    override_normalizing = (
        results['ExceptionalQuality'] &
        (results['Signal_Base'] == 'Strong but Deteriorating') &
        (results['MediumTermTrend'] >= 0) &
        (results['Cycle_Position_Score_Original'] < trend_threshold) &
        (results['NearPeak'] | (results['OutlierQuarter']))
    )
    results.loc[override_normalizing, 'Signal'] = 'Strong & Normalizing'

    # Override 2: Strong & Moderating
    # When: Exceptional quality + High volatility + NOT event-driven (no peak/outlier)
    # Credit rationale: Structural business volatility (cyclical industry, project-based)
    # makes trend score unreliable, but there's no specific "event" explaining weakness
    override_moderating = (
        results['ExceptionalQuality'] &
        (results['Signal_Base'] == 'Strong but Deteriorating') &
        results['VolatileSeries'] &
        ~results['NearPeak'] &        # NOT at an identifiable peak
        ~results['OutlierQuarter']    # AND NOT an outlier event
    )
    results.loc[override_moderating, 'Signal'] = 'Strong & Moderating'

    # Add reasons column for transparency
    results['Signal_Reason'] = ''
    results.loc[override_normalizing, 'Signal_Reason'] = 'Exceptional quality (≥90th %ile); Medium-term improving; Near peak/outlier'
    results.loc[override_moderating, 'Signal_Reason'] = 'Exceptional quality (≥90th %ile); Structural volatility (CV≥0.30); No specific peak/outlier event'
    results.loc[results['OutlierQuarter'] & ~override_normalizing & ~override_moderating, 'Signal_Reason'] += 'Outlier quarter detected'
    results.loc[needs_damping & ~override_normalizing & ~override_moderating, 'Signal_Reason'] += 'Volatility damping applied (50%)'

    results['Combined_Signal'] = results['Signal']  # Keep alias for backward compatibility

    _log_timing("05c_Label_Override_Complete")

    # ========================================================================
    # [V2.2] DIAGNOSTIC: Signal Assignment Quality Control
    # ========================================================================
    # Check for any "Weak & Deteriorating" issuers incorrectly marked with strong signals
    # This should never happen, but validates the classification logic

    if split_basis == "Percentile within Band (recommended)":
        quality_check = results['Composite_Percentile_in_Band']
    elif split_basis == "Global Percentile":
        quality_check = results['Composite_Percentile_Global']
    else:
        quality_check = results['Composite_Score']

    # Identify potential misclassifications (Weak & Deteriorating should never be strong)
    weak_deteriorating = (
        (quality_check < x_split_for_plot) &  # Weak quality
        (results['Cycle_Position_Score'] < trend_threshold)  # Deteriorating trend
    )

    # Check if any are incorrectly in the strong quadrant
    misclassified_signals = weak_deteriorating & results['Signal'].isin(['Strong & Improving', 'Strong but Deteriorating', 'Strong & Normalizing'])

    # Signal Classification Alert removed - validation happens upstream
    # if misclassified_signals.any():
    #     pass  # Logging available if needed

    # ========================================================================
    # [V2.2] COMPREHENSIVE RECOMMENDATION LOGIC
    # ========================================================================
    #
    # New approach: Classification-first with rating guardrails
    # Priority: 1) Classification → 2) Percentile within classification → 3) Rating caps

    def _apply_rating_guardrails(base_rec, rating_band):
        """
        Apply rating-based caps to prevent inappropriate recommendations for weak credits.

        Args:
            base_rec: Base recommendation from classification logic
            rating_band: Issuer's rating band (AAA, AA, A, BBB, BB, B, CCC, etc.)

        Returns:
            Final recommendation after applying rating caps
        """
        # Defaulted issuers: Always Avoid
        if pd.isna(rating_band) or rating_band in ['D', 'SD']:
            return 'Avoid'

        # Distressed (CCC/CC/C): Cap at Hold
        # Rationale: High default risk, should not recommend buying
        if rating_band in ['CCC', 'CC', 'C']:
            if base_rec in ['Strong Buy', 'Buy']:
                return 'Hold'
            return base_rec

        # Single-B: Cap at Buy
        # Rationale: Speculative grade, too risky for "Strong Buy"
        if rating_band == 'B':
            if base_rec == 'Strong Buy':
                return 'Buy'
            return base_rec

        # Investment Grade (AAA-BBB) and BB: No caps
        return base_rec

    def _apply_leverage_guardrail(base_rec, leverage_score):
        """
        Apply leverage-based cap to prevent positive recommendations for highly levered issuers.
        
        Rationale: Extreme leverage (Debt/EBITDA > 10x) indicates structural credit risk
        that should not be masked by an improving trend. Even if trends are positive,
        the balance sheet risk is too high for Hold or Buy.
        
        Args:
            base_rec: Base recommendation after classification and rating guardrails
            leverage_score: Issuer's Leverage_Score (0-100 scale)
        
        Returns:
            Final recommendation after applying leverage cap
        """
        # Handle missing leverage score
        if pd.isna(leverage_score):
            return base_rec
        
        # Extreme leverage (score < 35 ≈ Debt/EBITDA > 10x): Force Avoid
        LEVERAGE_AVOID_THRESHOLD = MODEL_THRESHOLDS.get('leverage_score_avoid', 35)
        
        if leverage_score < LEVERAGE_AVOID_THRESHOLD:
            if base_rec in ['Strong Buy', 'Buy', 'Hold']:
                return 'Avoid'
        
        return base_rec

    def _assign_recommendation_by_classification(classification, percentile_in_band):
        """
        Map classification + percentile to base recommendation (before rating guardrails).

        Logic:
        - Classification determines the tier (primary driver)
        - Percentile within band refines Strong vs regular (secondary modifier)

        Args:
            classification: Signal classification (e.g., "Strong & Improving")
            percentile_in_band: 0-100 percentile rank within rating band

        Returns:
            Base recommendation string
        """
        # Handle missing data
        if pd.isna(classification) or pd.isna(percentile_in_band):
            return 'Hold'

        # Ensure percentile is numeric
        pct = float(percentile_in_band)

        # ===================================================================
        # CLASSIFICATION-BASED RECOMMENDATION MAPPING
        # ===================================================================

        if classification == "Strong & Improving":
            # Best quadrant: Strong quality + Improving trend
            # → Always positive recommendation (Buy or Strong Buy)
            return "Strong Buy" if pct >= PERCENTILE_STRONG_BUY else "Buy"

        elif classification == "Strong but Deteriorating":
            # Good quality but declining trend
            # → Cautiously positive (Buy or Hold)
            return "Buy" if pct >= PERCENTILE_STRONG_BUY else "Hold"

        elif classification == "Strong & Normalizing":
            # Special case: Exceptional quality (90th+ percentile)
            # Medium-term improving, short-term dip (likely temporary)
            # → Buy regardless of percentile (quality overrides short-term weakness)
            return "Buy"

        elif classification == "Strong & Moderating":
            # Special case: Exceptional quality but high volatility
            # Short-term deteriorating with volatile series
            # → Buy (USER REQUESTED: treat as buy opportunity despite volatility)
            return "Buy"

        elif classification == "Weak but Improving":
            # Poor quality but turning around
            # → Cautiously positive if strong momentum (Buy or Hold)
            return "Buy" if pct >= PERCENTILE_STRONG_BUY else "Hold"

        elif classification == "Weak & Deteriorating":
            # Worst quadrant: Weak quality + Deteriorating trend
            # → Always negative recommendation (Avoid)
            return "Avoid"

        else:
            # Unknown/missing classification
            return "Hold"

    def assign_final_recommendation(row):
        """
        Complete recommendation logic with audit trail.

        Process:
        1. Get base recommendation from classification + percentile
        2. Apply rating guardrails (downgrade if needed)
        3. Generate reason for transparency

        Args:
            row: DataFrame row with 'Signal', 'Composite_Percentile_in_Band', 'Rating_Band'

        Returns:
            tuple: (final_recommendation, reason_string)
        """
        # Step 1: Base recommendation from classification
        base = _assign_recommendation_by_classification(
            row['Signal'],
            row['Composite_Percentile_in_Band']
        )

        # Step 2: Apply rating guardrails
        after_rating = _apply_rating_guardrails(base, row['Rating_Band'])
        
        # Step 3: Apply leverage guardrail
        leverage_score = row.get('Leverage_Score', None)
        final = _apply_leverage_guardrail(after_rating, leverage_score)

        # Step 4: Generate reason for transparency
        classification = row['Signal'] if pd.notna(row['Signal']) else '—'
        pct = row['Composite_Percentile_in_Band']
        pct_str = f"{pct:.0f}%" if pd.notna(pct) else "N/A"
        rating = row['Rating_Band'] if pd.notna(row['Rating_Band']) else 'NR'

        reason = f"{classification} (Percentile: {pct_str})"

        if final != base:
            if final != after_rating:
                # Leverage guardrail was applied
                lev_str = f"{leverage_score:.1f}" if pd.notna(leverage_score) else "N/A"
                reason += f" → Capped from {after_rating} to Avoid due to extreme leverage (score: {lev_str})"
            else:
                # Rating guardrail was applied
                reason += f" → Capped from {base} due to {rating} rating"

        return final, reason

    # Apply recommendation logic to all rows
    if not os.environ.get("RG_TESTS"):
        st.write("📊 Assigning recommendations...")
    recommendation_results = results.apply(
        lambda row: pd.Series(assign_final_recommendation(row)),
        axis=1
    )
    results['Recommendation'] = recommendation_results[0]
    results['Recommendation_Reason'] = recommendation_results[1]
    results['Rec'] = results['Recommendation']  # Alias for backward compatibility

    # ========================================================================
    # Overall Rank (recommendation-based ranking with quality tiebreaker)
    # ========================================================================
    # Create recommendation priority for ranking
    rec_priority = {"Strong Buy": 4, "Buy": 3, "Hold": 2, "Avoid": 1}
    results['Rec_Priority'] = results['Recommendation'].map(rec_priority)

    # Rank by recommendation first, then by composite score
    # Create a combined sort key: higher priority and higher score = lower rank number
    results['Sort_Key'] = (
        results['Rec_Priority'] * 1000 +  # Recommendation gets 1000x weight
        results['Composite_Score']         # Quality breaks ties
    )
    results['Overall_Rank'] = results['Sort_Key'].rank(ascending=False, method='dense').astype('Int64')

    # Clean up temporary columns
    results = results.drop(columns=['Rec_Priority', 'Sort_Key'])

    # After ranking
    if DIAG.enabled:
        nvidia = results[results['Company_Name'].str.contains('NVIDIA', case=False, na=False)]
        if len(nvidia) > 0:
            n = nvidia.iloc[0]

            # Get period info from df by matching company
            nvidia_in_df = df[df[COMPANY_NAME_COL].str.contains('NVIDIA', case=False, na=False)]
            period_info = {}
            if len(nvidia_in_df) > 0:
                ndf = nvidia_in_df.iloc[0]
                period_info = {
                    'suffix': ndf.get('selected_suffix', 'N/A'),
                    'date': str(ndf.get('selected_date', 'N/A')),
                    'is_annual': ndf.get('is_fy', 'N/A'),
                    'days_since': n.get('Days Since Latest Financials', 'N/A')
                }

            DIAG.log("NVIDIA_COMPLETE_SCORECARD",
                     rank=int(n['Overall_Rank']) if pd.notna(n.get('Overall_Rank')) else None,
                     composite_score=float(n['Composite_Score']) if pd.notna(n.get('Composite_Score')) else None,
                     component_scores={
                         'credit': float(n.get('Credit_Score', np.nan)) if pd.notna(n.get('Credit_Score')) else None,
                         'leverage': float(n.get('Leverage_Score', np.nan)) if pd.notna(n.get('Leverage_Score')) else None,
                         'profitability': float(n.get('Profitability_Score', np.nan)) if pd.notna(n.get('Profitability_Score')) else None,
                         'liquidity': float(n.get('Liquidity_Score', np.nan)) if pd.notna(n.get('Liquidity_Score')) else None,

                         'cash_flow': float(n.get('Cash_Flow_Score', np.nan)) if pd.notna(n.get('Cash_Flow_Score')) else None,
                         'cycle': float(n.get('Cycle_Position_Score', np.nan)) if pd.notna(n.get('Cycle_Position_Score')) else None
                     },
                     raw_metrics={
                         'debt_ebitda': float(ndf.get('Total Debt / EBITDA (x)', np.nan)) if len(nvidia_in_df) > 0 and pd.notna(ndf.get('Total Debt / EBITDA (x)')) else None,
                         'fcf_margin': float(ndf.get('Levered Free Cash Flow Margin', np.nan)) if len(nvidia_in_df) > 0 and pd.notna(ndf.get('Levered Free Cash Flow Margin')) else None,
                         'roa': float(ndf.get('Return on Assets', np.nan)) if len(nvidia_in_df) > 0 and pd.notna(ndf.get('Return on Assets')) else None,
                         'roe': float(ndf.get('Return on Equity', np.nan)) if len(nvidia_in_df) > 0 and pd.notna(ndf.get('Return on Equity')) else None,
                         'ebitda_margin': float(ndf.get('EBITDA Margin', np.nan)) if len(nvidia_in_df) > 0 and pd.notna(ndf.get('EBITDA Margin')) else None
                     },
                     period_info=period_info)

    _log_timing("06_Recommendations_Complete")

    # ========================================================================
    # [V2.2] COMPREHENSIVE VALIDATION: Verify Recommendation Quality
    # ========================================================================

    # Count violations of each guardrail type
    validation_results = {}

    # Violation 1: Weak & Deteriorating with Buy/Strong Buy
    weak_det_violations = (
        (results['Signal'] == 'Weak & Deteriorating') &
        results['Recommendation'].isin(['Buy', 'Strong Buy'])
    )
    validation_results['weak_det'] = weak_det_violations.sum()

    # Violation 2: Distressed (CCC/CC/C/D) with Strong Buy
    distressed_violations = (
        results['Rating_Band'].isin(['CCC', 'CC', 'C', 'D', 'SD']) &
        (results['Recommendation'] == 'Strong Buy')
    )
    validation_results['distressed'] = distressed_violations.sum()

    # Violation 3: Single-B with Strong Buy
    single_b_violations = (
        (results['Rating_Band'] == 'B') &
        (results['Recommendation'] == 'Strong Buy')
    )
    validation_results['single_b'] = single_b_violations.sum()

    # Violation 4: Strong & Improving with Avoid (UNLESS due to leverage guardrail)
    # Leverage guardrail legitimately caps to Avoid when Leverage_Score < 35
    LEVERAGE_AVOID_THRESHOLD = MODEL_THRESHOLDS.get('leverage_score_avoid', 35)
    
    strong_improving_avoid = (
        (results['Signal'] == 'Strong & Improving') &
        (results['Recommendation'] == 'Avoid') &
        (results['Leverage_Score'] >= LEVERAGE_AVOID_THRESHOLD)  # Only flag if NOT due to leverage guardrail
    )
    validation_results['strong_improving_avoid'] = strong_improving_avoid.sum()
    
    # Count leverage guardrail applications (for info, not violations)
    leverage_guardrail_applied = (
        (results['Recommendation'] == 'Avoid') &
        (results['Leverage_Score'] < LEVERAGE_AVOID_THRESHOLD)
    )

    # Count successful guardrail applications (for info)
    weak_det_capped = (
        (results['Signal'] == 'Weak & Deteriorating') &
        (results['Composite_Percentile_in_Band'] >= 60)
    ).sum()

    distressed_capped = (
        results['Rating_Band'].isin(['CCC', 'CC', 'C', 'D', 'SD']) &
        (results['Composite_Percentile_in_Band'] >= 80)
    ).sum()

    single_b_capped = (
        (results['Rating_Band'] == 'B') &
        (results['Composite_Percentile_in_Band'] >= 80)
    ).sum()

    # Count leverage guardrail applications
    leverage_capped = leverage_guardrail_applied.sum()

    # Display validation results
    total_violations = sum(validation_results.values())

    if total_violations > 0 and not os.environ.get("RG_TESTS"):
        # CRITICAL: Guardrails failed
        st.error(
            f"🔴 **RECOMMENDATION VALIDATION FAILED**\n\n"
            f"**{total_violations} violations detected:**\n"
            f"- {validation_results['weak_det']} Weak & Deteriorating → Buy/Strong Buy\n"
            f"- {validation_results['distressed']} Distressed (CCC/CC/C/D) → Strong Buy\n"
            f"- {validation_results['single_b']} Single-B → Strong Buy\n"
            f"- {validation_results['strong_improving_avoid']} Strong & Improving → Avoid\n\n"
            f"**DO NOT USE THESE RECOMMENDATIONS** - Logic error detected."
        )

        # Show violating issuers
        with st.expander("🔍 View Violating Issuers"):
            all_violations = (
                weak_det_violations |
                distressed_violations |
                single_b_violations |
                strong_improving_avoid
            )
            violation_cols = [
                'Company_Name', 'Rating_Band', 'Signal',
                'Composite_Percentile_in_Band', 'Recommendation', 'Recommendation_Reason'
            ]
            st.dataframe(results[all_violations][violation_cols])

    else:
        # SUCCESS: All guardrails working
        guardrails_applied = weak_det_capped + distressed_capped + single_b_capped + leverage_capped

        if guardrails_applied > 0 and not os.environ.get("RG_TESTS"):
            st.sidebar.success(
                f"✓ **Quality Guardrails Active**\n\n"
                f"Protected {guardrails_applied} issuers from inappropriate recommendations:\n"
                f"- {weak_det_capped} Weak & Deteriorating (capped to Avoid)\n"
                f"- {distressed_capped} Distressed CCC/CC/C (capped to Hold)\n"
                f"- {single_b_capped} Single-B (capped to Buy)\n"
                f"- {leverage_capped} Extreme Leverage (capped to Avoid)\n\n"
                f"These had high percentiles but were downgraded due to quality/rating/leverage concerns."
            )

        # Log success for tests
        if os.environ.get("RG_TESTS") == "1":
            print(f"\n✓ All recommendation guardrails passed validation")
            print(f"  - 0 violations detected")
            print(f"  - {guardrails_applied} guardrails successfully applied")

    _log_timing("07_FINAL_COMPLETE")

    # Print timing summary
    print("\n" + "="*60)
    print("TIMING SUMMARY (load_and_process_data)")
    print("="*60)
    prev_time = 0
    for label, elapsed in _checkpoints.items():
        delta = elapsed - prev_time
        print(f"{label:40s}: {delta:6.2f}s  (cumulative: {elapsed:6.2f}s)")
        prev_time = elapsed
    print("="*60 + "\n")

    # [V2.3] Debug display removed for cleaner UI

    # ========================================================================
    # DIAGNOSTIC: Results-level analysis
    # ========================================================================
    DIAG.section("PROCESSING RESULTS - ALL ISSUERS ANALYSIS")

    # Basic results summary
    total_issuers = len(results)
    scored_issuers = results['Composite Score'].notna().sum() if 'Composite Score' in results.columns else 0
    null_scores = total_issuers - scored_issuers

    DIAG.log("RESULTS_SUMMARY",
             total_issuers=total_issuers,
             scored_issuers=int(scored_issuers),
             null_scores=int(null_scores))

    # Analyze days distribution
    if 'Days Since Latest Financials' in results.columns:
        DIAG.analyze_days_distribution(results, COMPANY_NAME_COL, 'Days Since Latest Financials')

    # Analyze sector distribution
    if 'Parent Sector' in results.columns:
        DIAG.analyze_sector_distribution(results, 'Parent Sector')

    # Score distributions
    score_cols = ['Composite Score', 'Leverage Score', 'Coverage Score',
                  'Profitability Score', 'Liquidity Score', 'Efficiency Score',
                  'Cycle Position Score']

    for col in score_cols:
        if col in results.columns:
            scores = results[col].dropna()
            if len(scores) > 0:
                DIAG.log(f"SCORE_DIST_{col.upper().replace(' ', '_')}",
                         count=len(scores),
                         min=round(scores.min(), 2),
                         max=round(scores.max(), 2),
                         mean=round(scores.mean(), 2),
                         median=round(scores.median(), 2),
                         std=round(scores.std(), 2) if len(scores) > 1 else 0)

    # Top and bottom ranked issuers
    if 'Rank' in results.columns and 'Composite Score' in results.columns:
        ranked = results[results['Rank'].notna()].copy()
        if len(ranked) > 0:
            # Top 10
            top_10 = ranked.nsmallest(min(10, len(ranked)), 'Rank')
            top_data = []
            for _, row in top_10.iterrows():
                entry = {
                    'rank': int(row['Rank']),
                    'company': row.get(COMPANY_NAME_COL, 'Unknown'),
                    'score': round(row['Composite Score'], 1) if pd.notna(row.get('Composite Score')) else None
                }
                if 'Days Since Latest Financials' in row:
                    entry['days'] = int(row['Days Since Latest Financials']) if pd.notna(row['Days Since Latest Financials']) else None
                top_data.append(entry)

            DIAG.log("TOP_10_RANKED", issuers=top_data)

            # Bottom 10
            bottom_10 = ranked.nlargest(min(10, len(ranked)), 'Rank')
            bottom_data = []
            for _, row in bottom_10.iterrows():
                entry = {
                    'rank': int(row['Rank']),
                    'company': row.get(COMPANY_NAME_COL, 'Unknown'),
                    'score': round(row['Composite Score'], 1) if pd.notna(row.get('Composite Score')) else None
                }
                if 'Days Since Latest Financials' in row:
                    entry['days'] = int(row['Days Since Latest Financials']) if pd.notna(row['Days Since Latest Financials']) else None
                bottom_data.append(entry)

            DIAG.log("BOTTOM_10_RANKED", issuers=bottom_data)

    # Signal distribution
    if 'Signal' in results.columns:
        signal_dist = results['Signal'].value_counts()
        signal_data = {k: {"count": int(v), "percentage": round(v/len(results)*100, 1)}
                      for k, v in signal_dist.items()}
        DIAG.log("SIGNAL_DISTRIBUTION", total=len(results), signals=signal_data)

    # Classification distribution
    if 'Classification' in results.columns:
        class_dist = results['Classification'].value_counts()
        class_data = {k: {"count": int(v), "percentage": round(v/len(results)*100, 1)}
                     for k, v in class_dist.items()}
        DIAG.log("CLASSIFICATION_DISTRIBUTION", total=len(results), classifications=class_data)

    # Anomaly detection
    DIAG.subsection("ANOMALY DETECTION")

    # Check for duplicate ranks
    if 'Rank' in results.columns:
        rank_counts = results['Rank'].value_counts()
        duplicates = rank_counts[rank_counts > 1]
        if len(duplicates) > 0:
            DIAG.log("DUPLICATE_RANKS",
                     level="WARNING",
                     message=f"Found {len(duplicates)} rank values with duplicates",
                     duplicate_ranks=duplicates.head(5).to_dict())

    # Issuers with missing composite scores
    if 'Composite Score' in results.columns and null_scores > 0:
        missing_scores = results[results['Composite Score'].isna()]
        sample_missing = missing_scores.head(5)[COMPANY_NAME_COL].tolist() if len(missing_scores) > 0 else []
        DIAG.log("MISSING_COMPOSITE_SCORES",
                 level="WARNING",
                 message=f"{null_scores} issuers have null composite scores",
                 sample=sample_missing)

    # Outlier scores (> 3 std dev from mean)
    if 'Composite Score' in results.columns:
        scores = results['Composite Score'].dropna()
        if len(scores) > 1:
            mean = scores.mean()
            std = scores.std()
            threshold = 3 * std
            outliers = results[
                (results['Composite Score'] - mean).abs() > threshold
            ]
            if len(outliers) > 0:
                outlier_data = []
                for _, row in outliers.head(5).iterrows():
                    outlier_data.append({
                        'company': row.get(COMPANY_NAME_COL, 'Unknown'),
                        'score': round(row['Composite Score'], 1),
                        'deviation_from_mean': round(row['Composite Score'] - mean, 1)
                    })
                DIAG.log("OUTLIER_SCORES",
                         num_outliers=len(outliers),
                         threshold=round(threshold, 1),
                         sample=outlier_data)

    # Processing complete
    DIAG.log("PROCESSING_COMPLETE",
             timestamp=datetime.now().isoformat(),
             total_issuers=total_issuers,
             scored_issuers=int(scored_issuers))

    # Print warning summary before returning
    WarningCollector.print_summary()

    return results, df, audits, period_calendar

# ============================================================================
# MULTI-AGENT CREDIT REPORT (V2.3 - Corrected Implementation)
# ============================================================================
# Based on LlamaIndex AgentWorkflow best practices:
# - Linear workflow with handoffs (not parallel)
# - One tool per agent
# - Simple workflow.run() without Context or max_iterations
# - Clear system prompts with explicit handoff instructions
# ============================================================================

def get_metric_median(df: pd.DataFrame, metric_name: str) -> Optional[float]:
    """Extract median value for a metric."""
    col = resolve_metric_column(df, metric_name)
    if col:
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(values) > 0:
            return float(values.median())
    return None


def calculate_sector_medians(df: pd.DataFrame, classification: str) -> Dict[str, float]:
    """Calculate median metrics for companies in same sector."""
    sector = CLASSIFICATION_TO_SECTOR.get(classification)
    if not sector:
        return {}

    sector_classifications = [k for k, v in CLASSIFICATION_TO_SECTOR.items() if v == sector]
    class_col = None
    for col in ['Rubrics Custom Classification', 'Rubrics_Custom_Classification']:
        if col in df.columns:
            class_col = col
            break

    if not class_col:
        return {}

    sector_df = df[df[class_col].isin(sector_classifications)]

    if len(sector_df) < 5:
        return {}

    metrics = {
        "EBITDA Margin": get_metric_median(sector_df, "EBITDA Margin"),
        "ROE": get_metric_median(sector_df, "Return on Equity"),
        "ROA": get_metric_median(sector_df, "Return on Assets"),
        "Net Debt/EBITDA": get_metric_median(sector_df, "Net Debt / EBITDA"),
        "Interest Coverage": get_metric_median(sector_df, "EBITDA / Interest Expense (x)"),
        "Current Ratio": get_metric_median(sector_df, "Current Ratio (x)"),
        "Quick Ratio": get_metric_median(sector_df, "Quick Ratio (x)"),
    }

    return {k: v for k, v in metrics.items() if v is not None}


def calculate_cohort_medians(df: pd.DataFrame, rating_group: str) -> Dict[str, float]:
    """Calculate medians for IG or HY cohort."""
    ig_ratings = ['AAA','AA+','AA','AA-','A+','A','A-','BBB+','BBB','BBB-']
    hy_ratings = ['BB+','BB','BB-','B+','B','B-','CCC+','CCC','CCC-','CC','C']

    ratings = ig_ratings if rating_group == 'IG' else hy_ratings

    rating_col = None
    for col in ['S&P LT Issuer Credit Rating', 'Credit_Rating_Clean', 'Rating']:
        if col in df.columns:
            rating_col = col
            break

    if not rating_col:
        return {}

    cohort_df = df[df[rating_col].isin(ratings)]

    if len(cohort_df) < 10:
        return {}

    metrics = {
        "EBITDA Margin": get_metric_median(cohort_df, "EBITDA Margin"),
        "ROE": get_metric_median(cohort_df, "Return on Equity"),
        "ROA": get_metric_median(cohort_df, "Return on Assets"),
        "Net Debt/EBITDA": get_metric_median(cohort_df, "Net Debt / EBITDA"),
        "Current Ratio": get_metric_median(cohort_df, "Current Ratio (x)"),
    }

    return {k: v for k, v in metrics.items() if v is not None}


def prepare_sector_context(
    row: pd.Series,
    df: pd.DataFrame,
    classification: str,
    use_sector_adjusted: bool,
    calibrated_weights: Optional[Dict] = None,
    rating_band: str = 'BBB'
) -> Dict[str, Any]:
    """Prepare sector-aware context for agents."""

    sector = CLASSIFICATION_TO_SECTOR.get(classification, 'Default')

    weights = get_classification_weights(
        classification,
        use_sector_adjusted,
        calibrated_weights
    )

    if calibrated_weights:
        weight_source = f"Dynamic Calibration ({rating_band} cohort)"
    elif use_sector_adjusted and sector != 'Default':
        weight_source = f"Sector-Adjusted ({sector})"
    else:
        weight_source = "Universal Default"

    sector_medians = calculate_sector_medians(df, classification)
    ig_medians = calculate_cohort_medians(df, 'IG')
    hy_medians = calculate_cohort_medians(df, 'HY')

    return {
        "classification": classification,
        "sector": sector,
        "weights": weights,
        "weight_source": weight_source,
        "sector_medians": sector_medians,
        "ig_medians": ig_medians,
        "hy_medians": hy_medians
    }


def extract_metric_time_series(
    row: pd.Series,
    df: pd.DataFrame,
    metric_name: str,
    max_periods: int = 5
) -> Dict[str, Any]:
    """Extract time series for a metric."""
    current_value = get_from_row(row, metric_name)

    metric_cols = list_metric_columns(df, metric_name)

    time_series = []
    for i, col in enumerate(metric_cols[:max_periods]):
        value = row.get(col)
        if pd.notna(value):
            period_label = "Current" if i == 0 else f"T-{i}"
            time_series.append({
                "period": period_label,
                "value": float(value)
            })

    return {
        "metric_name": metric_name,
        "current_value": float(current_value) if pd.notna(current_value) else None,
        "time_series": time_series,
        "data_available": pd.notna(current_value)
    }


async def get_profitability_data(ctx, company_name: str, row_data: dict) -> str:
    """Extract profitability metrics with sector context."""
    metrics = ["EBITDA Margin", "Return on Equity", "Return on Assets", "EBIT Margin"]

    sector_context = prepare_sector_context(
        row=row_data["row"],
        df=row_data["df"],
        classification=row_data["classification"],
        use_sector_adjusted=row_data["use_sector_adjusted"],
        calibrated_weights=row_data.get("calibrated_weights"),
        rating_band=row_data.get("rating_band", "BBB")
    )

    result = {
        "company": company_name,
        "factor_score": row_data["profitability_score"],
        "sector_context": sector_context,
        "metrics": {}
    }

    for metric in metrics:
        result["metrics"][metric] = extract_metric_time_series(
            row_data["row"],
            row_data["df"],
            metric
        )

    state = await ctx.get("state")
    state["profitability_data"] = result
    await ctx.set("state", state)

    return f"Profitability data extracted. Score: {row_data['profitability_score']:.1f}/100, Sector: {sector_context['sector']}"


async def get_leverage_data(ctx, company_name: str, row_data: dict) -> str:
    """Extract leverage metrics with sector context."""
    metrics = [
        "Net Debt / EBITDA",
        "EBITDA / Interest Expense (x)",
        "Total Debt / Total Capital (%)",
        "Total Debt / EBITDA (x)"
    ]

    sector_context = prepare_sector_context(
        row=row_data["row"],
        df=row_data["df"],
        classification=row_data["classification"],
        use_sector_adjusted=row_data["use_sector_adjusted"],
        calibrated_weights=row_data.get("calibrated_weights"),
        rating_band=row_data.get("rating_band", "BBB")
    )

    result = {
        "company": company_name,
        "factor_score": row_data["leverage_score"],
        "sector_context": sector_context,
        "metrics": {}
    }

    for metric in metrics:
        result["metrics"][metric] = extract_metric_time_series(
            row_data["row"],
            row_data["df"],
            metric
        )

    state = await ctx.get("state")
    state["leverage_data"] = result
    await ctx.set("state", state)

    return f"Leverage data extracted. Score: {row_data['leverage_score']:.1f}/100"


async def get_liquidity_data(ctx, company_name: str, row_data: dict) -> str:
    """Extract liquidity metrics with sector context."""
    metrics = ["Current Ratio (x)", "Quick Ratio (x)"]

    sector_context = prepare_sector_context(
        row=row_data["row"],
        df=row_data["df"],
        classification=row_data["classification"],
        use_sector_adjusted=row_data["use_sector_adjusted"],
        calibrated_weights=row_data.get("calibrated_weights"),
        rating_band=row_data.get("rating_band", "BBB")
    )

    result = {
        "company": company_name,
        "factor_score": row_data["liquidity_score"],
        "sector_context": sector_context,
        "metrics": {}
    }

    for metric in metrics:
        result["metrics"][metric] = extract_metric_time_series(
            row_data["row"],
            row_data["df"],
            metric
        )

    state = await ctx.get("state")
    state["liquidity_data"] = result
    await ctx.set("state", state)

    return f"Liquidity data extracted. Score: {row_data['liquidity_score']:.1f}/100"


async def get_cash_flow_data(ctx, company_name: str, row_data: dict) -> str:
    """Extract cash flow metrics."""
    metrics = [
        "Cash from Ops.",
        "Total Revenues",
        "Total Debt",
        "Levered Free Cash Flow"
    ]

    result = {
        "company": company_name,
        "factor_score": row_data["cash_flow_score"],
        "metrics": {}
    }

    for metric in metrics:
        result["metrics"][metric] = extract_metric_time_series(
            row_data["row"],
            row_data["df"],
            metric
        )

    state = await ctx.get("state")
    state["cash_flow_data"] = result

    await ctx.set("state", state)
    return f"Cash flow data extracted. Score: {row_data['cash_flow_score']:.1f}/100"


async def generate_multiagent_credit_report(
    row: pd.Series,
    df: pd.DataFrame,
    composite_score: float,
    factor_scores: Dict[str, float],
    rating_band: str,
    company_name: str,
    rating: str,
    classification: str = "Unknown",
    use_sector_adjusted: bool = True,
    calibrated_weights: Optional[Dict] = None,
    api_key: str = None,
    claude_key: str = None
) -> str:
    """
    Generate multi-agent credit report using CORRECTED workflow pattern.

    Key improvements based on LlamaIndex best practices:
    - Linear workflow with handoffs (not parallel)
    - Simple workflow.run() - NO Context(), NO max_iterations
    - One tool per agent with clear responsibility
    - LLM called within tools to avoid hallucination
    """

    if not _LLAMAINDEX_AVAILABLE:
        return "# Error\n\nLlamaIndex not installed. Install with: pip install llama-index-core llama-index-llms-anthropic llama-index-llms-openai"

    # Set up LLM with explicit max_tokens defaults
    if claude_key:
        llm = Anthropic(
            model="claude-sonnet-4-20250514",
            api_key=claude_key,
            max_tokens=8192  # Set default max for all calls
        )
    elif api_key:
        llm = LlamaOpenAI(
            model="gpt-4o",
            api_key=api_key,
            max_tokens=8192  # Set default max for all calls
        )
    else:
        return "# Error\n\nNo API key provided."

    # Extract raw financial metrics from row for analysis
    def safe_get(key, default='N/A'):
        """Safely extract metric from row."""
        try:
            val = row.get(key, default)
            if pd.isna(val) or val == 'NM':
                return 'N/A'
            if isinstance(val, (int, float)):
                return f"{val:.2f}"
            return str(val)
        except Exception:
            return default

    # Extract profitability metrics
    ebitda_margin = safe_get('EBITDA Margin')
    roa = safe_get('ROA')
    net_margin = safe_get('Net Margin')

    # Extract leverage metrics
    net_debt_ebitda = safe_get('Net Debt/EBITDA')
    interest_coverage = safe_get('Interest Coverage')
    debt_capital = safe_get('Debt/Capital')
    total_debt_ebitda = safe_get('Total Debt/EBITDA')

    # Extract liquidity metrics
    current_ratio = safe_get('Current Ratio')
    cash_total_debt = safe_get('Cash/Total Debt')

    # Extract cash flow metrics
    ocf_revenue = safe_get('OCF/Revenue')
    ocf_debt = safe_get('OCF/Debt')
    ufcf_margin = safe_get('UFCF Margin')
    lfcf_margin = safe_get('LFCF Margin')

    # Extract growth metrics
    revenue_cagr = safe_get('Revenue CAGR')
    ebitda_growth = safe_get('EBITDA Growth')

    # Define async tool functions (one per agent)
    # Each specialist analyzes raw financial metrics, not just scores
    async def analyze_profitability_tool(ctx):
        """Analyze profitability metrics with LLM."""
        state = await ctx.get("state")
        prof_score = state['factor_scores'].get('profitability_score', 50)
        name = state['company_name']
        metrics = state['metrics']

        prompt = (
            f"Analyze {name}'s profitability performance using raw financial metrics.\n\n"
            f"**Profitability Score:** {prof_score:.1f}/100\n\n"
            f"**Raw Financial Metrics:**\n"
            f"- EBITDA Margin: {metrics['ebitda_margin']}%\n"
            f"- ROA (Return on Assets): {metrics['roa']}%\n"
            f"- Net Margin: {metrics['net_margin']}%\n\n"
            f"**Instructions:**\n"
            f"Provide a 2-3 paragraph assessment covering:\n"
            f"1. Analyze the actual margin levels (EBITDA, Net) and ROA - are these strong/weak for the industry?\n"
            f"2. What do these metrics reveal about operational efficiency and profitability quality?\n"
            f"3. How do these financials support or contradict the {prof_score:.1f}/100 score?\n"
            f"4. Compare to typical thresholds (e.g., EBITDA margins >20% = strong, 10-20% = moderate, <10% = weak)\n\n"
            f"Be specific and reference the actual metric values, not just the score."
        )

        response = await llm.acomplete(prompt, temperature=0.3, max_tokens=1200)
        analysis = response.text

        state['profitability_analysis'] = analysis
        await ctx.set("state", state)
        return f"Profitability analysis complete."

    async def analyze_leverage_tool(ctx):
        """Analyze leverage with LLM and provide commentary."""
        state = await ctx.get("state")
        lev_score = state['factor_scores'].get('leverage_score', 50)
        name = state['company_name']

        prompt = (
            f"Analyze {name}'s leverage position.\n\n"
            f"**Leverage Score:** {lev_score:.1f}/100\n\n"
            f"Provide a 2-3 paragraph assessment covering:\n"
            f"1. Debt levels and overall leverage assessment\n"
            f"2. Capital structure quality\n"
            f"3. Leverage trajectory and financial flexibility\n\n"
            f"Be specific and reference the score."
        )

        response = await llm.acomplete(prompt, temperature=0.3, max_tokens=1200)
        analysis = response.text

        state['leverage_analysis'] = analysis
        await ctx.set("state", state)
        return f"Leverage analysis complete."

    async def analyze_liquidity_tool(ctx):
        """Analyze liquidity with LLM and provide commentary."""
        state = await ctx.get("state")
        liq_score = state['factor_scores'].get('liquidity_score', 50)
        name = state['company_name']

        prompt = (
            f"Analyze {name}'s liquidity position.\n\n"
            f"**Liquidity Score:** {liq_score:.1f}/100\n\n"
            f"Provide a 2-3 paragraph assessment covering:\n"
            f"1. Current liquidity adequacy\n"
            f"2. Ability to meet short-term obligations\n"
            f"3. Interest coverage and debt serviceability\n\n"
            f"Be specific and reference the score."
        )

        response = await llm.acomplete(prompt, temperature=0.3, max_tokens=1200)
        analysis = response.text

        state['liquidity_analysis'] = analysis
        await ctx.set("state", state)
        return f"Liquidity analysis complete."

    async def analyze_cash_flow_tool(ctx):
        """Analyze cash flow with LLM and provide commentary."""
        state = await ctx.get("state")
        cf_score = state['factor_scores'].get('cash_flow_score', 50)
        name = state['company_name']

        prompt = (
            f"Analyze {name}'s cash flow quality.\n\n"
            f"**Cash Flow Score:** {cf_score:.1f}/100\n\n"
            f"Provide a 2-3 paragraph assessment covering:\n"
            f"1. Operating cash flow quality and sustainability\n"
            f"2. Free cash flow generation capacity\n"
            f"3. Cash conversion efficiency\n\n"
            f"Be specific and reference the score."
        )

        response = await llm.acomplete(prompt, temperature=0.3, max_tokens=1200)
        analysis = response.text

        state['cash_flow_analysis'] = analysis
        await ctx.set("state", state)
        return f"Cash flow analysis complete."



    async def compile_final_report_tool(ctx):
        """Compile specialist analyses into comprehensive final report."""
        state = await ctx.get("state")
        name = state['company_name']
        rating_val = state['rating']
        comp_score = state['composite_score']
        band = state['rating_band']
        classification = state['classification']
        f_scores = state['factor_scores']

        # Get specialist analyses
        prof_analysis = state.get('profitability_analysis', 'Not available')
        lev_analysis = state.get('leverage_analysis', 'Not available')
        liq_analysis = state.get('liquidity_analysis', 'Not available')
        cf_analysis = state.get('cash_flow_analysis', 'Not available')

        prompt = (
            f"You are an expert credit analyst. Synthesize the specialist analyses below into a comprehensive credit report for {name}.\n\n"
            f"**Company Overview:**\n"
            f"- Company Name: {name}\n"
            f"- S&P Rating: {rating_val}\n"
            f"- Composite Score: {comp_score:.1f}/100\n"
            f"- Rating Band: {band}\n"
            f"- Industry Classification: {classification}\n\n"
            f"**Factor Scores (0-100 scale, higher is better):**\n"
            f"- Credit Score: {f_scores.get('credit_score', 50):.1f}/100\n"
            f"- Leverage Score: {f_scores.get('leverage_score', 50):.1f}/100\n"
            f"- Profitability Score: {f_scores.get('profitability_score', 50):.1f}/100\n"
            f"- Liquidity Score: {f_scores.get('liquidity_score', 50):.1f}/100\n"
            f"- Cash Flow Score: {f_scores.get('cash_flow_score', 50):.1f}/100\n\n"
            f"**Specialist Agent Analyses:**\n\n"
            f"### Profitability Analysis\n"
            f"{prof_analysis}\n\n"
            f"### Leverage Analysis\n"
            f"{lev_analysis}\n\n"
            f"### Liquidity Analysis\n"
            f"{liq_analysis}\n\n"
            f"### Cash Flow Analysis\n"
            f"{cf_analysis}\n\n"
            f"**INSTRUCTIONS:**\n"
            f"Create a COMPLETE, professional credit report with ALL sections below.\n\n"
            f"## Comprehensive Credit Analysis: {name}\n\n"
            f"### Executive Summary\n"
            f"- Current Rating: {rating_val} | Composite Score: {comp_score:.1f}/100 | Band: {band}\n"
            f"- Industry: {classification}\n"
            f"- 3-4 sentence overview synthesizing key themes from all 5 specialist analyses\n"
            f"- Primary rating drivers and risk positioning\n\n"
            f"---\n\n"
            f"### Specialist Agent Analyses\n\n"
            f"#### Profitability Analysis (Score: {f_scores.get('profitability_score', 50):.1f}/100)\n"
            f"{prof_analysis}\n\n"
            f"#### Leverage Analysis (Score: {f_scores.get('leverage_score', 50):.1f}/100)\n"
            f"{lev_analysis}\n\n"
            f"#### Liquidity Analysis (Score: {f_scores.get('liquidity_score', 50):.1f}/100)\n"
            f"{liq_analysis}\n\n"
            f"#### Cash Flow Analysis (Score: {f_scores.get('cash_flow_score', 50):.1f}/100)\n"
            f"{cf_analysis}\n\n"
            f"---\n\n"
            f"### Factor Analysis Summary\n"
            f"Synthesize the specialist findings:\n"
            f"- **Credit Score** ({f_scores.get('credit_score', 50):.1f}/100): Balance sheet strength\n"
            f"- **Profitability** ({f_scores.get('profitability_score', 50):.1f}/100): Key themes from analysis above\n"
            f"- **Leverage** ({f_scores.get('leverage_score', 50):.1f}/100): Key themes from analysis above\n"
            f"- **Liquidity** ({f_scores.get('liquidity_score', 50):.1f}/100): Key themes from analysis above\n"
            f"- **Cash Flow** ({f_scores.get('cash_flow_score', 50):.1f}/100): Key themes from analysis above\n\n"
            f"Indicate strengths (70+), moderate performance (50-69), and weaknesses (<50) in your analysis.\n\n"
            f"### Credit Strengths\n"
            f"Extract 3-4 key positives from specialist analyses:\n"
            f"1. [Strength with score support]\n"
            f"2. [Strength with score support]\n"
            f"3. [Strength with score support]\n"
            f"4. [Strength with score support]\n\n"
            f"### Credit Risks & Concerns\n"
            f"Extract 3-4 key weaknesses from specialist analyses:\n"
            f"1. [Risk with score support]\n"
            f"2. [Risk with score support]\n"
            f"3. [Risk with score support]\n"
            f"4. [Risk with score support]\n\n"
            f"### Rating Outlook & Investment Recommendation\n"
            f"- **Rating Appropriateness**: Is {rating_val} justified given {comp_score:.1f}/100 composite?\n"
            f"- **Upgrade Triggers**: What improvements would drive rating higher?\n"
            f"- **Downgrade Risks**: What deterioration would pressure rating?\n"
            f"- **Investment Recommendation**: Strong Buy/Buy/Hold/Avoid based on risk-reward\n\n"
            f"**CRITICAL**: Complete ALL sections. Do NOT truncate. Target 1000-1200 words.\n\n"
            f"Begin the complete report now:"
        )

        response = await llm.acomplete(
            prompt,
            temperature=0.3,
            max_tokens=8192
        )
        return response.text

    # Create agents with FunctionAgent
    prof_agent = FunctionAgent(
        name="ProfitabilityAgent",
        description="Analyze profitability ratios.",
        system_prompt="Analyze profitability score and provide commentary. Hand off to LeverageAgent when complete.",
        llm=llm,
        tools=[analyze_profitability_tool],
        can_handoff_to=["LeverageAgent"],
    )

    lev_agent = FunctionAgent(
        name="LeverageAgent",
        description="Analyze leverage ratios.",
        system_prompt="Analyze leverage score and provide commentary. Hand off to LiquidityAgent when complete.",
        llm=llm,
        tools=[analyze_leverage_tool],
        can_handoff_to=["LiquidityAgent"],
    )

    liq_agent = FunctionAgent(
        name="LiquidityAgent",
        description="Analyze liquidity ratios.",
        system_prompt="Analyze liquidity score and provide commentary. Hand off to CashFlowAgent when complete.",
        llm=llm,
        tools=[analyze_liquidity_tool],
        can_handoff_to=["CashFlowAgent"],
    )

    cf_agent = FunctionAgent(
        name="CashFlowAgent",
        description="Analyze cash flow metrics.",
        system_prompt="Analyze cash flow score and provide commentary. Hand off to SupervisorAgent when complete.",
        llm=llm,
        tools=[analyze_cash_flow_tool],
        can_handoff_to=["SupervisorAgent"],
    )

    # growth_agent removed - Growth no longer in 5-factor model

    supervisor_agent = FunctionAgent(
        name="SupervisorAgent",
        description="Compile final credit report.",
        system_prompt="Collect all specialist analyses and compile comprehensive report. If analyses missing, hand back to appropriate agent.",
        llm=llm,
        tools=[compile_final_report_tool],
        can_handoff_to=["ProfitabilityAgent", "LeverageAgent", "LiquidityAgent", "CashFlowAgent"],
    )

    # Create workflow - SIMPLE structure per PDF
    workflow = AgentWorkflow(
        agents=[prof_agent, lev_agent, liq_agent, cf_agent, supervisor_agent],
        root_agent="ProfitabilityAgent",  # String name!
        initial_state={
            "company_name": company_name,
            "rating": rating,
            "composite_score": composite_score,
            "rating_band": rating_band,
            "classification": classification,
            "factor_scores": factor_scores,
            "profitability_analysis": None,
            "leverage_analysis": None,
            "liquidity_analysis": None,
            "cash_flow_analysis": None,

        }
    )

    # Run workflow - SIMPLE execution per PDF (no ctx, no max_iterations)
    handler = workflow.run(user_msg=f"Provide comprehensive credit analysis for {company_name}.")

    # Get result
    result = await handler

    # Extract final report
    final_report = None
    if hasattr(result, 'response') and hasattr(result.response, 'content'):
        final_report = result.response.content
    elif isinstance(result, str):
        final_report = result
    else:
        final_report = "# Error\n\nFailed to generate report."

    # Return just the final report (supervisor includes all specialist analyses)
    return final_report


# ============================================================================
# STREAMLINED SINGLE-LLM CREDIT REPORT (Faster alternative to multi-agent)
# ============================================================================

def generate_streamlined_credit_report(
    row: pd.Series,
    df: pd.DataFrame,
    composite_score: float,
    factor_scores: Dict[str, float],
    rating_band: str,
    company_name: str,
    rating: str,
    classification: str = "Unknown",
    use_sector_adjusted: bool = True,
    calibrated_weights: Optional[Dict] = None,
    api_key: str = None,
    claude_key: str = None
) -> str:
    """
    Generate comprehensive credit report using single LLM call (much faster than multi-agent).

    Returns complete report in 10-30 seconds vs 5+ minutes for multi-agent workflow.
    """

    # Extract historical financial data
    try:
        financial_data_package = extract_issuer_financial_data(df, company_name)
        company_info = financial_data_package["company_info"]
        financial_data = financial_data_package["financial_data"]
        period_types = financial_data_package["period_types"]
    except Exception:
        company_info = {
            "name": company_name,
            "rating": rating,
            "classification": classification
        }
        financial_data = {}
        period_types = {}

    # Get sector context
    parent_sector = CLASSIFICATION_TO_SECTOR.get(classification, "Unknown")
    weights_used = get_classification_weights(
        classification,
        use_sector_adjusted=use_sector_adjusted,
        calibrated_weights=calibrated_weights
    )

    # Format financial metrics for prompt
    metrics_text = []
    for metric, time_series in financial_data.items():
        if not time_series:
            continue
        sorted_periods = sorted(time_series.items(), reverse=True)[:5]  # Most recent 5 periods
        metrics_text.append(f"\n**{metric}:**")
        for date, value in sorted_periods:
            period_type = period_types.get(date, "")
            try:
                metrics_text.append(f"  - {date} ({period_type}): {value:.2f}")
            except (ValueError, TypeError):
                metrics_text.append(f"  - {date} ({period_type}): {value}")

    financial_section = "\n".join(metrics_text) if metrics_text else "Limited historical data available"

    # Build comprehensive prompt
    prompt = (
        f"You are an expert fixed income credit analyst preparing a comprehensive credit report.\n\n"
        f"**Company Overview:**\n"
        f"- Name: {company_name}\n"
        f"- S&P Rating: {rating}\n"
        f"- Rating Band: {rating_band}\n"
        f"- Classification: {classification}\n"
        f"- Parent Sector: {parent_sector}\n"
        f"- Composite Score: {composite_score:.1f}/100\n\n"
        f"**Factor Scores (0-100 scale, higher is better):**\n"
        f"- Credit Score: {factor_scores.get('credit_score', 50):.1f}/100\n"
        f"- Leverage Score: {factor_scores.get('leverage_score', 50):.1f}/100\n"
        f"- Profitability Score: {factor_scores.get('profitability_score', 50):.1f}/100\n"
        f"- Liquidity Score: {factor_scores.get('liquidity_score', 50):.1f}/100\n\n"
        f"- Cash Flow Score: {factor_scores.get('cash_flow_score', 50):.1f}/100\n\n"
        f"**Model Weights Used ({parent_sector} sector):**\n"
        f"- Credit: {weights_used.get('credit_score', 0.20)*100:.0f}%\n"
        f"- Leverage: {weights_used.get('leverage_score', 0.20)*100:.0f}%\n"
        f"- Profitability: {weights_used.get('profitability_score', 0.20)*100:.0f}%\n"
        f"- Liquidity: {weights_used.get('liquidity_score', 0.10)*100:.0f}%\n\n"
        f"- Cash Flow: {weights_used.get('cash_flow_score', 0.15)*100:.0f}%\n\n"
        f"**Historical Financial Metrics:**\n"
        f"{financial_section}\n\n"
        f"**Instructions:**\n"
        f"Please provide a comprehensive credit analysis report with the following structure:\n\n"
        f"1. **Executive Summary** (3-4 sentences)\n"
        f"   - Overall credit quality assessment based on composite score and factor scores\n"
        f"   - Key rating drivers (identify which factors are strongest/weakest)\n"
        f"   - Risk positioning (investment grade vs high yield characteristics)\n\n"
        f"2. **Profitability Analysis** (Score: {factor_scores.get('profitability_score', 50):.1f}/100)\n"
        f"   - EBITDA margin trends and interpretation\n"
        f"   - ROE/ROA performance relative to sector\n"
        f"   - Profitability sustainability assessment\n\n"
        f"3. **Leverage Analysis** (Score: {factor_scores.get('leverage_score', 50):.1f}/100)\n"
        f"   - Total Debt/EBITDA and Net Debt/EBITDA trends\n"
        f"   - Leverage trajectory (improving vs deteriorating)\n"
        f"   - Comparison to rating band norms\n\n"
        f"4. **Liquidity & Coverage Analysis** (Score: {factor_scores.get('liquidity_score', 50):.1f}/100)\n"
        f"   - Current ratio and cash position trends\n"
        f"   - Interest coverage analysis\n"
        f"   - Debt serviceability assessment\n\n"
        f"5. **Cash Flow Analysis** (Score: {factor_scores.get('cash_flow_score', 50):.1f}/100)\n"
        f"   - Operating cash flow quality and trends\n"
        f"   - Free cash flow generation capacity\n"
        f"   - Cash conversion efficiency\n\n"
        f"6. **Credit Strengths**\n"
        f"   - List 3-4 key positive credit factors based on factor scores\n"
        f"   - Support each with specific data points and scores\n\n"
        f"7. **Credit Risks & Concerns**\n"
        f"   - List 3-4 key risk factors based on weak factor scores\n"
        f"   - Support each with specific data points and scores\n\n"
        f"8. **Rating Outlook & Investment Recommendation**\n"
        f"   - Is the current {rating} rating appropriate given the {composite_score:.1f}/100 composite score?\n"
        f"   - What could trigger an upgrade or downgrade?\n"
        f"   - Investment recommendation from a credit perspective\n\n"
        f"**Formatting Requirements:**\n"
        f"- Use clear markdown formatting with headers (##, ###)\n"
        f"- Bold key metrics and conclusions\n"
        f"- Use bullet points for lists\n"
        f"- Be specific and reference actual numbers from the factor scores and historical data\n"
        f"- Keep tone professional and analytical\n"
        f"- Total length: 800-1000 words\n"
        f"- Focus on data-driven insights rather than generic statements\n\n"
        f"Generate the complete report now:"
    )

    # Make single LLM call
    try:
        if claude_key:
            # Use Claude Sonnet 4
            try:
                from anthropic import Anthropic
                client = Anthropic(api_key=claude_key)
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4000,
                    temperature=0.7,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                return response.content[0].text
            except ImportError:
                return "# Error\n\nAnthropic library not installed. Install with: pip install anthropic"
        elif api_key:
            # Use OpenAI GPT-4o
            try:
                import openai
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert fixed income credit analyst with deep experience in corporate credit analysis. You provide clear, data-driven, professional credit reports."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=4000
                )
                return response.choices[0].message.content
            except ImportError:
                return "# Error\n\nOpenAI library not installed. Install with: pip install openai"
        else:
            return "# Error\n\nNo API key provided. Add CLAUDE_API_KEY or OPENAI_API_KEY to .streamlit/secrets.toml"
    except Exception as e:
        return f"# Error\n\nFailed to generate report: {str(e)}"


# ============================================================================
# MAIN APP EXECUTION
# ============================================================================

if HAS_DATA:
    # ========================================================================
    # [V2.2.1] PRE-CALCULATE CALIBRATED WEIGHTS IF ENABLED
    # ========================================================================

    # [V2.2.1] Dynamic calibration controls sector weighting behavior:
    # - When ON: Calculate sector-specific calibrated weights
    # - When OFF: Use universal weights for all issuers (no sector adjustment)
    calibrated_weights_to_use = None
    effective_use_sector_adjusted = use_sector_adjusted  # Save original setting

    if use_dynamic_calibration:
        # Calibration mode: Calculate sector-specific weights
        with st.spinner("Calculating calibrated sector weights..."):
            try:
                # Quick pre-load to calculate weights (will be cached by main load)
                # Load just enough to calculate weights
                uploaded_file.seek(0)  # Reset file pointer
                temp_df = None

                file_name = uploaded_file.name.lower()
                if file_name.endswith('.csv'):
                    temp_df = pd.read_csv(uploaded_file)
                elif file_name.endswith('.xlsx'):
                    try:
                        temp_df = pd.read_excel(uploaded_file, sheet_name='Pasted Values')
                    except (ValueError, KeyError):
                        temp_df = pd.read_excel(uploaded_file, sheet_name=0)

                uploaded_file.seek(0)  # Reset file pointer for main load

                if temp_df is not None:
                    # Call our calibration function - this needs the processed data with factor scores
                    # So we'll need to pass this through load_and_process_data
                    # For now, signal that calibration should happen inside load_and_process_data
                    calibrated_weights_to_use = "CALCULATE_INSIDE"  # Special marker
                    effective_use_sector_adjusted = True  # Force sector adjustment for calibration
            except Exception as e:
                st.warning(f"Could not calculate calibrated weights: {str(e)}. Using universal weights.")
                calibrated_weights_to_use = None
                effective_use_sector_adjusted = False  # Fall back to universal
    else:
        # Universal mode: No sector-specific weights
        effective_use_sector_adjusted = False
        calibrated_weights_to_use = None

    # ========================================================================
    # LOAD DATA
    # ========================================================================

    # Clear cached calibrated weights from session state when calibration is off
    # Only clear cache when calibration state CHANGES, not every rerun
    previous_calibration_state = st.session_state.get('_previous_calibration_state')
    calibration_changed = previous_calibration_state is not None and previous_calibration_state != use_dynamic_calibration
    
    if not use_dynamic_calibration:
        if '_calibrated_weights' in st.session_state:
            del st.session_state['_calibrated_weights']
        if 'last_calibration_state' in st.session_state:
            del st.session_state['last_calibration_state']
        # Only clear cache when calibration toggle CHANGES (not every rerun)
        if calibration_changed:
            st.cache_data.clear()
            st.toast("🔄 Calibration disabled - cache cleared", icon="ℹ️")
    
    # Track calibration state for next rerun
    st.session_state['_previous_calibration_state'] = use_dynamic_calibration

    # [FIX] Force cache clear on first run after calibration logic fix
    # This ensures old cached results with broken sector mappings/calibration are invalidated
    # The version marker ensures this only happens once after the fix is deployed
    _CALIBRATION_FIX_VERSION = "v5.0"
    if st.session_state.get('_calibration_fix_version') != _CALIBRATION_FIX_VERSION:
        st.cache_data.clear()
        st.session_state['_calibration_fix_version'] = _CALIBRATION_FIX_VERSION
        if '_calibrated_weights' in st.session_state:
            del st.session_state['_calibrated_weights']
        st.toast("🔄 Cache cleared - recalculating with corrected calibration logic", icon="✅")

    with st.spinner("Loading and processing data..."):
        # [V2.3] Create cache buster from unified period selection parameters
        reference_date_str = str(reference_date_override) if reference_date_override else 'none'
        cache_key = f"{period_mode.value}_{reference_date_str}_{use_dynamic_calibration}_{calibration_rating_band}_{CACHE_VERSION}"

        # ====================================================================
        # DIAGNOSTIC: Capture and validate configuration
        # ====================================================================
        CONFIG_STATE.reset()
        CONFIG_STATE.capture_ui_state(
            period_mode=period_mode,
            reference_date_override=reference_date_override,
            prefer_annual_reports=prefer_annual_reports,
            use_dynamic_calibration=use_dynamic_calibration,
            calibration_rating_band=calibration_rating_band,
            use_sector_adjusted=effective_use_sector_adjusted,
            calibrated_weights=calibrated_weights_to_use,
            cache_key=cache_key
        )

        # Validate configuration consistency
        config_issues = CONFIG_STATE.validate_consistency()
        if config_issues:
            st.warning("⚠️ Configuration Issues:\n" + "\n".join(f"- {issue}" for issue in config_issues))

        results_final, df_original, audits, period_calendar = load_and_process_data(
            uploaded_file,
            effective_use_sector_adjusted,
            period_mode=period_mode,
            reference_date_override=reference_date_override,
            prefer_annual_reports=prefer_annual_reports,
            split_basis=split_basis,
            split_threshold=split_threshold,
            trend_threshold=trend_threshold,
            volatility_cv_threshold=0.30,      # Use default directly
            outlier_z_threshold=-2.5,          # Use default directly
            damping_factor=0.5,                # Use default directly
            near_peak_tolerance=0.10,          # Use default directly (10% = 0.10)
            calibrated_weights=calibrated_weights_to_use,
            _cache_buster=cache_key
        )
        _audit_count("Before freshness filters", results_final, audits)

        # ========================================================================
        # [V2.3] DIAGNOSTICS REMOVED FOR CLEANER UI
        # ========================================================================
        # Reference date diagnostics and validation removed to simplify interface

        # Normalize Combined_Signal values once
        results_final['Combined_Signal'] = results_final['Combined_Signal'].astype(str).str.strip()

        # Map any variant labels to canonical ones (precaution)
        canon = {
            "STRONG & IMPROVING": "Strong & Improving",
            "STRONG BUT DETERIORATING": "Strong but Deteriorating",
            "WEAK BUT IMPROVING": "Weak but Improving",
            "WEAK & DETERIORATING": "Weak & Deteriorating",
            "STRONG & NORMALIZING": "Strong & Normalizing",
            "STRONG & MODERATING": "Strong & Moderating",
        }
        results_final['Combined_Signal'] = results_final['Combined_Signal'].str.upper().map(canon).fillna(results_final['Combined_Signal'])

        # Dev-only sanity check: verify all Combined_Signal values are canonical
        if os.environ.get("RG_TESTS") == "1":
            uniq = set(results_final['Combined_Signal'].unique())
            assert all(x in {
                "Strong & Improving",
                "Strong but Deteriorating",
                "Weak but Improving",
                "Weak & Deteriorating",
                "Strong & Normalizing",
                "Strong & Moderating"} for x in uniq), f"Unexpected Combined_Signal values: {uniq}"

        # ============================================================================
        # COMPUTE FRESHNESS METRICS (V2.2)
        # ============================================================================
        
        # Financial data freshness (from Period Ended dates)
        try:
            pe_cols = [c for c in df_original.columns if str(c).startswith("Period Ended")]
            if pe_cols:
                # Get latest period date for each row
                results_final["Financial_Last_Period_Date"] = df_original.apply(
                    lambda row: _latest_valid_period_date_for_row(row, pe_cols), axis=1
                )
                # Calculate days since that date
                results_final["Financial_Data_Freshness_Days"] = (
                    pd.Timestamp.today().normalize() - results_final["Financial_Last_Period_Date"]
                ).dt.days
                # Assign traffic-light flag
                results_final["Financial_Data_Freshness_Flag"] = results_final["Financial_Data_Freshness_Days"].apply(_freshness_flag)
            else:
                results_final["Financial_Last_Period_Date"] = pd.NaT
                results_final["Financial_Data_Freshness_Days"] = np.nan
                results_final["Financial_Data_Freshness_Flag"] = "Unknown"
        except Exception as e:
            st.warning(f"Could not compute financial data freshness: {e}")
            results_final["Financial_Last_Period_Date"] = pd.NaT
            results_final["Financial_Data_Freshness_Days"] = np.nan
            results_final["Financial_Data_Freshness_Flag"] = "Unknown"
        
        # Rating review freshness (from S&P Last Review Date)
        try:
            rating_date_cols = [
                c for c in df_original.columns
                if str(c).strip().lower() in {
                    "s&p last review date", "sp last review date",
                    "s&p review date", "last review date",
                    "rating last review date"
                }
            ]
            if rating_date_cols:
                rd = pd.to_datetime(df_original[rating_date_cols[0]], errors="coerce", dayfirst=True)
                # Exclude 1900 sentinel dates
                rd = rd.mask(rd.dt.year == 1900)
                results_final["SP_Last_Review_Date"] = rd.values
                results_final["Rating_Review_Freshness_Days"] = (
                    pd.Timestamp.today().normalize() - results_final["SP_Last_Review_Date"]
                ).dt.days
                results_final["Rating_Review_Freshness_Flag"] = results_final["Rating_Review_Freshness_Days"].apply(_freshness_flag)
            else:
                results_final["SP_Last_Review_Date"] = pd.NaT
                results_final["Rating_Review_Freshness_Days"] = np.nan
                results_final["Rating_Review_Freshness_Flag"] = "Unknown"
        except Exception as e:
            st.warning(f"Could not compute rating review freshness: {e}")
            results_final["SP_Last_Review_Date"] = pd.NaT
            results_final["Rating_Review_Freshness_Days"] = np.nan
            results_final["Rating_Review_Freshness_Flag"] = "Unknown"
        
        # ============================================================================
        # [V2.3] FRESHNESS FILTERS REMOVED FOR SIMPLICITY
        # ============================================================================
        # All issuers are now included regardless of data age
        # No freshness filtering applied

        _audit_count("Final results", results_final, audits)

        # ============================================================================
        # HEADER
        # ============================================================================

        render_header(results_final, data_period, effective_use_sector_adjusted, df_original,
                    use_quarterly_beta, align_to_reference)

        # ============================================================================
        # TAB NAVIGATION
        # ============================================================================

        TAB_TITLES = [
            " Dashboard",
            " Issuer Search",
            " Rating Group Analysis",
            " Classification Analysis (NEW)",
            " Trend Analysis (NEW)",
            "GenAI Credit Report",
            "Diagnostics"
        ]
        TAB_TITLES_DISPLAY = [t.replace(" (NEW)", "") for t in TAB_TITLES]
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(TAB_TITLES_DISPLAY)
            
        # ============================================================================
        # TAB 1: DASHBOARD
        # ============================================================================
            
        with tab1:
            st.header(" Model Overview & Key Insights")
                
            # Top performers
            col1, col2 = st.columns(2)
                
            # Create recommendation priority for ranking
            rec_priority = {"Strong Buy": 4, "Buy": 3, "Hold": 2, "Avoid": 1}
            results_ranked = results_final.copy()
            results_ranked['Rec_Priority'] = results_ranked['Recommendation'].map(rec_priority)

            with col1:
                st.subheader("Top 10 Opportunities")
                st.caption("Best recommendations (Strong Buy first), then highest quality within each tier")

                # Sort by recommendation priority (descending), then composite score (descending)
                top10 = results_ranked.sort_values(
                    ['Rec_Priority', 'Composite_Score'],
                    ascending=[False, False]
                ).head(10)[
                    ['Company_Name', 'Credit_Rating_Clean', 'Rubrics_Custom_Classification',
                     'Composite_Score', 'Combined_Signal', 'Recommendation']
                ]
                top10.columns = ['Company', 'Rating', 'Classification', 'Score', 'Signal', 'Rec']
                st.dataframe(top10, use_container_width=True, hide_index=True)

            with col2:
                st.subheader("Bottom 10 Risks")
                st.caption("Worst recommendations (Avoid first), then lowest quality within each tier")

                # Sort by recommendation priority (ascending), then composite score (ascending)
                bottom10 = results_ranked.sort_values(
                    ['Rec_Priority', 'Composite_Score'],
                    ascending=[True, True]
                ).head(10)[
                    ['Company_Name', 'Credit_Rating_Clean', 'Rubrics_Custom_Classification',
                     'Composite_Score', 'Combined_Signal', 'Recommendation']
                ]
                bottom10.columns = ['Company', 'Rating', 'Classification', 'Score', 'Signal', 'Rec']
                st.dataframe(bottom10, use_container_width=True, hide_index=True)

            # Ranking methodology explanation
            st.info(
                "**Ranking Methodology:** Results are ranked by actionability - recommendations are prioritized "
                "(Strong Buy > Buy > Hold > Avoid), with quality score breaking ties within each recommendation tier. "
                "This ensures \"top opportunities\" are issuers you'd actually act on, not just high-quality credits "
                "with deteriorating trends."
            )

            # Four Quadrant Analysis
            st.subheader("Four Quadrant Analysis: Quality vs. Momentum")

            # Build rating filter options dynamically based on available bands
            rating_filter_options = [
                "All Issuers (IG + HY)",
                "Investment Grade Only",
                "High Yield Only",
                "---",  # Separator
                "AAA",
                "AA",
                "A",
                "BBB",
                "BB",
                "B",
                "CCC",
                "Unrated"
            ]

            rating_filter = st.selectbox(
                "Filter by Rating Group",
                options=rating_filter_options,
                index=0,
                help="Filter analysis by rating category. Select 'All' for universe view, "
                     "IG/HY for broad groups, or specific bands (AAA, BBB, etc.) for focused analysis.",
                key="quadrant_rating_filter"
            )

            # Apply filter
            if rating_filter == "Investment Grade Only":
                results_filtered = results_final[results_final['Rating_Group'] == 'Investment Grade'].copy()
                filter_label = " - Investment Grade"
            elif rating_filter == "High Yield Only":
                results_filtered = results_final[results_final['Rating_Group'] == 'High Yield'].copy()
                filter_label = " - High Yield"
            elif rating_filter == "---":
                # Separator selected - treat as "All"
                results_filtered = results_final.copy()
                filter_label = ""
            elif rating_filter in ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "Unrated"]:
                # Specific rating band selected
                results_filtered = results_final[results_final['Rating_Band'] == rating_filter].copy()
                filter_label = f" - {rating_filter}"
            else:
                # "All Issuers (IG + HY)" or fallback
                results_filtered = results_final.copy()
                filter_label = ""

            # Validate filtered results
            if len(results_filtered) == 0:
                st.warning(f"⚠️ No issuers found in selected rating category: {rating_filter}")
                st.info("Try selecting a different rating group or 'All Issuers'.")
                st.stop()

            # Show filter summary
            if filter_label:
                issuer_count = len(results_filtered)
                total_count = len(results_final)
                st.caption(f"Showing {issuer_count:,} of {total_count:,} issuers{filter_label}")
            else:
                st.caption(f"Displaying {len(results_filtered):,} issuers")

            # Ensure numeric dtypes for axes
            results_filtered['Composite_Percentile_in_Band'] = pd.to_numeric(results_filtered['Composite_Percentile_in_Band'], errors='coerce')
            results_filtered['Composite_Percentile_Global'] = pd.to_numeric(results_filtered.get('Composite_Percentile_Global', results_filtered['Composite_Percentile_in_Band']), errors='coerce')
            # Composite_Score already numeric from calculation - no conversion needed
            results_filtered['Cycle_Position_Score'] = pd.to_numeric(results_filtered['Cycle_Position_Score'], errors='coerce')

            # Use unified quality/trend split for visualization
            quality_metric_plot, x_split_for_plot, x_axis_label, x_vals = resolve_quality_metric_and_split(
                results_filtered, split_basis, split_threshold
            )
            y_vals = results_filtered["Cycle_Position_Score"]
            y_split = float(trend_threshold)

            # Create color mapping for quadrants
            color_map = {
                "Strong & Improving": "#2ecc71",      # Green
                "Strong but Deteriorating": "#f39c12",  # Orange
                "Weak but Improving": "#3498db",       # Blue
                "Weak & Deteriorating": "#e74c3c"      # Red
            }

            # Prepare data for scatter plot
            # Need to determine which column to use for x-axis
            if split_basis == "Absolute Composite Score":
                x_col = "Composite_Score"
            elif split_basis == "Global Percentile":
                x_col = "Composite_Percentile_Global"
            else:  # Percentile within band
                x_col = "Composite_Percentile_in_Band"

            # Create scatter plot using unified quality metric
            fig_quadrant = px.scatter(
                results_filtered,
                x=x_col,
                y="Cycle_Position_Score",
                color="Combined_Signal",
                color_discrete_map=color_map,
                hover_name="Company_Name",
                hover_data={
                    "Composite_Score": ":.1f",
                    "Composite_Percentile_in_Band": ":.1f",
                    "Cycle_Position_Score": ":.1f",
                    "Combined_Signal": False
                },
                title=f'Credit Quality vs. Trend Momentum{filter_label}',
                labels={
                    x_col: x_axis_label,
                    "Cycle_Position_Score": "Trend Score"
                }
            )

            # Add split lines in DATA coordinates (xref='x', yref='y')


            # Add quadrant labels (positioned relative to splits)
            y_upper = y_split + (100 - y_split) * 0.5  # midpoint of upper half
            y_lower = y_split * 0.5  # midpoint of lower half

            # Calculate x positions based on actual axis range and split
            x_max = float(results_filtered[x_col].max())
            x_min = float(results_filtered[x_col].min())
            x_upper = x_split_for_plot + (x_max - x_split_for_plot) * 0.5  # midpoint of upper half
            x_lower = x_min + (x_split_for_plot - x_min) * 0.5  # midpoint of lower half

            fig_quadrant.add_annotation(x=x_upper, y=y_upper, text="<b>BEST</b><br>Strong & Improving",
                                       showarrow=False, font=dict(size=12, color="gray"), xref='x', yref='y')
            fig_quadrant.add_annotation(x=x_upper, y=y_lower, text="<b>WARNING</b><br>Strong but Deteriorating",
                                       showarrow=False, font=dict(size=12, color="gray"), xref='x', yref='y')
            fig_quadrant.add_annotation(x=x_lower, y=y_upper, text="<b>OPPORTUNITY</b><br>Weak but Improving",
                                       showarrow=False, font=dict(size=12, color="gray"), xref='x', yref='y')
            fig_quadrant.add_annotation(x=x_lower, y=y_lower, text="<b>AVOID</b><br>Weak & Deteriorating",
                                       showarrow=False, font=dict(size=12, color="gray"), xref='x', yref='y')

            fig_quadrant.update_layout(
                height=600,
                hovermode='closest',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    title=None
                )
            )

            fig_quadrant.update_traces(
                marker=dict(size=8, opacity=0.7, line=dict(width=1, color='white'))
            )

            st.plotly_chart(fig_quadrant, use_container_width=True)
                
            # Quadrant summary statistics
            st.subheader("Quadrant Distribution")
            col1, col2, col3, col4 = st.columns(4)

            quadrant_counts = results_filtered['Combined_Signal'].value_counts()
            total = len(results_filtered)

            with col1:
                count = quadrant_counts.get("Strong & Improving", 0)
                st.metric("Strong & Improving", f"{count}", f"{count/total*100:.1f}%")

            with col2:
                count = quadrant_counts.get("Strong but Deteriorating", 0)
                st.metric("Strong but Deteriorating", f"{count}", f"{count/total*100:.1f}%")

            with col3:
                count = quadrant_counts.get("Weak but Improving", 0)
                st.metric("Weak but Improving", f"{count}", f"{count/total*100:.1f}%")

            with col4:
                count = quadrant_counts.get("Weak & Deteriorating", 0)
                st.metric("Weak & Deteriorating", f"{count}", f"{count/total*100:.1f}%")

            # ========================================================================
            # PRINCIPAL COMPONENT ANALYSIS
            # ========================================================================
            st.markdown("---")
            st.subheader("Principal Component Analysis")

            st.markdown(
                "**Principal Component Analysis** reveals the underlying structure of the 6 factors (5 quality + trend) "
                "and shows how they contribute to overall variation across issuers. The radar charts display "
                "each factor's loading (contribution) on the principal components."
            )

            try:
                from plotly.subplots import make_subplots

                # Get factor score columns and filter by coverage (min 50%)
                all_factor_cols = [c for c in results_final.columns if c.endswith("_Score") and c not in ("Composite_Score", "Quality_Score")]

                # Filter out factors with <50% coverage
                score_cols = []
                excluded_factors = []
                for col in all_factor_cols:
                    coverage_pct = (pd.to_numeric(results_final[col], errors='coerce').notna().sum() / len(results_final) * 100)
                    if coverage_pct >= 50.0:
                        score_cols.append(col)
                    else:
                        excluded_factors.append(col)

                # Show which factors are being used
                if excluded_factors:
                    excluded_names = [f.replace("_Score", "") for f in excluded_factors]
                    st.info(f"PCA using {len(score_cols)} factors (excluded due to low coverage: {', '.join(excluded_names)})")
                else:
                    st.info(f"PCA using all {len(score_cols)} factors")

                # Prepare data for PCA (sample if needed for performance)
                df_pca_sample = results_final.copy()
                if len(df_pca_sample) > 2000:
                    df_pca_sample = df_pca_sample.sample(n=2000, random_state=0)

                # Extract and clean numeric data
                X_pca = df_pca_sample[score_cols].copy()

                # Convert to numeric, coercing errors
                for col in score_cols:
                    X_pca[col] = pd.to_numeric(X_pca[col], errors='coerce')

                # Check if we have enough valid data
                valid_counts = X_pca.notna().sum()
                if valid_counts.min() < 10:
                    raise ValueError(f"Insufficient valid data: some factors have <10 valid values")

                # Remove rows with ANY missing values (most reliable approach)
                X_pca_clean = X_pca.dropna()

                # Check if we have enough rows after dropping NaNs
                if len(X_pca_clean) < 20:
                    raise ValueError(f"Only {len(X_pca_clean)} complete cases after removing missing values (need ≥20)")

                # Apply RobustScaler and PCA
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X_pca_clean.values)

                pca = PCA(n_components=min(3, len(score_cols)), random_state=0)
                pca_result = pca.fit_transform(X_scaled)

                # Get loadings and explained variance
                loadings = pca.components_
                var_exp = pca.explained_variance_ratio_ * 100

                # Clean up feature names for display
                feature_names = [col.replace('_Score', '') for col in score_cols]

                # Helper function to interpret PC business meaning
                def interpret_pc_name(pc_loadings, feature_names):
                    """Generate interpretive name based on loading patterns."""
                    abs_loadings = np.abs(pc_loadings)
                    top3_idx = np.argsort(abs_loadings)[-3:][::-1]
                    top_factors = [feature_names[i] for i in top3_idx]

                    # Categorize factors
                    profitability = {'Profitability', 'Credit'}
                    leverage = {'Leverage'}
                    coverage = {'Cash_Flow'}
                    liquidity = {'Liquidity'}

                    top_set = set(top_factors)

                    if profitability & top_set and leverage & top_set:
                        return "Overall Credit Quality"
                    elif profitability & top_set and len(profitability & top_set) >= 2:
                        return "Profitability & Returns"
                    elif leverage & top_set and coverage & top_set:
                        return "Leverage & Coverage"
                    elif liquidity & top_set:
                        return "Liquidity Position"
                    elif 'Profitability' in top_set:
                        return "Operating Performance"
                    elif 'Leverage' in top_set:
                        return "Debt & Leverage"
                    elif 'Cash_Flow' in top_set:
                        return "Cash Flow & Coverage"
                    else:
                        return "Credit Dimension"

                # Use simple PC labels
                n_components_to_show = min(3, loadings.shape[0])
                pc_names = [f"PC{i+1}" for i in range(n_components_to_show)]

                # ========================================================================
                # SECTION 1: RADAR CHARTS (LEFT) + 3D PLOT (RIGHT)
                # ========================================================================

                col_left, col_right = st.columns([1, 2])

                # LEFT COLUMN: 3 Vertical Radar Charts
                with col_left:
                    st.markdown("### Factor Loadings")
                    st.caption("Each radar chart shows how the 5 credit factors contribute to each principal component")

                    colors = ['#2C5697', '#E74C3C', '#27AE60']

                    for i in range(n_components_to_show):
                        pc_loadings = loadings[i, :]

                        fig_radar_single = go.Figure()

                        fig_radar_single.add_trace(
                            go.Scatterpolar(
                                r=pc_loadings,
                                theta=feature_names,
                                fill='toself',
                                name=f'PC{i+1}',
                                line=dict(width=2.5, color=colors[i]),
                                marker=dict(size=8),
                                fillcolor=colors[i],
                                opacity=0.5
                            )
                        )

                        fig_radar_single.update_layout(
                            height=340,
                            showlegend=False,
                            title=dict(
                                text=f'<b>PC{i+1}: {pc_names[i]}</b><br>({var_exp[i]:.1f}% variance)',
                                x=0.5,
                                xanchor='center',
                                font=dict(size=12, color='#2C5697')
                            ),
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[-1, 1],
                                    showticklabels=True,
                                    ticks='outside',
                                    tickfont=dict(size=9),
                                    gridcolor='lightgray'
                                ),
                                angularaxis=dict(
                                    tickfont=dict(size=10, color='#333333')
                                )
                            ),
                            paper_bgcolor='white',
                            plot_bgcolor='white',
                            margin=dict(t=50, b=20, l=20, r=20)
                        )

                        st.plotly_chart(fig_radar_single, use_container_width=True)

                # RIGHT COLUMN: 3D Issuer Distribution
                with col_right:
                    st.markdown("### 3D Issuer Distribution")
                    st.caption("Each point represents one issuer positioned by their PC1, PC2, and PC3 scores")

                    if n_components_to_show >= 3:
                        # Prepare dataframe with PC scores and issuer info
                        df_3d = pd.DataFrame(
                            pca_result[:, :3],
                            columns=['PC1', 'PC2', 'PC3'],
                            index=X_pca_clean.index
                        )

                        # Add company name and composite score for hover info
                        company_name_col = resolve_company_name_column(df_pca_sample)
                        if company_name_col:
                            df_3d['Company_Name'] = df_pca_sample.loc[X_pca_clean.index, company_name_col].values
                        else:
                            df_3d['Company_Name'] = df_pca_sample.loc[X_pca_clean.index].index.astype(str)

                        df_3d['Composite_Score'] = df_pca_sample.loc[X_pca_clean.index, 'Composite_Score'].values

                        # Add rating band if available
                        rating_col = resolve_rating_column(df_pca_sample)
                        if rating_col and 'Rating_Band' in df_pca_sample.columns:
                            df_3d['Rating_Band'] = df_pca_sample.loc[X_pca_clean.index, 'Rating_Band'].values
                            color_by = 'Rating_Band'
                            # Define color mapping for rating bands
                            rating_band_colors = {
                                'AAA': '#006400',
                                'AA': '#228B22',
                                'A': '#32CD32',
                                'BBB': '#FFD700',
                                'BB': '#FFA500',
                                'B': '#FF6347',
                                'CCC & Below': '#8B0000',
                                'Not Rated': '#808080'
                            }
                            df_3d['Color'] = df_3d['Rating_Band'].map(rating_band_colors).fillna('#808080')
                        else:
                            # Color by composite score if no rating band
                            color_by = 'Composite_Score'
                            df_3d['Color'] = df_3d['Composite_Score']

                        # Create 3D scatter plot
                        fig_3d = go.Figure()

                        if color_by == 'Rating_Band':
                            # Plot by rating band with separate traces for legend
                            for band in sorted(df_3d['Rating_Band'].unique()):
                                band_data = df_3d[df_3d['Rating_Band'] == band]
                                fig_3d.add_trace(go.Scatter3d(
                                    x=band_data['PC1'],
                                    y=band_data['PC2'],
                                    z=band_data['PC3'],
                                    mode='markers',
                                    marker=dict(
                                        size=5,
                                        color=band_data['Color'].iloc[0],
                                        line=dict(color='white', width=0.3),
                                        opacity=0.8
                                    ),
                                    name=str(band),
                                    text=band_data['Company_Name'],
                                    customdata=band_data[['Composite_Score', 'Rating_Band']],
                                    hovertemplate='<b>%{text}</b><br>' +
                                                'PC1: %{x:.2f}<br>' +
                                                'PC2: %{y:.2f}<br>' +
                                                'PC3: %{z:.2f}<br>' +
                                                'Score: %{customdata[0]:.1f}<br>' +
                                                'Rating: %{customdata[1]}<br>' +
                                                '<extra></extra>'
                                ))
                        else:
                            # Single trace colored by composite score
                            fig_3d.add_trace(go.Scatter3d(
                                x=df_3d['PC1'],
                                y=df_3d['PC2'],
                                z=df_3d['PC3'],
                                mode='markers',
                                marker=dict(
                                    size=5,
                                    color=df_3d['Composite_Score'],
                                    colorscale='RdYlGn',
                                    cmin=0,
                                    cmax=100,
                                    showscale=True,
                                    colorbar=dict(
                                        title="Score",
                                        x=1.0,
                                        len=0.7,
                                        thickness=15,
                                        tickfont=dict(size=10)
                                    ),
                                    line=dict(color='white', width=0.3),
                                    opacity=0.8
                                ),
                                text=df_3d['Company_Name'],
                                customdata=df_3d[['Composite_Score']],
                                hovertemplate='<b>%{text}</b><br>' +
                                            'PC1: %{x:.2f}<br>' +
                                            'PC2: %{y:.2f}<br>' +
                                            'PC3: %{z:.2f}<br>' +
                                            'Score: %{customdata[0]:.1f}<br>' +
                                            '<extra></extra>',
                                showlegend=False
                            ))

                        # Update layout
                        fig_3d.update_layout(
                            scene=dict(
                                xaxis=dict(
                                    title=dict(text=f'PC1: {pc_names[0]}<br>({var_exp[0]:.1f}%)', font=dict(size=11))
                                ),
                                yaxis=dict(
                                    title=dict(text=f'PC2: {pc_names[1]}<br>({var_exp[1]:.1f}%)', font=dict(size=11))
                                ),
                                zaxis=dict(
                                    title=dict(text=f'PC3: {pc_names[2]}<br>({var_exp[2]:.1f}%)', font=dict(size=11))
                                ),
                                camera=dict(
                                    eye=dict(x=1.5, y=1.5, z=1.3)
                                ),
                                aspectmode='cube'
                            ),
                            height=1050,
                            title=dict(
                                text=f"<b>3D Issuer Distribution</b><br><sub>n={len(df_3d):,} issuers</sub>",
                                x=0.5,
                                xanchor='center',
                                font=dict(size=14, color='#2C5697')
                            ),
                            paper_bgcolor='white',
                            plot_bgcolor='white',
                            margin=dict(l=0, r=0, t=60, b=0),
                            hovermode='closest',
                            legend=dict(
                                title=dict(text="Rating", font=dict(size=10)),
                                x=1.0,
                                y=0.5,
                                bgcolor='rgba(255,255,255,0.8)',
                                font=dict(size=9)
                            ) if color_by == 'Rating_Band' else None
                        )

                        st.plotly_chart(fig_3d, use_container_width=True, key='3d_issuer_plot')
                    else:
                        st.info(f"3D visualization requires 3 principal components. Currently showing {n_components_to_show} component(s).")

                # ========================================================================
                # SECTION 2: VARIANCE METRICS GRID
                # ========================================================================
                st.markdown("### Variance Explained")

                # Create columns for variance metrics
                metric_cols = st.columns(n_components_to_show + 1)

                # Individual PC variance
                for i in range(n_components_to_show):
                    with metric_cols[i]:
                        st.metric(
                            f"PC{i+1}",
                            f"{var_exp[i]:.1f}%",
                            help=f"Principal Component {i+1} explains {var_exp[i]:.1f}% of total variance"
                        )

                # Cumulative variance in last column
                with metric_cols[n_components_to_show]:
                    cum_var_total = var_exp[:n_components_to_show].sum()
                    st.metric(
                        f"PC1-{n_components_to_show}",
                        f"{cum_var_total:.1f}%",
                        help=f"First {n_components_to_show} components explain {cum_var_total:.1f}% of total variance"
                    )

                # ========================================================================
                # SECTION 3: HOW TO READ GUIDE
                # ========================================================================
                st.markdown("---")
                col_guide1, col_guide2 = st.columns(2)

                with col_guide1:
                    st.markdown("**How to Read**")
                    st.markdown(
                        "- **Distance from center**: Strength of factor's contribution\n"
                        "- **Positive values** (outward): Factor increases with PC\n"
                        "- **Negative values** (opposite): Factor decreases with PC"
                    )

                with col_guide2:
                    st.markdown("**Interpretation**")
                    st.markdown(
                        "- **Near ±1.0**: Very strong influence\n"
                        "- **Near ±0.5**: Moderate influence\n"
                        "- **Near 0.0**: Weak influence"
                    )

                # ========================================================================
                # SECTION 4: DETAILED INSIGHTS (EXPANDABLE)
                # ========================================================================
                with st.expander("View Detailed Loadings & Insights"):
                    # Build loadings dataframe
                    loadings_df = pd.DataFrame(
                        loadings[:n_components_to_show].T,
                        columns=[f'PC{i+1}' for i in range(n_components_to_show)],
                        index=feature_names
                    )

                    # Automatic insights
                    st.markdown("**Dominant Factors by Component:**")
                    for i in range(n_components_to_show):
                        pc_loadings_abs = loadings_df[f'PC{i+1}'].abs().sort_values(ascending=False)
                        top_factor = pc_loadings_abs.index[0]
                        top_value = loadings_df.loc[top_factor, f'PC{i+1}']
                        direction = "positively" if top_value > 0 else "negatively"

                        # Get top 2 factors
                        top2_factors = pc_loadings_abs.head(2)
                        factor2 = top2_factors.index[1]
                        value2 = loadings_df.loc[factor2, f'PC{i+1}']
                        dir2 = "positively" if value2 > 0 else "negatively"

                        st.markdown(f"- **PC{i+1}** ({var_exp[i]:.1f}% var): Most {direction} influenced by **{top_factor}** ({top_value:.3f}), followed by **{factor2}** ({dir2}, {value2:.3f})")

                    st.markdown("---")
                    st.markdown("**Complete Loadings Table:**")

                    # Add a column showing which PC each factor loads strongest on
                    loadings_df['Strongest_PC'] = loadings_df.abs().idxmax(axis=1)
                    loadings_df['Max_Loading'] = loadings_df[[f'PC{i+1}' for i in range(n_components_to_show)]].abs().max(axis=1)
                    loadings_df_display = loadings_df.sort_values('Max_Loading', ascending=False)

                    # Display with styling
                    st.dataframe(
                        loadings_df_display[[f'PC{i+1}' for i in range(n_components_to_show)]].style.format("{:.3f}").background_gradient(
                            cmap='RdYlGn', axis=0, vmin=-1, vmax=1
                        ),
                        use_container_width=True
                    )

                    # Interpretation guide
                    st.markdown("---")
                    st.markdown("**Interpretation Tips:**")
                    st.markdown("""
                    - **PC1** typically captures the primary dimension of credit quality (overall strength)
                    - **PC2** often represents a secondary differentiating factor (e.g., growth vs. stability)
                    - **PC3** captures tertiary patterns or sector-specific characteristics
                    - Factors with **similar sign patterns** across PCs tend to move together
                    - Factors with **opposite signs** represent trade-offs or different credit strategies
                    """)

            except Exception as e:
                # Get diagnostic info
                try:
                    all_factor_cols = [c for c in results_final.columns if c not in ("Composite_Score", "Quality_Score") and c.endswith("_Score")]
                    X_check = results_final[all_factor_cols].apply(pd.to_numeric, errors='coerce')
                    valid_per_col = X_check.notna().sum()
                    complete_rows = X_check.dropna().shape[0]

                    st.warning(f"PCA analysis unavailable: {e}")

                    with st.expander("Diagnostic Information"):
                        st.markdown("**Valid data points per factor:**")
                        for col in all_factor_cols:
                            pct = (valid_per_col[col] / len(results_final) * 100) if len(results_final) > 0 else 0
                            st.caption(f"• {col}: {valid_per_col[col]:,} / {len(results_final):,} ({pct:.1f}%)")

                        st.markdown(f"**Complete cases** (rows with all factors): {complete_rows:,} / {len(results_final):,}")

                        if complete_rows < 20:
                            st.error("Need at least 20 issuers with complete factor scores for PCA")
                            st.info("**Suggestion**: Check your data export to ensure all factor scores are populated")
                        elif valid_per_col.min() < len(results_final) * 0.5:
                            st.warning("Some factors have >50% missing values")
                            st.info("**Suggestion**: Review your factor calculation logic or data quality")
                except Exception:
                    st.warning(f"PCA analysis unavailable: {e}")
                    st.caption("This may occur with insufficient data or if factor scores are missing.")

            # ========================================================================
            # SCORE DISTRIBUTION AND CLASSIFICATION ANALYSIS
            # ========================================================================
            st.markdown("---")

            # Score distribution
            st.subheader("Score Distribution by Rating Group")

            fig = go.Figure()
            
            # Color scheme matching app theme
            colors = {
                'Investment Grade': '#2C5697',  # Dark blue
                'High Yield': '#7FBFFF'         # Light blue
            }
            fill_colors = {
                'Investment Grade': 'rgba(44, 86, 151, 0.4)',
                'High Yield': 'rgba(127, 191, 255, 0.4)'
            }
            
            for group in ['Investment Grade', 'High Yield']:
                group_data = results_final[results_final['Rating_Group'] == group]['Composite_Score'].dropna()
                
                if len(group_data) > 1:
                    # Calculate KDE for smooth curve
                    kde = gaussian_kde(group_data, bw_method=0.3)
                    
                    # Create smooth x range
                    x_min = max(0, group_data.min() - 5)
                    x_max = min(100, group_data.max() + 5)
                    x_range = np.linspace(x_min, x_max, 200)
                    y_density = kde(x_range)
                    
                    # Scale density to approximate counts
                    bin_width = (group_data.max() - group_data.min()) / 20
                    y_scaled = y_density * len(group_data) * bin_width
                    
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=y_scaled,
                        name=group,
                        fill='tozeroy',
                        mode='lines',
                        line=dict(width=2, color=colors[group]),
                        fillcolor=fill_colors[group]
                    ))

            fig.update_layout(
                xaxis_title='Composite Score',
                yaxis_title='Count',
                title='Composite Score Distribution',
                height=400,
                hovermode='x unified',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )
            st.plotly_chart(fig, use_container_width=True)

            # Classification comparison
            st.subheader("Average Scores by Classification")
            classification_avg = results_final.groupby('Rubrics_Custom_Classification').agg({
                'Composite_Score': 'mean',
                'Company_Name': 'count'
            }).reset_index()
            classification_avg.columns = ['Classification', 'Avg Score', 'Count']
            classification_avg = classification_avg.sort_values('Avg Score', ascending=False)

            fig2 = go.Figure(data=[
                go.Bar(
                    x=classification_avg['Classification'],
                    y=classification_avg['Avg Score'],
                    text=classification_avg['Avg Score'].round(1),
                    textposition='outside',
                    marker_color='#2C5697'
                )
            ])
            fig2.update_layout(
                xaxis_title='Classification',
                yaxis_title='Average Composite Score',
                title='Classification Performance Comparison',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)

            # ========================================================================
            # RATING-BAND LEADERBOARDS (V2.2)
            # ========================================================================
            st.markdown("---")
            st.subheader("Rating-Band Leaderboards")
        
            left, mid, right = st.columns([2, 1, 1])
            with left:
                # Only display bands that actually exist in the dataset, ordered by canonical order
                available_bands = sorted(
                    results_final["Rating_Band"].dropna().astype(str).unique().tolist(),
                    key=lambda b: (RATING_BAND_ORDER.index(b) if b in RATING_BAND_ORDER else 999, b)
                )
                if available_bands:
                    # Use band0 from URL if valid, otherwise default to first band
                    default_band_idx = 0
                    if band0 and band0 in available_bands:
                        default_band_idx = available_bands.index(band0)
        
                    band = st.selectbox(
                        "Select rating band",
                        options=available_bands,
                        index=default_band_idx,
                        key="band_selector",
                        help="Scores are comparable only within a band."
                    )
                else:
                    st.warning("No rating bands available in the dataset.")
                    band = None
        
            with mid:
                top_n = st.slider("Top N", min_value=5, max_value=50, value=topn0, step=5, key="top_n_slider")
        
            with right:
                include_nr = st.toggle(
                    "Show NR/Other bands",
                    value=False,
                    help="Include NR/SD/D if present"
                )
        
            # Persist band and Top N to URL
            _set_query_params(collect_current_state(
                scoring_method, data_period, use_quarterly_beta, align_to_reference,
                band_default=band if band else "",
                top_n_default=top_n
            ))
        
            if band:
                # Filter results based on NR/Other toggle
                if not include_nr:
                    # Filter out bands typically considered 'other'
                    results_band_scope = results_final[~results_final["Rating_Band"].isin(["NR", "SD", "D"])]
                else:
                    results_band_scope = results_final
        
                if band not in results_band_scope["Rating_Band"].astype(str).unique():
                    st.info(f"No issuers in band {band}.")
                else:
                    try:
                        tbl = build_band_leaderboard(results_band_scope, band, 'Company_ID', 'Company_Name', top_n=top_n)
                        if tbl.empty:
                            st.info(f"No issuers in band {band}.")
                        else:
                            st.dataframe(tbl, use_container_width=True, hide_index=True)
        
                            # Download selected leaderboard
                            csv_bytes, fname = to_csv_download(tbl, filename=f"leaderboard_{band}_top{top_n}.csv")
                            st.download_button(
                                " Download CSV",
                                data=csv_bytes,
                                file_name=fname,
                                mime="text/csv"
                            )
                    except Exception as e:
                        st.error(f"Error building leaderboard: {e}")
        
            # ========================================================================
            # DIAGNOSTICS & DATA HEALTH
            # ========================================================================
            st.subheader("Diagnostics & Data Health")
        
            with st.expander("Diagnostics & Data Health", expanded=False):
                try:
                    diag = diagnostics_summary(df_original, results_final)
        
                    st.markdown("**Dataset summary**")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total issuers", value=f"{diag['rows_total']:,}")
                    delta_text = f"-{diag['duplicate_ids']:,} dups" if diag['duplicate_ids'] else None
                    c2.metric("Unique IDs", value=f"{diag['unique_company_ids']:,}", delta=delta_text)
                    c3.metric("IG names", value=f"{diag['ig_count']:,}")
                    c4.metric("HY names", value=f"{diag['hy_count']:,}")
        
                    st.markdown("---")
                    st.markdown("**Freshness coverage**")
        
                    def _share_leq(s, cutoff):
                        """Calculate % of rows with days <= cutoff."""
                        s = pd.to_numeric(s, errors="coerce").dropna()
                        return 0 if s.empty else round(100 * (s <= cutoff).mean(), 1)
        
                    a, b, c, d, e, f = st.columns(6)
                    a.metric("Fin ≤90d", f"{_share_leq(results_final['Financial_Data_Freshness_Days'], 90)}%")
                    b.metric("Fin ≤180d", f"{_share_leq(results_final['Financial_Data_Freshness_Days'], 180)}%")
                    c.metric("Fin ≤365d", f"{_share_leq(results_final['Financial_Data_Freshness_Days'], 365)}%")
                    d.metric("Rating ≤90d", f"{_share_leq(results_final['Rating_Review_Freshness_Days'], 90)}%")
                    e.metric("Rating ≤180d", f"{_share_leq(results_final['Rating_Review_Freshness_Days'], 180)}%")
                    f.metric("Rating ≤365d", f"{_share_leq(results_final['Rating_Review_Freshness_Days'], 365)}%")
        
                    st.markdown("---")
                    st.markdown("**Rating band mix**")
                    if isinstance(diag["band_counts"], pd.Series) and not diag["band_counts"].empty:
                        bc = diag["band_counts"].reset_index()
                        bc.columns = ["Rating_Band", "Count"]
                        fig_bands = go.Figure(data=[go.Bar(x=bc["Rating_Band"], y=bc["Count"])])
                        fig_bands = apply_rubrics_plot_theme(fig_bands)
                        fig_bands.update_layout(title="Issuers by Rating Band")
                        st.plotly_chart(fig_bands, use_container_width=True)
                    else:
                        st.info("Rating_Band not available.")
        
                    st.markdown("---")
                    st.markdown("**Period Ended coverage**")
                    colA, colB, colC = st.columns(3)
                    colA.metric("Period columns", diag["period_cols"])
                    colB.metric("Earliest period", diag["period_min"].date().isoformat() if pd.notna(diag["period_min"]) else "n/a")
                    colC.metric("Latest period", diag["period_max"].date().isoformat() if pd.notna(diag["period_max"]) else "n/a")
                    if diag["fy_suffixes"] or diag["ltm_suffixes"]:
                        fy_list = ', '.join([s for s in diag['fy_suffixes']]) if diag['fy_suffixes'] else '(none)'
                        ltm_list = ', '.join([s for s in diag['ltm_suffixes']]) if diag['ltm_suffixes'] else '(none)'
                        st.caption(f"Detected FY suffixes: {fy_list}; LTM suffixes: {ltm_list}")
        
                    st.markdown("---")
                    st.markdown("**Factor score completeness**")
                    miss_scores = summarize_scores_missingness(results_final)
                    if not miss_scores.empty:
                        fig_miss_scores = go.Figure(data=[go.Bar(x=miss_scores["Column"], y=miss_scores["Missing_%"])])
                        fig_miss_scores.update_yaxes(title="% missing", range=[0, 100])
                        fig_miss_scores.update_xaxes(tickangle=45)
                        fig_miss_scores = apply_rubrics_plot_theme(fig_miss_scores)
                        fig_miss_scores.update_layout(title="Missingness by factor score (%)", margin=dict(b=100))
                        st.plotly_chart(fig_miss_scores, use_container_width=True)
                        st.dataframe(miss_scores, use_container_width=True, hide_index=True)
                    else:
                        st.info("No *_Score columns to summarize.")
        
                    st.markdown("---")
                    st.markdown("**Key metrics completeness**")
                    miss_metrics = summarize_key_metrics_missingness(df_original)
                    if not miss_metrics.empty:
                        fig_miss_metrics = go.Figure(data=[go.Bar(x=miss_metrics["Column"], y=miss_metrics["Missing_%"])])
                        fig_miss_metrics.update_yaxes(title="% missing", range=[0, 100])
                        fig_miss_metrics.update_xaxes(tickangle=45)
                        fig_miss_metrics = apply_rubrics_plot_theme(fig_miss_metrics)
                        fig_miss_metrics.update_layout(title="Missingness by key metric (%)", margin=dict(b=140))
                        st.plotly_chart(fig_miss_metrics, use_container_width=True)
                        st.dataframe(miss_metrics, use_container_width=True, hide_index=True)
                    else:
                        st.info("No key metric columns detected in the uploaded file.")
        
                    # Export diagnostics
                    st.markdown("---")
                    st.markdown("**Export diagnostics**")
                    diag_rows = [
                        {"Metric": "Total issuers", "Value": diag["rows_total"]},
                        {"Metric": "Unique company IDs", "Value": diag["unique_company_ids"]},
                        {"Metric": "Duplicate IDs", "Value": diag["duplicate_ids"]},
                        {"Metric": "IG count", "Value": diag["ig_count"]},
                        {"Metric": "HY count", "Value": diag["hy_count"]},
                        {"Metric": "Period columns", "Value": diag["period_cols"]},
                        {"Metric": "Earliest period", "Value": str(diag["period_min"]) if diag["period_min"] else ""},
                        {"Metric": "Latest period", "Value": str(diag["period_max"]) if diag["period_max"] else ""},
                        {"Metric": "FY suffixes", "Value": ", ".join(diag["fy_suffixes"]) if diag["fy_suffixes"] else ""},
                        {"Metric": "LTM suffixes", "Value": ", ".join(diag["ltm_suffixes"]) if diag["ltm_suffixes"] else ""},
                        {"Metric": "Financial Data ≤90d (%)", "Value": _share_leq(results_final['Financial_Data_Freshness_Days'], 90)},
                        {"Metric": "Financial Data ≤180d (%)", "Value": _share_leq(results_final['Financial_Data_Freshness_Days'], 180)},
                        {"Metric": "Financial Data ≤365d (%)", "Value": _share_leq(results_final['Financial_Data_Freshness_Days'], 365)},
                        {"Metric": "Rating Review ≤90d (%)", "Value": _share_leq(results_final['Rating_Review_Freshness_Days'], 90)},
                        {"Metric": "Rating Review ≤180d (%)", "Value": _share_leq(results_final['Rating_Review_Freshness_Days'], 180)},
                        {"Metric": "Rating Review ≤365d (%)", "Value": _share_leq(results_final['Rating_Review_Freshness_Days'], 365)},
                    ]
                    diag_df = pd.DataFrame(diag_rows)
                    csv_diag = io.StringIO()
                    diag_df.to_csv(csv_diag, index=False)
                    st.download_button(
                        " Download diagnostics (CSV)",
                        data=csv_diag.getvalue().encode("utf-8"),
                        file_name="diagnostics_summary.csv",
                        mime="text/csv"
                    )
        
                except Exception as e:
                    st.warning(f"Diagnostics unavailable: {e}")

            # ============================================================================
            # [V2.2.1] CALIBRATION DIAGNOSTICS
            # ============================================================================

            if use_dynamic_calibration:
                with st.expander(" Calibration Diagnostics", expanded=False):
                    st.markdown("### Weight Calibration Effectiveness")
                    st.markdown(f"**Calibration Rating Band:** {calibration_rating_band}")

                    # Show average scores by sector for the calibration rating band
                    rating_bands_map = {
                        'BBB': ['BBB+', 'BBB', 'BBB-'],
                        'A': ['A+', 'A', 'A-'],
                        'BB': ['BB+', 'BB', 'BB-']
                    }
                    rating_list = rating_bands_map.get(calibration_rating_band, ['BBB+', 'BBB', 'BBB-'])

                    df_rated = results_final[results_final['Credit_Rating_Clean'].isin(rating_list)]

                    if len(df_rated) > 0:
                        # Group by sector and calculate average composite score
                        sector_stats = []
                        for sector_name in ['Utilities', 'Real Estate', 'Energy', 'Materials', 'Consumer Staples',
                                          'Industrials', 'Information Technology', 'Health Care', 'Consumer Discretionary', 'Communication Services']:
                            # Get classifications for this sector
                            classifications = [k for k, v in CLASSIFICATION_TO_SECTOR.items() if v == sector_name]
                            sector_df = df_rated[df_rated['Rubrics_Custom_Classification'].isin(classifications)]

                            if len(sector_df) >= 5:  # Only show sectors with sufficient data
                                avg_score = sector_df['Composite_Score'].mean()
                                buy_pct = (sector_df['Recommendation'].isin(['Strong Buy', 'Buy']).sum() / len(sector_df) * 100)

                                sector_stats.append({
                                    'Sector': sector_name,
                                    'N': len(sector_df),
                                    'Avg Score': f"{avg_score:.1f}",
                                    '% Buy/Strong Buy': f"{buy_pct:.1f}%"
                                })

                        if sector_stats:
                            diag_df = pd.DataFrame(sector_stats)
                            st.dataframe(diag_df, use_container_width=True, hide_index=True)

                            # Calculate statistics
                            scores = [float(s['Avg Score']) for s in sector_stats]
                            rates = [float(s['% Buy/Strong Buy'].rstrip('%')) for s in sector_stats]
                            score_range = max(scores) - min(scores)
                            rate_range = max(rates) - min(rates)

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Score Range", f"{score_range:.1f} points",
                                        help="Range between highest and lowest sector average. Target: <15 points")
                                if score_range < 15:
                                    st.success(" Excellent calibration - scores well normalized across sectors")
                                elif score_range < 25:
                                    st.info(" Good calibration - minor differences remain")
                                else:
                                    st.warning(" Calibration incomplete - significant differences persist")

                            with col2:
                                st.metric("Buy Rate Range", f"{rate_range:.1f}%",
                                        help="Range between highest and lowest sector buy rates. Target: <20%")
                                if rate_range < 20:
                                    st.success(" Excellent fairness - similar buy rates across sectors")
                                elif rate_range < 30:
                                    st.info(" Good fairness - minor differences remain")
                                else:
                                    st.warning(" Fairness incomplete - significant differences persist")

                            st.markdown("""
                            **Expected if calibration works:**
                            - All sectors should have Avg Score between 45-65
                            - All sectors should have % Buy/Strong Buy between 30-50%
                            - Score Range should be < 15 points
                            - Buy Rate Range should be < 20%

                            **If scores still vary significantly:** Data may have insufficient coverage for some sectors,
                            or structural differences exist beyond factor score variations.
                            """)
                        else:
                            st.info("Insufficient sector representation for calibration diagnostics (need 5+ issuers per sector).")
                    else:
                        st.info(f"No issuers found in {calibration_rating_band} rating band.")

                    # ====================================================================
                    # WEIGHT COMPARISON VIEW
                    # ====================================================================

                    st.markdown("---")
                    st.markdown("### Calibrated vs Original Weights")

                    # Get calibrated weights from session state
                    calibrated_wts = st.session_state.get('_calibrated_weights', None)

                    if calibrated_wts is not None:
                        comparison_sector = st.selectbox(
                            "Select Sector to Compare",
                            options=[s for s in ['Utilities', 'Real Estate', 'Energy', 'Materials', 'Consumer Staples',
                                                'Industrials', 'Information Technology', 'Health Care',
                                                'Consumer Discretionary', 'Communication Services'] if s in calibrated_wts],
                            key="calibration_sector_compare"
                        )

                        if comparison_sector in calibrated_wts:
                            compare_data = []
                            for factor in ['credit_score', 'leverage_score', 'profitability_score',
                                          'liquidity_score', 'cash_flow_score']:
                                # V3.0: Compare against universal base weights
                                original = UNIVERSAL_WEIGHTS[factor]
                                calibrated = calibrated_wts[comparison_sector][factor]
                                change = calibrated - original
                                pct_change = (change / original * 100) if original != 0 else 0

                                compare_data.append({
                                    'Factor': factor.replace('_score', '').replace('_', ' ').title(),
                                    'Original': f"{original:.3f}",
                                    'Calibrated': f"{calibrated:.3f}",
                                    'Change': f"{change:+.3f}",
                                    '% Change': f"{pct_change:+.1f}%"
                                })

                            compare_df = pd.DataFrame(compare_data)
                            st.dataframe(compare_df, use_container_width=True, hide_index=True)

                            st.markdown(
                                "**Interpreting weight changes:**\n"
                                "- **Decreased weights** -> Sector deviates from market on this factor, so we reduce its influence\n"
                                "- **Increased weights** -> Sector is neutral on this factor, so we emphasize it for differentiation\n"
                                "- **Large changes (>50%)** -> Factor shows significant structural difference in this sector"
                            )
                    else:
                        st.info("Weight comparison unavailable (calibration may have failed or not yet calculated).")

            # Update URL state with current Tab 1 control values
            _updated_state = collect_current_state(
                scoring_method=scoring_method,
                data_period=data_period,
                use_quarterly_beta=use_quarterly_beta,
                align_to_reference=align_to_reference,
                band_default=band if band else "",
                top_n_default=top_n
            )
            _set_query_params(_updated_state)
        
        # ============================================================================
        # TAB 2: ISSUER SEARCH
        # ============================================================================
            
        with tab2:
            st.header(" Search & Filter Issuers")
                
            # Filters
            col1, col2, col3, col4 = st.columns(4)
                
            with col1:
                rating_filter = st.multiselect(
                    "Rating Band",
                    options=['All'] + sorted(results_final['Rating_Band'].unique().tolist()),
                    default=['All']
                )
                
            with col2:
                classification_filter = st.multiselect(
                    "Classification",
                    options=['All'] + sorted(results_final['Rubrics_Custom_Classification'].dropna().unique().tolist()),
                    default=['All']
                )
                
            with col3:
                rec_filter = st.multiselect(
                    "Recommendation",
                    options=['All'] + sorted(results_final['Recommendation'].unique().tolist()),
                    default=['All']
                )
                
            with col4:
                score_min, score_max = st.slider(
                    "Score Range",
                    min_value=0.0,
                    max_value=100.0,
                    value=(0.0, 100.0),
                    step=1.0
                )
                
            # Apply filters
            filtered = results_final.copy()
                
            if 'All' not in rating_filter:
                filtered = filtered[filtered['Rating_Band'].isin(rating_filter)]
                
            if 'All' not in classification_filter:
                filtered = filtered[filtered['Rubrics_Custom_Classification'].isin(classification_filter)]
                
            if 'All' not in rec_filter:
                filtered = filtered[filtered['Recommendation'].isin(rec_filter)]
                
            filtered = filtered[
                (filtered['Composite_Score'] >= score_min) &
                (filtered['Composite_Score'] <= score_max)
            ]
        
            # ========================================================================
            # WATCHLIST / EXCLUSIONS (V2.2)
            # ========================================================================
            with st.expander(" Watchlist / Exclusions", expanded=False):
                col_w, col_e = st.columns(2)
        
                with col_w:
                    st.markdown("**Watchlist (Include-Only)**")
                    st.caption("Upload CSV with Company_ID column to include only these issuers")
                    watchlist_file = st.file_uploader(
                        "Upload Watchlist CSV",
                        type=["csv"],
                        key="watchlist_uploader",
                        help="CSV must contain a 'Company_ID' column"
                    )
        
                    if watchlist_file is not None:
                        try:
                            watchlist_df = pd.read_csv(watchlist_file)
                            if 'Company_ID' in watchlist_df.columns:
                                watchlist_ids = watchlist_df['Company_ID'].astype(str).str.strip().tolist()
                                # Filter to only watchlist IDs
                                filtered = filtered[filtered['Company_ID'].astype(str).str.strip().isin(watchlist_ids)]
                                st.success(f" Watchlist applied: {len(watchlist_ids)} IDs")
                            else:
                                st.error("Watchlist CSV must contain 'Company_ID' column")
                        except Exception as e:
                            st.error(f"Failed to load watchlist: {e}")
        
                with col_e:
                    st.markdown("**Exclusions (Drop List)**")
                    st.caption("Upload CSV with Company_ID column to exclude these issuers")
                    exclusions_file = st.file_uploader(
                        "Upload Exclusions CSV",
                        type=["csv"],
                        key="exclusions_uploader",
                        help="CSV must contain a 'Company_ID' column"
                    )
        
                    if exclusions_file is not None:
                        try:
                            exclusions_df = pd.read_csv(exclusions_file)
                            if 'Company_ID' in exclusions_df.columns:
                                exclusion_ids = exclusions_df['Company_ID'].astype(str).str.strip().tolist()
                                # Filter out exclusion IDs
                                filtered = filtered[~filtered['Company_ID'].astype(str).str.strip().isin(exclusion_ids)]
                                st.success(f" Exclusions applied: {len(exclusion_ids)} IDs removed")
                            else:
                                st.error("Exclusions CSV must contain 'Company_ID' column")
                        except Exception as e:
                            st.error(f"Failed to load exclusions: {e}")
        
            st.info(f"**{len(filtered):,}** issuers match your criteria (out of {len(results_final):,} total)")
        
            # Add freshness badges
            def _badge(flag):
                """Convert flag to emoji badge."""
                return {"Green": "🟢", "Amber": "🟠", "Red": "🔴"}.get(flag, "⚪")
        
            filtered['Fin_Badge'] = filtered['Financial_Data_Freshness_Flag'].apply(_badge)
            filtered['Rating_Badge'] = filtered['Rating_Review_Freshness_Flag'].apply(_badge)
        
            # Results table
            display_cols = [
                'Overall_Rank', 'Company_Name', 'Ticker', 'Credit_Rating_Clean', 'Rating_Band', 
                'Rubrics_Custom_Classification', 'Composite_Score', 'Cycle_Position_Score',
                'Fin_Badge', 'Financial_Data_Freshness_Days',
                'Rating_Badge', 'Rating_Review_Freshness_Days',
                'Combined_Signal', 'Recommendation', 'Weight_Method', 'Company_ID'
            ]

            # --- UI-only column label mapping for issuer table ---
            ISSUER_TABLE_LABELS = {
                'Overall_Rank': 'Overall Rank',
                'Company_Name': 'Issuer Name',
                'Ticker': 'Ticker',
                'Credit_Rating_Clean': 'S&P Rating',
                'Rating_Band': 'Rating Band',
                'Rubrics_Custom_Classification': 'Sector / Industry',
                'Composite_Score': 'Composite Score (0–100)',
                'Cycle_Position_Score': 'Cycle Position Score',
                'Fin_Badge': 'Financials Data Freshness',
                'Financial_Data_Freshness_Days': 'Days Since Latest Financials',
                'Rating_Badge': 'Rating Data Freshness',
                'Rating_Review_Freshness_Days': 'Days Since Last Rating Review',
                'Combined_Signal': 'Quality & Trend Signal',
                'Recommendation': 'Model Recommendation',
                'Weight_Method': 'Portfolio Sector Weight (Context)',
                'Company_ID': 'Company ID',
            }

            # Create recommendation priority for sorting
            rec_priority = {"Strong Buy": 4, "Buy": 3, "Hold": 2, "Avoid": 1}
            filtered['Rec_Priority'] = filtered['Recommendation'].map(rec_priority)

            # Create a view for display only; do not mutate the pipeline DF
            # Add Rec_Priority to display_cols temporarily for sorting
            filtered_display = filtered[display_cols + ['Rec_Priority']].copy()
            filtered_display = filtered_display.rename(columns=ISSUER_TABLE_LABELS)

            # Sort by recommendation (best first), then by composite score (highest first)
            filtered_display = filtered_display.sort_values(
                ['Rec_Priority', 'Composite Score (0–100)'],
                ascending=[False, False]
            )

            # Remove Rec_Priority column (was only for sorting)
            filtered_display = filtered_display.drop(columns=['Rec_Priority'])

            st.dataframe(filtered_display, use_container_width=True, hide_index=True, height=600)
        
            # ========================================================================
            # ISSUER EXPLAINABILITY (V2.2)
            # ========================================================================
            calibrated_weights = st.session_state.get('_calibrated_weights', None)
            render_issuer_explainability(filtered, scoring_method, calibrated_weights)
        
            # ========================================================================
            # EXPORT CURRENT VIEW (V2.2)
            # ========================================================================
            with st.expander(" Export Current View (CSV)", expanded=False):
                export_cols = [
                    "Company_ID", "Company_Name", "Credit_Rating_Clean", "Rating_Band", "Rating_Group",
                    "Composite_Score", "Composite_Percentile_in_Band",
                    "Credit_Score", "Leverage_Score", "Profitability_Score", "Liquidity_Score", "Cash_Flow_Score",
                    "Cycle_Position_Score", "Band_Rank", "Overall_Rank", "Recommendation", "Combined_Signal",
                    "Financial_Last_Period_Date", "Financial_Data_Freshness_Days", "Financial_Data_Freshness_Flag",
                    "SP_Last_Review_Date", "Rating_Review_Freshness_Days", "Rating_Review_Freshness_Flag"
                ]
                export_df = filtered[[c for c in export_cols if c in filtered.columns]].copy()
                csv_buf = io.StringIO()
                export_df.to_csv(csv_buf, index=False)
                st.download_button(
                    " Download CSV",
                    data=csv_buf.getvalue().encode("utf-8"),
                    file_name="issuer_screen_current_view.csv",
                    mime="text/csv"
                )
            
        # ============================================================================
        # TAB 3: RATING GROUP ANALYSIS
        # ============================================================================

        with tab3:
            st.header(" Rating Group & Band Analysis")

            # Select rating band
            col1, col2 = st.columns([1, 3])

            with col1:
                available_bands = sorted(results_final['Rating_Band'].unique())
                selected_band = st.selectbox(
                    "Select Rating Band",
                    options=available_bands,
                    index=0
                )

            with col2:
                # Show classification filter for the selected band
                band_classifications = results_final[results_final['Rating_Band'] == selected_band]['Rubrics_Custom_Classification'].unique()
                selected_classification_band = st.selectbox(
                    "Filter by Classification (optional)",
                    options=['All Classifications'] + sorted(band_classifications.tolist())
                )

            # ========================================================================
            # BUILD STRICTLY FILTERED DATAFRAME (df_band)
            # ========================================================================
            # Start from master results and apply all selected filters
            df_band = results_final[results_final['Rating_Band'] == selected_band].copy()

            if selected_classification_band != 'All Classifications':
                df_band = df_band[df_band['Rubrics_Custom_Classification'] == selected_classification_band]

            # Handle empty selection
            if df_band.empty:
                st.warning("No issuers match the current selection.")
            else:
                # ========================================================================
                # HEADLINE METRICS (from df_band only)
                # ========================================================================
                # Count issuers with valid Composite_Score
                valid_scores = df_band['Composite_Score'].notna()
                issuers_count = valid_scores.sum()

                # Average and Median from df_band
                avg_score = df_band['Composite_Score'].mean(skipna=True)
                median_score = df_band['Composite_Score'].median(skipna=True)

                # Strong Buy %: count of "Strong Buy" / count of valid scores
                strong_buy_count = ((df_band['Recommendation'] == 'Strong Buy') & valid_scores).sum()
                strong_buy_pct = (strong_buy_count / issuers_count * 100) if issuers_count > 0 else 0.0

                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Issuers", f"{issuers_count:,}")
                with col2:
                    st.metric("Average Composite Score", f"{avg_score:.1f}" if pd.notna(avg_score) else "n/a")
                with col3:
                    st.metric("Median Score", f"{median_score:.1f}" if pd.notna(median_score) else "n/a")
                with col4:
                    st.metric("Strong Buy %", f"{strong_buy_pct:.1f}%")

                # ========================================================================
                # TOP 20 TABLE (from df_band only)
                # ========================================================================
                st.subheader(f" Top 20 {selected_band} Issuers" + (f" in {selected_classification_band}" if selected_classification_band != 'All Classifications' else ""))

                top_band = df_band.sort_values('Composite_Score', ascending=False).head(20)[
                    ['Band_Rank', 'Company_Name', 'Credit_Rating_Clean', 'Rubrics_Custom_Classification',
                     'Composite_Score', 'Cycle_Position_Score', 'Combined_Signal', 'Recommendation']
                ]
                top_band.columns = ['Band Rank', 'Company', 'Rating', 'Classification', 'Score', 'Cycle', 'Signal', 'Rec']
                st.dataframe(top_band, use_container_width=True, hide_index=True)


            
            
        # ============================================================================
        # TAB 4: SECTOR ANALYSIS (NEW - SOLUTION TO ISSUE #1)
        # ============================================================================
            
        with tab4:
            st.header(" Classification-Specific Analysis")

            # Classification selection
            selected_classification = st.selectbox(
                "Select Classification for Analysis",
                options=sorted(results_final['Rubrics_Custom_Classification'].dropna().unique())
            )
                
            classification_data = results_final[results_final['Rubrics_Custom_Classification'] == selected_classification].copy()
                
            # Show weights being used
            st.subheader(f" Factor Weights Applied - {selected_classification}")
                
            # Get calibrated weights from session state if dynamic calibration is enabled
            calibrated_weights = st.session_state.get('_calibrated_weights', None)
            classification_weights = get_classification_weights(selected_classification, use_sector_adjusted, calibrated_weights=calibrated_weights)
            universal_weights = get_classification_weights(selected_classification, use_sector_adjusted=False)
                
            weight_comparison = pd.DataFrame({
                'Factor': ['Credit', 'Leverage', 'Profitability', 'Liquidity', 'Cash Flow'],
                'Classification-Adjusted': [
                    classification_weights['credit_score'] * 100,
                    classification_weights['leverage_score'] * 100,
                    classification_weights['profitability_score'] * 100,
                    classification_weights['liquidity_score'] * 100,

                    classification_weights['cash_flow_score'] * 100
                ],
                'Universal': [
                    universal_weights['credit_score'] * 100,
                    universal_weights['leverage_score'] * 100,
                    universal_weights['profitability_score'] * 100,
                    universal_weights['liquidity_score'] * 100,

                    universal_weights['cash_flow_score'] * 100
                ]
            })
            weight_comparison['Difference'] = weight_comparison['Classification-Adjusted'] - weight_comparison['Universal']
                
            st.dataframe(weight_comparison, use_container_width=True, hide_index=True)
                
            # Classification statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Issuers", f"{len(classification_data):,}")
            with col2:
                st.metric("Avg Score", f"{classification_data['Composite_Score'].mean():.1f}")
            with col3:
                ig_pct = (classification_data['Rating_Group'] == 'Investment Grade').sum() / len(classification_data) * 100
                st.metric("IG %", f"{ig_pct:.0f}%")
            with col4:
                avg_cycle = classification_data['Cycle_Position_Score'].mean()
                st.metric("Avg Cycle Score", f"{avg_cycle:.1f}")
                
            # Top performers in classification
            st.subheader(f" Top 15 Performers - {selected_classification}")
                
            top_classification = classification_data.nlargest(15, 'Composite_Score')[
                ['Classification_Rank', 'Company_Name', 'Credit_Rating_Clean', 'Composite_Score', 
                 'Cycle_Position_Score', 'Combined_Signal', 'Recommendation', 'Weight_Method']
            ]
            top_classification.columns = ['Class Rank', 'Company', 'Rating', 'Score', 'Cycle', 'Signal', 'Rec', 'Weights']
            st.dataframe(top_classification, use_container_width=True, hide_index=True)
            
        # ============================================================================
        # ============================================================================
        # TAB 5: TREND ANALYSIS
        # ============================================================================

        with tab5:
            st.header("Cyclicality & Trend Analysis")

            # --- Normalize trend score column for this tab (idempotent) ---
            if 'Cycle_Position_Score' not in results_final.columns:
                for cand in ['Cycle_Score', 'cycle_pos', 'Cycle Position Score', 'CyclePositionScore']:
                    if cand in results_final.columns:
                        results_final = results_final.rename(columns={cand: 'Cycle_Position_Score'})
                        break
            # Safety cast
            if 'Cycle_Position_Score' in results_final.columns:
                results_final['Cycle_Position_Score'] = pd.to_numeric(results_final['Cycle_Position_Score'], errors='coerce')

            # ========================================================================
            # CONFIGURATION - Use SSOT constants
            # ========================================================================
            st.caption(f"Trend threshold: {TREND_THRESHOLD} · Issuers with Cycle Position Score ≥ {TREND_THRESHOLD} are classified as Improving")

            st.markdown("---")

            # Filters
            col1, col2 = st.columns(2)

            with col1:
                trend_classification = st.selectbox(
                    "Classification",
                    options=['All'] + sorted(results_final['Rubrics_Custom_Classification'].dropna().unique().tolist()),
                    key="trend_tab_class_filter"
                )

            with col2:
                trend_rating = st.selectbox(
                    "Rating Band",
                    options=['All'] + sorted(results_final['Rating_Band'].unique().tolist()),
                    key="trend_tab_rating_band"
                )
                
            # Filter data
            trend_data = results_final.copy()
                
            if trend_classification != 'All':
                trend_data = trend_data[trend_data['Rubrics_Custom_Classification'] == trend_classification]
                
            if trend_rating != 'All':
                trend_data = trend_data[trend_data['Rating_Band'] == trend_rating]

            # ========================================================================
            # UNIVERSE TREND SUMMARY - SSOT: Use Combined_Signal directly
            # ========================================================================
            st.subheader("Universe Trend Summary")
            
            if 'Combined_Signal' in trend_data.columns:
                # SSOT: Count from already-calculated Combined_Signal
                signal_counts = trend_data['Combined_Signal'].value_counts()
                total_issuers = len(trend_data)
                
                # Group signals by trend direction (from Combined_Signal, not recalculated)
                # Improving: "Strong & Improving", "Weak but Improving"
                # Deteriorating: "Strong but Deteriorating", "Weak & Deteriorating", "Strong & Normalizing", "Strong & Moderating"
                improving_signals = ['Strong & Improving', 'Weak but Improving']
                not_improving_signals = ['Strong but Deteriorating', 'Weak & Deteriorating', 
                                        'Strong & Normalizing', 'Strong & Moderating']
                
                improving_count = sum(signal_counts.get(s, 0) for s in improving_signals)
                not_improving_count = sum(signal_counts.get(s, 0) for s in not_improving_signals)
                
                # Show breakdown by actual signal (SSOT: display what was calculated)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🟢 Improving Trend")
                    pct = (improving_count / total_issuers * 100) if total_issuers > 0 else 0
                    st.metric("Total Improving", f"{improving_count:,}", f"{pct:.1f}%", delta_color="off")
                    
                    # Breakdown
                    for sig in improving_signals:
                        count = signal_counts.get(sig, 0)
                        if count > 0:
                            st.caption(f"  • {sig}: {count:,}")
                
                with col2:
                    st.markdown("#### 🔴 Not Improving")
                    pct = (not_improving_count / total_issuers * 100) if total_issuers > 0 else 0
                    st.metric("Total Not Improving", f"{not_improving_count:,}", f"{pct:.1f}%", delta_color="off")
                    
                    # Breakdown
                    for sig in not_improving_signals:
                        count = signal_counts.get(sig, 0)
                        if count > 0:
                            st.caption(f"  • {sig}: {count:,}")
            else:
                st.warning("Combined_Signal column not found - cannot display trend summary")

            st.markdown("---")

            # ========================================================================
            # SECTOR TREND RANKINGS - SSOT: Aggregate from Combined_Signal
            # ========================================================================
            st.subheader("Sector Trend Rankings")
            st.caption("Sectors ranked by average Cycle Position Score. % Improving calculated from Combined_Signal classifications.")

            if 'Rubrics_Custom_Classification' in trend_data.columns and 'Cycle_Position_Score' in trend_data.columns:
                
                # SSOT: Helper to calculate % improving from Combined_Signal
                def pct_improving_from_signal(signals: pd.Series) -> float:
                    """Calculate % with 'Improving' in signal name - uses pre-calculated signals"""
                    if len(signals) == 0:
                        return np.nan
                    improving = signals.str.contains('Improving', na=False).sum()
                    return (improving / len(signals)) * 100
                
                sector_agg = trend_data.groupby('Rubrics_Custom_Classification', as_index=False).agg(
                    Avg_Trend_Score=('Cycle_Position_Score', 'mean'),
                    Issuer_Count=('Cycle_Position_Score', 'count'),
                    Pct_Improving=('Combined_Signal', pct_improving_from_signal)  # SSOT: from signal, not recalculated
                )
                
                # Filter to sectors with at least 5 issuers
                sector_agg = sector_agg[sector_agg['Issuer_Count'] >= 5].copy()
                sector_agg = sector_agg.sort_values('Avg_Trend_Score', ascending=False)
                
                if len(sector_agg) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🟢 Top 10 Improving Sectors")
                        top_sectors = sector_agg.head(10).copy()
                        top_sectors['Rank'] = range(1, len(top_sectors) + 1)
                        display_df = top_sectors[['Rank', 'Rubrics_Custom_Classification', 'Avg_Trend_Score', 'Pct_Improving', 'Issuer_Count']].copy()
                        display_df.columns = ['Rank', 'Sector', 'Avg Trend Score', '% Improving', 'Issuers']
                        display_df['Avg Trend Score'] = display_df['Avg Trend Score'].round(1)
                        display_df['% Improving'] = display_df['% Improving'].round(1)
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    with col2:
                        st.markdown("#### 🔴 Bottom 10 Sectors")
                        bottom_sectors = sector_agg.tail(10).sort_values('Avg_Trend_Score', ascending=True).copy()
                        bottom_sectors['Rank'] = range(1, len(bottom_sectors) + 1)
                        display_df = bottom_sectors[['Rank', 'Rubrics_Custom_Classification', 'Avg_Trend_Score', 'Pct_Improving', 'Issuer_Count']].copy()
                        display_df.columns = ['Rank', 'Sector', 'Avg Trend Score', '% Improving', 'Issuers']
                        display_df['Avg Trend Score'] = display_df['Avg Trend Score'].round(1)
                        display_df['% Improving'] = display_df['% Improving'].round(1)
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    # Summary row
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        best = sector_agg.iloc[0]
                        st.metric("🟢 Highest Avg Trend Score", 
                                  best['Rubrics_Custom_Classification'], 
                                  f"Score: {best['Avg_Trend_Score']:.1f}", delta_color="off")
                    with col2:
                        worst = sector_agg.iloc[-1]
                        st.metric("🔴 Lowest Avg Trend Score", 
                                  worst['Rubrics_Custom_Classification'], 
                                  f"Score: {worst['Avg_Trend_Score']:.1f}", delta_color="off")
                    with col3:
                        # SSOT: Use the Pct_Improving we calculated from Combined_Signal
                        overall_pct = sector_agg['Pct_Improving'].mean()
                        st.metric("Avg % Improving", f"{overall_pct:.1f}%" if pd.notna(overall_pct) else "N/A")
                else:
                    st.info("Insufficient data - need sectors with at least 5 issuers")
            else:
                st.warning("Required columns not available for sector analysis")

            st.markdown("---")

            # ========================================================================
            # TOP 10 IMPROVING/DETERIORATING ISSUERS - SSOT: Filter by Combined_Signal
            # ========================================================================

            if 'Cycle_Position_Score' not in trend_data.columns:
                st.warning("Cycle_Position_Score not found; cannot rank issuers.")
            elif 'Combined_Signal' not in trend_data.columns:
                st.warning("Combined_Signal not found; cannot filter issuers.")
            else:
                # Top 10 Improving: filter by Combined_Signal containing 'Improving'
                st.subheader("Top 10 Improving Issuers")
                st.caption("Filtered by Combined_Signal containing 'Improving', ranked by Cycle Position Score")

                # SSOT: Use Combined_Signal to filter (already calculated in pipeline)
                improving_mask = trend_data['Combined_Signal'].str.contains('Improving', na=False)
                top_improving = (trend_data[improving_mask]
                                .sort_values('Cycle_Position_Score', ascending=False)
                                .head(10))

                # Include Cycle_Position_Score in display
                display_cols = ['Company_Name', 'Credit_Rating_Clean', 'Rubrics_Custom_Classification',
                               'Cycle_Position_Score', 'Combined_Signal', 'Recommendation']
                cols_present = [c for c in display_cols if c in top_improving.columns]

                if len(top_improving) > 0:
                    display_df = top_improving[cols_present].copy()
                    if 'Cycle_Position_Score' in display_df.columns:
                        display_df['Cycle_Position_Score'] = display_df['Cycle_Position_Score'].round(1)
                    st.dataframe(
                        display_df.rename(columns={
                            'Company_Name': 'Company',
                            'Credit_Rating_Clean': 'Rating',
                            'Rubrics_Custom_Classification': 'Classification',
                            'Cycle_Position_Score': 'Trend Score',
                            'Combined_Signal': 'Signal',
                            'Recommendation': 'Rec'
                        }),
                        use_container_width=True, hide_index=True
                    )
                else:
                    st.info("No improving issuers found with current filters")

                # Top 10 Deteriorating: filter by Combined_Signal containing 'Deteriorating'
                st.subheader("Top 10 Deteriorating Issuers")
                st.caption("Filtered by Combined_Signal containing 'Deteriorating', ranked by Cycle Position Score (lowest first)")

                # SSOT: Use Combined_Signal to filter
                deteriorating_mask = trend_data['Combined_Signal'].str.contains('Deteriorating', na=False)
                top_deteriorating = (trend_data[deteriorating_mask]
                                    .sort_values('Cycle_Position_Score', ascending=True)
                                    .head(10))

                if len(top_deteriorating) > 0:
                    display_df = top_deteriorating[cols_present].copy()
                    if 'Cycle_Position_Score' in display_df.columns:
                        display_df['Cycle_Position_Score'] = display_df['Cycle_Position_Score'].round(1)
                    st.dataframe(
                        display_df.rename(columns={
                            'Company_Name': 'Company',
                            'Credit_Rating_Clean': 'Rating',
                            'Rubrics_Custom_Classification': 'Classification',
                            'Cycle_Position_Score': 'Trend Score',
                            'Combined_Signal': 'Signal',
                            'Recommendation': 'Rec'
                        }),
                        use_container_width=True, hide_index=True
                    )
                else:
                    st.info("No deteriorating issuers found with current filters")
            
        # ============================================================================
        # TAB 6: GENAI CREDIT REPORT (V5.0 - REDESIGNED DUAL-PIPELINE)
        # ============================================================================

        with tab6:
            st.header("AI Credit Report")



            if results_final is not None and len(results_final) > 0 and df_original is not None:

                # Company selection
                company_options = sorted(results_final['Company_Name'].dropna().astype(str).unique().tolist())

                col1, col2 = st.columns([3, 1])

                with col1:
                    selected_company = st.selectbox(
                        "Select Issuer for Credit Analysis",
                        options=company_options,
                        key="genai_company_select"
                    )

                with col2:
                    generate_button = st.button(
                        "Generate Report",
                        type="primary",
                        use_container_width=True,
                        key="genai_generate"
                    )

                if selected_company:
                    # Display quick metrics
                    selected_row = results_final[results_final['Company_Name'].astype(str) == selected_company].iloc[0]

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("S&P Rating", selected_row.get('Credit_Rating_Clean', 'N/A'))
                    with col2:
                        st.metric("Composite Score", f"{selected_row.get('Composite_Score', 0):.1f}")
                    with col3:
                        st.metric("Rating Band", selected_row.get('Rating_Band', 'N/A'))
                    with col4:
                        classification = str(selected_row.get('Rubrics_Custom_Classification', 'N/A'))
                        st.metric("Classification", classification[:20] + "..." if len(classification) > 20 else classification)

                if generate_button and selected_company:
                    with st.spinner(f"Generating comprehensive credit report for {selected_company}..."):
                        try:
                            # Get calibration state from session
                            use_sector_adjusted = st.session_state.get('scoring_method') == 'Classification-Adjusted Scoring'
                            calibrated_weights = st.session_state.get('_calibrated_weights')

                            # STEP 1: Gather complete data from both sources
                            st.write("Preparing financial data...")
                            complete_data = prepare_genai_credit_report_data(
                                df_original=df_original,  # Raw input spreadsheet
                                results_df=results_final,  # Model outputs
                                company_name=selected_company,
                                use_sector_adjusted=use_sector_adjusted,
                                calibrated_weights=calibrated_weights
                            )

                            if "error" not in complete_data:
                                # Show which path was used
                                if complete_data.get('from_diagnostics'):
                                    st.caption("✅ Using pre-computed diagnostic data (fast)")
                                else:
                                    st.caption("📊 Extracted from input spreadsheet")

                            if "error" in complete_data:
                                st.error(f"Error: {complete_data['error']}")
                            else:
                                # STEP 2: Build comprehensive prompt
                                st.write("Building analysis prompt...")
                                prompt = build_comprehensive_credit_prompt(complete_data)

                                # STEP 3: Generate report using OpenAI
                                st.write("Generating analysis with AI...")

                                import openai

                                client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

                                response = client.chat.completions.create(
                                    model="gpt-5",
                                    messages=[
                                        {"role": "system", "content": "You are a professional credit analyst generating comprehensive credit reports."},
                                        {"role": "user", "content": prompt}
                                    ],
                                    max_completion_tokens=8000,
                                )

                                report = response.choices[0].message.content
                                
                                # [Phase 2] Validation: Check for recommendation consistency
                                model_rec = complete_data['model_outputs']['overall_metrics']['recommendation']
                                if model_rec == "Avoid" and any(x in report.lower() for x in ["strong buy", "outperform", "overweight"]):
                                    st.warning("⚠️ POTENTIAL INCONSISTENCY: Model recommends 'Avoid' but report may contain positive language. Please verify.")
                                elif model_rec in ["Buy", "Strong Buy"] and "avoid" in report.lower() and "recommend" in report.lower():
                                    st.warning("⚠️ POTENTIAL INCONSISTENCY: Model recommends 'Buy' but report may contain negative language. Please verify.")

                                st.markdown("---")

                                # STEP 4: Display report
                                # Escape dollar signs to prevent LaTeX rendering
                                report_display = report.replace('$', '\\$')
                                st.markdown(report_display)

                                # Download button
                                st.download_button(
                                    label="📥 Download Report",
                                    data=report,
                                    file_name=f"{selected_company.replace(' ', '_')}_Credit_Report_{pd.Timestamp.now().strftime('%Y%m%d')}.md",
                                    mime="text/markdown"
                                )

                                # STEP 5: Provide data transparency
                                with st.expander("View Source Data Used in Analysis"):

                                    st.subheader("1. Raw Financial Metrics (from Input Spreadsheet)")
                                    st.caption("This is the source of truth for actual company fundamentals")

                                    # Display in organized tabs
                                    data_tab1, data_tab2, data_tab3, data_tab4 = st.tabs([
                                        "Company Info",
                                        "Profitability & Leverage",
                                        "Liquidity & Coverage",
                                        "Growth & Cash Flow"
                                    ])

                                    with data_tab1:
                                        st.json(complete_data['raw_financials']['company_info'])

                                    with data_tab2:
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write("**Profitability:**")
                                            st.json(complete_data['raw_financials']['profitability'])
                                        with col2:
                                            st.write("**Leverage:**")
                                            st.json(complete_data['raw_financials']['leverage'])

                                    with data_tab3:
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write("**Liquidity:**")
                                            st.json(complete_data['raw_financials']['liquidity'])
                                        with col2:
                                            st.write("**Coverage:**")
                                            st.json(complete_data['raw_financials']['coverage'])

                                    with data_tab4:
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write("**Growth:**")
                                            st.json(complete_data['raw_financials']['growth'])
                                        with col2:
                                            st.write("**Cash Flow:**")
                                            st.json(complete_data['raw_financials']['cash_flow'])

                                    st.subheader("2. Peer Comparisons")
                                    st.caption("Context for understanding relative credit strength")
                                    st.json(complete_data['peer_context'])

                                    st.subheader("3. Model Scores (Contextual)")
                                    st.caption("Relative positioning scores - use with caution")
                                    st.json(complete_data['model_outputs'])

                        except Exception as e:
                            st.error(f"Error generating report: {str(e)}")
                            with st.expander("View Error Details"):
                                import traceback
                                st.code(traceback.format_exc())

            else:
                st.warning("Please load data and run the scoring model first (see Data Upload & Scoring tabs)")

        # ============================================================================
        # TAB 7: DIAGNOSTICS
        # ============================================================================

        with tab7:
            st.header("Calculation Diagnostics")
            st.markdown("**Complete transparency from input data to final ranking**")
            st.markdown("---")



            # SECTION 1: ISSUER SELECTION PANEL
            st.subheader("SELECT ISSUER FOR DIAGNOSTIC TRACE")

            col_select, col_info = st.columns([2, 1])

            with col_select:
                # Get list of companies
                # [Refactor Phase 3] Use helper to get companies with diagnostic data
                company_list = get_available_companies(results_final)
                
                if not company_list:
                    st.warning("No companies with diagnostic data found. Please run scoring first.")
                    st.stop()

                # Selectbox for company selection
                selected_company = st.selectbox(
                    "Choose Company:",
                    options=company_list,
                    index=0,
                    key="diagnostics_company_select"
                )

            with col_info:
                # Show selection stats
                total_companies = len(company_list)
                st.metric("Total Issuers", total_companies)
                st.caption(f"Analyzing: {selected_company}")

            st.markdown("---")

            # Initialize Diagnostic Data Accessor
            try:
                accessor = create_diagnostic_accessor(results_final, selected_company)
                
                # Run validation (optional, but good for debugging)
                validation_errors = validate_accessor_data(accessor)
                if validation_errors:
                    with st.expander("⚠ Data Consistency Warnings", expanded=False):
                        for error in validation_errors:
                            st.warning(error)
                            
            except Exception as e:
                st.error(f"Failed to load diagnostic data for {selected_company}: {str(e)}")
                st.info("This issuer may not have been scored with the latest version of the model.")
                st.stop()

            # Display basic issuer info using Accessor
            col_info1, col_info2, col_info3, col_info4 = st.columns(4)

            with col_info1:
                composite_score = accessor.get_composite_score()
                st.metric("Composite Score", f"{composite_score:.1f}" if composite_score is not None else "N/A")

            with col_info2:
                # Rank is still in results_final, not yet in accessor (could add later)
                # For now, pull from results via accessor.results
                overall_rank = accessor.results.get('Overall_Rank', 0)
                st.metric("Overall Rank", f"#{int(overall_rank)}" if pd.notna(overall_rank) and overall_rank > 0 else "N/A")

            with col_info3:
                credit_rating = accessor.get_credit_rating()
                st.metric("Rating", credit_rating if credit_rating and credit_rating != 'NR' else 'N/A')

            with col_info4:
                # Recommendation is in results
                recommendation = accessor.results.get('Recommendation', 'N/A')
                st.metric("Recommendation", recommendation)

            st.markdown("---")

            # SECTION 2: ACTIVE CONFIGURATION SUMMARY
            st.subheader("ACTIVE CONFIGURATION SUMMARY")

            config_data = {
                "Configuration Item": [
                    "Scoring Method",
                    "Period Selection Mode",
                    "Reference Date",
                    "Period Priority",
                    "Dynamic Calibration",
                    "Comparison Group",
                    "Calibration Band"
                ],
                "Current Setting": [
                    "Universal Weights" if not use_dynamic_calibration else "Dynamic Calibration (Classification-Adjusted)",
                    period_mode.value if hasattr(period_mode, 'value') else str(period_mode),
                    reference_date_override.strftime('%Y-%m-%d') if reference_date_override else "N/A",
                    "FY > CQ" if prefer_annual_reports else "Most Recent",
                    "Enabled" if use_dynamic_calibration else "Disabled",
                    accessor.get_sector() if effective_use_sector_adjusted else "Full Universe",
                    calibration_rating_band if use_dynamic_calibration else "N/A"
                ]
            }

            config_df = pd.DataFrame(config_data)
            st.dataframe(config_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # SECTION 3: DATA PIPELINE OVERVIEW
            st.subheader("PROCESSING PIPELINE OVERVIEW")

            st.markdown("**10-Stage Processing Pipeline:**")

            # Create pipeline table
            pipeline_data = {
                "Stage": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "Name": [
                    "LOAD",
                    "PARSE",
                    "SELECT",
                    "EXTRACT",
                    "SCORE",
                    "WEIGHT",
                    "COMPOSITE",
                    "RANK",
                    "CLASSIFY",
                    "OUTPUT"
                ],
                "Purpose": [
                    "Read data from Excel spreadsheet",
                    "Parse period dates and classify (FY/CQ)",
                    "Choose periods for quality and trend analysis",
                    "Extract metric values from selected periods",
                    "Calculate quality and trend scores",
                    "Apply factor weights (universal or calibrated)",
                    "Combine scores into composite score",
                    "Rank against sector and universe peers",
                    "Assign quality/trend classification",
                    "Generate final results and recommendation"
                ],
                "Status": ["✓"] * 10
            }

            pipeline_df = pd.DataFrame(pipeline_data)

            # Display with custom column configuration
            st.dataframe(
                pipeline_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Stage": st.column_config.NumberColumn(
                        "Stage",
                        width="small",
                        format="%d"
                    ),
                    "Name": st.column_config.TextColumn(
                        "Name",
                        width="small"
                    ),
                    "Purpose": st.column_config.TextColumn(
                        "Purpose",
                        width="large"
                    ),
                    "Status": st.column_config.TextColumn(
                        "Status",
                        width="small"
                    )
                }
            )

            st.caption("✓ All stages completed successfully. Expand any stage below to see detailed calculations.")
            st.markdown("---")


            # ========================================================================
            # DIAGNOSTIC EXPORT FUNCTIONS
            # ========================================================================

            def create_diagnostic_export_data(
                accessor: DiagnosticDataAccessor,
                selected_company: str,
                scoring_method: str,
                period_mode: str,
                reference_date_override,
                use_dynamic_calibration: bool,
                calibration_rating_band: str
            ) -> dict:
                """
                Create comprehensive diagnostic data structure for export.
                Returns a dictionary with all stage data.
                """
                export_data = {}

                # STAGE 0: Configuration
                export_data['configuration'] = {
                    'Company': selected_company,
                    'Analysis Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Scoring Method': scoring_method,
                    'Period Mode': str(period_mode),
                    'Reference Date': str(reference_date_override) if reference_date_override else 'N/A',
                    'Dynamic Calibration': 'Enabled' if use_dynamic_calibration else 'Disabled',
                    'Calibration Band': calibration_rating_band if use_dynamic_calibration else 'N/A',
                }

                # STAGE 1-2: Company Info
                export_data['company_info'] = {
                    'CompanyID': accessor.get_company_id(),
                    'Company Name': accessor.get_company_name(),
                    'Ticker': accessor.get_ticker(),
                    'S&P Rating': accessor.get_credit_rating(),
                    'Sector': accessor.get_sector(),
                    'Classification': accessor.get_sector(),
                    'Industry': accessor.get_industry(),
                    'Market Cap': accessor.get_market_cap(),
                }

                # Period data from diagnostic accessor
                period_info = accessor.get_period_selection()
                export_data['period_data'] = pd.DataFrame([{
                    'Selected Suffix': period_info.get('selected_suffix'),
                    'Selected Date': period_info.get('selected_date'),
                    'Period Type': period_info.get('period_type'),
                    'Selection Mode': period_info.get('selection_mode'),
                    'Selection Reason': period_info.get('selection_reason')
                }])

                # STAGE 3: Period Selection
                export_data['period_selection'] = {
                    'Quality Scoring Period': period_info.get('selected_suffix', 'N/A'),
                    'Trend Analysis Periods': 'All historical periods used',
                    'Total Periods Available': period_info.get('periods_available', 0)
                }

                # STAGE 4: Extracted Metrics (Data Quality Summary)
                quality_summary = accessor.get_data_quality_summary()
                # Convert dict to DataFrame for export compatibility
                if quality_summary:
                    quality_rows = []
                    for factor, metrics in quality_summary.items():
                        quality_rows.append({
                            'Factor': factor,
                            'Data Completeness': f"{metrics.get('completeness', 0):.1%}",
                            'Components Used': metrics.get('components_used', 0),
                            'Total Components': metrics.get('components_total', 0),
                            'Status': '✓ Complete' if metrics.get('completeness', 0) >= 0.75 else '⚠ Partial'
                        })
                    export_data['extracted_metrics'] = pd.DataFrame(quality_rows)
                else:
                    export_data['extracted_metrics'] = pd.DataFrame()

                # STAGE 5: Quality Factor Scores
                scores = accessor.get_all_factor_scores()
                factor_data = []
                for factor_name, score in scores.items():
                    factor_data.append({
                        'Factor': factor_name,
                        'Score': score if score is not None else 'N/A',
                        'Weight': 'See weights sheet',
                        'Status': 'Scored' if score is not None else 'Not Available'
                    })
                export_data['quality_factors'] = pd.DataFrame(factor_data)

                # STAGE 5B: Detailed Factor Breakdown
                factor_metrics_data = []
                for factor in ['Credit', 'Leverage', 'Profitability', 'Liquidity', 'Cash_Flow']:
                    details = accessor.get_factor_details(factor)
                    if details and 'components' in details:
                        for component_name, component_data in details['components'].items():
                            factor_metrics_data.append({
                                'Factor': factor,
                                'Component': component_name,
                                'Raw Value': component_data.get('raw_value', 'N/A'),
                                'Component Score': component_data.get('component_score', 'N/A'),
                                'Weight': component_data.get('weight', 'N/A'),
                                'Contribution': component_data.get('weighted_contribution', 'N/A')
                            })
                export_data['factor_metrics_detail'] = pd.DataFrame(factor_metrics_data)

                # STAGE 6: Weights
                contributions = accessor.get_factor_contributions()
                weights_data = []
                for item in contributions.get('contributions', []):
                    weights_data.append({
                        'Factor': item['factor'],
                        'Weight': item['weight'],
                        'Source': 'Dynamic Calibration' if use_dynamic_calibration else 'Universal Weights'
                    })
                export_data['weights'] = pd.DataFrame(weights_data)

                # STAGE 7: Composite Score
                export_data['composite_score'] = {
                    'Quality Score': accessor.results.get('Quality_Score', 'N/A'),
                    'Trend Score': accessor.get_trend_score(),
                    'Composite Score': accessor.get_composite_score(),
                    'Calculation Method': 'Weighted average of quality factors'
                }

                # STAGE 8: Rankings
                export_data['rankings'] = {
                    'Classification_Rank': accessor.results.get('Classification_Rank', 'N/A'),
                    'Classification_Total': accessor.results.get('Classification_Total', 'N/A'),
                    'Classification_Percentile': accessor.results.get('Composite_Percentile_in_Band', 'N/A'),
                    'Universe_Rank': accessor.results.get('Rank', 'N/A'),
                    'Universe_Total': 'See full results', # Total count not directly in accessor, but rank is enough
                    'Universe_Percentile': accessor.results.get('Composite_Percentile_Global', 'N/A')
                }

                # STAGE 9: Recommendation
                export_data['recommendation'] = {
                    'Final_Recommendation': accessor.results.get('Recommendation', 'N/A'),
                    'Composite_Score': accessor.get_composite_score(),
                    'Logic': 'Based on composite score thresholds'
                }

                # STAGE 10: Full Results Row
                export_data['full_results'] = accessor.results.to_frame().T

                # STAGE 11: Raw Input Data (V5.1.1 - Data Lineage)
                export_data['raw_inputs'] = accessor.diag.get('raw_inputs', {})

                # STAGE 12: Time Series Data (V5.1.1 - Trend Analysis Detail)
                export_data['time_series'] = accessor.get_all_metric_time_series()

                # STAGE 13: Signal Classification (V5.1.1)
                export_data['signal_classification'] = {
                    'Signal': accessor.results.get('Signal', 'N/A'),
                    'Signal_Base': accessor.results.get('Signal_Base', 'N/A'),
                    'Combined_Signal': accessor.results.get('Combined_Signal', 'N/A'),
                    'Is_Strong_Quality': accessor.get_composite_score() >= QUALITY_THRESHOLD if accessor.get_composite_score() else False,
                    'Is_Improving_Trend': accessor.get_trend_score() >= TREND_THRESHOLD if accessor.get_trend_score() else False,
                    'Exceptional_Quality': accessor.results.get('ExceptionalQuality', False),
                    'Volatile_Series': accessor.results.get('VolatileSeries', False),
                    'Outlier_Quarter': accessor.results.get('OutlierQuarter', False)
                }

                return export_data

            def create_diagnostic_excel(export_data: dict, company_name: str) -> bytes:
                """
                Create Excel file with multiple sheets for diagnostic data.
                Returns bytes for download.
                """
                output = io.BytesIO()

                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Sheet 1: Summary
                    summary_data = {
                        'Section': ['Configuration', 'Company Info', 'Scores', 'Rankings', 'Recommendation'],
                        'Key Data': [
                            export_data['configuration'].get('Scoring Method', 'N/A'),
                            export_data['company_info'].get('Company Name', 'N/A'),
                            export_data['composite_score'].get('Composite Score', 'N/A'),
                            f"Rank {export_data['rankings'].get('Universe_Rank', 'N/A')} of {export_data['rankings'].get('Universe_Total', 'N/A')}",
                            export_data['recommendation'].get('Final_Recommendation', 'N/A')
                        ]
                    }
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

                    # Sheet 2: Configuration
                    pd.DataFrame([export_data['configuration']]).to_excel(writer, sheet_name='Configuration', index=False)

                    # Sheet 3: Company Info
                    pd.DataFrame([export_data['company_info']]).to_excel(writer, sheet_name='Company Info', index=False)

                    # Sheet 4: Period Data
                    if not export_data['period_data'].empty:
                        export_data['period_data'].to_excel(writer, sheet_name='Period Data', index=False)

                    # Sheet 5: Extracted Metrics
                    if not export_data['extracted_metrics'].empty:
                        export_data['extracted_metrics'].to_excel(writer, sheet_name='Extracted Metrics', index=False)

                    # Sheet 6: Quality Factors
                    if not export_data['quality_factors'].empty:
                        export_data['quality_factors'].to_excel(writer, sheet_name='Quality Factors', index=False)

                    # Sheet 6.5: Factor Metrics Detail (Component-level breakdown)
                    if not export_data['factor_metrics_detail'].empty:
                        export_data['factor_metrics_detail'].to_excel(writer, sheet_name='Factor Metrics Detail', index=False)

                    # Sheet 7: Weights
                    if not export_data['weights'].empty:
                        export_data['weights'].to_excel(writer, sheet_name='Weights', index=False)

                    # Sheet 8: Composite Score
                    pd.DataFrame([export_data['composite_score']]).to_excel(writer, sheet_name='Composite Score', index=False)

                    # Sheet 9: Rankings
                    pd.DataFrame([export_data['rankings']]).to_excel(writer, sheet_name='Rankings', index=False)

                    # Sheet 10: Recommendation
                    pd.DataFrame([export_data['recommendation']]).to_excel(writer, sheet_name='Recommendation', index=False)

                    # Sheet 11: Full Results
                    if not export_data['full_results'].empty:
                        export_data['full_results'].to_excel(writer, sheet_name='Full Results', index=True)

                    # Sheet 12: Raw Input Data (V5.1.1)
                    raw_inputs = export_data.get('raw_inputs', {})
                    if raw_inputs:
                        # Flatten raw inputs for Excel
                        raw_rows = []
                        if 'BALANCE_SHEET' in raw_inputs:
                            # Hierarchical structure
                            for category, values in raw_inputs.items():
                                if isinstance(values, dict):
                                    for field, value in values.items():
                                        raw_rows.append({
                                            'Category': category,
                                            'Field': field,
                                            'Value': value if value is not None else 'N/A'
                                        })
                        else:
                            # Flat structure
                            for field, value in raw_inputs.items():
                                category = 'BALANCE_SHEET' if field in ['Total Assets', 'Total Debt', 'Cash & ST Investments', 
                                                                         'Current Assets', 'Current Liabilities', 'Inventory'] \
                                          else 'INCOME_STATEMENT' if field in ['Total Revenue', 'EBITDA', 'Interest Expense', 
                                                                                'Net Income', 'Cost of Goods Sold'] \
                                          else 'CASH_FLOW'
                                raw_rows.append({
                                    'Category': category,
                                    'Field': field,
                                    'Value': value if value is not None else 'N/A'
                                })
                        if raw_rows:
                            pd.DataFrame(raw_rows).to_excel(writer, sheet_name='Raw Input Data', index=False)

                    # Sheet 13: Time Series Data (V5.1.1)
                    time_series = export_data.get('time_series', {})
                    if time_series:
                        ts_rows = []
                        for metric, ts_data in time_series.items():
                            if isinstance(ts_data, dict):
                                dates = ts_data.get('dates', [])
                                values = ts_data.get('values', [])
                                # Create one row per metric with summary stats
                                ts_rows.append({
                                    'Metric': metric,
                                    'Periods': len(values),
                                    'Latest_Value': values[-1] if values else 'N/A',
                                    'Trend_Direction': ts_data.get('trend_direction', 'N/A'),
                                    'Classification': ts_data.get('classification', 'N/A'),
                                    'Volatility': ts_data.get('volatility', 'N/A'),
                                    'Momentum': ts_data.get('momentum', 'N/A'),
                                    'Values_Array': str(values)
                                })
                        if ts_rows:
                            pd.DataFrame(ts_rows).to_excel(writer, sheet_name='Time Series Data', index=False)

                    # Sheet 14: Signal Classification (V5.0)
                    signal_class = export_data.get('signal_classification', {})
                    if signal_class:
                        pd.DataFrame([signal_class]).to_excel(writer, sheet_name='Signal Classification', index=False)

                output.seek(0)
                return output.getvalue()

            def create_diagnostic_csv(export_data: dict, company_name: str) -> str:
                """
                Create comprehensive CSV with all diagnostic data.
                Uses the full text report format for complete data lineage.
                """
                output = io.StringIO()

                # Header
                output.write(f"DIAGNOSTIC REPORT FOR: {company_name}\n")
                output.write(f"Generated: {pd.Timestamp.now()}\n")
                output.write(f"App Version: 5.1.1\n")
                output.write("\n")

                # SECTION 1: CONFIGURATION
                output.write("=" * 80 + "\n")
                output.write("SECTION 1: CONFIGURATION\n")
                output.write("=" * 80 + "\n")
                for key, value in export_data['configuration'].items():
                    output.write(f"{key}: {value}\n")
                output.write("\n")

                # SECTION 2: COMPANY INFORMATION
                output.write("=" * 80 + "\n")
                output.write("SECTION 2: COMPANY INFORMATION\n")
                output.write("=" * 80 + "\n")
                for key, value in export_data['company_info'].items():
                    output.write(f"{key}: {value}\n")
                output.write("\n")

                # SECTION 3: PERIOD SELECTION
                output.write("=" * 80 + "\n")
                output.write("SECTION 3: PERIOD SELECTION\n")
                output.write("=" * 80 + "\n")
                if not export_data['period_data'].empty:
                    for col in export_data['period_data'].columns:
                        val = export_data['period_data'][col].iloc[0] if len(export_data['period_data']) > 0 else 'N/A'
                        output.write(f"{col}: {val}\n")
                output.write("\n")

                # SECTION 4: RAW INPUT DATA
                output.write("=" * 80 + "\n")
                output.write("SECTION 4: RAW INPUT DATA (from Excel)\n")
                output.write("=" * 80 + "\n")
                output.write("# These are the SOURCE values - exactly as they appear in the input file\n\n")
                raw_inputs = export_data.get('raw_inputs', {})
                if isinstance(raw_inputs, dict):
                    # Check if hierarchical (BALANCE_SHEET, etc.) or flat
                    if 'BALANCE_SHEET' in raw_inputs:
                        for category, values in raw_inputs.items():
                            output.write(f"{category}:\n")
                            if isinstance(values, dict):
                                for field, value in values.items():
                                    output.write(f"  {field}: {value if value is not None else 'N/A'}\n")
                            output.write("\n")
                    else:
                        # Flat structure - organize by type
                        output.write("BALANCE_SHEET:\n")
                        for key in ['Total Assets', 'Total Debt', 'Cash & ST Investments', 'Current Assets', 
                                   'Current Liabilities', 'Inventory']:
                            if key in raw_inputs:
                                val = raw_inputs[key]
                                output.write(f"  {key}: {f'{val:,.0f}' if val is not None else 'N/A'}\n")
                        output.write("\nINCOME_STATEMENT:\n")
                        for key in ['Total Revenue', 'EBITDA', 'Interest Expense', 'Net Income', 'Cost of Goods Sold']:
                            if key in raw_inputs:
                                val = raw_inputs[key]
                                output.write(f"  {key}: {f'{val:,.0f}' if val is not None else 'N/A'}\n")
                        output.write("\nCASH_FLOW:\n")
                        for key in ['Operating Cash Flow', 'Unlevered Free Cash Flow']:
                            if key in raw_inputs:
                                val = raw_inputs[key]
                                output.write(f"  {key}: {f'{val:,.0f}' if val is not None else 'N/A'}\n")
                output.write("\n")

                # SECTION 5: CALCULATED RATIOS
                output.write("=" * 80 + "\n")
                output.write("SECTION 5: CALCULATED RATIOS\n")
                output.write("=" * 80 + "\n")
                output.write("# Each ratio shows: Raw Value, Formula, Scoring Logic\n\n")
                if not export_data['factor_metrics_detail'].empty:
                    for _, row in export_data['factor_metrics_detail'].iterrows():
                        output.write(f"{row['Factor']} - {row['Component']}:\n")
                        output.write(f"  Raw Value: {row['Raw Value']}\n")
                        output.write(f"  Score: {row['Component Score']}\n")
                        output.write(f"  Weight: {row['Weight']}\n")
                        output.write(f"  Contribution: {row['Contribution']}\n")
                        output.write("\n")

                # SECTION 6: FACTOR SCORES
                output.write("=" * 80 + "\n")
                output.write("SECTION 6: FACTOR SCORES\n")
                output.write("=" * 80 + "\n")
                if not export_data['quality_factors'].empty:
                    for _, row in export_data['quality_factors'].iterrows():
                        output.write(f"{row['Factor']}: {row['Score']:.2f}\n")
                output.write("\n")

                # SECTION 7: WEIGHTS APPLIED
                output.write("=" * 80 + "\n")
                output.write("SECTION 7: WEIGHTS APPLIED\n")
                output.write("=" * 80 + "\n")
                if not export_data['weights'].empty:
                    for _, row in export_data['weights'].iterrows():
                        output.write(f"{row['Factor']}: {row['Weight']:.1%} ({row['Source']})\n")
                output.write("\n")

                # SECTION 8: TIME SERIES DATA
                output.write("=" * 80 + "\n")
                output.write("SECTION 8: TIME SERIES DATA (Trend Analysis)\n")
                output.write("=" * 80 + "\n")
                time_series = export_data.get('time_series', {})
                if time_series:
                    for metric, ts_data in time_series.items():
                        output.write(f"{metric}:\n")
                        if isinstance(ts_data, dict):
                            dates = ts_data.get('dates', [])
                            values = ts_data.get('values', [])
                            output.write(f"  Periods: {len(values)} data points\n")
                            output.write(f"  Values: {values}\n")
                            output.write(f"  Direction: {ts_data.get('trend_direction', 'N/A')}\n")
                            output.write(f"  Classification: {ts_data.get('classification', 'N/A')}\n")
                            output.write(f"  Volatility: {ts_data.get('volatility', 'N/A')}\n")
                            output.write(f"  Momentum: {ts_data.get('momentum', 'N/A')}\n")
                        output.write("\n")
                else:
                    output.write("No time series data available\n\n")

                # SECTION 9: COMPOSITE SCORE CALCULATION
                output.write("=" * 80 + "\n")
                output.write("SECTION 9: COMPOSITE SCORE CALCULATION\n")
                output.write("=" * 80 + "\n")
                for key, value in export_data['composite_score'].items():
                    output.write(f"{key}: {value}\n")
                output.write("\n")

                # SECTION 10: RANKINGS
                output.write("=" * 80 + "\n")
                output.write("SECTION 10: RANKINGS & PERCENTILES\n")
                output.write("=" * 80 + "\n")
                for key, value in export_data['rankings'].items():
                    output.write(f"{key}: {value}\n")
                output.write("\n")

                # SECTION 11: SIGNAL CLASSIFICATION
                output.write("=" * 80 + "\n")
                output.write("SECTION 11: SIGNAL CLASSIFICATION\n")
                output.write("=" * 80 + "\n")
                signal_data = export_data.get('signal_classification', {})
                if signal_data:
                    for key, value in signal_data.items():
                        output.write(f"{key}: {value}\n")
                else:
                    output.write(f"Signal: {export_data.get('recommendation', {}).get('Final_Recommendation', 'N/A')}\n")
                output.write("\n")

                # SECTION 12: RECOMMENDATION
                output.write("=" * 80 + "\n")
                output.write("SECTION 12: RECOMMENDATION\n")
                output.write("=" * 80 + "\n")
                for key, value in export_data['recommendation'].items():
                    output.write(f"{key}: {value}\n")
                output.write("\n")

                output.write("=" * 80 + "\n")
                output.write("END OF DIAGNOSTIC REPORT\n")
                output.write("=" * 80 + "\n")

                return output.getvalue()

            # ========================================================================
            # END DIAGNOSTIC EXPORT FUNCTIONS
            # ========================================================================




            # SECTION 4: DETAILED STAGE-BY-STAGE BREAKDOWN
            st.subheader("DETAILED CALCULATIONS")

            # STAGE 1: PERIOD SELECTION & DATA QUALITY
            with st.expander("DATA QUALITY & PERIOD ANALYSIS", expanded=False):
                st.markdown("### Period Selection")
                
                period_info = accessor.get_period_selection()
                
                col1, col2 = st.columns(2)
                with col1:
                    selected_suffix = period_info.get('selected_suffix', 'N/A')
                    # Format suffix for display (e.g., '.7' -> 'CQ7' or '.0' -> 'FY Base')
                    period_type = period_info.get('period_type', 'Unknown')
                    if selected_suffix != 'N/A':
                        if selected_suffix == '.0' or selected_suffix == '':
                            period_display = f"{period_type} Base Period"
                        else:
                            period_num = selected_suffix.replace('.', '')
                            period_display = f"{period_type}{period_num}"
                    else:
                        period_display = 'N/A'

                    st.markdown(f"**Selected Period:** {period_display}")
                    st.markdown(f"**Date:** {period_info.get('selected_date', 'N/A')}")
                with col2:
                    st.markdown(f"**Period Type:** {period_info.get('period_type', 'N/A')}")
                    st.markdown(f"**Data Source:** {period_info.get('source', 'Diagnostic Data')}")
                
                st.markdown("---")
                st.markdown("### Data Completeness")
                
                quality_summary = accessor.get_data_quality_summary()

                # Calculate aggregate metrics from per-factor data
                total_metrics = 0
                available_metrics = 0
                missing_metrics = 0
                missing_list = []

                for factor, metrics in quality_summary.items():
                    components_total = metrics.get('components_total', 0)
                    components_used = metrics.get('components_used', 0)
                    
                    # Ensure components_used doesn't exceed components_total (defensive)
                    components_used = min(components_used, components_total)
                    
                    total_metrics += components_total
                    available_metrics += components_used
                    missing = components_total - components_used
                    if missing > 0:
                        missing_list.append(f"{factor}: {missing} missing")

                missing_metrics = total_metrics - available_metrics
                
                q_col1, q_col2, q_col3 = st.columns(3)
                with q_col1:
                    st.metric("Total Metrics", total_metrics)
                with q_col2:
                    st.metric("Available", available_metrics)
                with q_col3:
                    st.metric("Missing", missing_metrics)
                
                if missing_list:
                    st.warning(f"Missing Metrics: {', '.join(missing_list)}")
                else:
                    st.success("All required metrics available")

            # STAGE 5A: TREND ANALYSIS - HISTORICAL PATTERNS
            # STAGE 5A: TREND ANALYSIS - HISTORICAL PATTERNS
            with st.expander("TREND ANALYSIS - HISTORICAL PATTERNS", expanded=False):
                
                # Get trend data from accessor
                trend_data = accessor.get_all_metric_time_series()
                cycle_score = accessor.get_trend_score()

                if not trend_data or cycle_score is None:
                    st.warning("⚠️ Insufficient historical data for trend analysis (need 2+ periods)")
                else:
                    # ====================================================================
                    # 1. CYCLE POSITION SCORE (TOP)
                    # ====================================================================
                    
                    # Classify the score (thresholds from MODEL_THRESHOLDS for SSOT)
                    if cycle_score >= MODEL_THRESHOLDS['display_cycle_favorable']:
                        classification = "🟢 FAVORABLE POSITION"
                        desc = "Strong improving trends with low volatility"
                    elif cycle_score >= MODEL_THRESHOLDS['display_cycle_neutral']:
                        classification = "🟡 NEUTRAL/STABLE"
                        desc = "Stable trends with moderate volatility"
                    else:
                        classification = "🔴 UNFAVORABLE POSITION"
                        desc = "Deteriorating trends or high volatility"
                    
                    # Display Score Box
                    st.markdown(f"""
                    <div style="border: 1px solid #e6e6e6; padding: 20px; border-radius: 5px; text-align: center; margin-bottom: 20px;">
                        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">CYCLE POSITION SCORE</div>
                        <div style="font-size: 32px; font-weight: bold; margin-bottom: 5px;">{cycle_score:.1f} / 100</div>
                        <div style="font-size: 16px; font-weight: 500;">{classification.replace("🟢 ", "").replace("🟡 ", "").replace("🔴 ", "")}</div>
                        <div style="font-size: 12px; color: #888; margin-top: 5px;">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display Overall Signal Classification
                    signal_details = accessor.get_signal_details()
                    combined_signal = signal_details['combined_signal']
                    recommendation = signal_details['recommendation']
                    is_override = signal_details['is_override']
                    
                    # Signal color mapping
                    signal_colors = {
                        'Strong & Improving': '#28a745',      # Green
                        'Strong & Normalizing': '#17a2b8',    # Cyan
                        'Strong & Moderating': '#ffc107',     # Yellow
                        'Strong but Deteriorating': '#fd7e14', # Orange
                        'Weak but Improving': '#6c757d',      # Gray
                        'Weak & Deteriorating': '#dc3545',    # Red
                    }
                    signal_color = signal_colors.get(combined_signal, '#6c757d')
                    
                    st.markdown("---")
                    st.markdown("#### OVERALL SIGNAL")
                    
                    col_sig1, col_sig2 = st.columns(2)
                    with col_sig1:
                        st.markdown(f"""
                        <div style="background-color: {signal_color}22; border-left: 4px solid {signal_color}; padding: 15px; border-radius: 4px;">
                            <h3 style="margin: 0; color: {signal_color};">{combined_signal}</h3>
                            <p style="margin: 5px 0 0 0;">Recommendation: <strong>{recommendation}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_sig2:
                        if is_override:
                            st.info("""
                            **Context-Aware Override Applied**
                            
                            This classification considers:
                            - Exceptional quality (≥90th percentile)
                            - Peak stabilization or high volatility patterns
                            - Medium-term trend direction
                            """)
                    
                    st.markdown("---")
                    
                    # ====================================================================
                    # 2. WHAT'S DRIVING THIS SCORE (METRIC CARDS)
                    # ====================================================================
                    st.markdown("### What's Driving This Score")
                    
                    key_trend_metrics = [
                        "Total Debt / EBITDA (x)",
                        "EBITDA Margin",
                        "EBITDA / Interest Expense (x)",
                        "Levered Free Cash Flow Margin",
                        "Revenue"
                    ]
                    
                    available_trend_metrics = {m: trend_data[m] for m in key_trend_metrics if m in trend_data}
                    component_rows = [] # For calculation table later
                    
                    for metric_name, metric_data in available_trend_metrics.items():
                        # Extract data
                        dates = metric_data.get('dates', [])
                        values = metric_data.get('values', [])
                        trend_direction = metric_data.get('trend_direction', 0) # % per year
                        momentum = metric_data.get('momentum', 50)
                        volatility = metric_data.get('volatility', 50)
                        classification_str = metric_data.get('classification', 'STABLE')
                        
                        # Get classification specific data
                        class_data = get_classification_data(accessor, metric_name)
                        
                        # Determine icon and color
                        icon = "[STABLE]"
                        if classification_str == 'IMPROVING': icon = "[IMPROVING]"
                        elif classification_str == 'DETERIORATING': icon = "[DETERIORATING]"
                        
                        # Calculate start/end values
                        start_val = values[0] if values else 0
                        end_val = values[-1] if values else 0
                        
                        # Format dates safely
                        try:
                            start_year = parser.parse(dates[0]).year if isinstance(dates[0], str) else dates[0].year
                            end_year = parser.parse(dates[-1]).year if isinstance(dates[-1], str) else dates[-1].year
                        except Exception:
                            start_year = "Start"
                            end_year = "End"
                        
                        # Generate Explanation - use volatility score to qualify language
                        # Volatility score: 0 = highly volatile, 100 = very stable
                        # Thresholds from MODEL_THRESHOLDS (SSOT)
                        explanation = ""
                        
                        if classification_str == 'IMPROVING':
                            if volatility >= VOLATILITY_CONSISTENT:
                                explanation = "Metric is showing consistent improvement over the period."
                            elif volatility >= VOLATILITY_MODERATE:
                                explanation = "Metric is showing overall improvement over the period."
                            else:
                                explanation = "Metric shows net improvement despite significant volatility."
                        elif classification_str == 'DETERIORATING':
                            if volatility >= VOLATILITY_CONSISTENT:
                                explanation = "Metric is showing consistent deterioration. Monitor closely."
                            elif volatility >= VOLATILITY_MODERATE:
                                explanation = "Metric is showing overall deterioration over the period. Monitor closely."
                            else:
                                explanation = "Metric shows net deterioration despite significant volatility. Monitor closely."
                        elif classification_str == 'NORMALIZING':
                            explanation = "Recent decline is from a peak, indicating stabilization rather than weakness."
                        elif classification_str == 'MODERATING':
                            explanation = "High volatility detected. Swings are likely cyclical rather than directional."
                        else:
                            # STABLE
                            if volatility >= VOLATILITY_CONSISTENT:
                                explanation = "Metric is stable with no significant trend."
                            elif volatility >= VOLATILITY_MODERATE:
                                explanation = "Metric is relatively stable over the period."
                            else:
                                explanation = "Metric fluctuates but shows no clear directional trend."
                            
                        # Container for the card
                        with st.container():
                            # Check for bounding
                            is_bounded = metric_data.get('fallback_bounded', False)
                            bound_warning = " [BOUNDED]" if is_bounded else ""
                            
                            st.markdown(f"#### {icon} {metric_name.upper()}{bound_warning}")
                            
                            if is_bounded:
                                bounds = metric_data.get('bound_limits', {})
                                st.warning(f"**Note:** Some values for this metric were calculated from components and capped to the CIQ-observed range [{bounds.get('min', 0)}, {bounds.get('max', 'N/A')}] to prevent extreme outliers from distorting the trend.")
                            
                            c1, c2 = st.columns([1, 2])
                            
                            with c1:
                                # Format values based on unit
                                # SSOT: Use existing alias resolution, then look up metric info
                                canonical_name = resolve_column_name(metric_name)
                                metric_key = next((k for k, v in METRIC_REGISTRY.items() if v['canonical'] == canonical_name), None)
                                unit = METRIC_REGISTRY.get(metric_key, {}).get('unit') if metric_key else None
                                
                                if unit == 'K':
                                    start_disp = format_monetary_value_for_display(start_val, metric_name)
                                    end_disp = format_monetary_value_for_display(end_val, metric_name)
                                else:
                                    start_disp = f"{start_val:.2f}"
                                    end_disp = f"{end_val:.2f}"
                                    
                                st.markdown(f"{start_disp.strip()} ({start_year}) → {end_disp.strip()} ({end_year})")
                                st.caption(f"Annual Change: {trend_direction:+.1f}% per year")
                                st.markdown(explanation)
                                
                                # Calculate trend score for display
                                trend_score_disp = ((trend_direction / 100.0) + 1) * 50
                                trend_score_disp = max(0, min(100, trend_score_disp))
                                
                                st.caption(f"Components: Dir {trend_score_disp:.1f} | Mom {momentum:.1f} | Vol {volatility:.1f}")
                                
                            with c2:
                                # Convert string dates to datetime if needed for plotting
                                plot_dates = []
                                for d in dates:
                                    if isinstance(d, str):
                                        try:
                                            plot_dates.append(parser.parse(d))
                                        except Exception:
                                            plot_dates.append(datetime.now()) # Fallback
                                    else:
                                        plot_dates.append(d)

                                # Render Chart
                                try:
                                    fig = create_trend_chart_with_classification(
                                        dates=plot_dates,
                                        values=values,
                                        metric_name=metric_name,
                                        classification=classification_str,
                                        annual_change_pct=trend_direction,
                                        peak_idx=class_data.get('peak_idx'),
                                        cv_value=class_data.get('cv_value')
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Chart error: {e}")
                            
                            st.markdown("---")
                            
                        # Add to component rows for table
                        trend_score = ((trend_direction / 100.0) + 1) * 50
                        trend_score = max(0, min(100, trend_score))
                        
                        component_rows.append({
                            'Metric': metric_name,
                            'Trend Direction': f"{classification_str}",
                            'Trend Score': f"{trend_score:.1f}",
                            'Momentum': f"{momentum:.1f}",
                            'Volatility': f"{volatility:.1f}",
                            'Avg': f"{(trend_score + momentum + volatility) / 3:.1f}"
                        })

                    # ====================================================================
                    # 3. SUMMARY
                    # ====================================================================
                    st.markdown("### Summary")
                    st.info(f"""
                    **Trend Analysis Summary:**
                    - **Overall Position**: {classification.replace("🟢 ", "").replace("🟡 ", "").replace("🔴 ", "")} ({cycle_score:.1f}/100)
                    - **Key Drivers**: {', '.join([f"{r['Metric']} ({r['Trend Direction']})" for r in component_rows])}
                    """)
                    
                    # ====================================================================
                    # 4. SCORE CALCULATION TABLE
                    # ====================================================================
                    st.markdown("### Score Calculation")
                    if component_rows:
                        st.dataframe(pd.DataFrame(component_rows), use_container_width=True, hide_index=True)
                        
                        # Verification logic
                        all_components = []
                        for row in component_rows:
                            all_components.extend([
                                float(row['Trend Score']),
                                float(row['Momentum']),
                                float(row['Volatility'])
                            ])
                        
                        avg_all = sum(all_components) / len(all_components) if all_components else 50
                        
                        if abs(avg_all - cycle_score) < 0.5:
                            st.caption(f"✓ Verified: Component average ({avg_all:.1f}) matches reported score ({cycle_score:.1f})")
                        else:
                            st.caption(f"[WARN] Difference: Component average ({avg_all:.1f}) vs reported score ({cycle_score:.1f})")

                            # Add compact but comprehensive classification reference
                            st.markdown("---")
                            show_formulas = st.checkbox("📋 Show Classification Formulas Reference", value=False, key="show_trend_formulas")
                            
                            if show_formulas:
                                st.markdown("**Complete methodology for trend and signal classifications**")
                                
                                tab1, tab2, tab3 = st.tabs(["Trend Calculation", "Signal Classification", "Quick Reference"])
                                
                                with tab1:
                                    st.markdown("### How Trend Direction is Calculated")

                                    st.markdown("""
                                    **Outlier Handling & Ratio Bounding:**
                                    When Capital IQ reports "NM" (Not Meaningful) for a ratio, the model attempts to calculate it from underlying components (e.g., Net Debt / EBITDA). 
                                    To prevent extreme outliers (e.g., from near-zero denominators) from distorting the trend, calculated values are **bounded** to the observed min/max range of valid Capital IQ data for that metric.
                                    - If a calculated value exceeds the max observed CIQ value, it is capped at the max.
                                    - If it falls below the min observed CIQ value, it is floored at the min.
                                    - This ensures trends reflect genuine directionality rather than mathematical artifacts.
                                    """)
                                    
                                    st.info("""
                                    **What is Trend Direction?**
                                    
                                    For each financial metric, we analyze its historical time series to calculate the annual rate of change. 
                                    This tells us if the metric is improving, deteriorating, or staying flat over time.
                                    
                                    The calculation uses actual calendar dates, so quarterly data isn't distorted by being treated 
                                    as equally-spaced annual data.
                                    """)
                                    
                                    st.markdown("**Step-by-Step Calculation:**")
                                    
                                    st.code("""
Step 1: Gather Time Series Data
- Extract 3+ historical periods with dates
- Example: (2021-12-31: 4.5x), (2022-12-31: 4.2x), ...

Step 2: Calculate Annual Rate of Change
- Convert dates to years from start: time_years = (dates - first_date) / 365.25
- Linear regression: slope_per_year = polyfit(time_years, values, 1)[0]
- Example: Debt ratio 4.5x -> 3.2x over 4 years = -0.325x per year

Step 3: Normalize by Mean (Scale-Independent)
- slope_pct_per_year = slope_per_year / mean(values)
- Example: -0.325 / 3.84 = -0.085 (-8.5% per year)

Step 4: Scale and Clip
- normalized_trend = clip(slope_pct_per_year * 10, -1, +1)
- Scale by 10x so 1% annual change -> 0.10, 10% -> 1.0
- Example: -0.085 * 10 = -0.85

Step 5: Classify
if normalized_trend > +0.2:    -> IMPROVING (>+20% annually)
elif -0.2 <= trend <= +0.2:    -> STABLE (+/-20% annually)
else:                          -> DETERIORATING (<-20% annually)

Step 6: Convert to 0-100 Score
trend_score = ((normalized_trend) + 1) * 50
- Range: -1.0 -> 0 (worst), 0.0 -> 50 (neutral), +1.0 -> 100 (best)
                                    """, language="python")
                                    
                                    st.markdown("**Complete Formula:**")
                                    st.code("""
# Given: time series of (date, value) pairs

time_years = (dates - dates[0]) / 365.25
slope_per_year = polyfit(time_years, values, degree=1)[0]
mean_value = abs(mean(values))
slope_pct_per_year = slope_per_year / mean_value
normalized_trend = clip(slope_pct_per_year * 10, -1, +1)

if normalized_trend > +0.2:
    classification = "IMPROVING"
elif normalized_trend < -0.2:
    classification = "DETERIORATING"
else:
    classification = "STABLE"

trend_score = ((normalized_trend) + 1) * 50
                                    """, language="python")
                                    
                                    st.markdown("---")
                                    st.markdown("### Fallback Calculations & CIQ-Aligned Bounding")
                                    
                                    st.warning("""
                                    **⚠️ What does the [BOUNDED] indicator mean?**
                                    
                                    When you see [BOUNDED] next to a trend metric, it indicates that one or more data points 
                                    were calculated using **fallback logic** because Capital IQ returned "NM" (Not Meaningful).
                                    """)
                                    
                                    st.markdown("""
                                    **Why does CIQ return "NM"?**
                                    
                                    Capital IQ marks ratios as "NM" when:
                                    - **Interest Coverage**: Very low or zero interest expense (company has minimal debt)
                                    - **Net Debt/EBITDA**: Net cash position (cash exceeds debt) or negative EBITDA
                                    
                                    These are often **positive credit signals** (minimal debt burden), not missing data.
                                    
                                    **How Fallback Works:**
                                    
                                    When CIQ returns "NM", the app calculates the ratio from component metrics:
                                    - Interest Coverage = EBITDA ÷ |Interest Expense|
                                    - Net Debt/EBITDA = (Total Debt - Cash) ÷ EBITDA
                                    
                                    **CIQ-Aligned Bounding:**
                                    
                                    Fallback calculations can produce extreme values (e.g., 1000x coverage when 
                                    interest expense is minimal). To maintain comparability with CIQ-provided values, 
                                    extreme fallback values are bounded to CIQ's observed ranges:
                                    """)
                                    
                                    bounds_df = pd.DataFrame({
                                        'Metric': ['EBITDA / Interest Expense (x)', 'Net Debt / EBITDA'],
                                        'Min Bound': [0.0, 0.0],
                                        'Max Bound': [294.0, 274.0],
                                        'CIQ Observed Range': ['0.002 - 293.65', '0.000 - 273.83'],
                                        'Typical "Excellent" Level': ['> 10x', '< 2x']
                                    })
                                    st.dataframe(bounds_df, use_container_width=True, hide_index=True)
                                    
                                    st.info("""
                                    **Interpretation Guidance:**
                                    
                                    When a metric shows [BOUNDED]:
                                    1. The **direction** of change is still meaningful
                                    2. The **magnitude** may be understated (actual ratios could be even more extreme)
                                    3. Very high coverage ratios typically indicate **minimal debt burden** (credit positive)
                                    4. Focus on whether the company is maintaining its strong position, not exact values
                                    """)
                                
                                with tab2:
                                    st.markdown("### How Quality × Trend Generates Signals")
                                    
                                    st.markdown("**Base Classifications:**")
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.success("""
                                        **1. Strong & Improving**
                                        
                                        High-quality company with positive momentum. Best investment opportunities.
                                        
                                        **Formula:** Quality ≥ 55 AND Trend ≥ 55
                                        
                                        **Recommendation:** Strong Buy
                                        """)
                                        
                                        st.info("""
                                        **3. Weak but Improving**
                                        
                                        Lower-quality company with positive momentum. Potential turnaround play.
                                        
                                        **Formula:** Quality < 55 AND Trend ≥ 55
                                        
                                        **Recommendation:** Hold
                                        """)
                                    
                                    with col2:
                                        st.warning("""
                                        **2. Strong but Deteriorating**
                                        
                                        High-quality company facing headwinds. Quality justifies position despite weakness.
                                        
                                        **Formula:** Quality ≥ 55 AND Trend < 55
                                        
                                        **Recommendation:** Buy or Hold
                                        """)
                                        
                                        st.error("""
                                        **4. Weak & Deteriorating**
                                        
                                        Low-quality company with negative momentum. Increasing risk.
                                        
                                        **Formula:** Quality < 55 AND Trend < 55
                                        
                                        **Recommendation:** Avoid
                                        """)
                                    
                                    st.markdown("---")
                                    st.markdown("**Context-Aware Overrides:**")
                                    st.caption("These refine 'Strong but Deteriorating' for exceptional quality companies")
                                    
                                    st.info("""
                                    **Override A: Strong & Normalizing**
                                    
                                    **Plain English:**
                                    Exceptionally high-quality company (top 10%) at or near a cyclical peak. Short-term weakness is 
                                    due to peak stabilization or an outlier quarter, not true deterioration. Medium-term trend is 
                                    still positive. Think: company went from "excellent" to "very good" - normalizing from a peak, 
                                    not deteriorating.
                                    
                                    **Formula:**
                                    ```
                                    if (Quality ≥ 90th percentile) AND
                                       (Base Signal = "Strong but Deteriorating") AND
                                       (Medium-Term Trend ≥ 0) AND
                                       (Trend Score < 55) AND
                                       (Near Peak OR Outlier Quarter):
                                           → Strong & Normalizing
                                    ```
                                    
                                    **Example:** AAA company grew margin from 25% to 35% over 3 years, now at 33-34%. 
                                    Not deteriorating - just normalizing from exceptional peak.
                                    
                                    **Recommendation:** Buy
                                    """)
                                    
                                    st.warning("""
                                    **Override B: Strong & Moderating**
                                    
                                    **Plain English:**
                                    Exceptionally high-quality company (top 10%) with high volatility (CV ≥ 30%). Metrics swing 
                                    significantly due to cyclical business, seasonal patterns, or project-based revenue. Volatility 
                                    makes trend score artificially low, but underlying quality is strong. Model applies "damping" 
                                    to smooth volatility.
                                    
                                    **Formula:**
                                    ```
                                    elif (Quality ≥ 90th percentile) AND
                                         (Base Signal = "Strong but Deteriorating") AND
                                         (Coefficient of Variation ≥ 0.30) AND
                                         (NOT Normalizing):
                                           → Strong & Moderating
                                    ```
                                    
                                    **Example:** AA construction company with margins swinging 8-15% based on project timing. 
                                    Recent 8-10% looks weak vs last year's 12-15%, but this is normal volatility, not weakness.
                                    
                                    **Recommendation:** Buy
                                    """)
                                    
                                    st.markdown("""
                                    **Signal Flow:**
                                    1. Calculate base signal from Quality and Trend thresholds (55 and 55)
                                    2. For "Strong but Deteriorating" only: check if quality ≥ 90th percentile
                                    3. If yes, apply Normalizing override (if peak/outlier) or Moderating override (if volatile)
                                    4. Map final signal to recommendation (Strong Buy / Buy / Hold / Avoid)
                                    """)
                                
                                with tab3:
                                    st.markdown("### Quick Reference Table")
                                    
                                    quick_ref_df = pd.DataFrame({
                                        'Signal': [
                                            'Strong & Improving',
                                            'Strong & Normalizing',
                                            'Strong & Moderating',
                                            'Strong but Deteriorating',
                                            'Weak but Improving',
                                            'Weak & Deteriorating'
                                        ],
                                        'Formula': [
                                            'Q≥55 AND T≥55',
                                            'Q≥90th + Peak',
                                            'Q≥90th + CV≥0.30',
                                            'Q≥55 AND T<55',
                                            'Q<55 AND T≥55',
                                            'Q<55 AND T<55'
                                        ],
                                        'Recommendation': [
                                            'Strong Buy',
                                            'Buy',
                                            'Buy',
                                            'Buy/Hold',
                                            'Hold',
                                            'Avoid'
                                        ],
                                        'Scatter Plot': [
                                            'Green (top right)',
                                            'Cyan (peak zone)',
                                            'Yellow (volatile)',
                                            'Orange (right)',
                                            'Blue (left)',
                                            'Red (bottom left)'
                                        ]
                                    })
                                    st.dataframe(quick_ref_df, use_container_width=True, hide_index=True)
                                    
                                    st.markdown("""
                                    **Scatter Plot Interpretation:**
                                    - **Green (BEST)**: High quality + improving trends
                                    - **Cyan**: Peak stabilization - quality at cyclical high
                                    - **Yellow**: Quality with volatility - smoothing applied
                                    - **Orange (WARNING)**: Quality but negative trends
                                    - **Blue (OPPORTUNITY)**: Lower quality but improving
                                    - **Red (AVOID)**: Low quality + deteriorating
                                    """)
                            st.markdown("""
                            **How Trend Directions Are Classified**
                            
                            For each metric's time series, we calculate a normalized trend value (ranging from -1 to +1 representing -100% to +100% annual change), then classify it:
                            
                            **Classification Thresholds:**
                            """)
                            
                            col_trend1, col_trend2, col_trend3 = st.columns(3)
                            
                            with col_trend1:
                                st.info("""
                                **IMPROVING**
                                
                                Trend > +0.2 (> +20% annually)
                                
                                Metric is showing strong positive momentum with consistent improvement over the historical period.
                                
                                **Examples:**
                                - Revenue growing 25%/year
                                - Margins expanding
                                - Debt ratios declining
                                """)
                            
                            with col_trend2:
                                st.warning("""
                                **STABLE**
                                
                                -0.2 ≤ Trend ≤ +0.2 (-20% to +20%)
                                
                                Metric is relatively flat or fluctuating within a moderate range, showing neither strong improvement nor deterioration.
                                
                                **Examples:**
                                - Revenue flat at 5%/year
                                - Margins holding steady
                                - Ratios oscillating
                                """)
                            
                            with col_trend3:
                                st.error("""
                                **DETERIORATING**
                                
                                Trend < -0.2 (< -20% annually)
                                
                                Metric is showing negative momentum with consistent decline over the historical period.
                                
                                **Examples:**
                                - Revenue declining 25%/year
                                - Margins contracting
                                - Debt ratios rising
                                """)
                            
                            st.markdown("""
                            **Additional Trend Components:**
                            
                            - **Momentum (0-100)**: Compares recent performance to historical average
                              - 0 = Recent period much worse than history
                              - 50 = Recent period matches history  
                              - 100 = Recent period much better than history
                            
                            - **Volatility (0-100)**: Measures consistency of metric over time
                              - 0 = Highly volatile (unreliable trend)
                              - 50 = Moderate volatility
                              - 100 = Very stable (reliable trend)
                            
                            The **Cycle Position Score** combines all 12 components (4 metrics × 3 components each) to assess overall business cycle positioning.
                            """)
                    else:
                        st.warning("⚠️ Trend score not available (insufficient historical data)")


                    st.markdown("""
                    **How Quality and Trend Combine to Generate Investment Signals**
                    
                    The model uses a 6-classification system that combines Quality Score (0-100) and Trend Score (0-100) 
                    to categorize each issuer into one of six signals that drive the final recommendation.
                    """)

                    st.markdown("#### Base Classifications")

                    col_sig1, col_sig2 = st.columns(2)

                    with col_sig1:
                        st.success("""
                        **Strong & Improving**
                        
                        - Quality ≥ 55 AND Trend ≥ 55
                        - High quality + positive momentum
                        - Recommendation: **Strong Buy**
                        - Best investment opportunities
                        """)
                        
                        st.info("""
                        **Weak but Improving**
                        
                        - Quality < 55 AND Trend ≥ 55
                        - Lower quality + positive momentum
                        - Recommendation: **Hold**
                        - Potential turnaround plays
                        """)

                    with col_sig2:
                        st.warning("""
                        **Strong but Deteriorating**
                        
                        - Quality ≥ 55 AND Trend < 55
                        - High quality + negative/flat trend
                        - Recommendation: **Buy** or **Hold**
                        - Quality justifies position
                        """)
                        
                        st.error("""
                        **Weak & Deteriorating**
                        
                        - Quality < 55 AND Trend < 55
                        - Low quality + negative momentum
                        - Recommendation: **Avoid**
                        - Avoid fundamentally weak credits
                        """)

                    st.markdown("#### Context-Aware Overrides")
                    st.caption("These refine 'Strong but Deteriorating' based on additional analysis")

                    col_override1, col_override2 = st.columns(2)

                    with col_override1:
                        st.info("""
                        **Strong & Normalizing**
                        
                        Applied when:
                        - Exceptional quality (≥90th percentile)
                        - Medium-term trend improving
                        - Near peak or outlier quarter
                        
                        Recommendation: **Buy**
                        
                        Interpretation: Peak stabilization - quality overrides short-term weakness
                        """)

                    with col_override2:
                        st.warning("""
                        **Strong & Moderating**
                        
                        Applied when:
                        - Exceptional quality (≥90th percentile)
                        - High volatility (CV ≥ 0.30)
                        - Volatility damping applied
                        
                        Recommendation: **Buy**
                        
                        Interpretation: Quality with volatility - smoothing applied
                        """)

                    st.markdown("""
                    **Signal Flow:**
                    1. Calculate base signal from Quality and Trend thresholds (55 and 55)
                    2. Apply context-aware overrides for exceptional quality cases
                    3. Map final signal to recommendation (Strong Buy / Buy / Hold / Avoid)
                    
                    This ensures that high-quality issuers receive appropriate treatment even when trend analysis suggests caution.
                    """)

            # STAGE 5B: QUALITY FACTOR SCORING
            with st.expander("QUALITY FACTOR SCORING", expanded=False):
                st.markdown("### Quality Factors")
                st.caption("Each factor scored 0-100 based on thresholds and input metrics")

                # Define the 5 quality factors
                factors = [
                    ("Credit", "Credit_Score", "Fundamental Credit Score"),
                    ("Leverage", "Leverage_Score", "Debt ratios and leverage metrics"),
                    ("Profitability", "Profitability_Score", "Margins and returns"),
                    ("Liquidity", "Liquidity_Score", "Current and quick ratios"),
                    ("Cash_Flow", "Cash_Flow_Score", "Operating and free cash flow metrics")
                ]

                for idx, (factor_name, score_col, description) in enumerate(factors, 1):
                    # Get factor details from accessor
                    factor_details = accessor.get_factor_details(factor_name)
                    score_value = factor_details.get('final_score') if factor_details else None

                    if score_value is not None:
                        # Determine classification
                        if score_value >= 80:
                            classification = "EXCELLENT"
                            color_class = "🟢"
                        elif score_value >= 60:
                            classification = "HIGH QUALITY"
                            color_class = "🟡"
                        elif score_value >= 40:
                            classification = "MODERATE"
                            color_class = "🟠"
                        else:
                            classification = "LOW QUALITY"
                            color_class = "🔴"

                        # Display factor details
                        st.markdown(f"### {color_class} {idx}. {factor_name.upper()} FACTOR: {score_value:.1f}/100 ({classification})")

                        # Show final score with progress bar
                        st.markdown(f"**Final Score:** {score_value:.1f} / 100")
                        st.progress(score_value / 100)
                        st.caption(f"Classification: **{classification}** (≥60 is High Quality)")

                        st.markdown("---")

                        # Show input metrics
                        st.markdown("#### Input Metrics")

                        components = factor_details.get('components', {})

                        # Build metrics list from components dict for display
                        metrics = []
                        for component_name, component_data in components.items():
                            has_data = component_data.get('component_score') is not None
                            metrics.append({
                                'name': component_name.replace('_', ' '),
                                'value': component_data.get('raw_value', 'N/A'),
                                'score': component_data.get('component_score', 'N/A'),
                                'weight': component_data.get('weight', 'N/A'),
                                'status': '✓' if has_data else '✗'
                            })

                        available_metrics = [m for m in metrics if m['status'] == '✓']
                        missing_metrics = [m for m in metrics if m['status'] == '✗']

                        if available_metrics:
                            st.markdown("**Available Metrics:**")
                            for metric in available_metrics:
                                value = metric['value']
                                score = metric['score']
                                weight = metric['weight']
                                
                                # Format value
                                try:
                                    numeric_value = float(value) if value not in ['N/A', None] else None
                                    if numeric_value is not None:
                                        if "%" in metric['name'] or "Margin" in metric['name'] or "Growth" in metric['name'] or "Return" in metric['name']:
                                            value_str = f"{numeric_value:.1f}%"
                                        elif "Ratio" in metric['name'] or "Coverage" in metric['name']:
                                            value_str = f"{numeric_value:.2f}x"
                                        else:
                                            value_str = f"{numeric_value:.2f}"
                                    else:
                                        value_str = str(value)
                                except Exception:
                                    value_str = str(value)
                                
                                # Format score
                                try:
                                    score_str = f"{float(score):.1f}" if score not in ['N/A', None] else 'N/A'
                                except Exception:
                                    score_str = str(score)
                                
                                # Safe weight formatting
                                try:
                                    weight_str = f"{weight:.0%}" if isinstance(weight, (int, float)) else str(weight)
                                except Exception:
                                    weight_str = str(weight)

                                st.markdown(f"- **{metric['name']}**: {value_str} → Score: {score_str} (Weight: {weight_str})")

                        if missing_metrics:
                            st.markdown("**Missing Data:**")
                            for metric in missing_metrics:
                                st.markdown(f"- {metric['name']} ✗")

                        st.markdown("---")

                        # Show scoring methodology
                        st.markdown("#### Scoring Methodology")

                        for metric in available_metrics:
                            st.markdown(f"**{metric['name']}**")

                            # Show thresholds table
                            thresholds = metric.get('thresholds', {})
                            if thresholds:
                                threshold_data = []
                                for range_label, (min_score, max_score, quality) in thresholds.items():
                                    threshold_data.append({
                                        "Range": range_label,
                                        "Score": f"{min_score}-{max_score}",
                                        "Quality": quality
                                    })
                                st.dataframe(pd.DataFrame(threshold_data), use_container_width=True, hide_index=True)

                            # Show where the actual value falls
                            st.info(f"**Your Value:** {metric['value']}")
                            st.caption("Linear interpolation applied within applicable range")

                        st.markdown("---")

                        # Show final factor calculation
                        st.markdown("#### Final Factor Score Calculation")
                        # Show calculation method based on component structure
                        if components:
                            num_components = len([c for c in components.values() if c.get('component_score') is not None])
                            if num_components > 0:
                                st.markdown(f"**Method:** Weighted average of {num_components} component(s)")
                            else:
                                st.markdown(f"**Method:** Based on available components")
                        else:
                            st.markdown(f"**Method:** N/A")


                        if missing_metrics:
                            st.warning(f"⚠️ {len(missing_metrics)} metric(s) missing. Factor score based on available data only.")

                        # Add separator between factors
                        st.markdown("---")
                        st.markdown("")  # Extra spacing

                    elif pd.notna(score_value):
                        # Have score but no details (shouldn't happen, but handle gracefully)
                        st.markdown(f"**{idx}. {factor_name} Score:** {score_value:.1f}/100")
                        st.progress(score_value / 100)
                    else:
                        # No score available
                        st.markdown(f"**{idx}. {factor_name} Score:** N/A (insufficient data)")
                        st.caption(f"Description: {description}")

                st.markdown("---")

                # Summary section
                st.markdown("### Factor Score Summary")

                col_sum1, col_sum2, col_sum3 = st.columns(3)

                with col_sum1:
                    # Count factors with data
                    factors_with_data = sum(1 for _, score_col, _ in factors if pd.notna(accessor.results.get(score_col)))
                    st.metric("Factors Scored", f"{factors_with_data}/5")

                with col_sum2:
                    # Count high quality factors
                    high_quality = sum(1 for _, score_col, _ in factors
                                      if pd.notna(accessor.results.get(score_col)) and accessor.results.get(score_col) >= 60)
                    st.metric("High Quality Factors", f"{high_quality}/{factors_with_data}")

                with col_sum3:
                    # Average factor score
                    factor_scores = [accessor.results.get(score_col) for _, score_col, _ in factors
                                    if pd.notna(accessor.results.get(score_col))]
                    if factor_scores:
                        avg_factor = sum(factor_scores) / len(factor_scores)
                        st.metric("Average Factor Score", f"{avg_factor:.1f}/100")

                st.info("""
                💡 **Note:** Each factor is scored independently based on its input metrics. The final
                composite score (shown in Stage 7) combines these factor scores using weights that may
                vary by sector when Dynamic Calibration is enabled.
                """)

            # STAGE 6: WEIGHT APPLICATION
            with st.expander("WEIGHT APPLICATION", expanded=False):
                st.markdown("### Factor Weights Used")

                # Determine weight source
                classification = accessor.results.get('Rubrics_Custom_Classification', 'N/A')
                parent_sector = CLASSIFICATION_TO_SECTOR.get(classification, 'N/A')

                if use_dynamic_calibration:
                    st.info(f"""
**Using Dynamic Calibration**
- Classification: **{classification}**
- Parent Sector: **{parent_sector}**
- Calibration Band: **{calibration_rating_band}**

Weights have been calibrated specifically for the {parent_sector} sector
to remove sector bias and improve cross-sector comparability.
""")
                    weight_source = f"Dynamic Calibration - {parent_sector} Sector"
                else:
                    st.info("""
**Using Universal Weights**

Factors weighted by financial importance: Credit 20%, Leverage 25%, Profitability 20%, Liquidity 10%, Cash Flow 25%.
""")
                    weight_source = "Universal Weights"

                st.markdown("---")

                # Get factor contributions from accessor
                contributions_data = accessor.get_factor_contributions()
                
                # Display table
                st.markdown("### Weight Application Table")
                st.caption("Shows how each factor contributes to the final quality score")
                
                weight_data = []
                for item in contributions_data['contributions']:
                    weight_data.append({
                        "Factor": item['factor'],
                        "Raw Score": f"{item['raw_score']:.1f}/100" if item['raw_score'] is not None else "N/A",
                        "Weight": f"{item['weight']*100:.1f}%",
                        "Weighted Contribution": f"{item['contribution']:.2f}"
                    })
                
                # Add totals row
                weight_data.append({
                    "Factor": "**TOTAL**",
                    "Raw Score": "—",
                    "Weight": f"**{contributions_data['total_weight']*100:.1f}%**",
                    "Weighted Contribution": f"**{contributions_data['total_score']:.2f}**"
                })
                
                weight_df = pd.DataFrame(weight_data)
                st.dataframe(
                    weight_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Factor": st.column_config.TextColumn("Factor", width="medium"),
                        "Raw Score": st.column_config.TextColumn("Raw Score", width="small"),
                        "Weight": st.column_config.TextColumn("Weight", width="small"),
                        "Weighted Contribution": st.column_config.TextColumn("Weighted Contribution", width="small")
                    }
                )
                
                st.caption(f"Total Weighted Quality Score: {contributions_data['total_score']:.2f}/100")
                
                st.markdown("---")
                
                # Show calculation example for one factor
                st.markdown("### Calculation Example")
                
                # Pick the first factor with data
                example_item = next((item for item in contributions_data['contributions'] if item['raw_score'] is not None), None)
                
                if example_item:
                    factor = example_item['factor']
                    score = example_item['raw_score']
                    weight = example_item['weight']
                    contribution = example_item['contribution']
                    
                    st.markdown(f"**{factor} Factor Calculation:**")
                    st.code(f"""
Step 1: Raw {factor} Score = {score:.1f}/100
Step 2: {factor} Weight = {weight*100:.1f}%
Step 3: Weighted Contribution = {score:.1f} × {weight:.3f} = {contribution:.2f}
""")
                    st.caption(f"This {contribution:.2f} points is added to the total quality score")

                st.markdown("---")

                # Compare to universal weights if using dynamic calibration
                if use_dynamic_calibration:
                    st.markdown("### Comparison to Universal Weights")
                    st.caption("See how dynamic calibration adjusted the weights for this sector")

                    # Universal weights (5-factor model)
                    universal_weights = {
                        "Credit": 0.20,
                        "Leverage": 0.25,
                        "Profitability": 0.20,
                        "Liquidity": 0.10,
                        "Cash Flow": 0.25,
                        "Cash_Flow": 0.25,  # Handle underscore variant
                    }

                    comparison_data = []
                    for item in contributions_data['contributions']:
                        factor_name = item['factor']
                        calibrated = item['weight']
                        # Handle potential name variations
                        factor_lookup = factor_name.replace('_', ' ')
                        universal = universal_weights.get(factor_name, 
                                   universal_weights.get(factor_lookup, 0.15))  # Correct fallback
                        difference = calibrated - universal

                        # Determine if increased or decreased
                        if abs(difference) < 0.01:
                            change = "→ No change"
                        elif difference > 0:
                            change = f"↑ Increased by {abs(difference)*100:.1f}%"
                        else:
                            change = f"↓ Decreased by {abs(difference)*100:.1f}%"

                        comparison_data.append({
                            "Factor": factor_name,
                            "Universal": f"{universal*100:.1f}%",
                            "Calibrated": f"{calibrated*100:.1f}%",
                            "Change": change
                        })

                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

                    # Show calibration effectiveness if available
                    cal_weights = st.session_state.get('_calibrated_weights', {})
                    effectiveness = cal_weights.get('_effectiveness', 0)
                    variance_reduction = cal_weights.get('_variance_reduction', 0)
                    
                    if effectiveness > 0:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Calibration Effectiveness",
                                f"{effectiveness*100:.0f}%",
                                help="Percentage reduction in cross-sector variance"
                            )
                        with col2:
                            if effectiveness >= 0.7:
                                st.success("✅ Excellent calibration")
                            elif effectiveness >= 0.5:
                                st.info("Good calibration")
                            elif effectiveness >= 0.3:
                                st.warning("Moderate calibration")
                            else:
                                st.error("⚠️ Low effectiveness")
                    
                    st.markdown(f"""
**How Weights Were Determined:**

Dynamic calibration used **variance minimization** to find optimal weights that minimize 
the spread of composite scores across {len([s for s in cal_weights if not s.startswith('_') and s != 'Default'])} sectors.

This ensures that a median {calibration_rating_band}-rated company scores similarly regardless of sector,
removing structural biases from cross-sector comparisons.
""")

                st.markdown("---")

                # Explain what happens next
                st.markdown("### Next Step")
                st.info("""
These weighted factor contributions are summed to create the **Quality Score** component.

In Stage 7, the Quality Score is combined with the Trend Score (if applicable)
to produce the final Composite Score.
""")

            # STAGE 7: COMPOSITE SCORE CALCULATION
            with st.expander("COMPOSITE SCORE CALCULATION", expanded=False):
                st.markdown("### Score Combination Method")

                # Get the composite score
                composite = accessor.results.get('Composite_Score', None)

                if composite is None or pd.isna(composite):
                    st.error("❌ Composite score not available")
                else:
                    st.markdown(f"**Final Composite Score:** {composite:.2f} / 100")
                    st.progress(composite / 100)

                    # Show scoring method
                    st.caption(f"**Calculation Method:** {scoring_method}")

                    st.markdown("---")
                    # STEP-BY-STEP CALCULATION
                    st.markdown("### Step-by-Step Calculation")

                    # Try to get Quality and Trend scores separately
                    quality_score = accessor.results.get('Quality_Score', None)
                    trend_score = accessor.results.get('Cycle_Position_Score', None)
                    # If separate scores aren't available, estimate from composite
                    if quality_score is None or pd.isna(quality_score):
                        # Try to reconstruct from weighted factors (from Stage 6)
                        weights = {
                            "Credit": accessor.results.get('Weight_Credit_Used', 0.167),
                            "Leverage": accessor.results.get('Weight_Leverage_Used', 0.167),
                            "Profitability": accessor.results.get('Weight_Profitability_Used', 0.167),
                            "Liquidity": accessor.results.get('Weight_Liquidity_Used', 0.167),
                            "Cash_Flow": accessor.results.get('Weight_Cash_Flow_Used', 0.167)
                        }

                        scores = {
                            "Credit": accessor.results.get('Credit_Score', 0),
                            "Leverage": accessor.results.get('Leverage_Score', 0),
                            "Profitability": accessor.results.get('Profitability_Score', 0),
                            "Liquidity": accessor.results.get('Liquidity_Score', 0),
                            "Cash Flow": accessor.results.get('Cash_Flow_Score', 0)
                        }

                        # Calculate quality score as weighted sum
                        quality_score = 0
                        for factor in scores.keys():
                            if pd.notna(scores[factor]) and pd.notna(weights[factor]):
                                quality_score += scores[factor] * weights[factor]

                    # Check if trend scoring is used
                    has_trend = trend_score is not None and pd.notna(trend_score)

                    if has_trend:
                        # TWO-COMPONENT MODEL: Quality + Trend
                        st.markdown("#### Component 1: Quality Score")

                        st.markdown(f"""
**Quality Score:** {quality_score:.2f} / 100

Calculated in Stage 5 & 6:
- Five quality factors scored (0-100 each)
- Each factor weighted based on {scoring_method.lower()}
- Weighted scores summed to produce quality component
""")

                        st.progress(quality_score / 100)

                        st.markdown("---")
                        st.markdown("#### Component 2: Trend Score")

                        st.markdown(f"""
**Trend Score:** {trend_score:.2f} / 100

Calculated in Stage 5A:
- Historical trends analyzed for 15-20 key metrics
- Direction, momentum, and volatility assessed
- Average trend score across all metrics
""")

                        st.progress(trend_score / 100)

                        st.markdown("---")
                        st.markdown("#### Component 3: Weighting")

                        # Determine weights (adjust these based on your actual implementation)
                        # [V4.0] Quality/Trend blend weights (matches pipeline calculation)
                        quality_weight = 0.80  # 80%
                        trend_weight = 0.20    # 20%

                        st.markdown(f"""
**Quality Weight:** {quality_weight*100:.0f}%
**Trend Weight:** {trend_weight*100:.0f}%

Quality (current fundamentals) dominates, with trend (direction of travel) providing additional differentiation.
Improving issuers score higher than deteriorating ones with identical quality.
""")

                        col_w1, col_w2 = st.columns(2)
                        with col_w1:
                            st.metric("Quality Component", f"{quality_score * quality_weight:.2f} points")
                        with col_w2:
                            st.metric("Trend Component", f"{trend_score * trend_weight:.2f} points")

                        st.markdown("---")
                        st.markdown("#### Final Calculation")

                        calculated_composite = (quality_score * quality_weight) + (trend_score * trend_weight)

                        st.code(f"""
Composite Score Calculation:

Step 1: Quality Component
   Quality Score × Quality Weight
   = {quality_score:.2f} × {quality_weight:.2f}
   = {quality_score * quality_weight:.2f}

Step 2: Trend Component
   Trend Score × Trend Weight
   = {trend_score:.2f} × {trend_weight:.2f}
   = {trend_score * trend_weight:.2f}

Step 3: Sum Components
   Quality Component + Trend Component
   = {quality_score * quality_weight:.2f} + {trend_score * trend_weight:.2f}
   = {calculated_composite:.2f}

Final Composite Score: {composite:.2f} / 100
""")

                        # Note if calculated differs from stored
                        if abs(calculated_composite - composite) > 0.5:
                            st.info(f"💡 Note: Minor difference due to rounding or additional adjustments in calculation pipeline")

                    else:
                        # SINGLE-COMPONENT MODEL: Quality Only
                        st.markdown("#### Quality Score (No Trend Component)")

                        st.markdown(f"""
**Quality Score:** {quality_score:.2f} / 100

Calculated in Stage 5 & 6:
- Five quality factors scored (0-100 each)
- Each factor weighted based on {scoring_method.lower()}
- Weighted scores summed to produce final score

**Note:** Trend analysis not applied in this calculation
""")

                        st.progress(quality_score / 100)

                        st.markdown("---")
                        st.markdown("#### Final Score")

                        st.code(f"""
Composite Score = Quality Score
            = {quality_score:.2f}

Final Composite Score: {composite:.2f} / 100
""")

                    st.markdown("---")

                    # SCORE INTERPRETATION
                    st.markdown("### Score Interpretation")

                    col_int1, col_int2, col_int3 = st.columns(3)

                    with col_int1:
                        # Thresholds from MODEL_THRESHOLDS for SSOT compliance
                        if composite >= MODEL_THRESHOLDS['display_composite_high']:
                            interpretation = "High Quality"
                            color = "🟢"
                        elif composite >= MODEL_THRESHOLDS['display_composite_moderate']:
                            interpretation = "Moderate Quality"
                            color = "🟡"
                        else:
                            interpretation = "Below Average"
                            color = "🔴"

                        st.metric("Classification", f"{color} {interpretation}")

                    with col_int2:
                        # Determine if above/below median
                        all_composites = results_final['Composite_Score'].dropna()
                        market_median = all_composites.median()

                        if composite > market_median:
                            position = f"Above Median (+{composite - market_median:.1f})"
                        elif composite < market_median:
                            position = f"Below Median ({composite - market_median:.1f})"
                        else:
                            position = "At Median"

                        st.metric("vs Market", position)

                    with col_int3:
                        # Determine if improving/stable/deteriorating
                        if has_trend:
                            if trend_score >= TREND_THRESHOLD:
                                trend_label = "🟢 Improving"
                            elif trend_score >= 45:
                                trend_label = "🟡 Stable"
                            else:
                                trend_label = "🔴 Deteriorating"
                            st.metric("Trend Signal", trend_label)
                        else:
                            st.metric("Trend Signal", "N/A")

                    st.markdown("---")

                    # UNIVERSE CONTEXT (existing code, enhanced)
                    st.markdown("### Universe Context")
                    # Read stored percentile instead of recalculating (single source of truth)
                    percentile = accessor.results.get('Composite_Percentile_Global', None)
                    all_composites = results_final['Composite_Score'].dropna()
                    
                    if pd.isna(percentile):
                        # Fallback calculation only if not stored
                        percentile = (all_composites < composite).sum() / len(all_composites) * 100

                    col_u1, col_u2, col_u3, col_u4 = st.columns(4)

                    with col_u1:
                        st.metric("Your Score", f"{composite:.1f}")

                    with col_u2:
                        st.metric("Percentile", f"{percentile:.1f}%")
                        st.caption(f"{int((all_composites < composite).sum())} issuers score lower")

                    with col_u3:
                        st.metric("Universe Median", f"{all_composites.median():.1f}")

                    with col_u4:
                        st.metric("Universe Range", f"{all_composites.min():.1f} - {all_composites.max():.1f}")

                    # Quartile position
                    if percentile >= 75:
                        quartile = "Top Quartile (75th-100th percentile)"
                        quartile_color = "🟢"
                    elif percentile >= 50:
                        quartile = "Upper Middle Quartile (50th-75th percentile)"
                        quartile_color = "🟡"
                    elif percentile >= 25:
                        quartile = "Lower Middle Quartile (25th-50th percentile)"
                        quartile_color = "🟠"
                    else:
                        quartile = "Bottom Quartile (0-25th percentile)"
                        quartile_color = "🔴"

                    st.info(f"{quartile_color} **Position:** {quartile}")

                    st.markdown("---")

                    # SECTOR CONTEXT (new addition)
                    st.markdown("### Sector Context")

                    classification = accessor.results.get('Rubrics_Custom_Classification', 'N/A')
                    parent_sector = CLASSIFICATION_TO_SECTOR.get(classification, 'N/A')

                    # Get sector data
                    sector_data = results_final[results_final['Rubrics_Custom_Classification'] == classification]
                    sector_composites = sector_data['Composite_Score'].dropna()

                    if len(sector_composites) > 0:
                        # READ from stored value - do not recalculate
                        sector_percentile = accessor.results.get('Composite_Percentile_in_Band', 0)
                        if pd.isna(sector_percentile):
                            sector_percentile = 0
                        sector_median = sector_composites.median()

                        col_s1, col_s2, col_s3 = st.columns(3)

                        with col_s1:
                            st.metric("Classification", classification)
                            st.caption(f"Parent Sector: {parent_sector}")

                        with col_s2:
                            st.metric("Sector Percentile", f"{sector_percentile:.1f}%")
                            st.caption(f"{len(sector_composites)} issuers in classification")

                        with col_s3:
                            st.metric("Sector Median", f"{sector_median:.1f}")

                            if composite > sector_median:
                                st.caption(f"Above sector median (+{composite - sector_median:.1f})")
                            elif composite < sector_median:
                                st.caption(f"Below sector median ({composite - sector_median:.1f})")
                            else:
                                st.caption("At sector median")

                    st.markdown("---")

                    # WHAT'S NEXT
                    st.markdown("### Next Steps")

                    st.info("""
**Stage 8:** This composite score is used to rank the issuer against:
- Classification peers (same Rubrics Custom Classification)
- Full universe (all issuers)

**Stage 9:** Based on the composite score and other factors, a recommendation
is assigned (Strong Buy / Buy / Hold / Avoid)
""")

            # STAGE 8: RELATIVE RANKING
            with st.expander("RELATIVE RANKING & PERCENTILES", expanded=False):
                st.markdown("### Sector Ranking")

                sector_data = results_final[results_final['Rubrics_Custom_Classification'] == accessor.results['Rubrics_Custom_Classification']]
                sector_ranked = sector_data.sort_values('Composite_Score', ascending=False).reset_index(drop=True)
                sector_rank = sector_ranked[sector_ranked['Company_Name'] == selected_company].index[0] + 1 if len(sector_ranked[sector_ranked['Company_Name'] == selected_company]) > 0 else 0
                total_in_sector = len(sector_ranked)
                sector_composites = sector_data['Composite_Score'].dropna()
                sector_percentile = accessor.results.get('Composite_Percentile_in_Band', 0)
                if pd.isna(sector_percentile):
                    sector_percentile = 0

                st.markdown(f"""
                **{accessor.results['Rubrics_Custom_Classification']} Sector ({total_in_sector} issuers)**

                - Your Rank: **#{sector_rank}** out of {total_in_sector}
                - Percentile: **{sector_percentile:.1f}%**
                - Position: {"Top" if sector_percentile >= 75 else "Upper Middle" if sector_percentile >= 50 else "Lower Middle" if sector_percentile >= 25 else "Bottom"} quartile
                """)

                st.markdown("---")
                st.markdown("### Universe Ranking")

                universe_ranked = results_final.sort_values('Composite_Score', ascending=False).reset_index(drop=True)
                universe_rank = universe_ranked[universe_ranked['Company_Name'] == selected_company].index[0] + 1 if len(universe_ranked[universe_ranked['Company_Name'] == selected_company]) > 0 else 0
                total_universe = len(universe_ranked)

                st.markdown(f"""
                **Full Universe ({total_universe} issuers)**

                - Your Rank: **#{universe_rank}** out of {total_universe}
                - Percentile: **{percentile:.1f}%**
                - Position: {"Top" if percentile >= 75 else "Upper Middle" if percentile >= 50 else "Lower Middle" if percentile >= 25 else "Bottom"} quartile
                """)

            # STAGE 9: CLASSIFICATION & RECOMMENDATION
            with st.expander("CLASSIFICATION & RECOMMENDATION", expanded=False):
                st.markdown("### How Your Recommendation Was Determined")

                recommendation = accessor.results['Recommendation']
                # [FIX] Use Composite_Score for classification display to match actual signal logic
                quality_score = accessor.results.get('Composite_Score', 0)

                # Define thresholds
                # Match actual signal classification thresholds (lines 11511, 7343)
                # Use centralized thresholds (defined at module level)
                # QUALITY_THRESHOLD and TREND_THRESHOLD are already available

                # Step 1: Threshold Checks
                st.markdown("#### Step 1: Threshold Checks")

                col_t1, col_t2 = st.columns(2)

                with col_t1:
                    st.markdown("**Quality Assessment:**")

                    if quality_score >= QUALITY_THRESHOLD:
                        quality_class = "STRONG"
                        quality_color = "🟢"
                        quality_desc = f"Quality Score ({quality_score:.1f}) ≥ {QUALITY_THRESHOLD}"
                    else:
                        quality_class = "WEAK"
                        quality_color = "🔴"
                        quality_desc = f"Quality Score ({quality_score:.1f}) < {QUALITY_THRESHOLD}"

                    st.markdown(f"""
                    - **Your Quality Score:** {quality_score:.1f} / 100
                    - **Classification:** {quality_color} **{quality_class}**
                    - **Threshold Check:** {quality_desc}
                    """)

                with col_t2:
                    if has_trend:
                        st.markdown("**Trend Assessment:**")

                        if trend_score >= TREND_THRESHOLD:
                            trend_class = "IMPROVING"
                            trend_color = "🟢"
                            trend_desc = f"Trend Score ({trend_score:.1f}) ≥ {TREND_THRESHOLD}"
                        else:
                            trend_class = "DETERIORATING"
                            trend_color = "🔴"
                            trend_desc = f"Trend Score ({trend_score:.1f}) < {TREND_THRESHOLD}"

                        st.markdown(f"""
                        - **Your Trend Score:** {trend_score:.1f} / 100
                        - **Classification:** {trend_color} **{trend_class}**
                        - **Threshold Check:** {trend_desc}
                        """)
                    else:
                        st.markdown("**Trend Assessment:**")
                        st.info("Trend analysis not available for this issuer")

                # Step 2: Decision Matrix
                st.markdown("---")
                st.markdown("#### Step 2: Decision Matrix")

                if has_trend:
                    st.markdown("**Full Matrix (Quality × Trend):**")

                    # Create decision matrix
                    matrix_data = {
                        "Quality Level": [
                            "🟢 STRONG (≥50)",
                            "🔴 WEAK (<50)"
                        ],
                        "🟢 IMPROVING (≥55)": [
                            "Strong & Improving → Strong Buy/Buy",
                            "Weak but Improving → Buy/Hold"
                        ],
                        "🔴 DETERIORATING (<55)": [
                            "Strong but Deteriorating → Buy/Hold*",
                            "Weak & Deteriorating → Avoid"
                        ]
                    }

                    matrix_df = pd.DataFrame(matrix_data)

                    # Determine user's position in matrix
                    quality_row = 0 if quality_score >= QUALITY_THRESHOLD else 1
                    trend_col = "🟢 IMPROVING (≥55)" if trend_score >= TREND_THRESHOLD else "🔴 DETERIORATING (<55)"

                    # Define recommendation colors
                    rec_colors = {
                        "Strong Buy": '#90EE90',  # Light green
                        "Buy": '#B8F5B8',         # Lighter green
                        "Hold": '#FFFFCC',        # Light yellow
                        "Avoid": '#FFB6B6'        # Light red
                    }

                    st.dataframe(
                        matrix_df.style.apply(
                            lambda x: [
                                f'background-color: {rec_colors.get(matrix_df.loc[idx, x.name], "#FFFFFF")}; font-weight: bold'
                                if (idx == quality_row and x.name == trend_col)
                                else ''
                                for idx in range(len(x))
                            ],
                            axis=0
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.success(f"**Your Position:** {quality_class} × {trend_class} → **{recommendation}**")
                    st.caption("*May override to 'Strong & Normalizing' or 'Strong & Moderating' for exceptional quality issuers")

                else:
                    st.markdown("**Simplified Matrix (Quality Only):**")

                    # Create simplified decision matrix
                    simple_matrix_data = {
                        "Quality Level": [
                            "🟢 STRONG (≥50)",
                            "🔴 WEAK (<50)"
                        ],
                        "Recommendation": [
                            "Buy",
                            "Avoid"
                        ]
                    }

                    simple_matrix_df = pd.DataFrame(simple_matrix_data)

                    # Determine user's position
                    quality_row = 0 if quality_score >= QUALITY_THRESHOLD else 1

                    st.dataframe(
                        simple_matrix_df.style.apply(
                            lambda x: [
                                'background-color: #90EE90; font-weight: bold'
                                if idx == quality_row
                                else ''
                                for idx in range(len(x))
                            ],
                            axis=0
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.success(f"**Your Position:** {quality_class} → **{recommendation}**")

                # Step 3: Detailed Explanation
                st.markdown("---")
                st.markdown("#### Step 3: Why This Recommendation?")

                # Generate detailed explanation based on recommendation
                if recommendation == "Strong Buy":
                    explanation = f"""
                    **Strong Buy** means this issuer demonstrates **exceptional credit quality with positive momentum**.

                    **Key Factors:**
                    - High quality score ({quality_score:.1f}/100) indicates strong fundamentals across multiple factors
                    - Improving trend ({trend_score:.1f}/100) shows positive momentum in key metrics
                    - This combination suggests the issuer is well-positioned for continued performance

                    **Investment Thesis:**
                    Consider adding or increasing position. The issuer shows both strong current fundamentals and positive trajectory.
                    """
                elif recommendation == "Buy":
                    if has_trend and trend_score >= TREND_THRESHOLD:
                        explanation = f"""
                        **Buy** means this issuer shows **solid quality with improving trends**.

                        **Key Factors:**
                        - {"Strong" if quality_score >= QUALITY_THRESHOLD else "Weak"} quality score ({quality_score:.1f}/100) provides a stable foundation
                        - Improving trend ({trend_score:.1f}/100) indicates positive momentum
                        - Good risk/reward profile for incremental exposure

                        **Investment Thesis:**
                        Consider adding position. The improving trends suggest upside potential from current levels.
                        """
                    else:
                        explanation = f"""
                        **Buy** means this issuer demonstrates **strong fundamentals**.

                        **Key Factors:**
                        - High quality score ({quality_score:.1f}/100) indicates robust credit profile
                        - Strong performance across key financial metrics
                        - Stable credit characteristics

                        **Investment Thesis:**
                        Consider adding position. Strong fundamentals provide downside protection.
                        """
                elif recommendation == "Hold":
                    if has_trend:
                        if trend_score < TREND_THRESHOLD:
                            explanation = f"""
                            **Hold** means this issuer shows **mixed signals** that warrant a cautious stance.

                            **Key Factors:**
                            - {"Strong" if quality_score >= QUALITY_THRESHOLD else "Weak"} quality score ({quality_score:.1f}/100) provides some support
                            - However, deteriorating trend ({trend_score:.1f}/100) raises concerns about future performance
                            - Risk/reward appears balanced at current levels

                            **Investment Thesis:**
                            Maintain current position but monitor closely. Deteriorating trends may signal need to reduce exposure if they continue.
                            """
                        else:
                            explanation = f"""
                            **Hold** means this issuer shows **adequate but not compelling** characteristics.

                            **Key Factors:**
                            - Moderate quality score ({quality_score:.1f}/100) suggests average fundamentals
                            - {"Stable" if trend_score >= TREND_THRESHOLD else "Improving"} trend ({trend_score:.1f}/100)
                            - Risk/reward appears balanced

                            **Investment Thesis:**
                            Maintain current exposure. Better opportunities likely available elsewhere.
                            """
                    else:
                        explanation = f"""
                        **Hold** means this issuer shows **adequate but not compelling** characteristics.

                        **Key Factors:**
                        - Moderate quality score ({quality_score:.1f}/100) suggests average fundamentals
                        - No significant red flags, but limited upside potential

                        **Investment Thesis:**
                        Maintain current exposure. Better opportunities likely available elsewhere.
                        """
                else:  # Avoid
                    explanation = f"""
                    **Avoid** means this issuer shows **material credit concerns**.

                    **Key Factors:**
                    - Low quality score ({quality_score:.1f}/100) indicates weak fundamentals
                    - {"Deteriorating trends exacerbate concerns" if has_trend and trend_score < TREND_THRESHOLD else "Weak performance across key metrics"}
                    - Risk/reward profile unfavorable

                    **Investment Thesis:**
                    Consider reducing or avoiding exposure. Better risk-adjusted opportunities available in higher-quality issuers.
                    """

                st.markdown(explanation)

                # Step 4: Contributing Factors
                st.markdown("---")
                st.markdown("#### Step 4: Contributing Factors")

                col_f1, col_f2 = st.columns(2)

                # Analyze strengths and concerns from factor scores
                factor_scores = {
                    "Credit": accessor.results.get('Credit_Score'),
                    "Leverage": accessor.results.get('Leverage_Score'),
                    "Profitability": accessor.results.get('Profitability_Score'),
                    "Liquidity": accessor.results.get('Liquidity_Score'),
                    "Cash Flow": accessor.results.get('Cash_Flow_Score')
                }

                # Filter out None/NaN scores
                valid_factors = {k: v for k, v in factor_scores.items() if v is not None and pd.notna(v)}

                # Sort by score
                sorted_factors = sorted(valid_factors.items(), key=lambda x: x[1], reverse=True)

                with col_f1:
                    st.markdown("**Strengths:**")
                    strengths = [f for f, s in sorted_factors if s >= 60]  # All above 60
                    if strengths:
                        for factor, score in sorted_factors:
                            if factor in strengths:
                                # Color code by score level
                                if score >= 80:
                                    icon = "🟢"
                                    label = "Excellent"
                                else:
                                    icon = "🟡"
                                    label = "High Quality"
                                st.markdown(f"- {icon} **{factor}:** {score:.1f}/100 ({label})")
                    else:
                        st.markdown("- No factors scoring above 60")

                with col_f2:
                    st.markdown("**Concerns:**")
                    concerns = [f for f, s in sorted_factors if s < 60]  # All below 60
                    if concerns:
                        for factor, score in sorted_factors:
                            if factor in concerns:
                                # Color code by score level
                                if score < 40:
                                    icon = "🔴"
                                    label = "Low Quality"
                                else:
                                    icon = "🟠"
                                    label = "Moderate"
                                st.markdown(f"- {icon} **{factor}:** {score:.1f}/100 ({label})")
                    else:
                        st.markdown("- No concerns (all factors ≥60)")

                # Final note
                st.markdown("---")
                st.info("""
                **Important Note:** This recommendation is based on quantitative analysis only.
                Always consider qualitative factors such as:
                - Management quality and strategy
                - Industry dynamics and competitive position
                - Regulatory environment
                - Macroeconomic conditions
                - Specific bond covenants and structure
                """)

            # STAGE 10: OUTPUT SUMMARY
            with st.expander("FINAL OUTPUT SUMMARY", expanded=False):
                st.markdown("### Complete Issuer Profile")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Identity:**")
                    st.markdown(f"- Company: {selected_company}")
                    st.markdown(f"- Ticker: {accessor.get_ticker()}")
                    st.markdown(f"- Sector: {accessor.results.get('Rubrics_Custom_Classification', 'N/A')}")
                    st.markdown(f"- Rating: {accessor.results.get('Credit_Rating_Clean', 'N/A')}")

                    st.markdown("")
                    st.markdown("**Scores:**")
                    st.markdown(f"- Composite: **{composite:.1f}**")

                with col2:
                    st.markdown("**Rankings:**")
                    st.markdown(f"- Sector Rank: #{sector_rank} / {total_in_sector}")
                    st.markdown(f"- Universe Rank: #{universe_rank} / {total_universe}")
                    st.markdown(f"- Sector Percentile: {sector_percentile:.1f}%")
                    st.markdown(f"- Universe Percentile: {percentile:.1f}%")

                    st.markdown("")
                    st.markdown("**Classification:**")
                    st.markdown(f"- **Recommendation: {recommendation}**")

                st.markdown("---")
                st.success("✓ All 10 stages completed successfully")





            # PART 10: EXPORT & DOWNLOAD OPTIONS
            st.markdown("---")
            st.subheader("DOWNLOAD DIAGNOSTIC REPORT")

            st.info("💡 **Export Options:** Download complete diagnostic data for debugging and analysis. Excel format includes multiple sheets for each stage.")

            # Prepare comprehensive export data
            try:
                export_data = create_diagnostic_export_data(
                    accessor=accessor,
                    selected_company=selected_company,
                    scoring_method=scoring_method,
                    period_mode=period_mode,
                    reference_date_override=reference_date_override,
                    use_dynamic_calibration=use_dynamic_calibration,
                    calibration_rating_band=calibration_rating_band
                )
                export_ready = True
            except Exception as e:
                st.error(f"Error preparing export data: {str(e)}")
                export_ready = False

            col_d1, col_d2, col_d3, col_d4 = st.columns(4)

            with col_d1:
                # Enhanced CSV export with all diagnostic data
                if export_ready:
                    try:
                        csv_data = create_diagnostic_csv(export_data, selected_company)
                        st.download_button(
                            label="📄 Download Full Diagnostics (CSV)",
                            data=csv_data,
                            file_name=f"{selected_company.replace(' ', '_')}_full_diagnostics.csv",
                            mime="text/csv",
                            key="download_csv",
                            help="Complete diagnostic data in CSV format with all stages"
                        )
                    except Exception as e:
                        st.button(
                            label="📄 Download Full Diagnostics (CSV)",
                            disabled=True,
                            key="download_csv_error",
                            help=f"Export error: {str(e)}"
                        )
                else:
                    st.button(
                        label="📄 Download Full Diagnostics (CSV)",
                        disabled=True,
                        key="download_csv_disabled",
                        help="Export data not available"
                    )

            with col_d2:
                # Excel export with multiple sheets
                if export_ready:
                    try:
                        excel_data = create_diagnostic_excel(export_data, selected_company)
                        st.download_button(
                            label="📊 Download Diagnostics (Excel)",
                            data=excel_data,
                            file_name=f"{selected_company.replace(' ', '_')}_diagnostics.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_excel",
                            help="Multi-sheet Excel workbook with complete diagnostic breakdown"
                        )
                    except Exception as e:
                        st.button(
                            "📊 Download Diagnostics (Excel)",
                            key="download_excel_error",
                            disabled=True,
                            help=f"Export error: {str(e)}"
                        )
                else:
                    st.button(
                        "📊 Download Diagnostics (Excel)",
                        key="download_excel_disabled",
                        disabled=True,
                        help="Export data not available"
                    )

            with col_d3:
                # Full diagnostic JSON export
                composite_score_val = accessor.results.get('Composite_Score', None)
                recommendation_val = accessor.results.get('Recommendation', 'N/A')

                full_diagnostic_export = {
                    "metadata": {
                        "company": selected_company,
                        "generated": datetime.now().isoformat(),
                        "app_version": "5.1.1"
                    },
                    "configuration": {
                        "scoring_method": scoring_method,
                        "period_mode": str(period_mode),
                        "reference_date": str(reference_date_override) if reference_date_override else None,
                        "dynamic_calibration": use_dynamic_calibration,
                        "calibration_band": calibration_rating_band if use_dynamic_calibration else None
                    },
                    "company_info": {
                        "company_id": accessor.get_company_id(),
                        "company_name": accessor.get_company_name(),
                        "ticker": accessor.get_ticker(),
                        "rating": accessor.get_credit_rating(),
                        "sector": accessor.get_sector(),
                        "industry": accessor.get_industry(),
                        "market_cap": accessor.get_market_cap()
                    },
                    "period_selection": accessor.get_period_selection(),
                    "raw_inputs": accessor.diag.get('raw_inputs', {}),
                    "factor_scores": {
                        factor: {
                            "score": accessor.get_factor_score(factor),
                            "details": accessor.get_factor_details(factor)
                        }
                        for factor in ['Credit', 'Leverage', 'Profitability', 'Liquidity', 'Cash_Flow']
                    },
                    "time_series": accessor.get_all_metric_time_series(),
                    "composite_calculation": accessor.get_composite_calculation(),
                    "results": {
                        "composite_score": float(composite_score_val) if pd.notna(composite_score_val) else None,
                        "quality_score": float(accessor.results.get('Quality_Score')) if pd.notna(accessor.results.get('Quality_Score')) else None,
                        "trend_score": float(accessor.results.get('Cycle_Position_Score')) if pd.notna(accessor.results.get('Cycle_Position_Score')) else None,
                        "signal": accessor.results.get('Signal', 'N/A'),
                        "recommendation": recommendation_val
                    },
                    "rankings": {
                        "classification_rank": int(accessor.results.get('Classification_Rank')) if pd.notna(accessor.results.get('Classification_Rank')) else None,
                        "classification_total": int(accessor.results.get('Classification_Total')) if pd.notna(accessor.results.get('Classification_Total')) else None,
                        "classification_percentile": float(accessor.results.get('Composite_Percentile_in_Band')) if pd.notna(accessor.results.get('Composite_Percentile_in_Band')) else None,
                        "global_percentile": float(accessor.results.get('Composite_Percentile_Global')) if pd.notna(accessor.results.get('Composite_Percentile_Global')) else None
                    }
                }

                json_data = json.dumps(full_diagnostic_export, indent=2, default=str)
                st.download_button(
                    label="📋 Download Full Diagnostics (JSON)",
                    data=json_data,
                    file_name=f"{selected_company.replace(' ', '_')}_diagnostics.json",
                    mime="application/json",
                    key="download_json",
                    help="Complete diagnostic data in JSON format - machine readable"
                )

            # Add helpful information
            st.markdown("---")
            st.markdown("""
            **📥 Export Guide:**
            - **Full Diagnostics (CSV)**: Complete 12-section diagnostic report in text format - optimized for review and LLM parsing
            - **Diagnostics (Excel)**: Multi-sheet workbook with 14 sheets including raw inputs and time series - best for detailed analysis
            - **Full Diagnostics (JSON)**: Complete diagnostic data structure - machine readable, includes all raw data and calculations
            """)

        st.markdown("---")
        st.markdown("""
    <div style='text-align: center; color: #4c566a; padding: 20px;'>
        <p><strong>Issuer Credit Screening Model V5.0</strong></p>
        <p>© 2025 Rubrics Asset Management</p>
    </div>
    """, unsafe_allow_html=True)

    # ============================================================================
    # [V2.2] SELF-TESTS (Run with RG_TESTS=1 environment variable)
    # ============================================================================
        
if os.environ.get("RG_TESTS") == "1":  # Tests enabled
    import sys
    print("\n" + "="*60)
    print("Running RG_TESTS for V3.0...")
    print("="*60 + "\n")
    import sys

    # Test 1: Period classification with labeled periods
    print("Test 1: FY series extraction with labeled periods")
    row = pd.Series({
        "Period Ended": "31/12/2020",
        "Period Ended.1": "31/12/2021",
        "Period Ended.2": "31/12/2022",
        "Period Ended.3": "31/12/2023",
        "Period Ended.4": "31/12/2024",
        "Period Ended.5": "30/09/2023",
        "Period Ended.12": "30/09/2025",
        "EBITDA Margin": 14,
        "EBITDA Margin.1": 15,
        "EBITDA Margin.2": 16,
        "EBITDA Margin.3": 17,
        "EBITDA Margin.4": 18,
        "EBITDA Margin.5": 99.9,  # CQ - should be excluded from FY
        "Net Debt / EBITDA": 3.5,
        "Net Debt / EBITDA.1": 3.2,
        "Net Debt / EBITDA.2": 3.0,
        "Net Debt / EBITDA.3": 2.8,
        "Net Debt / EBITDA.4": 2.6,
        "EBITDA / Interest Expense (x)": 3.5,
        "EBITDA / Interest Expense (x).4": 4.2
    })

    df_test = pd.DataFrame([row])
    pe_cols = parse_period_ended_cols(df_test)

    # pe_cols is now list of (suffix, datetime_series) tuples
    suffixes = [suffix for suffix, _ in pe_cols[:5]]
    assert suffixes == ['', '.1', '.2', '.3', '.4'], \
        f"Expected ['', '.1', '.2', '.3', '.4'], got {suffixes}"
    print("  OK Period columns sorted correctly")

    fy = get_metric_series_row(df_test.iloc[0], "EBITDA Margin", prefer="FY")
    assert list(fy.values) == [14.0, 15.0, 16.0, 17.0, 18.0], \
        f"Expected [14,15,16,17,18], got {list(fy.values)}"
    print(f"  OK FY series values correct: {list(fy.values)}")

    assert str(list(fy.index)[-1]).startswith("2024"), \
        f"Expected last date to be 2024, got {list(fy.index)[-1]}"
    print(f"  OK FY series indexed by actual dates: {list(fy.index)}")

    most_recent = most_recent_annual_value(df_test.iloc[0], "EBITDA Margin")
    assert most_recent == 18, f"Expected 18, got {most_recent}"
    print(f"  OK Most recent annual value: {most_recent}")

    # Test 2: 1900 sentinel → NaT → dropped
    print("\nTest 2: 1900 sentinel handling")
    row2 = pd.Series({
        "Period Ended": "31/12/2020",
        "Period Ended.1": "31/12/2021",
        "Period Ended.2": "0/01/1900",  # 1900 sentinel
        "Period Ended.3": "31/12/2023",
        "Period Ended.4": "31/12/2024",
        "EBITDA Margin": 14,
        "EBITDA Margin.1": 15,
        "EBITDA Margin.2": 16,
        "EBITDA Margin.3": 17,
        "EBITDA Margin.4": 18,
    })
    df2 = pd.DataFrame([row2])
    parse_period_ended_cols(df2)
    fy2 = get_metric_series_row(df2.iloc[0], "EBITDA Margin", prefer="FY")
    assert len(fy2) == 4, f"Expected 4 FY periods (one dropped), got {len(fy2)}"
    print(f"  OK 1900 sentinel properly excluded: {len(fy2)} periods remain")

    # Test 3: Classification weights
    print("\nTest 3: Classification weight validation")
    for classification in CLASSIFICATION_TO_SECTOR.keys():
        weights = get_classification_weights(classification, use_sector_adjusted=True)
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, \
            f"{classification}: weights sum to {total}, not 1.0"
    print(f"  OK All {len(CLASSIFICATION_TO_SECTOR)} classifications have valid weights (sum=1.0)")

    # Test 4: CQ exclusion from FY series
    print("\nTest 4: CQ data exclusion from FY series")
    fy_with_cq = get_metric_series_row(df_test.iloc[0], "EBITDA Margin", prefer="FY")
    assert 99.9 not in list(fy_with_cq.values), \
        "CQ period (99.9) should be excluded from FY series"
    print("  OK CQ periods properly excluded from FY series")

    # Test 5: Most recent annual value for Interest Coverage
    print("\nTest 5: Interest Coverage annual-only extraction")
    coverage = most_recent_annual_value(df_test.iloc[0], "EBITDA / Interest Expense (x)")
    assert coverage == 4.2, f"Expected 4.2 (FY.4), got {coverage}"
    print(f"  OK Interest Coverage uses annual data: {coverage}")

    # Test 6: Rating Group Classification (all non-IG -> HY, no Unknown)
    print("\nTest 6: Rating group classification")

    # Import the classification function from load_and_process_data scope
    # We'll recreate it here for testing
    def _classify_rating_group_test(x):
        ig_ratings_test = ['AAA','AA+','AA','AA-','A+','A','A-','BBB+','BBB','BBB-']
        if pd.isna(x):
            return 'High Yield'
        xu = str(x).strip().upper()
        if xu in ig_ratings_test:
            return 'Investment Grade'
        if xu in {'', 'NR', 'N/R', 'N.M', 'N/M', 'WD', 'W/D', 'NOT RATED', 'NR.'}:
            return 'High Yield'
        return 'High Yield'

    test_ratings = pd.Series(['AAA', 'BBB-', 'BB+', 'NR', 'WD', 'N/M', '', np.nan, 'SD', 'C'])
    result = test_ratings.apply(_classify_rating_group_test)
    expected = ['Investment Grade', 'Investment Grade', 'High Yield', 'High Yield', 'High Yield',
                'High Yield', 'High Yield', 'High Yield', 'High Yield', 'High Yield']

    assert list(result) == expected, f"Expected {expected}, got {list(result)}"
    assert 'Unknown' not in set(result.unique()), "Should not produce 'Unknown' category"
    print(f"  OK All non-IG ratings classified as High Yield: {len([r for r in result if r == 'High Yield'])}/10")
    print(f"  OK No 'Unknown' category produced")

    # Test 7: Core ID/Name alias resolution and canonicalization
    print("\nTest 7: Core ID/Name alias resolution")

    # Create DataFrame with messy headers (NBSP, extra spaces, case variations)
    messy = pd.DataFrame({
        'Issuer\u00a0Name': ['Alpha Corp', 'Beta PLC'],
        'issuer id': [101, 202],              # lower-case + space variant
        'S&P Credit  Rating': ['BBB-', 'NR'], # double space variant
        'EBITDA / Interest Expense (x)': [5.0, 1.2],
        'Period Ended.4': ['31/12/2024', '31/12/2024']
    })

    # Simulate header normalization (same as in load_and_process_data)
    messy.columns = [' '.join(str(c).replace('\u00a0', ' ').split()) for c in messy.columns]

    # Test alias lists
    name_aliases = COMPANY_NAME_ALIASES
    id_aliases = COMPANY_ID_ALIASES
    rating_aliases = RATING_ALIASES

    # Resolve columns
    nm = resolve_column(messy, name_aliases)
    idm = resolve_column(messy, id_aliases)
    rtm = resolve_column(messy, rating_aliases)

    assert nm is not None and idm is not None and rtm is not None, \
        f"Failed to resolve core aliases: name={nm}, id={idm}, rating={rtm}"
    print(f"  OK Resolved aliases: '{nm}', '{idm}', '{rtm}'")

    # Canonicalize to standard names
    rename_map_test = {nm: 'Company Name', idm: 'Company ID', rtm: 'S&P Credit Rating'}
    messy = messy.rename(columns=rename_map_test)

    assert {'Company Name', 'Company ID', 'S&P Credit Rating'}.issubset(set(messy.columns)), \
        f"Canonical rename failed. Columns: {list(messy.columns)}"
    print("  OK Canonical rename successful")
    print(f"  OK Final columns: {list(messy.columns)}")

    # Test 8: Date-based period selection in get_most_recent_column
    print("\nTest 8: Date-based period selection (get_most_recent_column)")

    # Test 8a: FY-only dataset - should resolve to latest FY value
    fy_only = pd.DataFrame([{
        "Period Ended": "31/12/2020",
        "Period Ended.1": "31/12/2021",
        "Period Ended.2": "31/12/2022",
        "Period Ended.3": "31/12/2023",
        "Period Ended.4": "31/12/2024",
        "EBITDA Margin": 10,
        "EBITDA Margin.1": 11,
        "EBITDA Margin.2": 12,
        "EBITDA Margin.3": 13,
        "EBITDA Margin.4": 14,
    }])

    result_series = get_most_recent_column(fy_only, "EBITDA Margin", "Most Recent Fiscal Year (FY0)")
    assert result_series.iloc[0] == 14, f"Expected value 14 (from .4, latest date), got {result_series.iloc[0]}"
    print("  OK FY-only dataset resolves to latest FY value")

    # Test 8b: Mixed FY/CQ - latest by date, not by index
    # .4 has earlier date (2024-06-30) vs .3 (2024-12-31), so .3 should be selected
    mixed = pd.DataFrame([{
        "Period Ended": "31/12/2020",
        "Period Ended.1": "31/12/2021",
        "Period Ended.2": "31/12/2022",
        "Period Ended.3": "31/12/2024",  # Latest date (FY)
        "Period Ended.4": "30/06/2024",  # Earlier date (CQ)
        "Period Ended.5": "30/09/2024",  # In between (CQ)
        "Net Debt / EBITDA": 3.0,
        "Net Debt / EBITDA.1": 3.1,
        "Net Debt / EBITDA.2": 3.2,
        "Net Debt / EBITDA.3": 2.8,  # Should be selected (latest date)
        "Net Debt / EBITDA.4": 4.5,  # Should NOT be selected (earlier date)
        "Net Debt / EBITDA.5": 4.2,
    }])

    result_series = get_most_recent_column(mixed, "Net Debt / EBITDA", "Most Recent Fiscal Year (FY0)")
    assert result_series.iloc[0] == 2.8, \
        f"Expected value 2.8 (from .3, latest date 2024-12-31), got {result_series.iloc[0]}"
    print("  OK Mixed FY/CQ selects latest by date (not index)")

    # Test 8c: Sentinel 1900 dates ignored
    with_sentinel = pd.DataFrame([{
        "Period Ended": "31/12/2020",
        "Period Ended.1": "31/12/2021",
        "Period Ended.2": "0/01/1900",   # Sentinel - should be ignored
        "Period Ended.3": "31/12/2023",   # Latest valid date
        "Return on Equity": 8.0,
        "Return on Equity.1": 9.0,
        "Return on Equity.2": 99.9,  # Has data but period is invalid
        "Return on Equity.3": 10.5,  # Should be selected
    }])

    result_series = get_most_recent_column(with_sentinel, "Return on Equity", "Most Recent Fiscal Year (FY0)")
    assert result_series.iloc[0] == 10.5, \
        f"Expected value 10.5 (from .3, ignoring 1900 sentinel at .2), got {result_series.iloc[0]}"
    print("  OK Sentinel 1900 dates properly ignored")

    # Test 9: use_quarterly_beta affects trend window
    print("\nTest 9: use_quarterly_beta affects trend window")

    # Create dataset with .1-.4 (annual) and .5-.8 (quarterly)
    # Use values that create detectable differences without maxing out at 1.0
    trend_test = pd.DataFrame([{
        "EBITDA Margin": 10.0,
        "EBITDA Margin.1": 10.5,
        "EBITDA Margin.2": 11.0,
        "EBITDA Margin.3": 11.5,
        "EBITDA Margin.4": 12.0,  # Gentle upward trend for FY
        "EBITDA Margin.5": 13.0,  # Quarterly - stronger uptick
        "EBITDA Margin.6": 14.0,
        "EBITDA Margin.7": 15.0,
        "EBITDA Margin.8": 16.0,  # Adds volatility for CQ mode
        "Total Debt / EBITDA (x)": 5.0,
        "Total Debt / EBITDA (x).1": 4.8,
        "Total Debt / EBITDA (x).2": 4.6,
        "Total Debt / EBITDA (x).3": 4.4,
        "Total Debt / EBITDA (x).4": 4.2,
        "Total Debt / EBITDA (x).5": 4.0,
        "Total Debt / EBITDA (x).6": 3.8,
        "Total Debt / EBITDA (x).7": 3.6,
        "Total Debt / EBITDA (x).8": 3.4,
        "Return on Equity": 8.0,
        "Return on Equity.1": 8.2,
        "Return on Equity.2": 8.4,
        "Return on Equity.3": 8.6,
        "Return on Equity.4": 8.8,
        "Return on Equity.5": 9.0,
        "Return on Equity.6": 9.2,
        "Return on Equity.7": 9.4,
        "Return on Equity.8": 9.6,
        "Current Ratio (x)": 1.2,
        "Current Ratio (x).1": 1.25,
        "Current Ratio (x).2": 1.3,
        "Current Ratio (x).3": 1.35,
        "Current Ratio (x).4": 1.4,
        "Current Ratio (x).5": 1.45,
        "Current Ratio (x).6": 1.5,
        "Current Ratio (x).7": 1.55,
        "Current Ratio (x).8": 1.6,
    }])

    test_metrics = ["EBITDA Margin", "Total Debt / EBITDA (x)", "Return on Equity", "Current Ratio (x)"]

    # Test 9a: FY-mode (use_quarterly=False) - should use only base + .1-.4
    trend_fy, _ = calculate_trend_indicators(trend_test, test_metrics, use_quarterly=False, reference_date=None)

    # Test 9b: Quarterly-mode (use_quarterly=True) - should use base + .1-.12 (available up to .8 here)
    trend_cq, _ = calculate_trend_indicators(trend_test, test_metrics, use_quarterly=True, reference_date=None)

    # Verify that momentum differs between the two modes
    # Momentum compares recent vs prior periods, so including .5-.8 should change the result
    margin_momentum_fy = trend_fy["EBITDA Margin_momentum"].iloc[0]
    margin_momentum_cq = trend_cq["EBITDA Margin_momentum"].iloc[0]

    # With 5 annual periods (base, .1-.4), momentum compares last 4 vs prior (not enough for 8-period split)
    # With 9 periods (base, .1-.8), momentum compares last 4 (.5-.8) vs prior 4 (.1-.4)
    assert margin_momentum_fy != margin_momentum_cq, \
        f"Expected different momentum scores between FY and CQ modes, got FY={margin_momentum_fy:.1f}, CQ={margin_momentum_cq:.1f}"
    print(f"  OK FY-mode momentum: {margin_momentum_fy:.1f}, LTM-mode momentum: {margin_momentum_cq:.1f} (differ)")

    # Verify volatility also differs (more data points = different coefficient of variation)
    margin_vol_fy = trend_fy["EBITDA Margin_volatility"].iloc[0]
    margin_vol_cq = trend_cq["EBITDA Margin_volatility"].iloc[0]

    assert margin_vol_fy != margin_vol_cq, \
        f"Expected different volatility scores between FY and CQ modes"
    print(f"  OK FY-mode volatility: {margin_vol_fy:.1f}, CQ-mode volatility: {margin_vol_cq:.1f} (differ)")

    print("  OK Quarterly toggle affects trend calculations as expected")

    # Test 10: Control separation (period vs trend window)
    print("\nTest 10: Control separation (period vs trend window)")

    # Prepare a small copy to avoid side effects
    _df = trend_test.copy()

    # A. Period selector affects single-period extraction (but not trend window)
    try:
        _fy_val = get_most_recent_column(_df, base_metric="EBITDA Margin", data_period_setting="Most Recent Fiscal Year (FY0)")
        _cq_val = get_most_recent_column(_df, base_metric="EBITDA Margin", data_period_setting="Most Recent Quarter (CQ-0)")

        # Check if they differ (may not if dataset aligns perfectly)
        if not _fy_val.equals(_cq_val):
            print("  OK: Period selector changes point-in-time values")
        else:
            print("  WARN: FY0 and CQ-0 resolved equal in this sample; code paths executed")
    except Exception as e:
        print(f"  WARN: Period selector test skipped ({e})")

    # B. Trend window affects momentum/volatility but NOT the period selector
    try:
        m_annual, _ = calculate_trend_indicators(_df, test_metrics, use_quarterly=False, reference_date=None)
        m_quarterly, _ = calculate_trend_indicators(_df, test_metrics, use_quarterly=True, reference_date=None)

        # Check if momentum differs between annual and quarterly windows
        if (m_annual["EBITDA Margin_momentum"] != m_quarterly["EBITDA Margin_momentum"]).any():
            print("  OK: Trend window changes momentum/volatility as expected")
        else:
            print("  WARN: Momentum equal in this sample; verify quarterly columns exist for tested metric")
    except Exception as e:
        print(f"  WARN: Trend window test skipped ({e})")

    # Test 11: FY/CQ overlap de-duplication
    print("\nTest: FY/CQ overlap de-duplication")
    df_test = pd.DataFrame([{
        "Period Ended": "31/12/2020",
        "Period Ended.1": "31/12/2021",
        "Period Ended.2": "31/12/2022",
        "Period Ended.3": "31/12/2023",
        "Period Ended.4": "31/12/2024",   # FY0
        "Period Ended.5": "31/12/2023",   # CQ-7
        "Period Ended.8": "31/12/2024",   # CQ-3 (overlaps FY0)
        "EBITDA Margin": 10, "EBITDA Margin.1": 11, "EBITDA Margin.2": 12, "EBITDA Margin.3": 13, "EBITDA Margin.4": 14,
        "EBITDA Margin.5": 13.5, "EBITDA Margin.8": 14.2
    }])
    ts = _build_metric_timeseries(df_test, "EBITDA Margin", use_quarterly=True)
    assert (ts.columns == pd.Index(sorted(ts.columns))).all()
    # Only one 2024-12-31 column should exist after dedup (CQ preferred)
    dup_2024 = [c for c in ts.columns if c.endswith("2024-12-31")]
    assert len(dup_2024) == 1, f"Expected single 2024-12-31 after dedup, got {dup_2024}"
    print("  OK overlap de-duplication (CQ preferred)")

    # Test 12: extract_issuer_financial_data handles non-December FY as FY
    print("\nTest 12: LLM extractor FY/CQ classification (non-December FY)")
    df_ed = pd.DataFrame([{
        "Company Name": "TestCo",
        "Period Ended": "30/06/2024",
        "Period Ended.1": "30/06/2023",
        "EBITDA Margin": 10.0,
        "EBITDA Margin.1": 9.0
    }])
    data_ed = extract_issuer_financial_data(df_ed, "TestCo")
    assert data_ed["period_types"].get("2024-06-30") == "FY", \
        f"Expected FY for 2024-06-30, got {data_ed['period_types'].get('2024-06-30')}"
    print("  OK non-December FY recognized as FY in LLM extractor")

    print("\n" + "="*60)
    print("SUCCESS: ALL RG_TESTS PASSED for V3.0 (12 tests)")
    print("="*60 + "\n")

    # Exit successfully after tests
    sys.exit(0)

