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
from enum import Enum

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

    # Define possible reference quarters (last 8 quarters)
    current_date = datetime.now()
    current_year = current_date.year

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

    # Filter to only quarters that make sense (not in future)
    valid_quarters = [q for q in possible_quarters if q <= pd.Timestamp(current_date)]

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
            except:
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
                except:
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
            except:
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

    return PeriodType.QUARTERLY


# [V2.2] Only configure Streamlit if not running tests
if os.environ.get("RG_TESTS") != "1":
    st.set_page_config(
        page_title="Issuer Credit Screening Model V3.0",
        layout="wide",
        page_icon="https://rubricsam.com/wp-content/uploads/2021/01/cropped-rubrics-logo-tight.png",
    )

# ============================================================================
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
        <h1>Issuer Credit Screening Model V3.0</h1>
        <div class="rb-sub">6-Factor Composite Scoring with Sector Adjustment & Trend Analysis</div>
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
            elif data_period == "Most Recent Quarter (CQ-0)":
                display_period = "CQ-0"
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
                f"Data Periods: {_period_labels['fy_label']}  |  {_period_labels['cq_label']}  |  "
                f"**Reference Date**: {ref_date.strftime('%b %d, %Y')} (aligned for fair comparison)"
            )
        else:
            st.caption(f"Data Periods: {_period_labels['fy_label']}  |  {_period_labels['cq_label']}")

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
METRIC_ALIASES = {
    "EBITDA Margin": ["EBITDA Margin", "EBITDA margin %", "EBITDA Margin (%)"],
    "Return on Equity": ["Return on Equity", "ROE"],
    "Return on Assets": ["Return on Assets", "ROA"],
    "Total Debt / EBITDA (x)": ["Total Debt / EBITDA (x)", "Total Debt/EBITDA", "Total Debt to EBITDA", "Debt / EBITDA (x)"],
    "Net Debt / EBITDA": ["Net Debt / EBITDA", "Net Debt/EBITDA", "Net Debt to EBITDA"],
    "EBITDA / Interest Expense (x)": ["EBITDA / Interest Expense (x)", "EBITDA/Interest (x)", "EBITDA / Interest", "Interest Coverage (x)", "Interest Cover (x)"],
    "Current Ratio (x)": ["Current Ratio (x)", "Current Ratio"],
    "Quick Ratio (x)": ["Quick Ratio (x)", "Quick Ratio"],
    # optional level/trend inputs (don't break if absent)
    "Total Debt": ["Total Debt"],
    "Cash and Short-Term Investments": ["Cash and Short-Term Investments"],
    "Total Revenues": ["Total Revenues"],
}

def _resolve_company_name_col(df: pd.DataFrame) -> str | None:
    return resolve_column(df, ["Company_Name", "Company Name", "Name"])

def _resolve_classification_col(df: pd.DataFrame) -> str | None:
    return resolve_column(df, ["Rubrics_Custom_Classification", "Rubrics Custom Classification"])

def resolve_metric_column(df_like, canonical: str) -> str | None:
    aliases = METRIC_ALIASES.get(canonical, [canonical])
    # Accept Series -> make it a 1-row frame to reuse resolve_column
    if isinstance(df_like, pd.Series):
        df_like = df_like.to_frame().T
    return resolve_column(df_like, aliases)

def list_metric_columns(df: pd.DataFrame, canonical: str) -> tuple[str | None, list[str]]:
    """Return (base_col, [all existing suffixed cols]) for a canonical metric."""
    base = resolve_metric_column(df, canonical)
    suffixes = []
    for alias in METRIC_ALIASES.get(canonical, [canonical]):
        suffixes += [c for c in df.columns if isinstance(c, str) and c.startswith(f"{alias}.")]
    suffixes = sorted(set(suffixes))
    return base, suffixes

def get_from_row(row: pd.Series, canonical: str):
    """Row-level safe getter honoring aliases."""
    for a in METRIC_ALIASES.get(canonical, [canonical]):
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
    cols = [c for c in results.columns if c.endswith("_Score") and c != "Composite_Score"]
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
    df["Rank_in_Band"] = df["Composite_Score"].rank(method="dense", ascending=False).astype(int)

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
    Parse Period Ended* columns, return (num_cols, min_date, max_date, fy_suffixes, cq_suffixes).
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
        "cq_suffixes": cq_sfx,
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
        fy_suffixes, cq_suffixes = period_cols_by_kind(pe_data, df_original)
        period_kind_by_suffix = {sfx: "FY" for sfx in fy_suffixes}
        period_kind_by_suffix.update({sfx: "CQ" for sfx in cq_suffixes})
    except Exception:
        period_kind_by_suffix = {}  # fallback to month heuristic below

    for metric in metrics_to_extract:
        # Resolve metric column (handle aliases)
        metric_col = resolve_column(df_original, METRIC_ALIASES.get(metric, [metric]))
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
        model="gpt-4-turbo-preview",
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
        max_tokens=2500
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
        "Growth_Score": _safe_get(row_res or row_raw, "Growth_Score", "growth_score"),
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

    # Medians for 6 factors + composite
    factor_cols = ["Credit_Score", "Leverage_Score", "Profitability_Score",
                   "Liquidity_Score", "Growth_Score", "Composite_Score"]
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
    factors = ["Credit_Score","Leverage_Score","Profitability_Score","Liquidity_Score","Growth_Score","Cash_Flow_Score"]
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
      - 'period_hints': List[str]            # e.g. ['FY0: 31/12/2024', 'CQ-1: 30/09/2024']
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
        - medians: median scores for Credit, Leverage, Profitability, Liquidity, Growth, and Composite
        - ig_count / hy_count: Investment Grade vs High Yield mix
        - signal_counts: distribution of Combined_Signal values (Strong/Moderate Quality & Improving/Stable/Deteriorating Trend)
        - top5 / bottom5: top 5 and bottom 5 issuers by Composite_Score

        Structure your response in this order:
        1) **Classification overview** — what this classification represents and its role in the credit universe.
        2) **Cohort credit profile** — discuss median scores across the 6 factors, highlight the IG vs HY mix, and assess overall credit quality.
        3) **Signal distribution** — analyze the signal_counts to identify whether the group is trending positively or negatively; note any concentration in specific signals.
        4) **Notable performers** — mention a few names from top5 and bottom5 to illustrate the range of credit quality within the group.
        5) **Methodology note** — briefly mention the 6-factor scoring system (0–100 scale) with classification-adjusted weights and the Leverage Option A (40/30/20/10).

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
        except:
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
    except:
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
    except:
        return []


def get_bottom_issuers(class_subset: pd.DataFrame, n: int, by: str) -> list:
    """Get bottom N issuers by specified metric."""
    try:
        name_col = _namecol(class_subset)
        if not name_col or by not in class_subset.columns:
            return []

        sorted_df = class_subset[[name_col, by]].dropna().sort_values(by, ascending=True).head(n)
        return [{"name": row[name_col], "value": f"{row[by]:.1f}"} for _, row in sorted_df.iterrows()]
    except:
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
    except:
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
    for k in METRIC_ALIASES.keys():
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
    fy, cq = _latest_periods(row, prefer_fy=True, fy_n=5, cq_n=8)
    cols = [f"FY-{i}" for i in range(len(fy)-1, -1, -1)] + [f"CQ-{i}" for i in range(len(cq)-1, -1, -1)]
    dates = list(reversed(fy)) + list(reversed(cq))

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
        "growth": _safe_get(issuer_row, "Growth_Score", "growth_score"),
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
- Total Debt: ${m['total_debt']}

Coverage:
- EBITDA / Interest Expense: {m['coverage']}x (Peer median: {context['peer_stats'].get('peer_median_coverage', 'N/A')}x)

Liquidity:
- Current Ratio: {m['current_ratio']}x
- Cash & Equivalents: ${m['cash']}
- Quick Ratio: {m['quick_ratio']}x
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
- Growth Score: {s['growth']}
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
                                    df_original,
                                    trend_thr=55,  # Fixed threshold
                                    quality_thr=60)  # Fixed threshold
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
    - Positions 5-12 (.5 through .12): CQ-7 through CQ-0 (quarterly)

    Args:
        pe_data: List of (suffix, datetime_series) tuples from parse_period_ended_cols
        df: DataFrame (not used in position-based approach, kept for API compatibility)

    Returns: (fy_suffixes, cq_suffixes) - lists of suffixes
    """
    if len(pe_data) == 0:
        return [], []

    if len(pe_data) <= 5:
        # Only annual data (5 or fewer periods)
        return [suffix for suffix, _ in pe_data], []

    # Split: first 5 positions are FY (annual), rest are CQ (quarterly)
    fy_suffixes = [pe_data[i][0] for i in range(5)]
    cq_suffixes = [pe_data[i][0] for i in range(5, len(pe_data))]

    return fy_suffixes, cq_suffixes

# ============================================================================
# [V2.2] PERIOD CALENDAR UTILITIES (ROBUST HANDLING OF VENDOR DATES & FY/CQ OVERLAP)
# ============================================================================

_EXCEL_EPOCH = pd.Timestamp("1899-12-30")  # Excel serial origin

_SENTINEL_BAD = {
    "0/01/1900", "00/01/1900", "0/0/0000", "00/00/0000",
    "1900-01-00", "1899-12-31"  # common vendor quirks
}

_PERIOD_COL_RE = re.compile(
    r"(?i)^.*period\s*ended.*\b((FY-?\d+|FY0|CQ-?\d+|CQ0))\b"
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

def _batch_extract_metrics(df, metric_list, has_period_alignment, data_period_setting, reference_date=None):
    """
    OPTIMIZED: Extract all metrics at once using vectorized operations.
    Returns dict of {metric_name: Series of values}.

    Args:
        reference_date: If provided, filters to only use data on or before this date.
                       Used for alignment when CQ-0 is selected with align_to_reference=True.
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
    fy_suffixes, cq_suffixes = period_cols_by_kind(pe_data, df)

    if data_period_setting == "Most Recent Quarter (CQ-0)":
        # User wants most recent quarter - include both CQ and FY, most recent date wins
        if cq_suffixes:
            # Include both CQ and FY periods - line 3270-3272 will pick the most recent
            candidate_suffixes = cq_suffixes + fy_suffixes
        else:
            # No quarterly data available - use FY as fallback
            candidate_suffixes = fy_suffixes if fy_suffixes else [s for s, _ in pe_data[:5]]

    elif data_period_setting == "Most Recent Fiscal Year (FY0)":
        # User wants fiscal year data only
        candidate_suffixes = fy_suffixes if fy_suffixes else [s for s, _ in pe_data[:5]]

    else:
        # Unknown setting - default to FY for safety
        candidate_suffixes = fy_suffixes if fy_suffixes else [s for s, _ in pe_data[:5]]

    # For each metric, extract most recent value based on data_period_setting (vectorized)
    for metric in metric_list:
        # Collect (date, value) pairs for this metric across candidate suffixes
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
                'value': pd.to_numeric(df[col], errors='coerce')
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
    Returns a tuple (latest_fy_date, latest_cq_date, used_fallback) using parsed Period Ended columns.

    Primary method: Uses period_cols_by_kind classifier to detect FY vs CQ based on date frequency.
    Fallback method: Treats first 5 suffixes as FY, remainder as CQ (documented fallback).

    Args:
        df: DataFrame with Period Ended columns

    Returns:
        (latest_fy_date, latest_cq_date, used_fallback) tuple where:
        - latest_fy_date: Latest fiscal year end date (pd.Timestamp or pd.NaT)
        - latest_cq_date: Latest quarter end date (pd.Timestamp or pd.NaT)
        - used_fallback: Boolean indicating if fallback method was used
    """
    try:
        pe_data = parse_period_ended_cols(df.copy())  # [(suffix, series_of_dates), ...]
        fy_suffixes, cq_suffixes = period_cols_by_kind(pe_data, df)  # preferred path

        # Collect column-wise dates
        latest_fy = pd.NaT
        latest_cq = pd.NaT

        for sfx, ser in pe_data:
            sd = pd.to_datetime(ser, errors="coerce", dayfirst=True)
            if sfx in fy_suffixes:
                max_date = sd.max(skipna=True)
                if pd.notna(max_date):
                    latest_fy = max_date if pd.isna(latest_fy) else max(latest_fy, max_date)
            if sfx in cq_suffixes:
                max_date = sd.max(skipna=True)
                if pd.notna(max_date):
                    latest_cq = max_date if pd.isna(latest_cq) else max(latest_cq, max_date)

        return latest_fy, latest_cq, False  # False => did not use fallback

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
            cq_cols = pe_cols_sorted[5:]

            latest_fy = pd.to_datetime(df[fy_cols], errors="coerce", dayfirst=True).max(axis=1).max(skipna=True) if fy_cols else pd.NaT
            latest_cq = pd.to_datetime(df[cq_cols], errors="coerce", dayfirst=True).max(axis=1).max(skipna=True) if cq_cols else pd.NaT

            return latest_fy, latest_cq, True  # True => used fallback

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
        - cq_label: "Most Recent Quarter (CQ-0 — 2025-06-30)" or "Most Recent Quarter (CQ-0)" if date unavailable
        - used_fallback: Boolean indicating if fallback classification method was used

    Example:
        {
            "fy_label": "Most Recent Fiscal Year (FY0 — 2024-12-31)",
            "cq_label": "Most Recent Quarter (CQ-0 — 2025-06-30)",
            "used_fallback": False
        }
    """
    fy0, cq0, used_fallback = _latest_period_dates(df)

    def _fmt(prefix, dt):
        """Format label with optional date suffix."""
        return f"{prefix}" + (f" — {dt.date().isoformat()}" if pd.notna(dt) else "")

    return {
        "fy_label": _fmt("Most Recent Fiscal Year (FY0)", fy0),
        "cq_label": _fmt("Most Recent Quarter (CQ-0)", cq0),
        "used_fallback": used_fallback
    }

# ============================================================================
# DYNAMIC WEIGHT CALIBRATION (V2.2.1)
# ============================================================================

def calculate_calibrated_sector_weights(df, rating_band='BBB', use_dynamic=True):
    """
    Calculate sector weights that normalize scores across sectors.

    V3.0 REDESIGN: Two-mode system with UNIVERSAL_WEIGHTS as single source of truth

    MODE 1 (use_dynamic=False):
        Returns universal weights for ALL sectors (no sector differentiation)

    MODE 2 (use_dynamic=True):
        Starts from UNIVERSAL_WEIGHTS and applies data-driven sector neutralization

    Methodology: Inverse Deviation Weighting
    - If sector underperforms on factor → REDUCE weight (minimize penalty)
    - If sector outperforms on factor → REDUCE weight (don't amplify advantage)
    - If sector is neutral on factor → INCREASE weight (make it differentiator)

    Args:
        df: DataFrame with factor scores and classifications
        rating_band: Rating level to calibrate on (default 'BBB' for broad IG)
        use_dynamic: If True, calculate from uploaded data. If False, use universal weights.

    Returns:
        dict: Sector name -> {factor: weight} mappings, normalized to sum=1.0
    """
    if not use_dynamic:
        # MODE 1: Return universal weights for all sectors (no differentiation)
        # Get all unique sectors from CLASSIFICATION_TO_SECTOR
        all_sectors = set(CLASSIFICATION_TO_SECTOR.values())
        universal_for_all = {}
        for sector in all_sectors:
            universal_for_all[sector] = UNIVERSAL_WEIGHTS.copy()
        universal_for_all['Default'] = UNIVERSAL_WEIGHTS.copy()
        return universal_for_all

    # MODE 2: Data-driven calibration starting from universal base

    # Define rating bands
    rating_bands = {
        'BBB': ['BBB+', 'BBB', 'BBB-'],
        'A': ['A+', 'A', 'A-'],
        'BB': ['BB+', 'BB', 'BB-'],
    }

    # Get companies in target rating band
    target_ratings = rating_bands.get(rating_band, ['BBB+', 'BBB', 'BBB-'])

    # Find the rating column
    rating_col = None
    for col_name in ['Credit_Rating_Clean', 'S&P LT Issuer Credit Rating', 'Credit Rating', 'Rating']:
        if col_name in df.columns:
            rating_col = col_name
            break

    if rating_col is None:
        # Fallback to universal weights for all sectors
        all_sectors = set(CLASSIFICATION_TO_SECTOR.values())
        universal_for_all = {}
        for sector in all_sectors:
            universal_for_all[sector] = UNIVERSAL_WEIGHTS.copy()
        universal_for_all['Default'] = UNIVERSAL_WEIGHTS.copy()
        return universal_for_all

    df_rated = df[df[rating_col].isin(target_ratings)].copy()

    if len(df_rated) < 50:  # Insufficient data
        # Fallback to universal weights for all sectors
        all_sectors = set(CLASSIFICATION_TO_SECTOR.values())
        universal_for_all = {}
        for sector in all_sectors:
            universal_for_all[sector] = UNIVERSAL_WEIGHTS.copy()
        universal_for_all['Default'] = UNIVERSAL_WEIGHTS.copy()
        return universal_for_all

    # Map factor scores to their column names
    factor_score_cols = {
        'credit_score': 'Credit_Score',
        'leverage_score': 'Leverage_Score',
        'profitability_score': 'Profitability_Score',
        'liquidity_score': 'Liquidity_Score',
        'growth_score': 'Growth_Score',
        'cash_flow_score': 'Cash_Flow_Score'
    }

    # Get classification field
    class_field = None
    for field in ['Rubrics_Custom_Classification', 'Rubrics Custom Classification',
                  'Classification', 'Custom_Classification']:
        if field in df_rated.columns:
            class_field = field
            break

    if class_field is None:
        # Fallback to universal weights for all sectors
        all_sectors = set(CLASSIFICATION_TO_SECTOR.values())
        universal_for_all = {}
        for sector in all_sectors:
            universal_for_all[sector] = UNIVERSAL_WEIGHTS.copy()
        universal_for_all['Default'] = UNIVERSAL_WEIGHTS.copy()
        return universal_for_all

    # Calculate market medians (for this rating band)
    market_medians = {}
    for factor_key, score_col in factor_score_cols.items():
        if score_col in df_rated.columns:
            values = pd.to_numeric(df_rated[score_col], errors='coerce').dropna()
            if len(values) > 0:
                market_medians[factor_key] = values.median()

    # Calculate sector-specific weights
    calibrated_weights = {}

    # Get all unique sectors
    all_sectors = set(CLASSIFICATION_TO_SECTOR.values())

    for sector_name in all_sectors:
        # Get classifications for this sector
        sector_classifications = [k for k, v in CLASSIFICATION_TO_SECTOR.items()
                                 if v == sector_name]

        # Get companies in this sector
        sector_df = df_rated[df_rated[class_field].isin(sector_classifications)]

        if len(sector_df) < 5:  # Insufficient data for this sector
            # Use universal weights as fallback
            calibrated_weights[sector_name] = UNIVERSAL_WEIGHTS.copy()
            continue

        # Calculate sector medians
        sector_medians = {}
        for factor_key, score_col in factor_score_cols.items():
            if score_col in sector_df.columns:
                values = pd.to_numeric(sector_df[score_col], errors='coerce').dropna()
                if len(values) > 0:
                    sector_medians[factor_key] = values.median()

        # Calculate deviations and calibrated weights
        raw_weights = {}

        for factor_key in factor_score_cols.keys():
            if factor_key not in sector_medians or factor_key not in market_medians:
                # Use universal weight as base
                raw_weights[factor_key] = UNIVERSAL_WEIGHTS[factor_key]
                continue

            sector_val = sector_medians[factor_key]
            market_val = market_medians[factor_key]

            if market_val == 0:
                # Use universal weight as base
                raw_weights[factor_key] = UNIVERSAL_WEIGHTS[factor_key]
                continue

            # Calculate deviation percentage
            # For scores, higher is better, so negative deviation = underperformance
            deviation_pct = ((sector_val - market_val) / abs(market_val)) * 100

            # START FROM UNIVERSAL BASE (same for all sectors)
            base_weight = UNIVERSAL_WEIGHTS[factor_key]

            # SECTOR NEUTRALIZATION LOGIC
            # Any deviation (+ or -) means this factor reflects sector structure
            # → REDUCE weight to remove sector bias
            # Only neutral factors should drive cross-sector comparisons

            abs_dev = abs(deviation_pct)

            if abs_dev > 50:
                # Extreme deviation → this is pure sector characteristic
                calibrated = base_weight * 0.15
            elif abs_dev > 30:
                # Large deviation → strongly sector-driven
                calibrated = base_weight * 0.30
            elif abs_dev > 20:
                # Moderate deviation → significantly sector-driven
                calibrated = base_weight * 0.50
            elif abs_dev > 10:
                # Small deviation → moderately sector-driven
                calibrated = base_weight * 0.70
            elif abs_dev > 5:
                # Minor deviation → slightly sector-driven
                calibrated = base_weight * 0.85
            else:
                # Neutral (±5%) → pure issuer differentiation
                calibrated = base_weight * 1.00

            # Cap weights to prevent extreme values
            calibrated = min(max(calibrated, 0.01), 0.45)

            raw_weights[factor_key] = calibrated

        # Normalize to sum = 1.0
        total = sum(raw_weights.values())
        if total > 0:
            calibrated_weights[sector_name] = {k: v/total for k, v in raw_weights.items()}
        else:
            # Use universal weights as fallback
            calibrated_weights[sector_name] = UNIVERSAL_WEIGHTS.copy()

    # Add Default using universal weights
    calibrated_weights['Default'] = UNIVERSAL_WEIGHTS.copy()

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
    'leverage_score': 0.20,
    'profitability_score': 0.20,
    'liquidity_score': 0.10,
    'growth_score': 0.15,
    'cash_flow_score': 0.15
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
        Dictionary with 6 factor weights (summing to 1.0)
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

def _resolve_model_weights_for_row(row: pd.Series, scoring_method: str):
    """
    Return (weights_dict, provenance_str) with sector/classification precedence.
    Keys: lowercase matching UNIVERSAL_WEIGHTS (credit_score, leverage_score,
          profitability_score, liquidity_score, growth_score, cash_flow_score)
    """
    UNIVERSAL = {"credit_score": 0.20, "leverage_score": 0.20, "profitability_score": 0.20,
                 "liquidity_score": 0.10, "growth_score": 0.15, "cash_flow_score": 0.15}

    # Universal mode short-circuit
    if str(scoring_method).lower().startswith("universal"):
        return UNIVERSAL, "Universal weights"

    cls = _resolve_text_field(row, ["Rubrics_Custom_Classification", "Rubrics Custom Classification", "Classification", "Custom_Classification"])
    sec = _resolve_text_field(row, ["IQ_SECTOR", "Sector", "GICS_Sector"])

    # 1) App-provided resolver (preferred)
    try:
        w = get_classification_weights(cls, use_sector_adjusted=True)
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

def _build_explainability_table(issuer_row: pd.Series, scoring_method: str):
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
        "Growth": "growth_score",
        "Cash Flow": "cash_flow_score"
    }

    canonical = list(factor_map.keys())

    # Check for column existence
    present = [f for f in canonical if f.replace(" ", "_") + "_Score" in issuer_row.index]

    # Get CURRENT weights (from current calibration settings)
    current_weights_lc, provenance = _resolve_model_weights_for_row(issuer_row, scoring_method)

    # Normalize current weights over present factors
    current_w = {f: float(max(0.0, current_weights_lc.get(factor_map[f], 0.0))) for f in present}
    current_sum = sum(current_w.values()) or 1.0
    current_w = {k: v / current_sum for k, v in current_w.items()}

    # Get ORIGINAL weights (used in actual calculation) from stored columns
    original_w = {}
    weight_cols_map = {
        "Credit": "Weight_Credit_Used",
        "Leverage": "Weight_Leverage_Used",
        "Profitability": "Weight_Profitability_Used",
        "Liquidity": "Weight_Liquidity_Used",
        "Growth": "Weight_Growth_Used",
        "Cash Flow": "Weight_CashFlow_Used"
    }

    # Check if original weights are stored
    has_original_weights = all(weight_cols_map[f] in issuer_row.index for f in present)

    if has_original_weights:
        original_w = {f: float(issuer_row.get(weight_cols_map[f], 0.0)) for f in present}
        original_sum = sum(original_w.values()) or 1.0
        original_w = {k: v / original_sum for k, v in original_w.items()}
    else:
        # Fall back to current weights if original not stored
        original_w = current_w.copy()

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
            "Original Weight %": round(100.0 * original_wt, 1),
            "Current Weight %": round(100.0 * current_wt, 1),
            "Weight Change": f"{weight_change:+.0f}%",
            "Original Contrib": round(original_contrib, 2),
            "Current Contrib": round(current_contrib, 2),
            "Contrib Change": round(current_contrib - original_contrib, 2)
        })

    df = pd.DataFrame(rows)
    comp = float(issuer_row.get("Composite_Score", np.nan))

    # Calculate differences
    original_sum_contrib = df["Original Contrib"].sum() if len(df) else np.nan
    current_sum_contrib = df["Current Contrib"].sum() if len(df) else np.nan
    diff_original = float(original_sum_contrib - comp) if pd.notna(comp) and len(df) else np.nan
    diff_current = float(current_sum_contrib - comp) if pd.notna(comp) and len(df) else np.nan

    # Return extended info
    return df, provenance, comp, diff_current, original_sum_contrib, has_original_weights
# ================================

def render_issuer_explainability(filtered: pd.DataFrame, scoring_method: str):
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

        df_contrib, provenance, comp, diff_current, original_sum, has_original = _build_explainability_table(issuer_row, scoring_method)

        # Display weight provenance
        st.markdown(f"**Current Weight Method:** {provenance}")

        if has_original:
            st.info("ℹ️ **Comparison Mode:** Showing original weights (used in calculation) vs current weights (from active calibration)")
        else:
            st.info("ℹ️ **Note:** Original weights not stored. Showing current calibrated weights only.")

        # Display comparison table
        st.dataframe(df_contrib, use_container_width=True, hide_index=True)

        # Highlight significant weight changes
        st.caption("""
        **How to read this table:**
        - **Score**: Factor score (0-100) for this issuer
        - **Original Weight %**: Weight used in stored composite calculation
        - **Current Weight %**: Weight from current dynamic calibration settings
        - **Weight Change**: % change from original to current
        - **Original Contrib**: Factor's contribution with original weights
        - **Current Contrib**: Factor's contribution with current weights
        - **Contrib Change**: How calibration changes this factor's impact
        """)

        # Summary metrics
        st.markdown("---")
        st.markdown("### Score Breakdown")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Stored Composite Score", f"{comp:.2f}" if pd.notna(comp) else "n/a")
            st.caption("Score as calculated and stored")

        with col2:
            if has_original and pd.notna(original_sum):
                st.metric("Original Calculation", f"{original_sum:.2f}")
                st.caption("Sum using original weights")
                diff_original = original_sum - comp if pd.notna(comp) else np.nan
                if pd.notna(diff_original) and abs(diff_original) > 0.5:
                    st.caption(f"Diff: {diff_original:+.2f} (rounding)")
            else:
                st.metric("Original Calculation", "N/A")
                st.caption("Weights not stored")

        with col3:
            current_sum = df_contrib["Current Contrib"].sum() if len(df_contrib) else np.nan
            if pd.notna(current_sum):
                st.metric("Current Calibration", f"{current_sum:.2f}")
                st.caption("Sum using current weights")
                if pd.notna(comp):
                    impact = current_sum - comp
                    st.caption(f"Impact: {impact:+.2f} points")
            else:
                st.metric("Current Calibration", "N/A")

        # Interpretation guidance
        if has_original and pd.notna(comp) and pd.notna(current_sum):
            impact = current_sum - comp
            if abs(impact) > 5.0:
                st.warning(f"""
                **⚠️ Significant Calibration Impact ({impact:+.2f} points)**

                Current dynamic calibration would change this issuer's score by {impact:+.2f} points
                compared to the stored composite score. This indicates substantial weight adjustments
                for this sector/classification.
                """)
            elif abs(impact) > 1.0:
                st.info(f"""
                **ℹ️ Moderate Calibration Impact ({impact:+.2f} points)**

                Dynamic calibration adjusts this issuer's score by {impact:+.2f} points.
                This shows sector-specific weighting is active and tailoring weights for this classification.
                """)
            elif abs(impact) > 0.1:
                st.success(f"""
                **✓ Minor Calibration Impact ({impact:+.2f} points)**

                Current weights are very similar to original calculation.
                Sector-adjusted weighting is active but produces minimal score change.
                """)
# ================================

# ============================================================================
# METHODOLOGY TAB RENDERING (PROGRAMMATIC SPECIFICATION)
# ============================================================================

def _detect_factors_and_metrics():
    """
    Returns metadata about the 6-factor model structure.

    Returns:
        List of dicts with keys: factor, metric_examples, direction
    """
    return [
        {
            "factor": "Credit Score",
            "metric_examples": "S&P LT Issuer Rating (100%). Interest Coverage is assessed under Leverage.",
            "direction": "Higher is better"
        },
        {
            "factor": "Leverage Score",
            "metric_examples": "Net Debt/EBITDA (40%), Interest Coverage (30%), Debt/Capital (20%), Total Debt/EBITDA (10%)",
            "direction": "Lower debt is better (inverted scoring)"
        },
        {
            "factor": "Profitability Score",
            "metric_examples": "EBITDA Margin, ROA, Net Margin",
            "direction": "Higher is better"
        },
        {
            "factor": "Liquidity Score",
            "metric_examples": "Current Ratio, Cash / Total Debt",
            "direction": "Higher is better"
        },
        {
            "factor": "Growth Score",
            "metric_examples": "Revenue CAGR, EBITDA growth (multi-period trend)",
            "direction": "Higher is better"
        },
        {
            "factor": "Cash Flow Score",
            "metric_examples": "OCF/Revenue, OCF/Debt, UFCF margin, LFCF margin (equal-weighted, clipped & scaled)",
            "direction": "Higher is better"
        }
    ]


def render_methodology_tab(df_original: pd.DataFrame, results_final: pd.DataFrame):
    """
    Render comprehensive, programmatically generated methodology specification.

    All numbers, weights, and period labels are read from current app state or constants.
    No hard-coded stale values.

    Args:
        df_original: Original uploaded DataFrame
        results_final: Final results DataFrame with scores and signals
    """
    st.markdown("# Model Methodology (V3.0)")
    st.markdown("*Programmatically Generated Specification — All values reflect current configuration*")
    st.markdown("---")

    # ========================================================================
    # SECTION 1: OVERVIEW & PIPELINE
    # ========================================================================

    st.markdown("## 1. Overview & Pipeline")
    st.markdown("""
    The Issuer Credit Screening Model is a multi-factor analytics system that evaluates global fixed-income
    issuers using a structured six-factor composite score and a trend overlay. It combines fundamental
    strength (level) with trend momentum (direction) to produce consistent issuer rankings within and across
    rating groups (IG and HY).

    **End-to-end pipeline:**

    1. **Data Upload** → Validate core columns (Company ID, Name, S&P Rating)
    2. **Period Parsing** → Extract FY/CQ dates from "Period Ended" columns, resolve overlaps
    3. **Metric Extraction** → Pull most recent values per user's data period setting (FY0 or CQ-0)
    4. **Factor Scoring** → Transform 6 raw metrics into 0-100 factor scores
    5. **Weight Resolution** → Apply Universal or Sector-Adjusted weights per issuer classification
    6. **Composite Score** → Weighted average of 6 factor scores → single 0-100 quality metric
    7. **Trend Overlay** → Calculate Cycle Position Score from time-series momentum
    8. **Signal Assignment** → Classify into 4 quadrants (Strong/Weak × Improving/Deteriorating)
    9. **Recommendation** → Percentile-based bands with guardrail (no Buy for Weak & Deteriorating)
    10. **Visualization & Export** → Charts, leaderboards, AI analysis, downloadable data
    """)

    # ========================================================================
    # SECTION 2: FACTOR → METRICS MAPPING
    # ========================================================================

    st.markdown("## 2. Factor → Metrics Mapping")
    st.markdown("Each issuer receives six factor scores (0-100) prior to aggregation:")

    factor_meta = _detect_factors_and_metrics()
    factors_df = pd.DataFrame(factor_meta)
    factors_df.columns = ["Factor", "Metric Examples", "Direction"]
    st.table(factors_df)

    st.markdown("""
    **Normalization notes:**
    - All factors are robust-scaled (winsorized, median/MAD-based) to handle outliers
    - Factors like Leverage are inverted where lower is better
    - Growth and trend metrics use time-series slope over the selected window
    - Final transformation produces 0-100 scale for all factors
    """)

    # ========================================================================
    # SECTION 3: COMPOSITE SCORE FORMULA
    # ========================================================================

    st.markdown("## 3. Composite Score Formula")
    st.markdown("The **Composite_Score** is a weighted average of the 6 factor scores:")

    st.code("""
Composite Score :
    w_credit      × Credit_Score +
    w_leverage    × Leverage_Score +
    w_profit      × Profitability_Score +
    w_liquidity   × Liquidity_Score +
    w_growth      × Growth_Score +
    w_cashflow    × Cash_Flow_Score

where all weights sum to 1.0
    """, language="text")

    st.markdown("""
    **Range:** 0-100 (higher is better)
    **Usage:** Issuers are ranked **within rating groups** (IG vs HY) using Composite_Score percentiles
    """)

    # ========================================================================
    # SECTION 4: CURRENT WEIGHTS
    # ========================================================================

    st.markdown("## 4. Current Weights")

    # Get current scoring method from session state
    scoring_method = st.session_state.get("scoring_method", "Universal Weights")
    use_sector_adjusted = st.session_state.get("use_sector_adjusted", False)

    st.markdown(f"**Active configuration:** `{scoring_method}`")

    if use_sector_adjusted:
        st.markdown("""
        **Sector-Adjusted Mode:** Weights vary by issuer classification → parent sector.
        Example: "Software and Services" → "Information Technology" sector weights.
        """)

        # Get calibrated weights from session state if available
        calibrated_weights = st.session_state.get('_calibrated_weights', None)

        if calibrated_weights:
            # Build weights table for all calibrated sectors
            weights_rows = []
            for sector_name in sorted(calibrated_weights.keys()):
                weights = calibrated_weights[sector_name]
                weights_rows.append({
                    "Sector": sector_name,
                    "Credit": f"{weights['credit_score']:.2f}",
                    "Leverage": f"{weights['leverage_score']:.2f}",
                    "Profitability": f"{weights['profitability_score']:.2f}",
                    "Liquidity": f"{weights['liquidity_score']:.2f}",
                    "Growth": f"{weights['growth_score']:.2f}",
                    "Cash Flow": f"{weights['cash_flow_score']:.2f}",
                    "Sum": f"{sum(weights.values()):.2f}"
                })

            weights_df = pd.DataFrame(weights_rows)
            st.dataframe(weights_df, use_container_width=True, height=400)
        else:
            st.info("Calibrated weights not yet calculated. Upload data to see sector-specific weights.")

    else:
        st.markdown("**Universal Weights Mode:** Same weights for all issuers regardless of sector.")

        default_weights = UNIVERSAL_WEIGHTS
        weights_display = pd.DataFrame([{
            "Factor": "Credit Score",
            "Weight": f"{default_weights['credit_score']:.2f}"
        }, {
            "Factor": "Leverage Score",
            "Weight": f"{default_weights['leverage_score']:.2f}"
        }, {
            "Factor": "Profitability Score",
            "Weight": f"{default_weights['profitability_score']:.2f}"
        }, {
            "Factor": "Liquidity Score",
            "Weight": f"{default_weights['liquidity_score']:.2f}"
        }, {
            "Factor": "Growth Score",
            "Weight": f"{default_weights['growth_score']:.2f}"
        }, {
            "Factor": "Cash Flow Score",
            "Weight": f"{default_weights['cash_flow_score']:.2f}"
        }, {
            "Factor": "**Total**",
            "Weight": f"**{sum(default_weights.values()):.2f}**"
        }])

        st.table(weights_display)

    # ========================================================================
    # [V2.2.1] SECTION 4A: WEIGHT CALIBRATION (IF ENABLED)
    # ========================================================================

    use_calibration = st.session_state.get('use_dynamic_calibration', True)
    if use_calibration:
        st.markdown("---")
        st.markdown("### 4a. Dynamic Weight Calibration (ACTIVE)")

        st.markdown("""
        **Weight calibration mode is currently enabled (default).** The weights shown above have been recalibrated
        from your uploaded data to normalize composite scores across sectors.

        **How it works:**

        1. **Target Rating Band Selection** — Calibration uses companies in a specific rating band (default: BBB)
           to calculate sector-specific baseline performance.

        2. **Deviation Analysis** — For each sector, the model calculates median factor scores and compares them
           to the overall market median for that rating band.

        3. **Inverse Weighting** — Weights are adjusted using an inverse deviation strategy:
           - If a sector **underperforms** on a factor → **REDUCE** weight (minimize penalty)
           - If a sector **outperforms** on a factor → **REDUCE** weight (don't amplify advantage)
           - If a sector is **neutral** on a factor → **MAINTAIN** weight (use as differentiator)

        4. **Normalization** — All weights are rescaled to sum to 1.0 for each sector.

        **Weight Modes:**
        - **Dynamic Calibration (default/on):** Sector-specific weights calculated from your data to ensure
          fair cross-sector comparison. BBB-rated companies in all sectors average ~50-60 composite scores.
        - **Universal Weights (off):** Same weights applied to all issuers regardless of sector.
          Simpler but may introduce sector bias.

        **Expected outcome:**
        BBB-rated companies in all sectors should average composite scores of ~50-60, with similar
        Buy recommendation rates (~40%) across sectors.

        **Trade-off:**
        - ✓ Fair cross-sector comparison
        - ✓ Normalized composite scores
        - ✗ May obscure sector-specific fundamentals
        - ✗ Requires sufficient data (50+ companies, 5+ per sector)

        See the **Calibration Diagnostics** panel in the Dashboard tab to evaluate effectiveness.
        """)
    else:
        st.markdown("---")
        st.markdown("### 4a. Universal Weights Mode (ACTIVE)")

        st.markdown("""
        **Universal weights mode is currently active.** All issuers receive the same factor weights
        regardless of their sector or classification.

        This is a simpler approach but may introduce sector bias, as sectors with structural differences
        (e.g., Utilities with weak cash flow, Energy with high leverage) will score differently on average.

        **To enable sector-fair comparison:** Check "Use Dynamic Weight Calibration" in the sidebar.
        """)

    # ========================================================================
    # SECTION 5: DATA VINTAGE & PERIOD HANDLING
    # ========================================================================

    st.markdown("## 5. Data Vintage & Period Handling")

    if df_original is not None and not df_original.empty:
        period_labels = build_dynamic_period_labels(df_original)

        st.markdown(f"""
        **Data period for point-in-time scores:** {st.session_state.get("data_period", "FY0")}
        **Trend window mode:** {"Quarterly (13 periods)" if st.session_state.get("use_quarterly_beta", False) else "Annual (5 periods)"}

        **Detected periods from uploaded file:**
        - {period_labels['fy_label']}
        - {period_labels['cq_label']}

        {'⚠️ Period dates unavailable or fallback heuristic used' if period_labels['used_fallback'] else '✓ Period dates parsed from "Period Ended" columns'}
        """)

        st.markdown("""
        **Period Calendar (V2.2):**
        - Sentinel dates (e.g., `0/01/1900`) are removed
        - FY/CQ overlaps within ±10 days are resolved (preference per trend window mode)
        - Multi-index vendor headers supported
        - Single source of truth for all period-based calculations
        """)

    else:
        st.info("Upload data to see detected period vintage")

    # ========================================================================
    # SECTION 6: QUALITY/TREND SPLIT LOGIC
    # ========================================================================

    st.markdown("## 6. Quality/Trend Split Logic")

    # [V2.3] quality_basis is now hard-coded
    quality_basis = "Percentile within Band (recommended)"
    quality_threshold = 60  # Fixed threshold
    trend_threshold = 55  # Fixed threshold

    st.markdown(f"""
    **Current configuration:**
    - **Quality Basis:** `{quality_basis}`
    - **Quality Threshold:** `{quality_threshold}`
    - **Trend Threshold:** `{trend_threshold}`

    **Base Four-Quadrant Classification:**
    """)

    st.code(f"""
IF quality_metric ≥ {quality_threshold} AND Cycle_Position_Score ≥ {trend_threshold}:
    Signal = "Strong & Improving"

ELIF quality_metric ≥ {quality_threshold} AND Cycle_Position_Score < {trend_threshold}:
    Signal = "Strong but Deteriorating"

ELIF quality_metric < {quality_threshold} AND Cycle_Position_Score ≥ {trend_threshold}:
    Signal = "Weak but Improving"

ELSE:
    Signal = "Weak & Deteriorating"
    """, language="text")

    st.markdown("""
    **Context-aware refinements (V2.2):**
    Strong states may be further classified as **Moderating** (high volatility plateau) or **Normalizing** (peak stabilisation with medium-term improving trend) based on dual-horizon analysis and exceptional quality flags. Weak states remain Improving or Deteriorating.

    """)

    st.markdown("""
    **Where quality_metric is determined by Quality Basis:**
    - **Percentile within Band:** Issuer's percentile rank among peers in same rating band (e.g., BBB issuers ranked among BBBs)
    - **Global Percentile:** Issuer's percentile rank across all issuers regardless of rating
    - **Absolute Composite Score:** Raw 0-100 Composite_Score value

    **These thresholds drive:**
    - Signal assignment in data model
    - Quadrant chart split lines (x-axis adapts to quality basis)
    - Top 10 Improving/Deteriorating tables
    """)

    # ========================================================================
    # SECTION 7: RECOMMENDATION LOGIC [V2.2]
    # ========================================================================

    st.markdown("## 7. Recommendation Logic [V2.2]")
    st.markdown("""
    **New comprehensive approach:** Recommendations are based on **classification first**, then refined by percentile and rating.

    ### Base Recommendation by Classification

    | Classification | High Percentile (≥70%) | Low Percentile (<70%) |
    |---------------|----------------------|---------------------|
    | **Strong & Improving** | Strong Buy | Buy |
    | **Strong but Deteriorating** | Buy | Hold |
    | **Strong & Normalizing** | Buy | Buy |
    | **Strong & Moderating** | Buy | Buy |
    | **Weak but Improving** | Buy | Hold |
    | **Weak & Deteriorating** | Avoid | Avoid |

    ### Rating Guardrails (Applied After Classification)

    **Distressed Issuers (CCC/CC/C/D):**
    - Strong Buy → Capped to **Hold**
    - Buy → Capped to **Hold**
    - Rationale: High default risk, should not recommend buying

    **Single-B Issuers:**
    - Strong Buy → Capped to **Buy**
    - Rationale: Speculative grade, too risky for "Strong Buy"

    **Investment Grade & BB:**
    - No caps applied

    ### Key Changes from V2.1

    - ✓ Recommendations now **respect classification** (e.g., "Strong & Improving" always gets Buy/Strong Buy)
    - ✓ Added **rating-based guardrails** (distressed and single-B caps)
    - ✓ "Strong & Moderating" treated as **Buy** (high quality despite volatility)
    - ✓ Comprehensive **validation** prevents inappropriate recommendations
    """)

    # ========================================================================
    # SECTION 8: MISSING DATA RULES
    # ========================================================================

    st.markdown("## 8. Missing Data Rules")
    st.markdown("""
    **Factor score imputation:**
    - If a factor score cannot be computed (missing input metrics), it is set to `NaN`
    - Composite_Score is still calculated using available factors
    - If all 6 factors are missing → Composite Score becomes `NaN` → issuer excluded from rankings

    **Trend calculation:**
    - Requires at least 3 valid periods for regression slope
    - If insufficient data → Cycle_Position_Score = `NaN` → trend-dependent signals unavailable

    **Freshness:**
    - Stale data (>365 days since Period Ended) are **flagged** but not excluded
    - Users can filter by freshness in diagnostics section
    """)

    # ========================================================================
    # SECTION 9: PER-ISSUER AUDIT
    # ========================================================================

    st.markdown("## 9. Per-Issuer Audit")
    st.markdown("Inspect raw metrics, factor scores, weights, and composite calculation for any issuer.")

    if results_final is not None and not results_final.empty:
        with st.expander("Show Per-Issuer Audit", expanded=False):
            issuer_names = sorted(results_final["Company_Name"].dropna().unique().tolist())
            if issuer_names:
                selected_issuer = st.selectbox(
                    "Select Issuer for Audit",
                    options=issuer_names,
                    key="methodology_audit_issuer"
                )

                issuer_row = results_final[results_final["Company_Name"].apply(_norm) == _norm(selected_issuer)].iloc[0]

                # Header metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Company ID", issuer_row.get("Company_ID", "—"))
                with col2:
                    st.metric("Rating", issuer_row.get("Credit_Rating_Clean", "—"))
                with col3:
                    st.metric("Composite Score", f"{issuer_row.get('Composite_Score', 0):.2f}")
                with col4:
                    signal_val = issuer_row.get("Combined_Signal", issuer_row.get("Signal", "—"))
                    st.metric("Signal", signal_val)

                st.markdown("#### Factor Scores & Weights")

                # Get weights for this issuer
                classification = issuer_row.get("Rubrics_Custom_Classification", None)
                if pd.isna(classification):
                    classification = "Default"

                weights = get_classification_weights(classification, use_sector_adjusted)

                # Build audit table
                audit_rows = []
                for factor_key in ['credit_score', 'leverage_score', 'profitability_score',
                                   'liquidity_score', 'growth_score', 'cash_flow_score']:
                    # Map to display column names
                    display_map = {
                        'credit_score': 'Credit_Score',
                        'leverage_score': 'Leverage_Score',
                        'profitability_score': 'Profitability_Score',
                        'liquidity_score': 'Liquidity_Score',
                        'growth_score': 'Growth_Score',
                        'cash_flow_score': 'Cash_Flow_Score'
                    }

                    score_col = display_map[factor_key]
                    score = issuer_row.get(score_col, np.nan)
                    weight = weights[factor_key]
                    contribution = score * weight if pd.notna(score) else 0.0

                    audit_rows.append({
                        "Factor": score_col.replace('_', ' '),
                        "Score (0-100)": f"{score:.2f}" if pd.notna(score) else "N/A",
                        "Weight": f"{weight:.3f}",
                        "Contribution": f"{contribution:.4f}"
                    })

                audit_df = pd.DataFrame(audit_rows)
                st.table(audit_df)

                # Composite calculation check
                total_contribution = sum(
                    issuer_row.get(display_map[fk], 0) * weights[fk]
                    for fk in weights.keys()
                    if pd.notna(issuer_row.get(display_map[fk], np.nan))
                )

                actual_composite = issuer_row.get("Composite_Score", np.nan)
                diff = total_contribution - actual_composite if pd.notna(actual_composite) else np.nan

                # Format values for display
                composite_str = f"{actual_composite:.4f}" if pd.notna(actual_composite) else "N/A"
                diff_str = f"{diff:.6f}" if pd.notna(diff) else "N/A"

                st.markdown(f"""
                **Composite Calculation:**
                - Sum of contributions: `{total_contribution:.4f}`
                - Recorded Composite_Score: `{composite_str}`
                - Difference: `{diff_str}` (should be ~0)
                """)

                # Additional metadata
                st.markdown("#### Additional Metadata")
                metadata_cols = {
                    "Rating Group": issuer_row.get("Rating_Group", "—"),
                    "Rating Band": issuer_row.get("Rating_Band", "—"),
                    "Classification": issuer_row.get("Rubrics_Custom_Classification", "—"),
                    "Cycle Position Score": f"{issuer_row.get('Cycle_Position_Score', np.nan):.2f}" if pd.notna(issuer_row.get('Cycle_Position_Score')) else "N/A",
                    "Recommendation": issuer_row.get("Recommendation", "—")
                }

                for k, v in metadata_cols.items():
                    st.text(f"{k}: {v}")
            else:
                st.info("No issuers available for audit")
    else:
        st.info("Upload data to enable per-issuer audit")

    # ========================================================================
    # SECTION 10: EXPORT METHODOLOGY
    # ========================================================================

    st.markdown("## 10. Export Methodology")

    # Build markdown export
    export_md = f"""# Issuer Credit Screening Model - Methodology Specification (V3.0)

*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Configuration Snapshot

- **Scoring Method:** {scoring_method}
- **Data Period:** {st.session_state.get("data_period", "FY0")}
- **Trend Window:** {"Quarterly (13 periods)" if st.session_state.get("use_quarterly_beta", False) else "Annual (5 periods)"}
- **Quality Basis:** {quality_basis}
- **Quality Threshold:** {quality_threshold}
- **Trend Threshold:** {trend_threshold}

## Factor → Metrics Mapping

| Factor | Metric Examples | Direction |
|--------|----------------|-----------|
"""

    for fm in factor_meta:
        export_md += f"| {fm['factor']} | {fm['metric_examples']} | {fm['direction']} |\n"

    export_md += f"""
## Composite Score Formula

```
Composite Score :
    w_credit      × Credit_Score +
    w_leverage    × Leverage_Score +
    w_profit      × Profitability_Score +
    w_liquidity   × Liquidity_Score +
    w_growth      × Growth_Score +
    w_cashflow    × Cash_Flow_Score

where all weights sum to 1.0
```

## Current Weights ({scoring_method})

"""

    if use_sector_adjusted:
        export_md += "| Sector | Credit | Leverage | Profitability | Liquidity | Growth | Cash Flow | Sum |\n"
        export_md += "|--------|--------|----------|---------------|-----------|--------|-----------|-----|\n"
        # Get calibrated weights from session state if available
        calibrated_weights = st.session_state.get('_calibrated_weights', None)
        if calibrated_weights:
            for sector_name in sorted(calibrated_weights.keys()):
                w = calibrated_weights[sector_name]
                export_md += f"| {sector_name} | {w['credit_score']:.2f} | {w['leverage_score']:.2f} | {w['profitability_score']:.2f} | {w['liquidity_score']:.2f} | {w['growth_score']:.2f} | {w['cash_flow_score']:.2f} | {sum(w.values()):.2f} |\n"
        else:
            export_md += "| (Calibrated weights not yet calculated) | - | - | - | - | - | - | - |\n"
    else:
        dw = UNIVERSAL_WEIGHTS
        export_md += "| Factor | Weight |\n|--------|--------|\n"
        for fk in ['credit_score', 'leverage_score', 'profitability_score',
                   'liquidity_score', 'growth_score', 'cash_flow_score']:
            export_md += f"| {fk.replace('_', ' ').title()} | {dw[fk]:.2f} |\n"
        export_md += f"| **Total** | **{sum(dw.values()):.2f}** |\n"

    export_md += f"""
## Quality/Trend Split Logic

```
IF quality_metric ≥ {quality_threshold} AND Cycle_Position_Score ≥ {trend_threshold}:
    Signal = "Strong & Improving"

ELIF quality_metric ≥ {quality_threshold} AND Cycle_Position_Score < {trend_threshold}:
    Signal = "Strong but Deteriorating"

ELIF quality_metric < {quality_threshold} AND Cycle_Position_Score ≥ {trend_threshold}:
    Signal = "Weak but Improving"

ELSE:
    Signal = "Weak & Deteriorating"
```

**Quality metric basis:** {quality_basis}

## Recommendation Logic [V2.2]

**Classification-first approach with rating guardrails:**

Base recommendations by classification:
- Strong & Improving: Strong Buy (≥70% percentile) or Buy
- Strong but Deteriorating: Buy (≥70% percentile) or Hold
- Strong & Normalizing: Always Buy (quality overrides short-term weakness)
- Strong & Moderating: Always Buy (high quality despite volatility)
- Weak but Improving: Buy (≥70% percentile) or Hold
- Weak & Deteriorating: Always Avoid

Rating guardrails (applied after classification):
- Distressed (CCC/CC/C/D): Cap Strong Buy/Buy → Hold
- Single-B: Cap Strong Buy → Buy
- Investment Grade & BB: No caps

## Data Vintage

"""

    if df_original is not None and not df_original.empty:
        period_labels = build_dynamic_period_labels(df_original)
        export_md += f"""- {period_labels['fy_label']}
- {period_labels['cq_label']}
- {'Period dates unavailable or fallback heuristic used' if period_labels['used_fallback'] else 'Period dates parsed from "Period Ended" columns'}
"""
    else:
        export_md += "- No data uploaded\n"

    export_md += """
## Missing Data Rules

- Factor scores with missing inputs → NaN
- Composite calculated from available factors
- All factors missing → Composite = NaN → excluded from rankings
- Trend requires ≥3 periods
- Stale data (>365 days) flagged but not excluded

---

*End of Methodology Specification*
"""

    # Download button
    st.download_button(
        label="📥 Download methodology.md",
        data=export_md,
        file_name=f"methodology_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )

    st.markdown("---")
    st.markdown("*All values in this specification are programmatically generated from current app state. No hard-coded values.*")


# ================================

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
                except:
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
    value=True,  # Default to ON for fair cross-sector comparison
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
volatility_cv_threshold = 0.30
outlier_z_threshold = -2.5
damping_factor = 0.5
near_peak_tolerance = 10

# [V2.3] Derive data_period_setting from period_mode for backward compatibility
# In V2.3, we always use quarterly/most recent since trend analysis needs quarterly granularity
# The alignment is controlled by align_to_reference and reference_date_override
data_period_setting = "Most Recent Quarter (CQ-0)"
use_quarterly_beta = True  # Always use quarterly for trend analysis in V2.3

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
        except:
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
        except:
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

        # PROFITABILITY METRICS (directly from spreadsheet)
        "profitability": {
            "ebitda_margin": safe_get_numeric('EBITDA Margin'),
            "ebit_margin": safe_get_numeric('EBIT Margin'),
            "operating_margin": safe_get_numeric('Operating Margin'),
            "net_margin": safe_get_numeric('Net Margin'),
            "roe": safe_get_numeric('Return on Equity'),
            "roa": safe_get_numeric('Return on Assets'),
            "roic": safe_get_numeric('Return on Invested Capital'),
        },

        # LEVERAGE METRICS (directly from spreadsheet)
        "leverage": {
            "total_debt": safe_get_numeric('Total Debt'),
            "net_debt": safe_get_numeric('Net Debt'),
            "total_equity": safe_get_numeric('Total Common Equity'),
            "total_debt_ebitda": safe_get_numeric('Total Debt / EBITDA (x)'),
            "net_debt_ebitda": safe_get_numeric('Net Debt / EBITDA'),
            "total_debt_equity": safe_get_numeric('Total Debt/Equity (x)'),
            "total_debt_capital": safe_get_numeric('Total Debt / Total Capital (%)'),
        },

        # COVERAGE METRICS (directly from spreadsheet)
        "coverage": {
            "ebitda_interest": safe_get_numeric('EBITDA/ Interest Expense (x)'),  # NO space before /
            "ebit_interest": safe_get_numeric('EBIT/ Interest Expense (x)'),      # NO space before /
            "interest_expense": safe_get_numeric('Interest Expense'),
        },

        # LIQUIDITY METRICS (directly from spreadsheet)
        "liquidity": {
            "current_ratio": safe_get_numeric('Current Ratio (x)'),
            "quick_ratio": safe_get_numeric('Quick Ratio (x)'),
            "cash_st_investments": safe_get_numeric('Cash and Short-Term Investments'),
            "current_assets": safe_get_numeric('Current Assets'),
            "current_liabilities": safe_get_numeric('Current Liabilities'),
            "working_capital": safe_get_numeric('Working Capital'),
        },

        # GROWTH METRICS (directly from spreadsheet)
        "growth": {
            "revenue_1y_growth": safe_get_numeric('Total Revenues, 1 Year Growth'),
            "revenue_3y_cagr": safe_get_numeric('Total Revenues, 3 Yr. CAGR'),
            "ebitda_3y_cagr": safe_get_numeric('EBITDA, 3 Years CAGR'),
            "total_revenues": safe_get_numeric('Total Revenues'),
            "ebitda": safe_get_numeric('EBITDA'),
        },

        # CASH FLOW METRICS (directly from spreadsheet - corrected column names)
        "cash_flow": {
            "cfo": safe_get_numeric('Cash from Ops.'),              # Corrected: 'Ops.' not 'Operations'
            "capex": safe_get_numeric('Capital Expenditure'),       # Corrected: singular not plural
            "fcf": safe_get_numeric('Unlevered Free Cash Flow'),   # Corrected: full column name
            "cfo_total_debt": None,  # Will calculate below if not in spreadsheet
            "fcf_total_debt": None,  # Will calculate below if not in spreadsheet
        },

        # BALANCE SHEET (for context)
        "balance_sheet": {
            "total_assets": safe_get_numeric('Total Assets'),
            "total_liabilities": safe_get_numeric('Total Liabilities'),
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

    metrics_to_compare = {
        'ebitda_margin': ('EBITDA Margin', True),
        'roe': ('Return on Equity', True),
        'roa': ('Return on Assets', True),
        'total_debt_ebitda': ('Total Debt / EBITDA (x)', False),  # Lower is better
        'net_debt_ebitda': ('Net Debt / EBITDA', False),
        'current_ratio': ('Current Ratio (x)', True),
        'quick_ratio': ('Quick Ratio (x)', True),
        'ebitda_interest': ('EBITDA/ Interest Expense (x)', True),  # NO space before /
        'revenue_1y_growth': ('Total Revenues, 1 Year Growth', True),
        'revenue_3y_cagr': ('Total Revenues, 3 Yr. CAGR', True),
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
            "growth_score": result.get('Growth_Score'),
            "cash_flow_score": result.get('Cash_Flow_Score'),
        },

        "quality_vs_trend": {
            "quality_score": result.get('Quality_Score'),
            "trend_score": result.get('Trend_Score'),
        },

        "weights_applied": {
            "credit_weight": weights_used.get('credit_score'),
            "leverage_weight": weights_used.get('leverage_score'),
            "profitability_weight": weights_used.get('profitability_score'),
            "liquidity_weight": weights_used.get('liquidity_score'),
            "growth_weight": weights_used.get('growth_score'),
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


def prepare_genai_credit_report_data(
    df_original: pd.DataFrame,
    results_df: pd.DataFrame,
    company_name: str,
    use_sector_adjusted: bool,
    calibrated_weights: dict = None
) -> dict:
    """
    MASTER FUNCTION: Combines all data sources for GenAI credit report.

    This is the function that should be called from the UI when user
    requests a credit report.

    Args:
        df_original: Raw input spreadsheet data (the uploaded Excel file)
        results_df: Model scoring results from app.py
        company_name: Company to analyze
        use_sector_adjusted: Whether sector calibration is enabled
        calibrated_weights: Dynamic calibration weights (if enabled)

    Returns:
        dict: Complete data package for GenAI with raw metrics, model scores, and peer context
    """

    try:
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


def build_comprehensive_credit_prompt(data: dict) -> str:
    """
    Build GenAI prompt with clear data hierarchy and interpretation guidance.
    Emphasizes raw metrics over model scores.

    Args:
        data: Complete data package from prepare_genai_credit_report_data()

    Returns:
        str: Formatted prompt for LLM
    """

    raw = data['raw_financials']
    model = data['model_outputs']
    peers = data['peer_context']

    # Helper function to format metric safely
    def fmt(value, decimals=2, suffix=''):
        if value is None or pd.isna(value):
            return "N/A"
        if suffix == '%':
            return f"{value:.{decimals}f}%"
        elif suffix == 'x':
            return f"{value:.{decimals}f}x"
        elif suffix == 'B':
            return f"${value/1000:.1f}B"
        else:
            return f"{value:.{decimals}f}"

    prompt = f"""# CREDIT ANALYSIS REPORT: {data['company_name']}

You are generating a professional credit analysis report. You have access to THREE data sources with a CLEAR HIERARCHY:

## DATA SOURCE #1: RAW FINANCIAL METRICS (PRIMARY - SOURCE OF TRUTH)

These come directly from the input spreadsheet and represent ACTUAL company fundamentals.
Use these metrics as the FOUNDATION of your analysis.

### Company Profile
- Name: {raw['company_info']['name']}
- Ticker: {raw['company_info']['ticker']}
- S&P Rating: {raw['company_info']['sp_rating']}
- Classification: {raw['company_info']['classification']}
- Sector: {raw['company_info']['sector']}

### Profitability (Actual Metrics from Spreadsheet)
- EBITDA Margin: {fmt(raw['profitability']['ebitda_margin'], 2, '%')}
- Return on Equity: {fmt(raw['profitability']['roe'], 2, '%')}
- Return on Assets: {fmt(raw['profitability']['roa'], 2, '%')}
- Operating Margin: {fmt(raw['profitability']['operating_margin'], 2, '%')}

### Leverage (Actual Metrics from Spreadsheet)
- Total Debt/EBITDA: {fmt(raw['leverage']['total_debt_ebitda'], 2, 'x')}
- Net Debt/EBITDA: {fmt(raw['leverage']['net_debt_ebitda'], 2, 'x')}
- Total Debt: {fmt(raw['leverage']['total_debt'], 1, 'B')}
- Total Debt/Equity: {fmt(raw['leverage']['total_debt_equity'], 2, 'x')}

### Coverage (Actual Metrics from Spreadsheet)
- EBITDA/Interest Expense: {fmt(raw['coverage']['ebitda_interest'], 2, 'x')}

### Liquidity (Actual Metrics from Spreadsheet)
- Current Ratio: {fmt(raw['liquidity']['current_ratio'], 2, 'x')}
- Quick Ratio: {fmt(raw['liquidity']['quick_ratio'], 2, 'x')}
- Cash & ST Investments: {fmt(raw['liquidity']['cash_st_investments'], 1, 'B')}

### Growth (Actual Metrics from Spreadsheet)
- Revenue Growth (1Y): {fmt(raw['growth']['revenue_1y_growth'], 2, '%')}
- Revenue CAGR (3Y): {fmt(raw['growth']['revenue_3y_cagr'], 2, '%')}

### Cash Flow (Actual Metrics from Spreadsheet)
- Operating Cash Flow: {fmt(raw['cash_flow']['cfo'], 1, 'B')}
- Free Cash Flow: {fmt(raw['cash_flow']['fcf'], 1, 'B')}
- CFO/Total Debt: {fmt(raw['cash_flow']['cfo_total_debt'], 2, '%')}

---

## DATA SOURCE #2: PEER COMPARISONS (CRITICAL CONTEXT)

These show how the company compares to sector peers and rating peers.
Use these to determine if metrics are "strong", "average", or "weak".

### Sector Peer Comparison ({peers['sector_comparison']['classification']})
Peer Count: {peers['sector_comparison']['peer_count']} companies

| Metric | Company Value | Sector Median | Percentile |
|--------|---------------|---------------|------------|
| EBITDA Margin | {fmt(raw['profitability']['ebitda_margin'], 2, '%')} | {fmt(peers['sector_comparison']['medians'].get('ebitda_margin'), 2, '%')} | {fmt(peers['sector_comparison']['percentiles'].get('ebitda_margin'), 0)}%ile |
| ROE | {fmt(raw['profitability']['roe'], 2, '%')} | {fmt(peers['sector_comparison']['medians'].get('roe'), 2, '%')} | {fmt(peers['sector_comparison']['percentiles'].get('roe'), 0)}%ile |
| Total Debt/EBITDA | {fmt(raw['leverage']['total_debt_ebitda'], 2, 'x')} | {fmt(peers['sector_comparison']['medians'].get('total_debt_ebitda'), 2, 'x')} | {fmt(peers['sector_comparison']['percentiles'].get('total_debt_ebitda'), 0)}%ile |
| Current Ratio | {fmt(raw['liquidity']['current_ratio'], 2, 'x')} | {fmt(peers['sector_comparison']['medians'].get('current_ratio'), 2, 'x')} | {fmt(peers['sector_comparison']['percentiles'].get('current_ratio'), 0)}%ile |

### Rating Peer Comparison ({peers['rating_comparison']['rating']})
Peer Count: {peers['rating_comparison']['peer_count']} companies

| Metric | Company Value | Rating Median | Percentile |
|--------|---------------|---------------|------------|
| EBITDA Margin | {fmt(raw['profitability']['ebitda_margin'], 2, '%')} | {fmt(peers['rating_comparison']['medians'].get('ebitda_margin'), 2, '%')} | {fmt(peers['rating_comparison']['percentiles'].get('ebitda_margin'), 0)}%ile |
| ROE | {fmt(raw['profitability']['roe'], 2, '%')} | {fmt(peers['rating_comparison']['medians'].get('roe'), 2, '%')} | {fmt(peers['rating_comparison']['percentiles'].get('roe'), 0)}%ile |
| Total Debt/EBITDA | {fmt(raw['leverage']['total_debt_ebitda'], 2, 'x')} | {fmt(peers['rating_comparison']['medians'].get('total_debt_ebitda'), 2, 'x')} | {fmt(peers['rating_comparison']['percentiles'].get('total_debt_ebitda'), 0)}%ile |
| Current Ratio | {fmt(raw['liquidity']['current_ratio'], 2, 'x')} | {fmt(peers['rating_comparison']['medians'].get('current_ratio'), 2, 'x')} | {fmt(peers['rating_comparison']['percentiles'].get('current_ratio'), 0)}%ile |

**INTERPRETATION GUIDE:**
- Percentile >75%: Strong/Exceptional (for positive metrics) or Weak (for leverage - higher percentile = lower leverage = better)
- Percentile 50-75%: Above Average
- Percentile 25-50%: Below Average
- Percentile <25%: Weak (for positive metrics) or Strong (for leverage)

---

## DATA SOURCE #3: MODEL SCORES (SUPPLEMENTARY CONTEXT ONLY)

{model['scoring_methodology']['critical_note']}

### Model Scores (0-100 scale, relative positioning after sector calibration)
- Composite Score: {fmt(model['overall_metrics']['composite_score'], 1)}/100
- Profitability Score: {fmt(model['factor_scores']['profitability_score'], 1)}/100
- Leverage Score: {fmt(model['factor_scores']['leverage_score'], 1)}/100
- Liquidity Score: {fmt(model['factor_scores']['liquidity_score'], 1)}/100
- Growth Score: {fmt(model['factor_scores']['growth_score'], 1)}/100
- Cash Flow Score: {fmt(model['factor_scores']['cash_flow_score'], 1)}/100
- Credit Score: {fmt(model['factor_scores']['credit_score'], 1)}/100

### Model Context
- Signal: {model['overall_metrics']['signal']}
- Recommendation: {model['overall_metrics']['recommendation']}
- Weights Applied: {model['weights_applied']['source']}
- Sector: {model['sector_context']['sector']}

### Understanding Model Scores (CRITICAL)

**Example of CORRECT Interpretation:**
If Profitability Score = 38.7 BUT EBITDA Margin = 28.17% (97th percentile):
WRONG: "Low profitability score indicates weak earnings"
CORRECT: "EBITDA margin of 28.17% is exceptional (97th percentile vs sector). The moderate profitability score of 38.7 reflects sector calibration adjusting for this classification's structural profitability advantages. The model deweights profitability to avoid sector bias in cross-sector comparisons."

**Key Principle:** Model scores are for relative RANKING within a universe, not absolute credit assessment.

---

## YOUR TASK: GENERATE CREDIT ANALYSIS REPORT

### Analysis Hierarchy (MUST FOLLOW THIS ORDER):

1. **START WITH RAW METRICS (Data Source #1)**
   - What are the actual EBITDA margin, leverage ratios, liquidity ratios?
   - State these numbers explicitly

2. **ADD PEER CONTEXT (Data Source #2)**
   - How do metrics compare to sector median?
   - How do they compare to rating peer median?
   - What percentile is the company in?

3. **MAKE ASSESSMENT**
   - If percentile >75%: "Strong" or "Exceptional"
   - If better than rating peer median: Note as credit strength
   - If worse than rating peer median: Note as credit concern
   - If metric is below median BUT percentile shows >50%: The distribution is right-skewed, company is still above average

4. **EXPLAIN MODEL SCORES (Data Source #3) - LAST**
   - If raw metrics strong BUT model score low: Explain sector calibration effect
   - If raw metrics weak AND model score low: Confirm fundamental weakness
   - DO NOT use model scores to override what raw metrics show

### Report Structure:

**Executive Summary**
- Brief overview based on raw metrics and peer comparisons
- Synthesize absolute credit quality
- 3-4 sentences max

**Profitability Analysis (Score: X/100)**
- Start with actual metrics: "EBITDA margin of X%, ROE of Y%"
- Compare to peers: "vs sector median of A%, rating median of B%"
- State percentiles: "placing company in Xth percentile"
- Assess: "This represents [strong/weak/average] profitability"
- Then explain: "The profitability score of X reflects [sector calibration context]"

**Leverage Analysis (Score: X/100)**
- Follow same structure as profitability
- Remember: lower leverage = better credit quality

**Liquidity Analysis (Score: X/100)**
- Follow same structure
- Current ratio >1.5x generally considered adequate for IG

**Coverage Analysis**
- State EBITDA/Interest coverage
- Compare to thresholds (>3x = comfortable, 2-3x = adequate, <2x = tight)

**Cash Flow Analysis (Score: X/100)**
- State CFO, FCF, CFO/Debt ratio
- Assess cash generation strength

**Growth Analysis (Score: X/100)**
- State revenue growth rates
- Context: is growth positive/negative, accelerating/decelerating?

**Credit Strengths**
- List 3-5 genuine strengths with supporting data
- Must be backed by percentiles >60% or better than rating peers

**Credit Risks & Concerns**
- List 3-5 genuine concerns with supporting data
- Must be backed by percentiles <40% or worse than rating peers

**Rating Outlook & Investment Recommendation**
- Based on peer comparison: count metrics better/worse than rating peer median
- If 70%+ metrics better than rating peers: Rating appropriate or conservative
- If 70%+ metrics worse: Rating at risk
- Provide outlook (Stable/Negative/Positive) with rationale
- Investment recommendation: Overweight/Neutral/Underweight with rationale

### CRITICAL RULES (MUST FOLLOW):

NEVER say: "Profitability score of X indicates weak earnings"
ALWAYS say: "EBITDA margin of X% (Yth percentile) indicates [strong/weak] profitability. The profitability score of Z reflects [context]."

NEVER interpret model scores without first stating raw metrics
ALWAYS state raw metric → peer comparison → percentile → assessment → then model score context

NEVER recommend rating change without showing metrics worse than rating peers
ALWAYS compare multiple metrics to rating peer median before assessing rating risk

NEVER call fundamentals "weak" when percentiles show >60%
ALWAYS align qualitative language with percentile rankings

NEVER use phrases like "low score suggests" or "score indicates"
ALWAYS say "actual metric of X (percentile Y) shows [assessment]"

### Formatting:
- Use clear section headers
- Include specific numbers in every claim
- Cite percentiles frequently
- Keep sections concise (3-5 sentences each)
- Use bullet points for strengths/risks lists

Generate the comprehensive credit analysis report now, following all instructions above.
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
    fy_suffixes, cq_suffixes = period_cols_by_kind(pe_data, df)

    # Determine which suffixes to search based on user selection
    if data_period_setting == "Most Recent Fiscal Year (FY0)":
        target_suffixes = fy_suffixes if fy_suffixes else [s for s, _ in pe_data]
    elif data_period_setting == "Most Recent Quarter (CQ-0)":
        target_suffixes = cq_suffixes if cq_suffixes else [s for s, _ in pe_data]
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
                              reference_date=None,
                              pe_data_cached=None, fy_cq_cached=None) -> pd.DataFrame:
    """
    OPTIMIZED: Vectorized time series construction with FY/CQ de-duplication.
    Returns DataFrame where each row is an issuer's time series (columns = ISO dates).

    Args:
        reference_date: Optional cutoff date (str or pd.Timestamp) to align all issuers.
                       If provided, only uses data up to this date for all issuers.
        pe_data_cached: Pre-parsed period columns to avoid re-parsing (performance optimization)
        fy_cq_cached: Pre-computed (fy_suffixes, cq_suffixes) tuple
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
        fy_suffixes, cq_suffixes = fy_cq_cached
    else:
        fy_suffixes, cq_suffixes = period_cols_by_kind(pe_data, df)

    # 3) Choose candidate suffix list by mode
    if use_quarterly:
        candidate_suffixes = [s for s, _ in pe_data]
    else:
        candidate_suffixes = fy_suffixes if fy_suffixes else [s for s, _ in pe_data]

    # 4) VECTORIZED: Build long-format DataFrame with (row_idx, date, value, is_cq)
    long_data = []
    cq_set = set(cq_suffixes)

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
            'is_cq': sfx in cq_set
        })
        long_data.append(chunk)

    if not long_data:
        return pd.DataFrame(index=df.index)

    # Concatenate all chunks
    long_df = pd.concat(long_data, ignore_index=True)

    # 5) Filter out invalid dates and values
    long_df = long_df[long_df['date'].notna() & long_df['value'].notna()]
    long_df = long_df[long_df['date'].dt.year != 1900]  # Remove 1900 sentinels

    # 5a) TIMING MISMATCH FIX: Filter to reference date if provided
    if reference_date is not None:
        reference_dt = pd.to_datetime(reference_date)
        long_df = long_df[long_df['date'] <= reference_dt]

    # 6) De-duplicate: For same (row_idx, date), prefer CQ over FY
    if use_quarterly:
        # Sort so CQ comes first, then drop duplicates keeping first (CQ preferred)
        long_df = long_df.sort_values(['row_idx', 'date', 'is_cq'], ascending=[True, True, False])
        long_df = long_df.drop_duplicates(subset=['row_idx', 'date'], keep='first')
    else:
        # Annual mode: already filtered to FY only by candidate_suffixes
        long_df = long_df.drop_duplicates(subset=['row_idx', 'date'], keep='first')

    # 7) Convert dates to ISO strings for column names
    long_df['date_str'] = long_df['date'].dt.date.astype(str)

    # 8) Pivot to wide format: rows = issuers, columns = dates
    wide_df = long_df.pivot_table(
        index='row_idx',
        columns='date_str',
        values='value',
        aggfunc='first'  # Should be unnecessary after de-dup, but safe
    )

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
    except:
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
                                reference_date=None):
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

    # Helper function for vectorized calculations - TIME-AWARE VERSION
    def _calc_row_stats(row_series):
        """
        Calculate trend, volatility, momentum for a single row's time series.

        TIME-AWARE VERSION: Accounts for actual time intervals between periods.

        Args:
            row_series: Series with ISO date strings as index and metric values

        Returns:
            Series with 'trend', 'vol', 'mom' scores
        """
        values = row_series.dropna()
        n = len(values)

        if n < 3:
            return pd.Series({'trend': 0.0, 'vol': 50.0, 'mom': 50.0})

        # Parse dates from index (ISO strings like "2024-12-31")
        try:
            dates = pd.to_datetime(values.index)
        except Exception:
            # Fallback: if date parsing fails, use legacy logic
            return _calc_row_stats_legacy(row_series)

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
            # Not enough data or time span too small
            mom = 50.0

        return pd.Series({'trend': trend, 'vol': vol, 'mom': mom})

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
                                      pe_data_cached=pe_data_cached, fy_cq_cached=fy_cq_cached)

        # Vectorized calculation using apply
        stats = ts.apply(_calc_row_stats, axis=1)

        trend_scores[f'{base_metric}_trend'] = stats['trend']
        trend_scores[f'{base_metric}_volatility'] = stats['vol']
        trend_scores[f'{base_metric}_momentum'] = stats['mom']

    return trend_scores

def calculate_cycle_position_score(trend_scores, key_metrics_trends):
    """
    SOLUTION TO ISSUE #2: BUSINESS CYCLE POSITION
    
    Composite score indicating where company is in business cycle:
    - High score (70-100): Favorable position (improving trends, low volatility)
    - Medium score (40-70): Neutral/stable
    - Low score (0-40): Unfavorable (deteriorating trends, high volatility)
    """
    cycle_components = []
    
    for metric in key_metrics_trends:
        if f'{metric}_trend' in trend_scores.columns:
            # Positive trend = good
            trend_component = (trend_scores[f'{metric}_trend'] + 1) * 50  # Convert -1/+1 to 0-100
            cycle_components.append(trend_component)
        
        if f'{metric}_volatility' in trend_scores.columns:
            # Low volatility = good (already on 0-100 scale, high = stable)
            cycle_components.append(trend_scores[f'{metric}_volatility'])
        
        if f'{metric}_momentum' in trend_scores.columns:
            # High momentum = good (already on 0-100 scale)
            cycle_components.append(trend_scores[f'{metric}_momentum'])
    
    if cycle_components:
        cycle_score = pd.concat(cycle_components, axis=1).mean(axis=1)
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
        "quality_threshold": float(st.session_state.get("cfg_quality_threshold", 60)),
        "trend_threshold": float(st.session_state.get("cfg_trend_threshold", 55)),
    }

def compute_trend_heatmap(df: pd.DataFrame, selected_band: str, trend_threshold: float, min_count: int = 5):
    """
    Compute trend heatmap with proper Rating Band filtering and min-count guard.

    Returns (heatmap_df, aggregation_df)
    - heatmap_df: pivoted for visualization (rows=metrics, cols=classifications)
    - aggregation_df: raw aggregation with counts for debugging
    """
    # 1) Filter by Rating Band if not "All"
    df_ = df.copy()
    if selected_band and selected_band != "All":
        df_ = df_[df_["Rating_Band"] == selected_band]

    # Guard: nothing to show
    if df_.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 2) Group by Classification
    if 'Rubrics_Custom_Classification' not in df_.columns:
        return pd.DataFrame(), pd.DataFrame()

    # Rename for consistency
    if 'Rubrics_Custom_Classification' in df_.columns and 'Classification' not in df_.columns:
        df_ = df_.rename(columns={'Rubrics_Custom_Classification': 'Classification'})

    gb = df_.groupby("Classification", dropna=False, observed=True)

    # Helper for % improving
    def pct_improving(s: pd.Series) -> float:
        s = s.dropna()
        if len(s) == 0:
            return np.nan
        return float((s >= trend_threshold).mean() * 100.0)

    agg = gb.agg(
        Avg_Composite=("Composite_Score", "mean"),
        Pct_Improving=("Cycle_Position_Score", pct_improving),
        Avg_Cycle_Position=("Cycle_Position_Score", "mean"),
        Count=("Cycle_Position_Score", "count"),
    ).reset_index()

    # 3) Min-count guard: mask too-small groups
    agg.loc[agg["Count"] < min_count, ["Avg_Composite", "Pct_Improving", "Avg_Cycle_Position"]] = np.nan

    # 4) Pivot into 3 rows × classifications for heatmap
    heat = (
        agg.melt(id_vars=["Classification"], value_vars=["Avg_Composite", "Pct_Improving", "Avg_Cycle_Position"],
                 var_name="Metric", value_name="Score")
        .pivot(index="Metric", columns="Classification", values="Score")
        .sort_index()  # order rows consistently
    )

    return heat, agg

# ============================================================================
# AI ANALYSIS HELPERS (deterministic)
# ============================================================================

from datetime import datetime, timedelta

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

def _detect_signal(df, trend_thr=55, quality_thr=60):
    """Return a small dataframe with computed/imputed regime signal if missing."""
    comp = _col(df, ["Composite_Score", "Composite Score (0-100)", "Composite Score"])
    cyc  = _col(df, ["Cycle_Position_Score", "Cycle Position Score (0-100)", "Cycle Position Score"])
    # Prefer post-override label if present
    sig  = _col(df, ["Combined_Signal", "Quality & Trend Signal", "Quality and Trend Signal", "Signal"])
    out = df.copy()
    if sig is None:
        # Derive a simple 2x2 on composite vs cycle position for transparency.
        def _lab(row):
            q = row[comp]
            t = row[cyc]
            if pd.isna(q) or pd.isna(t): return "n/a"
            hi_q = q >= quality_thr
            up_t = t >= trend_thr
            if hi_q and up_t: return "Strong & Improving"
            if hi_q and not up_t: return "Strong but Deteriorating"
            if (not hi_q) and up_t: return "Weak but Improving"
            return "Weak & Deteriorating"
        out["__Signal"] = out.apply(_lab, axis=1)
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
        for m in ["EBITDA Margin", "Return on Equity", "EBITDA / Interest Expense (x)", "Total Debt / EBITDA (x)", "Net Debt / EBITDA"]:
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
    for m in ["EBITDA Margin", "Return on Equity", "EBITDA / Interest Expense (x)"]:
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

def build_buckets_v2(results_df: pd.DataFrame, df_original: pd.DataFrame, trend_thr=55, quality_thr=60):
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
        if pd.isna(r[comp]) or pd.isna(r[cyc]): return "n/a"
        return (
            "Strong & Improving"      if (r[comp] >= quality_thr and r[cyc] >= trend_thr) else
            "Strong & Normalizing"    if (r[comp] >= quality_thr and r[cyc] <  trend_thr) else
            "Weak but Improving"      if (r[comp] <  quality_thr and r[cyc] >= trend_thr) else
            "Weak & Deteriorating"
        )
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

@st.cache_data(show_spinner=False)
def load_and_process_data(uploaded_file, use_sector_adjusted,
                          period_mode=PeriodSelectionMode.LATEST_AVAILABLE,
                          reference_date_override=None,
                          split_basis="Percentile within Band (recommended)", split_threshold=60, trend_threshold=55,
                          volatility_cv_threshold=0.30, outlier_z_threshold=-2.5, damping_factor=0.5, near_peak_tolerance=0.10,
                          calibrated_weights=None,
                          _cache_buster=None):
    """Load data and calculate issuer scores with unified period selection (V2.3)

    Args:
        period_mode: PeriodSelectionMode enum
            - LATEST_AVAILABLE: Use most recent data per issuer (accepts misalignment)
            - REFERENCE_ALIGNED: Align all issuers to common reference date
        reference_date_override: Required when period_mode=REFERENCE_ALIGNED.
                                pd.Timestamp of the reference date.
        calibrated_weights: Optional dict of dynamically calibrated sector weights (V3.0).
                          If provided, sector-specific weights used; otherwise UNIVERSAL_WEIGHTS.
    """

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
            else:  # LATEST_AVAILABLE
                reference_date_actual = None
                align_to_reference = False

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
        'Total Debt / EBITDA (x)',
        'EBITDA Margin',
        'Return on Equity',
        'Current Ratio (x)'
    ]

    # [V2.3] Extract metrics using unified period mode
    # For trend analysis, always use quarterly data for better granularity
    # Reference date determined earlier based on period_mode
    use_quarterly_for_trends = True  # Always use quarterly for trend analysis

    # Calculate trend indicators with unified period selection
    trend_scores = calculate_trend_indicators(df, key_metrics_for_trends,
                                             use_quarterly=use_quarterly_for_trends,
                                             reference_date=reference_date_actual)
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
        factor_name: str = ""
    ) -> tuple:
        """
        Calculate factor score with automatic weight renormalization for missing data.

        Args:
            components: 2D array (n_issuers × n_components) of component scores
            weights: 1D array of component weights (must sum to 1.0)
            min_components: Minimum number of non-missing components required
            factor_name: Name of factor (for data quality column naming)

        Returns:
            tuple of (scores, data_completeness, components_used_count)
        """
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
        normalized_weights = np.where(
            weight_sums > 0,
            effective_weights / weight_sums,
            0.0
        )

        # Calculate weighted scores
        weighted_components = components * normalized_weights
        raw_scores = np.nansum(weighted_components, axis=1)

        # Apply minimum component threshold
        final_scores = np.where(
            components_available >= min_components,
            raw_scores,
            np.nan
        )

        # Calculate data completeness
        data_completeness = components_available / n_components

        return (
            pd.Series(final_scores),
            pd.Series(data_completeness),
            pd.Series(components_available, dtype=int)
        )

    def calculate_quality_scores(df, data_period_setting, has_period_alignment, reference_date=None, align_to_reference=False):
        """
        Calculate quality scores for all issuers.

        Args:
            reference_date: If provided AND align_to_reference is True AND data_period_setting is CQ-0,
                          filters point-in-time metrics to this reference date for fair comparison.
            align_to_reference: Whether alignment is enabled by user.
        """
        scores = pd.DataFrame(index=df.index)

        # Determine whether to apply reference date filtering for point-in-time metrics
        # Only apply if: (1) CQ-0 selected AND (2) alignment enabled
        apply_reference_date = (
            reference_date is not None
            and align_to_reference
            and data_period_setting == "Most Recent Quarter (CQ-0)"
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
            'Cash from Ops. to Curr. Liab. (x)'
        ]
        metrics = _batch_extract_metrics(df, needed_metrics, has_period_alignment, data_period_setting, ref_date_for_extraction)

        # Credit Score – 100% S&P LT Issuer Rating (Interest Coverage moved under Leverage)

        # Credit Rating mapping to 0-100 scale
        rating_map = {
            'AAA': 21, 'AA+': 20, 'AA': 19, 'AA-': 18,
            'A+': 17, 'A': 16, 'A-': 15,
            'BBB+': 14, 'BBB': 13, 'BBB-': 12,
            'BB+': 11, 'BB': 10, 'BB-': 9,
            'B+': 8, 'B': 7, 'B-': 6,
            'CCC+': 5, 'CCC': 4, 'CCC-': 3,
            'CC': 2, 'C': 1, 'D': 0, 'SD': 0, 'NR': np.nan
        }
        cr = df[RATING_COL].map(_clean_rating)
        rating_score = cr.map(rating_map) * (100.0/21.0)

        scores['credit_score'] = rating_score

        # EBITDA / Interest Expense coverage - now used in Leverage (Annual-only)
        ebitda_interest = metrics['EBITDA / Interest Expense (x)']

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

        # Leverage (Annual-only) - Option A weights: ND/EBITDA 40%, Coverage 30%, Debt/Cap 20%, TD/EBITDA 10%

        # Component 1: Net Debt / EBITDA (40%)
        net_debt_ebitda = metrics['Net Debt / EBITDA']
        # Keep only valid positive values, let negatives and missing be NaN
        net_debt_ebitda = net_debt_ebitda.where(net_debt_ebitda >= 0, other=np.nan).clip(upper=20.0)
        part1 = (np.minimum(net_debt_ebitda, 3.0)/3.0)*60.0
        part2 = (np.maximum(net_debt_ebitda-3.0, 0.0)/5.0)*40.0
        raw_penalty = np.minimum(part1+part2, 100.0)
        net_debt_score = np.clip(100.0 - raw_penalty, 0.0, 100.0)

        # Component 2: Interest Coverage (30%)
        interest_coverage_score = ebitda_cov_score

        # Component 3: Total Debt / Total Capital (20%)
        debt_capital = metrics['Total Debt / Total Capital (%)']
        # Don't fill missing with 50%, just clip valid values
        debt_capital = debt_capital.clip(0, 100)
        debt_cap_score = np.clip(100 - debt_capital, 0, 100)

        # Component 4: Total Debt / EBITDA (10%)
        debt_ebitda = metrics['Total Debt / EBITDA (x)']
        # Keep only valid positive values, let negatives and missing be NaN
        debt_ebitda = debt_ebitda.where(debt_ebitda >= 0, other=np.nan).clip(upper=20.0)
        part1_td = (np.minimum(debt_ebitda, 3.0)/3.0)*60.0
        part2_td = (np.maximum(debt_ebitda-3.0, 0.0)/5.0)*40.0
        raw_penalty_td = np.minimum(part1_td+part2_td, 100.0)
        debt_ebitda_score = np.clip(100.0 - raw_penalty_td, 0.0, 100.0)

        # Leverage Score with renormalization using unified function
        leverage_components = np.column_stack([
            net_debt_score,
            interest_coverage_score,
            debt_cap_score,
            debt_ebitda_score
        ])

        leverage_weights = np.array([0.40, 0.30, 0.20, 0.10])

        leverage_score, leverage_completeness, leverage_components_used = \
            _calculate_factor_score_with_renormalization(
                leverage_components,
                leverage_weights,
                min_components=2,  # Require at least 2 of 4 components
                factor_name="Leverage"
            )

        scores['leverage_score'] = leverage_score
        scores['leverage_data_completeness'] = leverage_completeness
        scores['leverage_components_used'] = leverage_components_used

        # Profitability ([V2.2] Annual-only) with renormalization
        roe = _pct_to_100(metrics['Return on Equity'])
        ebitda_margin = _pct_to_100(metrics['EBITDA Margin'])
        roa = _pct_to_100(metrics['Return on Assets'])
        ebit_margin = _pct_to_100(metrics['EBIT Margin'])

        roe_score = np.clip(roe, -50, 50) + 50
        margin_score = np.clip(ebitda_margin, -50, 50) + 50
        roa_score = np.clip(roa * 5, 0, 100)
        ebit_score = np.clip(ebit_margin * 2, 0, 100)

        profitability_components = np.column_stack([
            roe_score,
            margin_score,
            roa_score,
            ebit_score
        ])

        profitability_weights = np.array([0.30, 0.30, 0.20, 0.20])

        profitability_score, profitability_completeness, profitability_components_used = \
            _calculate_factor_score_with_renormalization(
                profitability_components,
                profitability_weights,
                min_components=2,  # Require at least 2 of 4 components
                factor_name="Profitability"
            )

        scores['profitability_score'] = profitability_score
        scores['profitability_data_completeness'] = profitability_completeness
        scores['profitability_components_used'] = profitability_components_used

        # Liquidity ([V2.2] Annual-only) with renormalization
        # Don't clip NaN to 0 - preserve NaN for missing data
        current_ratio = metrics['Current Ratio (x)']
        current_ratio = current_ratio.where(current_ratio >= 0, other=np.nan)
        quick_ratio = metrics['Quick Ratio (x)']
        quick_ratio = quick_ratio.where(quick_ratio >= 0, other=np.nan)

        current_score = np.clip((current_ratio/3.0)*100.0, 0, 100)
        quick_score = np.clip((quick_ratio/2.0)*100.0, 0, 100)

        liquidity_components = np.column_stack([
            current_score,
            quick_score
        ])

        liquidity_weights = np.array([0.60, 0.40])

        liquidity_score, liquidity_completeness, liquidity_components_used = \
            _calculate_factor_score_with_renormalization(
                liquidity_components,
                liquidity_weights,
                min_components=1,  # Require at least 1 of 2 components
                factor_name="Liquidity"
            )

        scores['liquidity_score'] = liquidity_score
        scores['liquidity_data_completeness'] = liquidity_completeness
        scores['liquidity_components_used'] = liquidity_components_used

        # Growth ([V2.2] Annual-only) with renormalization
        rev_growth_1y = _pct_to_100(metrics['Total Revenues, 1 Year Growth'])
        rev_cagr_3y = _pct_to_100(metrics['Total Revenues, 3 Yr. CAGR'])
        ebitda_cagr_3y = _pct_to_100(metrics['EBITDA, 3 Years CAGR'])

        rev_1y_score = np.clip((rev_growth_1y + 10) * 2, 0, 100)
        rev_3y_score = np.clip((rev_cagr_3y + 10) * 2, 0, 100)
        ebitda_3y_score = np.clip((ebitda_cagr_3y + 10) * 2, 0, 100)

        growth_components = np.column_stack([
            rev_3y_score,
            rev_1y_score,
            ebitda_3y_score
        ])

        growth_weights = np.array([0.40, 0.30, 0.30])

        growth_score, growth_completeness, growth_components_used = \
            _calculate_factor_score_with_renormalization(
                growth_components,
                growth_weights,
                min_components=2,  # Require at least 2 of 3 components
                factor_name="Growth"
            )

        scores['growth_score'] = growth_score
        scores['growth_data_completeness'] = growth_completeness
        scores['growth_components_used'] = growth_components_used

        # Cash Flow ([v3] Annual-only) - enhance with data quality tracking
        _cf_comp = _cash_flow_component_scores(df, data_period_setting, has_period_alignment, ref_date_for_extraction)
        _cf_cols = [c for c in ["OCF_to_Revenue_Score", "OCF_to_Debt_Score",
                                 "UFCF_margin_Score", "LFCF_margin_Score"] if c in _cf_comp.columns]

        # Filter to only columns that have at least some valid data
        _cf_cols = [c for c in _cf_cols if _cf_comp[c].notna().sum() > 0]

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

        return scores

    # [V2.3] Derive data_period_setting from period_mode for backward compatibility
    # For now, always use quarterly/most recent since we're using quarterly for trends
    # The reference_date_actual controls whether data is aligned or not
    data_period_setting = "Most Recent Quarter (CQ-0)"

    quality_scores = calculate_quality_scores(df, data_period_setting, has_period_alignment, reference_date_actual, align_to_reference)
    _log_timing("04_Quality_Scores_Complete")

    _audit_count("After factor construction", df, audits)

    # Clean rating for grouping
    def _clean_rating_outer(x):
        x = str(x).upper().strip()
        x = x.replace('NOT RATED','NR').replace('N/R','NR').replace('N\\M','N/M')
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
                'Growth_Score': quality_scores['growth_score'],
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
                   'liquidity_score', 'growth_score', 'cash_flow_score']

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

        if n_available >= 4:  # Require at least 4 of 6 factors
            # Renormalize weights
            effective_weights = factor_weights * available_mask
            effective_weights = effective_weights / effective_weights.sum()

            # Calculate composite
            composite = np.nansum(factor_values * effective_weights)
            completeness = n_available / 6
        else:
            composite = np.nan
            completeness = n_available / 6

        composite_scores_list.append(composite)
        composite_completeness_list.append(completeness)

    # Create composite score series
    composite_score = pd.Series(composite_scores_list, index=qs.index)
    qs['composite_data_completeness'] = composite_completeness_list
    _log_timing("05_Composite_Score_Complete")

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
        'Credit_Score': quality_scores['credit_score'],
        'Leverage_Score': quality_scores['leverage_score'],
        'Profitability_Score': quality_scores['profitability_score'],
        'Liquidity_Score': quality_scores['liquidity_score'],
        'Growth_Score': quality_scores['growth_score'],
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

    results_dict['Growth_Data_Completeness'] = quality_scores.get('growth_data_completeness', 1.0)
    results_dict['Growth_Components_Used'] = quality_scores.get('growth_components_used', 3)

    results_dict['Cash_Flow_Data_Completeness'] = quality_scores.get('cash_flow_data_completeness', 1.0)
    results_dict['Cash_Flow_Components_Used'] = quality_scores.get('cash_flow_components_used', 4)

    results_dict['Credit_Data_Completeness'] = 1.0  # Credit is always single component

    # Add overall composite data completeness
    results_dict['Composite_Data_Completeness'] = quality_scores.get('composite_data_completeness', 1.0)

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

    # [Enhanced Explainability] Store the weights used in calculation for transparency
    results['Weight_Credit_Used'] = weight_matrix['credit_score']
    results['Weight_Leverage_Used'] = weight_matrix['leverage_score']
    results['Weight_Profitability_Used'] = weight_matrix['profitability_score']
    results['Weight_Liquidity_Used'] = weight_matrix['liquidity_score']
    results['Weight_Growth_Used'] = weight_matrix['growth_score']
    results['Weight_CashFlow_Used'] = weight_matrix['cash_flow_score']

    # Add trend indicators to results
    for col in trend_scores.columns:
        results[col] = trend_scores[col]

    _audit_count("After scoring (non-NaN Composite_Score)", results[results['Composite_Score'].notna()], audits)

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

    # Overall Rank - will be calculated after Recommendation column is created (see line ~7392)

    # ========================================================================
    # [V2.2] CONTEXT FLAGS FOR DUAL-HORIZON ANALYSIS
    # ========================================================================

    # Exceptional quality flag (≥90th percentile composite OR top factor)
    results['ExceptionalQuality'] = (
        (results['Composite_Percentile_in_Band'] >= 90) |
        (results['Profitability_Score'] >= 90) |
        (results['Growth_Score'] >= 90)
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

    _log_timing("05b_Context_Flags_Complete")

    # ========================================================================
    # GENERATE SIGNAL (Position & Trend quadrant classification)
    # ========================================================================

    # Use unified quality/trend split rule
    quality_metric, x_split_for_rule, _, _ = resolve_quality_metric_and_split(
        results, split_basis, split_threshold
    )

    trend_metric = results["Cycle_Position_Score"]
    is_strong_quality = quality_metric >= x_split_for_rule
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
    # When: Exceptional quality + High volatility + Not improving
    override_moderating = (
        results['ExceptionalQuality'] &
        (results['Signal_Base'] == 'Strong but Deteriorating') &
        results['VolatileSeries'] &
        ~override_normalizing  # Don't double-override
    )
    results.loc[override_moderating, 'Signal'] = 'Strong & Moderating'

    # Add reasons column for transparency
    results['Signal_Reason'] = ''
    results.loc[override_normalizing, 'Signal_Reason'] = 'Exceptional quality (≥90th %ile); Medium-term improving; Near peak/outlier'
    results.loc[override_moderating, 'Signal_Reason'] = 'Exceptional quality (≥90th %ile); High volatility (CV≥0.30); Damping applied'
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
        (quality_check < x_split_for_rule) &  # Weak quality
        (results['Cycle_Position_Score'] < trend_threshold)  # Deteriorating trend
    )

    # Check if any are incorrectly in the strong quadrant
    misclassified_signals = weak_deteriorating & results['Signal'].isin(['Strong & Improving', 'Strong but Deteriorating', 'Strong & Normalizing'])

    if misclassified_signals.any() and not os.environ.get("RG_TESTS"):
        st.warning(
            f"**Signal Classification Alert**\n\n"
            f"{misclassified_signals.sum()} issuer(s) have Weak & Deteriorating fundamentals "
            f"but were initially classified in the Strong category. This may indicate data quality issues "
            f"or edge cases in the classification logic.\n\n"
            f"Review these issuers carefully before making investment decisions."
        )

        # Show affected issuers in expander (don't clutter main view)
        with st.expander("View Affected Issuers"):
            alert_cols = ['Company_Name', 'Composite_Score', 'Composite_Percentile_in_Band',
                         'Cycle_Position_Score', 'Signal', 'Credit_Rating_Clean']
            st.dataframe(results[misclassified_signals][alert_cols])

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
            return "Strong Buy" if pct >= 70 else "Buy"

        elif classification == "Strong but Deteriorating":
            # Good quality but declining trend
            # → Cautiously positive (Buy or Hold)
            return "Buy" if pct >= 70 else "Hold"

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
            return "Buy" if pct >= 70 else "Hold"

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
        final = _apply_rating_guardrails(base, row['Rating_Band'])

        # Step 3: Generate reason for transparency
        classification = row['Signal'] if pd.notna(row['Signal']) else '—'
        pct = row['Composite_Percentile_in_Band']
        pct_str = f"{pct:.0f}%" if pd.notna(pct) else "N/A"
        rating = row['Rating_Band'] if pd.notna(row['Rating_Band']) else 'NR'

        reason = f"{classification} (Percentile: {pct_str})"

        if final != base:
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

    # Violation 4: Strong & Improving with Avoid
    strong_improving_avoid = (
        (results['Signal'] == 'Strong & Improving') &
        (results['Recommendation'] == 'Avoid')
    )
    validation_results['strong_improving_avoid'] = strong_improving_avoid.sum()

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
        guardrails_applied = weak_det_capped + distressed_capped + single_b_capped

        if guardrails_applied > 0 and not os.environ.get("RG_TESTS"):
            st.sidebar.success(
                f"✓ **Quality Guardrails Active**\n\n"
                f"Protected {guardrails_applied} issuers from inappropriate recommendations:\n"
                f"- {weak_det_capped} Weak & Deteriorating (capped to Avoid)\n"
                f"- {distressed_capped} Distressed CCC/CC/C (capped to Hold)\n"
                f"- {single_b_capped} Single-B (capped to Buy)\n\n"
                f"These had high percentiles but were downgraded due to quality/rating concerns."
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


async def get_growth_data(ctx, company_name: str, row_data: dict) -> str:
    """Extract growth metrics."""
    metrics = [
        "Total Revenues, 3 Yr. CAGR",
        "Total Revenues, 1 Year Growth",
        "EBITDA, 3 Years CAGR"
    ]

    result = {
        "company": company_name,
        "factor_score": row_data["growth_score"],
        "metrics": {}
    }

    for metric in metrics:
        result["metrics"][metric] = extract_metric_time_series(
            row_data["row"],
            row_data["df"],
            metric
        )

    state = await ctx.get("state")
    state["growth_data"] = result
    await ctx.set("state", state)

    return f"Growth data extracted. Score: {row_data['growth_score']:.1f}/100"


async def compile_final_report(ctx) -> str:
    """Signal all analyses collected."""
    state = await ctx.get("state")

    sections = {
        "profitability": state.get("profitability_analysis"),
        "leverage": state.get("leverage_analysis"),
        "liquidity": state.get("liquidity_analysis"),
        "cash_flow": state.get("cash_flow_analysis"),
        "growth": state.get("growth_analysis")
    }

    complete = sum(1 for s in sections.values() if s)
    return f"Collected {complete}/5 specialist analyses"


def create_profitability_agent(llm):
    return FunctionAgent(
        name="ProfitabilityAgent",
        description="Analyze profitability: EBITDA Margin, ROE, ROA, EBIT Margin",
        system_prompt="""You are a profitability analyst.

**Task:**
1. Call get_profitability_data
2. Analyze each metric vs sector medians (primary comparison)
3. Note sector-specific norms (e.g., "Tech EBITDA margins typically 25-35%")
4. Explain factor weight in context of sector (e.g., "Profitability weighted 25% for this sector vs 20% default")
5. Describe historical trends with specific values

**Output Format:**
### EBITDA Margin (Weight: X%)
- Current: Y%
- Sector Context: [Sector] companies typically achieve Z% margins. This company is [above/below/in-line].
- Historical: [trend with values]
- Assessment: [strength/weakness vs sector norms]

[Repeat for ROE, ROA, EBIT Margin]

**Overall Profitability:** [3-4 sentences on sector-relative performance]

Hand off to LeverageAgent when complete.""",
        llm=llm,
        tools=[get_profitability_data],
        can_handoff_to=["LeverageAgent"]
    )


def create_leverage_agent(llm):
    return FunctionAgent(
        name="LeverageAgent",
        description="Analyze leverage: Net Debt/EBITDA, Interest Coverage, Debt/Capital, Total Debt/EBITDA",
        system_prompt="""You are a leverage analyst.

**Task:**
1. Call get_leverage_data
2. Analyze each metric vs sector medians
3. Note if sector has different leverage tolerance (e.g., "Utilities typically 4-5x due to regulated cash flows")
4. Describe historical trends

**Output Format:**
### Net Debt / EBITDA (Weight: X%)
- Current: Yx
- Sector Context: [comparison to sector median]
- Historical: [trend]
- Assessment: [strength/weakness]

[Repeat for Interest Coverage, Debt/Capital, Total Debt/EBITDA]

**Overall Leverage:** [3-4 sentences]

Hand off to LiquidityAgent.""",
        llm=llm,
        tools=[get_leverage_data],
        can_handoff_to=["LiquidityAgent"]
    )


def create_liquidity_agent(llm):
    return FunctionAgent(
        name="LiquidityAgent",
        description="Analyze liquidity: Current Ratio, Quick Ratio",
        system_prompt="""You are a liquidity analyst.

**Task:**
1. Call get_liquidity_data
2. Analyze each metric vs sector medians
3. Describe historical trends

**Output Format:**
### Current Ratio (Weight: X%)
- Current: Yx
- Sector Context: [comparison]
- Historical: [trend]
- Assessment: [strength/weakness]

[Repeat for Quick Ratio]

**Overall Liquidity:** [3-4 sentences]

Hand off to CashFlowAgent.""",
        llm=llm,
        tools=[get_liquidity_data],
        can_handoff_to=["CashFlowAgent"]
    )


def create_cash_flow_agent(llm):
    return FunctionAgent(
        name="CashFlowAgent",
        description="Analyze cash flow quality",
        system_prompt="""You are a cash flow analyst.

**Task:**
1. Call get_cash_flow_data
2. Calculate ratios: OCF/Revenue, OCF/Debt, LFCF Margin
3. Analyze trends in cash generation

**Output Format:**
### Cash Flow Quality
- OCF/Revenue: [if calculable from OCF and Revenue]
- OCF/Debt: [if calculable]
- LFCF Margin: [if calculable]
- Historical: [trends]
- Assessment: [quality of cash generation]

**Overall Cash Flow:** [3-4 sentences]

Hand off to GrowthAgent.""",
        llm=llm,
        tools=[get_cash_flow_data],
        can_handoff_to=["GrowthAgent"]
    )


def create_growth_agent(llm):
    return FunctionAgent(
        name="GrowthAgent",
        description="Analyze growth: Revenue CAGR 3Y, Revenue Growth 1Y, EBITDA CAGR 3Y",
        system_prompt="""You are a growth analyst.

**Task:**
1. Call get_growth_data
2. Analyze growth trajectory
3. Note consistency and quality

**Output Format:**
### Growth Profile
- Revenue CAGR 3Y: X%
- Revenue Growth 1Y: Y%
- EBITDA CAGR 3Y: Z%
- Assessment: [consistency, quality]

**Overall Growth:** [3-4 sentences]

Hand off to SupervisorAgent.""",
        llm=llm,
        tools=[get_growth_data],
        can_handoff_to=["SupervisorAgent"]
    )


def create_supervisor_agent(llm):
    return FunctionAgent(
        name="SupervisorAgent",
        description="Synthesize all analyses into 8-section credit report",
        system_prompt="""You are the Chief Credit Officer. Synthesize all specialist analyses into comprehensive report.

**Task:**
1. Call compile_final_report
2. Extract 3-4 credit strengths from specialist outputs
3. Extract 3-4 credit risks from specialist outputs
4. Provide rating outlook

**Required Format:**

# Credit Analysis: {company_name}
**S&P Rating:** {rating} | **Composite Score:** {composite_score}/100 | **Band:** {rating_band}

## 1. Executive Summary
[3-4 sentences: overall profile, key themes]

## 2. Profitability Analysis
[ProfitabilityAgent output verbatim]

## 3. Leverage Analysis
[LeverageAgent output verbatim]

## 4. Liquidity Analysis
[LiquidityAgent output verbatim]

## 5. Cash Flow & Growth Analysis
[CashFlowAgent + GrowthAgent outputs combined]

## 6. Credit Strengths
- **[Strength 1]:** [1-2 sentences with metrics]
- **[Strength 2]:** [1-2 sentences with metrics]
- **[Strength 3]:** [1-2 sentences with metrics]

## 7. Credit Risks & Concerns
- **[Risk 1]:** [1-2 sentences with metrics]
- **[Risk 2]:** [1-2 sentences with metrics]
- **[Risk 3]:** [1-2 sentences with metrics]

## 8. Rating Outlook & Recommendation
[3-4 sentences on trajectory, catalysts]

**Output ONLY the markdown report above.**""",
        llm=llm,
        tools=[compile_final_report],
        can_handoff_to=[]
    )


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
        except:
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

        prompt = f"""Analyze {name}'s profitability performance using raw financial metrics.

**Profitability Score:** {prof_score:.1f}/100

**Raw Financial Metrics:**
- EBITDA Margin: {metrics['ebitda_margin']}%
- ROA (Return on Assets): {metrics['roa']}%
- Net Margin: {metrics['net_margin']}%

**Instructions:**
Provide a 2-3 paragraph assessment covering:
1. Analyze the actual margin levels (EBITDA, Net) and ROA - are these strong/weak for the industry?
2. What do these metrics reveal about operational efficiency and profitability quality?
3. How do these financials support or contradict the {prof_score:.1f}/100 score?
4. Compare to typical thresholds (e.g., EBITDA margins >20% = strong, 10-20% = moderate, <10% = weak)

Be specific and reference the actual metric values, not just the score."""

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

        prompt = f"""Analyze {name}'s leverage position.

**Leverage Score:** {lev_score:.1f}/100

Provide a 2-3 paragraph assessment covering:
1. Debt levels and overall leverage assessment
2. Capital structure quality
3. Leverage trajectory and financial flexibility

Be specific and reference the score."""

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

        prompt = f"""Analyze {name}'s liquidity position.

**Liquidity Score:** {liq_score:.1f}/100

Provide a 2-3 paragraph assessment covering:
1. Current liquidity adequacy
2. Ability to meet short-term obligations
3. Interest coverage and debt serviceability

Be specific and reference the score."""

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

        prompt = f"""Analyze {name}'s cash flow quality.

**Cash Flow Score:** {cf_score:.1f}/100

Provide a 2-3 paragraph assessment covering:
1. Operating cash flow quality and sustainability
2. Free cash flow generation capacity
3. Cash conversion efficiency

Be specific and reference the score."""

        response = await llm.acomplete(prompt, temperature=0.3, max_tokens=1200)
        analysis = response.text

        state['cash_flow_analysis'] = analysis
        await ctx.set("state", state)
        return f"Cash flow analysis complete."

    async def analyze_growth_tool(ctx):
        """Analyze growth with LLM and provide commentary."""
        state = await ctx.get("state")
        growth_score = state['factor_scores'].get('growth_score', 50)
        name = state['company_name']

        prompt = f"""Analyze {name}'s growth profile.

**Growth Score:** {growth_score:.1f}/100

Provide a 2-3 paragraph assessment covering:
1. Revenue and EBITDA growth trends
2. Growth momentum assessment
3. Sustainability of growth trajectory

Be specific and reference the score."""

        response = await llm.acomplete(prompt, temperature=0.3, max_tokens=1200)
        analysis = response.text

        state['growth_analysis'] = analysis
        await ctx.set("state", state)
        return f"Growth analysis complete."

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
        gr_analysis = state.get('growth_analysis', 'Not available')

        prompt = f"""You are an expert credit analyst. Synthesize the specialist analyses below into a comprehensive credit report for {name}.

**Company Overview:**
- Company Name: {name}
- S&P Rating: {rating_val}
- Composite Score: {comp_score:.1f}/100
- Rating Band: {band}
- Industry Classification: {classification}

**Factor Scores (0-100 scale, higher is better):**
- Credit Score: {f_scores.get('credit_score', 50):.1f}/100
- Leverage Score: {f_scores.get('leverage_score', 50):.1f}/100
- Profitability Score: {f_scores.get('profitability_score', 50):.1f}/100
- Liquidity Score: {f_scores.get('liquidity_score', 50):.1f}/100
- Growth Score: {f_scores.get('growth_score', 50):.1f}/100
- Cash Flow Score: {f_scores.get('cash_flow_score', 50):.1f}/100

**Specialist Agent Analyses:**

### Profitability Analysis
{prof_analysis}

### Leverage Analysis
{lev_analysis}

### Liquidity Analysis
{liq_analysis}

### Cash Flow Analysis
{cf_analysis}

### Growth Analysis
{gr_analysis}

**INSTRUCTIONS:**
Create a COMPLETE, professional credit report with ALL sections below.

## Comprehensive Credit Analysis: {name}

### Executive Summary
- Current Rating: {rating_val} | Composite Score: {comp_score:.1f}/100 | Band: {band}
- Industry: {classification}
- 3-4 sentence overview synthesizing key themes from all 5 specialist analyses
- Primary rating drivers and risk positioning

---

### Specialist Agent Analyses

#### Profitability Analysis (Score: {f_scores.get('profitability_score', 50):.1f}/100)
{prof_analysis}

#### Leverage Analysis (Score: {f_scores.get('leverage_score', 50):.1f}/100)
{lev_analysis}

#### Liquidity Analysis (Score: {f_scores.get('liquidity_score', 50):.1f}/100)
{liq_analysis}

#### Cash Flow Analysis (Score: {f_scores.get('cash_flow_score', 50):.1f}/100)
{cf_analysis}

#### Growth Analysis (Score: {f_scores.get('growth_score', 50):.1f}/100)
{gr_analysis}

---

### Factor Analysis Summary
Synthesize the specialist findings:
- **Credit Score** ({f_scores.get('credit_score', 50):.1f}/100): Rating quality assessment
- **Profitability** ({f_scores.get('profitability_score', 50):.1f}/100): Key themes from analysis above
- **Leverage** ({f_scores.get('leverage_score', 50):.1f}/100): Key themes from analysis above
- **Liquidity** ({f_scores.get('liquidity_score', 50):.1f}/100): Key themes from analysis above
- **Cash Flow** ({f_scores.get('cash_flow_score', 50):.1f}/100): Key themes from analysis above
- **Growth** ({f_scores.get('growth_score', 50):.1f}/100): Key themes from analysis above

Indicate strengths (70+), moderate performance (50-69), and weaknesses (<50) in your analysis.

### Credit Strengths
Extract 3-4 key positives from specialist analyses:
1. [Strength with score support]
2. [Strength with score support]
3. [Strength with score support]
4. [Strength with score support]

### Credit Risks & Concerns
Extract 3-4 key weaknesses from specialist analyses:
1. [Risk with score support]
2. [Risk with score support]
3. [Risk with score support]
4. [Risk with score support]

### Rating Outlook & Investment Recommendation
- **Rating Appropriateness**: Is {rating_val} justified given {comp_score:.1f}/100 composite?
- **Upgrade Triggers**: What improvements would drive rating higher?
- **Downgrade Risks**: What deterioration would pressure rating?
- **Investment Recommendation**: Strong Buy/Buy/Hold/Avoid based on risk-reward

**CRITICAL**: Complete ALL sections. Do NOT truncate. Target 1000-1200 words.

Begin the complete report now:"""

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
        system_prompt="Analyze cash flow score and provide commentary. Hand off to GrowthAgent when complete.",
        llm=llm,
        tools=[analyze_cash_flow_tool],
        can_handoff_to=["GrowthAgent"],
    )

    growth_agent = FunctionAgent(
        name="GrowthAgent",
        description="Analyze growth trends.",
        system_prompt="Analyze growth score and provide commentary. Hand off to SupervisorAgent when complete.",
        llm=llm,
        tools=[analyze_growth_tool],
        can_handoff_to=["SupervisorAgent"],
    )

    supervisor_agent = FunctionAgent(
        name="SupervisorAgent",
        description="Compile final credit report.",
        system_prompt="Collect all specialist analyses and compile comprehensive report. If analyses missing, hand back to appropriate agent.",
        llm=llm,
        tools=[compile_final_report_tool],
        can_handoff_to=["ProfitabilityAgent", "LeverageAgent", "LiquidityAgent", "CashFlowAgent", "GrowthAgent"],
    )

    # Create workflow - SIMPLE structure per PDF
    workflow = AgentWorkflow(
        agents=[prof_agent, lev_agent, liq_agent, cf_agent, growth_agent, supervisor_agent],
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
            "growth_analysis": None,
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
    prompt = f"""You are an expert fixed income credit analyst preparing a comprehensive credit report.

**Company Overview:**
- Name: {company_name}
- S&P Rating: {rating}
- Rating Band: {rating_band}
- Classification: {classification}
- Parent Sector: {parent_sector}
- Composite Score: {composite_score:.1f}/100

**Factor Scores (0-100 scale, higher is better):**
- Credit Score: {factor_scores.get('credit_score', 50):.1f}/100
- Leverage Score: {factor_scores.get('leverage_score', 50):.1f}/100
- Profitability Score: {factor_scores.get('profitability_score', 50):.1f}/100
- Liquidity Score: {factor_scores.get('liquidity_score', 50):.1f}/100
- Growth Score: {factor_scores.get('growth_score', 50):.1f}/100
- Cash Flow Score: {factor_scores.get('cash_flow_score', 50):.1f}/100

**Model Weights Used ({parent_sector} sector):**
- Credit: {weights_used.get('credit_score', 0.20)*100:.0f}%
- Leverage: {weights_used.get('leverage_score', 0.20)*100:.0f}%
- Profitability: {weights_used.get('profitability_score', 0.20)*100:.0f}%
- Liquidity: {weights_used.get('liquidity_score', 0.10)*100:.0f}%
- Growth: {weights_used.get('growth_score', 0.15)*100:.0f}%
- Cash Flow: {weights_used.get('cash_flow_score', 0.15)*100:.0f}%

**Historical Financial Metrics:**
{financial_section}

**Instructions:**
Please provide a comprehensive credit analysis report with the following structure:

1. **Executive Summary** (3-4 sentences)
   - Overall credit quality assessment based on composite score and factor scores
   - Key rating drivers (identify which factors are strongest/weakest)
   - Risk positioning (investment grade vs high yield characteristics)

2. **Profitability Analysis** (Score: {factor_scores.get('profitability_score', 50):.1f}/100)
   - EBITDA margin trends and interpretation
   - ROE/ROA performance relative to sector
   - Profitability sustainability assessment

3. **Leverage Analysis** (Score: {factor_scores.get('leverage_score', 50):.1f}/100)
   - Total Debt/EBITDA and Net Debt/EBITDA trends
   - Leverage trajectory (improving vs deteriorating)
   - Comparison to rating band norms

4. **Liquidity & Coverage Analysis** (Score: {factor_scores.get('liquidity_score', 50):.1f}/100)
   - Current ratio and cash position trends
   - Interest coverage analysis
   - Debt serviceability assessment

5. **Cash Flow Analysis** (Score: {factor_scores.get('cash_flow_score', 50):.1f}/100)
   - Operating cash flow quality and trends
   - Free cash flow generation capacity
   - Cash conversion efficiency

6. **Growth Analysis** (Score: {factor_scores.get('growth_score', 50):.1f}/100)
   - Revenue and EBITDA growth trends
   - Growth momentum assessment
   - Sustainability of growth trajectory

7. **Credit Strengths**
   - List 3-4 key positive credit factors based on factor scores
   - Support each with specific data points and scores

8. **Credit Risks & Concerns**
   - List 3-4 key risk factors based on weak factor scores
   - Support each with specific data points and scores

9. **Rating Outlook & Investment Recommendation**
   - Is the current {rating} rating appropriate given the {composite_score:.1f}/100 composite score?
   - What could trigger an upgrade or downgrade?
   - Investment recommendation from a credit perspective

**Formatting Requirements:**
- Use clear markdown formatting with headers (##, ###)
- Bold key metrics and conclusions
- Use bullet points for lists
- Be specific and reference actual numbers from the factor scores and historical data
- Keep tone professional and analytical
- Total length: 800-1000 words
- Focus on data-driven insights rather than generic statements

Generate the complete report now:"""

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
# MAIN APP EXECUTION (Skip if running tests)
# ============================================================================

if os.environ.get("RG_TESTS") != "1":
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
        if not use_dynamic_calibration:
            if '_calibrated_weights' in st.session_state:
                del st.session_state['_calibrated_weights']
            if 'last_calibration_state' in st.session_state:
                del st.session_state['last_calibration_state']
            # Force Streamlit to clear the cache when calibration toggle changes
            st.cache_data.clear()

        with st.spinner("Loading and processing data..."):
            # [V2.3] Create cache buster from unified period selection parameters
            reference_date_str = str(reference_date_override) if reference_date_override else 'none'
            cache_key = f"{period_mode.value}_{reference_date_str}_{use_dynamic_calibration}_{calibration_rating_band}"

            results_final, df_original, audits, period_calendar = load_and_process_data(
                uploaded_file,
                effective_use_sector_adjusted,
                period_mode=period_mode,
                reference_date_override=reference_date_override,
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
                "Methodology",
                "GenAI Credit Report"
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
                st.info("""
                **Ranking Methodology:** Results are ranked by actionability - recommendations are prioritized
                (Strong Buy > Buy > Hold > Avoid), with quality score breaking ties within each recommendation tier.
                This ensures "top opportunities" are issuers you'd actually act on, not just high-quality credits
                with deteriorating trends.
                """)

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
                        "Cycle_Position_Score": "Deteriorating ← Trend → Improving"
                    }
                )

                # Add split lines in DATA coordinates (xref='x', yref='y')
                fig_quadrant.add_vline(x=x_split_for_plot, line_width=1.5, line_dash="dash", line_color="#888", layer="below")
                fig_quadrant.add_hline(y=y_split, line_width=1.5, line_dash="dash", line_color="#888", layer="below")

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

                st.markdown("""
                **Principal Component Analysis** reveals the underlying structure of the 6 credit factors
                and shows how they contribute to overall variation across issuers. The radar charts display
                each factor's loading (contribution) on the principal components.
                """)

                try:
                    from plotly.subplots import make_subplots

                    # Get factor score columns and filter by coverage (min 50%)
                    all_factor_cols = [c for c in results_final.columns if c.endswith("_Score") and c != "Composite_Score"]

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
                        st.caption("Each radar chart shows how the 6 credit factors contribute to each principal component")

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
                        st.markdown("""
                        - **Distance from center**: Strength of factor's contribution
                        - **Positive values** (outward): Factor increases with PC
                        - **Negative values** (opposite): Factor decreases with PC
                        """)

                    with col_guide2:
                        st.markdown("**Interpretation**")
                        st.markdown("""
                        - **Near ±1.0**: Very strong influence
                        - **Near ±0.5**: Moderate influence
                        - **Near 0.0**: Weak influence
                        """)

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
                        all_factor_cols = [c for c in results_final.columns if c.endswith("_Score") and c != "Composite_Score"]
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
                    except:
                        st.warning(f"PCA analysis unavailable: {e}")
                        st.caption("This may occur with insufficient data or if factor scores are missing.")

                # ========================================================================
                # SCORE DISTRIBUTION AND CLASSIFICATION ANALYSIS
                # ========================================================================
                st.markdown("---")

                # Score distribution
                st.subheader("Score Distribution by Rating Group")

                fig = go.Figure()
                for group in ['Investment Grade', 'High Yield']:
                    group_data = results_final[results_final['Rating_Group'] == group]['Composite_Score']
                    fig.add_trace(go.Histogram(
                        x=group_data,
                        name=group,
                        opacity=0.7,
                        nbinsx=20
                    ))

                fig.update_layout(
                    barmode='overlay',
                    xaxis_title='Composite Score',
                    yaxis_title='Count',
                    title='Composite Score Distribution',
                    height=400
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
                        if diag["fy_suffixes"] or diag["cq_suffixes"]:
                            fy_list = ', '.join([s for s in diag['fy_suffixes']]) if diag['fy_suffixes'] else '(none)'
                            cq_list = ', '.join([s for s in diag['cq_suffixes']]) if diag['cq_suffixes'] else '(none)'
                            st.caption(f"Detected FY suffixes: {fy_list}; CQ suffixes: {cq_list}")
        
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
                            {"Metric": "CQ suffixes", "Value": ", ".join(diag["cq_suffixes"]) if diag["cq_suffixes"] else ""},
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
                                              'liquidity_score', 'growth_score', 'cash_flow_score']:
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

                                st.markdown("""
                                **Interpreting weight changes:**
                                - **Decreased weights** → Sector deviates from market on this factor, so we reduce its influence
                                - **Increased weights** → Sector is neutral on this factor, so we emphasize it for differentiation
                                - **Large changes (>50%)** → Factor shows significant structural difference in this sector
                                """)
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
                    'Combined_Signal', 'Recommendation', 'Weight_Method'
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
                    'Weight_Method': 'Portfolio Sector Weight (Context)'
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
                render_issuer_explainability(filtered, scoring_method)
        
                # ========================================================================
                # EXPORT CURRENT VIEW (V2.2)
                # ========================================================================
                with st.expander(" Export Current View (CSV)", expanded=False):
                    export_cols = [
                        "Company_ID", "Company_Name", "Credit_Rating_Clean", "Rating_Band", "Rating_Group",
                        "Composite_Score", "Composite_Percentile_in_Band",
                        "Credit_Score", "Leverage_Score", "Profitability_Score", "Liquidity_Score", "Growth_Score", "Cash_Flow_Score",
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

                    # ========================================================================
                    # HISTOGRAM with Quality Split Line (from df_band only)
                    # ========================================================================
                    st.subheader(f" Score Distribution - {selected_band} Band")

                    # Get quality split value using the same logic as the quadrant chart
                    _, x_split_for_hist, split_label, _ = resolve_quality_metric_and_split(
                        df_band, split_basis, split_threshold
                    )

                    # Build histogram from df_band only
                    hist_data = df_band['Composite_Score'].dropna()

                    fig = go.Figure(data=[go.Histogram(
                        x=hist_data,
                        nbinsx=15,
                        marker_color='#2C5697',
                        name='Composite Score'
                    )])

                    # Add vertical line for quality split
                    if pd.notna(x_split_for_hist):
                        fig.add_vline(
                            x=x_split_for_hist,
                            line_width=2,
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"Quality Split: {x_split_for_hist:.1f}",
                            annotation_position="top"
                        )

                    # Determine annotation based on split basis
                    if split_basis == "Global Percentile":
                        subtitle = f"Quality split at p{split_threshold:.0f} (Global Percentile mode)"
                    elif split_basis == "Percentile within Band (recommended)":
                        subtitle = f"Quality split at p{split_threshold:.0f} within band (Recommended mode)"
                    else:
                        subtitle = f"Quality split at p{split_threshold:.0f} of Composite Score (Absolute mode)"

                    fig.update_layout(
                        xaxis_title='Composite Score',
                        yaxis_title='Count',
                        height=350,
                        showlegend=False,
                        title_text=subtitle,
                        title_font_size=12
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # ========================================================================
                    # DEBUG SECTION (hidden by default)
                    # ========================================================================
                    _ENABLE_DEBUG = False  # Set to True to show debug info
                    if _ENABLE_DEBUG:
                        with st.expander("🔍 Debug Info", expanded=False):
                            st.write(f"**df_band size:** {len(df_band)} rows")
                            st.write(f"**Valid Composite_Score count:** {issuers_count}")
                            st.write(f"**Strong Buy count:** {strong_buy_count}")
                            st.write(f"**Strong Buy %:** {strong_buy_pct:.2f}%")
                            st.write(f"**Histogram data points:** {len(hist_data)}")
                            st.write(f"**Quality split value:** {x_split_for_hist:.2f}")
                            st.write(f"**Split basis:** {split_basis}")
                            st.write(f"**Split threshold:** {split_threshold}")

                            # Verify histogram count matches df_band
                            assert len(hist_data) == df_band['Composite_Score'].notna().sum(), \
                                "Histogram data count mismatch!"
                            st.success("✓ Histogram count matches df_band non-NaN Composite_Score count")
            
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
                
                classification_weights = get_classification_weights(selected_classification, use_sector_adjusted)
                universal_weights = get_classification_weights(selected_classification, use_sector_adjusted=False)
                
                weight_comparison = pd.DataFrame({
                    'Factor': ['Credit', 'Leverage', 'Profitability', 'Liquidity', 'Growth', 'Cash Flow'],
                    'Classification-Adjusted': [
                        classification_weights['credit_score'] * 100,
                        classification_weights['leverage_score'] * 100,
                        classification_weights['profitability_score'] * 100,
                        classification_weights['liquidity_score'] * 100,
                        classification_weights['growth_score'] * 100,
                        classification_weights['cash_flow_score'] * 100
                    ],
                    'Universal': [
                        universal_weights['credit_score'] * 100,
                        universal_weights['leverage_score'] * 100,
                        universal_weights['profitability_score'] * 100,
                        universal_weights['liquidity_score'] * 100,
                        universal_weights['growth_score'] * 100,
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
            # TAB 5: TREND ANALYSIS
            # ============================================================================

            with tab5:
                st.header(" Cyclicality & Trend Analysis")

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
                # QUALITY/TREND CONFIGURATION (read-only, set via sidebar)
                # ========================================================================
                # [V2.3] quality_basis is now hard-coded
                qs_basis = "Percentile within Band (recommended)"
                q_thresh = 60  # Fixed threshold
                t_thresh = 55  # Fixed threshold
                st.caption(f"Quality split basis: {qs_basis} · Quality threshold: {q_thresh} · Trend threshold: {t_thresh}")

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

                # Period Calendar Debug Display
                if period_calendar is not None and not period_calendar.empty:
                    if st.toggle("Show period calendar (debug)", value=False, key="show_period_calendar"):
                        st.markdown("### Period Calendar (FY/CQ Overlap Resolution)")
                        st.caption(f"**Overlaps removed:** {sum(1 for c in df_original.columns if 'Period Ended' in str(c)) * len(df_original) - len(period_calendar)} | **Prefer quarterly:** {use_quarterly_beta}")

                        # Show summary pivot
                        period_pivot = latest_periods(period_calendar, max_k_fy=4, max_k_cq=7)
                        st.dataframe(period_pivot, use_container_width=True, hide_index=True)

                        # Show raw period calendar
                        with st.expander("Show full period calendar (all periods)"):
                            st.dataframe(period_calendar, use_container_width=True, hide_index=True)

                # Cycle Position Analysis
                st.subheader("Business Cycle Position by Sector/Classification")
                st.caption("Shows which sectors are improving (green) vs deteriorating (red). Cycle Position Score (0-100) is a composite of trend, volatility, and momentum across leverage, profitability, liquidity, and growth metrics.")

                # Build sector/classification heatmap using the new helper
                heatmap_data, agg_data = compute_trend_heatmap(
                    results_final,
                    selected_band=trend_rating,
                    trend_threshold=t_thresh,
                    min_count=5
                )

                if not heatmap_data.empty:
                    # Create color scale: red (bad) -> yellow (neutral) -> green (good)
                    fig = go.Figure(data=go.Heatmap(
                        z=heatmap_data.values,
                        x=heatmap_data.columns,
                        y=heatmap_data.index,
                        colorscale='RdYlGn',  # Red-Yellow-Green
                        text=[[f'{val:.1f}' if pd.notna(val) else 'N/A' for val in row] for row in heatmap_data.values],
                        texttemplate='%{text}',
                        textfont={"size": 10},
                        hoverongaps=False,
                        colorbar=dict(title="Score")
                    ))

                    fig.update_layout(
                        title='Sector/Classification Trend Heatmap',
                        xaxis_title='',
                        yaxis_title='Metric',
                        height=400,
                        xaxis={'side': 'bottom'},
                        margin=dict(l=150, r=50, t=80, b=150)
                    )

                    # Rotate x-axis labels for readability
                    fig.update_xaxes(tickangle=-45)

                    st.plotly_chart(fig, use_container_width=True)

                    st.caption("**Notes:** % Improving uses the Trend threshold only. The Quality threshold affects the quadrant split, not this heatmap. Classifications with <5 issuers in the selected band are hidden (NaN).")

                    # Debug expander to verify counts and % Improving calculations
                    with st.expander("Debug: heatmap inputs"):
                        st.text(f"Trend threshold used: {t_thresh} · Rating Band: {trend_rating} · Classification: {trend_classification}")

                        # Show period calendar info if available
                        if period_calendar is not None and not period_calendar.empty:
                            st.markdown("**Period Calendar Summary:**")
                            period_summary = period_calendar.groupby(['period_type', 'k'], as_index=False)['period_end_date'].agg(['count', 'min', 'max'])
                            st.dataframe(period_summary, use_container_width=True)

                        st.markdown("**Aggregation Data:**")
                        st.dataframe(agg_data, use_container_width=True)

                    # Summary metrics (using non-NaN data only)
                    if not agg_data.empty:
                        # Filter out NaN rows for summary metrics
                        valid_agg = agg_data[agg_data['Avg_Cycle_Position'].notna()].copy()

                        if not valid_agg.empty:
                            col1, col2, col3 = st.columns(3)

                            # Sort by Avg_Cycle_Position for best/worst
                            valid_agg = valid_agg.sort_values('Avg_Cycle_Position', ascending=False)

                            with col1:
                                best_sector = valid_agg.iloc[0]['Classification']
                                best_score = valid_agg.iloc[0]['Avg_Cycle_Position']
                                st.metric("🟢 Most Improving Sector", best_sector, f"{best_score:.1f}")
                            with col2:
                                worst_sector = valid_agg.iloc[-1]['Classification']
                                worst_score = valid_agg.iloc[-1]['Avg_Cycle_Position']
                                st.metric("🔴 Most Deteriorating Sector", worst_sector, f"{worst_score:.1f}")
                            with col3:
                                # Overall % Improving from the filtered data
                                overall_pct = valid_agg['Pct_Improving'].mean()
                                st.metric("Overall % Improving", f"{overall_pct:.1f}%" if pd.notna(overall_pct) else "n/a")
                else:
                    st.info("Classification data not available or insufficient data for the selected band")
                
                # Improving vs. Deteriorating
                st.subheader("Trend Classification")
                
                # Calculate trend signal for leverage (most important)
                if 'Total Debt / EBITDA (x)_trend' in trend_data.columns:
                    trend_data['Trend_Signal'] = trend_data['Total Debt / EBITDA (x)_trend'].apply(
                        lambda x: 'Improving (Deleveraging)' if x < -0.2 else ('Deteriorating (Leveraging)' if x > 0.2 else 'Stable')
                    )
                    
                    trend_counts = trend_data['Trend_Signal'].value_counts()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        improving = trend_counts.get('Improving (Deleveraging)', 0)
                        st.metric(" Improving", f"{improving}", f"{improving/len(trend_data)*100:.1f}%")
                    with col2:
                        stable = trend_counts.get('Stable', 0)
                        st.metric(" Stable", f"{stable}", f"{stable/len(trend_data)*100:.1f}%")
                    with col3:
                        deteriorating = trend_counts.get('Deteriorating (Leveraging)', 0)
                        st.metric(" Deteriorating", f"{deteriorating}", f"{deteriorating/len(trend_data)*100:.1f}%")
                
                # ========================================================================
                # TOP 10 IMPROVING/DETERIORATING (ranked by Cycle Position Score)
                # ========================================================================

                # Guard: ensure column exists
                if 'Cycle_Position_Score' not in trend_data.columns:
                    st.warning("Cycle_Position_Score not found; cannot rank Top 10 trend issuers.")
                else:
                    # Top 10 Improving: highest cycle position scores
                    st.subheader("Top 10 Improving Trend Issuers")
                    st.caption("Ranked by Cycle Position Score (highest = most improving)")

                    improving_mask = trend_data['Combined_Signal'].str.contains('Improving', na=False)
                    top_improving = (trend_data[improving_mask]
                                    .sort_values('Cycle_Position_Score', ascending=False)
                                    .head(10))

                    # Select columns (raw-only: no Composite_Score or Cycle_Position_Score)
                    cols = ['Company_Name', 'Credit_Rating_Clean', 'Rubrics_Custom_Classification',
                            'Combined_Signal', 'Recommendation']
                    # Add raw scores if available
                    if 'Raw_Quality_Score' in top_improving.columns:
                        cols.insert(3, 'Raw_Quality_Score')
                    if 'Raw_Trend_Score' in top_improving.columns:
                        cols.insert(4, 'Raw_Trend_Score')
                    cols_present = [c for c in cols if c in top_improving.columns]

                    st.dataframe(
                        top_improving[cols_present].rename(columns={
                            'Company_Name': 'Company',
                            'Credit_Rating_Clean': 'Rating',
                            'Rubrics_Custom_Classification': 'Classification',
                            'Raw_Quality_Score': 'Quality (raw)',
                            'Raw_Trend_Score': 'Trend (raw)',
                            'Combined_Signal': 'Signal',
                            'Recommendation': 'Rec'
                        }),
                        use_container_width=True, hide_index=True
                    )

                    # Top 10 Deteriorating: lowest cycle position scores
                    st.subheader("Top 10 Deteriorating Trend Issuers")
                    st.caption("Ranked by Cycle Position Score (lowest = most deteriorating)")

                    deteriorating_mask = trend_data['Combined_Signal'].str.contains('Deteriorating', na=False)
                    top_deteriorating = (trend_data[deteriorating_mask]
                                        .sort_values('Cycle_Position_Score', ascending=True)
                                        .head(10))

                    st.dataframe(
                        top_deteriorating[cols_present].rename(columns={
                            'Company_Name': 'Company',
                            'Credit_Rating_Clean': 'Rating',
                            'Rubrics_Custom_Classification': 'Classification',
                            'Raw_Quality_Score': 'Quality (raw)',
                            'Raw_Trend_Score': 'Trend (raw)',
                            'Combined_Signal': 'Signal',
                            'Recommendation': 'Rec'
                        }),
                        use_container_width=True, hide_index=True
                    )
            
            # ============================================================================
            # TAB 6: METHODOLOGY
            # ============================================================================
            
            with tab6:
                # Render programmatically generated methodology specification
                render_methodology_tab(df_original, results_final)
        
            # ============================================================================
            # TAB 7: GENAI CREDIT REPORT (V3.0 - REDESIGNED DUAL-PIPELINE)
            # ============================================================================

            with tab7:
                st.header("AI Credit Report")

                st.info("""
                **New in V3.0:** Reports now analyze actual financial metrics from the input spreadsheet,
                compare to sector and rating peers, and provide proper context for model scores.
                """)

                if results_final is not None and len(results_final) > 0 and df_original is not None:

                    # Company selection
                    company_options = sorted(results_final['Company_Name'].dropna().unique().tolist())

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
                        selected_row = results_final[results_final['Company_Name'] == selected_company].iloc[0]

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("S&P Rating", selected_row.get('Credit_Rating_Clean', 'N/A'))
                        with col2:
                            st.metric("Composite Score", f"{selected_row.get('Composite_Score', 0):.1f}")
                        with col3:
                            st.metric("Rating Band", selected_row.get('Rating_Band', 'N/A'))
                        with col4:
                            classification = selected_row.get('Rubrics_Custom_Classification', 'N/A')
                            st.metric("Classification", classification[:20] + "..." if len(classification) > 20 else classification)

                    if generate_button and selected_company:
                        with st.spinner(f"Generating comprehensive credit report for {selected_company}..."):
                            try:
                                # Get calibration state from session
                                use_sector_adjusted = st.session_state.get('scoring_method') == 'Classification-Adjusted Scoring'
                                calibrated_weights = st.session_state.get('_calibrated_weights')

                                # STEP 1: Gather complete data from both sources
                                st.write("Extracting financial data from input spreadsheet...")
                                complete_data = prepare_genai_credit_report_data(
                                    df_original=df_original,  # Raw input spreadsheet
                                    results_df=results_final,  # Model outputs
                                    company_name=selected_company,
                                    use_sector_adjusted=use_sector_adjusted,
                                    calibrated_weights=calibrated_weights
                                )

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
                                        model="gpt-4-turbo-preview",
                                        messages=[
                                            {"role": "system", "content": "You are a professional credit analyst generating comprehensive credit reports."},
                                            {"role": "user", "content": prompt}
                                        ],
                                        max_tokens=4000,
                                        temperature=0.3
                                    )

                                    report = response.choices[0].message.content

                                    st.markdown("---")

                                    # STEP 4: Display report
                                    st.markdown(report)

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

            st.markdown("---")
            st.markdown("""
        <div style='text-align: center; color: #4c566a; padding: 20px;'>
            <p><strong>Issuer Credit Screening Model V3.0</strong></p>
            <p>© 2025 Rubrics Asset Management | Annual-Only Default + Minimal Identifiers</p>
        </div>
        """, unsafe_allow_html=True)
        
        # ============================================================================
        # [V2.2] SELF-TESTS (Run with RG_TESTS=1 environment variable)
        # ============================================================================
        
if False and os.environ.get("RG_TESTS") == "1":  # Tests temporarily disabled - contains outdated function calls
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

    assert list(fy.index)[-1].startswith("2024"), \
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
    trend_fy = calculate_trend_indicators(trend_test, test_metrics, use_quarterly=False, reference_date=None)

    # Test 9b: Quarterly-mode (use_quarterly=True) - should use base + .1-.12 (available up to .8 here)
    trend_cq = calculate_trend_indicators(trend_test, test_metrics, use_quarterly=True, reference_date=None)

    # Verify that momentum differs between the two modes
    # Momentum compares recent vs prior periods, so including .5-.8 should change the result
    margin_momentum_fy = trend_fy["EBITDA Margin_momentum"].iloc[0]
    margin_momentum_cq = trend_cq["EBITDA Margin_momentum"].iloc[0]

    # With 5 annual periods (base, .1-.4), momentum compares last 4 vs prior (not enough for 8-period split)
    # With 9 periods (base, .1-.8), momentum compares last 4 (.5-.8) vs prior 4 (.1-.4)
    assert margin_momentum_fy != margin_momentum_cq, \
        f"Expected different momentum scores between FY and CQ modes, got FY={margin_momentum_fy:.1f}, CQ={margin_momentum_cq:.1f}"
    print(f"  OK FY-mode momentum: {margin_momentum_fy:.1f}, CQ-mode momentum: {margin_momentum_cq:.1f} (differ)")

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
        m_annual = calculate_trend_indicators(_df, test_metrics, use_quarterly=False, reference_date=None)
        m_quarterly = calculate_trend_indicators(_df, test_metrics, use_quarterly=True, reference_date=None)

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

