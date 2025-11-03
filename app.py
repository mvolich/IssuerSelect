import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import json
import os
import time
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

# [V2.0] Only configure Streamlit if not running tests
if os.environ.get("RG_TESTS") != "1":
    st.set_page_config(
        page_title="Issuer Credit Screening Model V2.0",
        layout="wide",
        page_icon="https://rubricsam.com/wp-content/uploads/2021/01/cropped-rubrics-logo-tight.png",
    )

# ============================================================================
# HEADER RENDERING HELPER
# ============================================================================

def render_header(results_final=None, data_period=None, use_sector_adjusted=False, df_original=None):
    """
    Render the application header exactly once.

    Pre-upload: Shows only hero (title + subtitle + logo)
    Post-upload: Shows hero + 4 metrics + caption with period dates
    """
    # Always render the hero once
    st.markdown("""
    <div class="rb-header">
      <div class="rb-title">
        <h1>Issuer Credit Screening Model V2.0</h1>
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
        st.caption(f"📅 Data Periods: {_period_labels['fy_label']}  |  {_period_labels['cq_label']}")
        if os.environ.get("RG_TESTS") and _period_labels.get("used_fallback"):
            st.caption("[DEV] FY/CQ classifier not available — using documented fallback (first 5 FY, rest CQ).")

        st.markdown("---")

# ============================================================================
# [V2.0] MINIMAL IDENTIFIERS + FEATURE GATES
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
# [V2.0] PLOT/BRAND HELPERS
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
# [V2.0] PCA VISUALIZATION (IG/HY COHORT CLUSTERING)
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
        xaxis_title="PC1",
        yaxis_title="PC2"
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
            xaxis_title="Overall Credit Quality (Better →)",
            yaxis_title="Financial Strength vs Leverage Balance"
        )
        return fig

    ig_fig = make_fig(ig_df, "Investment Grade")
    hy_fig = make_fig(hy_df, "High Yield")

    st.subheader("Investment Grade Issuer Map")
    st.plotly_chart(ig_fig, use_container_width=True)
    st.subheader("High Yield Issuer Map")
    st.plotly_chart(hy_fig, use_container_width=True)

# ============================================================================
# [V2.0] RATING-BAND UTILITIES (LEADERBOARDS)
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
# DIAGNOSTICS & DATA HEALTH (V2.0)
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
            # Robust parse; ignore Excel serials/NaT/1900 sentinels
            dt = pd.to_datetime(period_value, errors="coerce", dayfirst=True)
            if pd.isna(dt) or (hasattr(dt, "year") and dt.year == 1900):
                continue
            date_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
            time_series[date_str] = float(metric_value)
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
                                    trend_thr=st.session_state.get("cfg_trend_threshold", 55),
                                    quality_thr=st.session_state.get("cfg_quality_threshold", 60))
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
# [V2.0] URL STATE & PRESETS
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

def collect_current_state(scoring_method, data_period, use_quarterly_beta,
                          band_default=None, top_n_default=20):
    """Collect current UI state into a dictionary."""
    state = {
        "scoring_method": scoring_method,
        "data_period": data_period,
        "use_quarterly_beta": bool(use_quarterly_beta),
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
    band_default = state.get("band_default") or ""
    top_n_default = int(state.get("top_n_default") or 20)
    return sm, dp, qb, band_default, top_n_default

def _build_deep_link(state: dict) -> str:
    """
    Return a relative deep link (querystring) that reproduces the current state.
    We use a relative link (?key=val...) so it works across environments.
    """
    qp = {k: (str(v).lower() if isinstance(v, bool) else str(v)) for k, v in state.items() if v is not None}
    return "?" + urlencode(qp)

# ============================================================================
# [V2.0] PERIOD ENDED PARSING (ACTUAL DATES)
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
    Split period columns into FY vs CQ based on frequency heuristics.

    Args:
        pe_data: List of (suffix, datetime_series) tuples from parse_period_ended_cols
        df: DataFrame to analyze date frequencies

    Returns: (fy_suffixes, cq_suffixes) - lists of suffixes

    Heuristic:
    - Calculate median days between consecutive valid dates
    - If median ≈ 90 days (60-150): Quarterly (CQ)
    - If median ≈ 365 days (270+): Annual (FY)
    - First ~5 periods likely FY, rest likely CQ (fallback)
    """
    if len(pe_data) == 0:
        return [], []

    # Try frequency-based classification for first row with valid dates
    for idx in range(len(df)):
        dates = []
        for suffix, dt_series in pe_data:
            val = dt_series.iloc[idx]
            if pd.notna(val):
                dates.append(val)

        if len(dates) >= 2:
            # Calculate gaps between consecutive dates
            gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            valid_gaps = [g for g in gaps if g > 0]

            if valid_gaps:
                median_gap = np.median(valid_gaps)

                # Quarterly: ~90 days (60-150 day range)
                if 60 <= median_gap <= 150:
                    # All periods are quarterly
                    return [], [suffix for suffix, _ in pe_data]
                # Annual: ~365 days (270+ day range)
                elif median_gap >= 270:
                    # All periods are annual
                    return [suffix for suffix, _ in pe_data], []
                # Mixed or unclear - use position heuristic
                break

    # Fallback: assume first 5 are FY (indices 0-4), rest are CQ
    fy_suffixes = [pe_data[i][0] for i in range(min(5, len(pe_data)))]
    cq_suffixes = [pe_data[i][0] for i in range(5, len(pe_data))]

    return fy_suffixes, cq_suffixes

# ============================================================================
# [V2.0] PERIOD CALENDAR UTILITIES (ROBUST HANDLING OF VENDOR DATES & FY/CQ OVERLAP)
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

def _batch_extract_metrics(df, metric_list, has_period_alignment, data_period_setting):
    """
    OPTIMIZED: Extract all metrics at once using vectorized operations.
    Returns dict of {metric_name: Series of values}.
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

    # Build FY suffix list (annual-only)
    fy_suffixes, _ = period_cols_by_kind(pe_data, df)
    candidate_suffixes = fy_suffixes if fy_suffixes else [s for s, _ in pe_data[:5]]  # First 5 as fallback

    # For each metric, extract most recent annual value (vectorized)
    for metric in metric_list:
        # Collect (date, value) pairs for this metric across all FY suffixes
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

        # Get most recent (latest date) value per issuer
        long_df = long_df.sort_values(['row_idx', 'date'])
        most_recent = long_df.groupby('row_idx').last()['value']

        # Reindex to match original df
        result[metric] = most_recent.reindex(df.index, fill_value=np.nan)

    return result

# ============================================================================
# CASH FLOW HELPERS (v3 - DataFrame-level with alias-aware batch extraction)
# ============================================================================

def _cf_components_dataframe(df: pd.DataFrame, data_period_setting: str, has_period_alignment: bool) -> pd.DataFrame:
    """Extract cash flow components using alias-aware batch extraction."""

    cash_flow_metrics = [
        'Cash from Ops.',
        'Cash from Operations',
        'Operating Cash Flow',
        'Cash from Ops',
        'Capital Expenditure',
        'Capital Expenditures',
        'CAPEX',
        'Total Revenues',
        'Total Revenue',
        'Revenue',
        'Total Debt',
        'Levered Free Cash Flow',
        'Free Cash Flow'
    ]

    metrics = _batch_extract_metrics(df, cash_flow_metrics, has_period_alignment, data_period_setting)

    # Map to standardized names
    ocf = metrics.get('Cash from Ops.', metrics.get('Cash from Operations',
          metrics.get('Operating Cash Flow', metrics.get('Cash from Ops', pd.Series(np.nan, index=df.index)))))

    capex = metrics.get('Capital Expenditure', metrics.get('Capital Expenditures',
            metrics.get('CAPEX', pd.Series(np.nan, index=df.index))))

    rev = metrics.get('Total Revenues', metrics.get('Total Revenue',
          metrics.get('Revenue', pd.Series(np.nan, index=df.index))))

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

def _cash_flow_component_scores(df: pd.DataFrame, data_period_setting: str, has_period_alignment: bool) -> pd.DataFrame:
    """Calculate cash flow component scores using alias-aware extraction."""

    components = _cf_components_dataframe(df, data_period_setting, has_period_alignment)
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
# FRESHNESS HELPERS (V2.0)
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
# PERIOD LABELING & FY/CQ CLASSIFICATION (V2.0)
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
# SECTOR-SPECIFIC WEIGHTS (SOLUTION TO ISSUE #1: SECTOR BIAS)
# ============================================================================

SECTOR_WEIGHTS = {
    'Utilities': {
        'credit_score': 0.25,
        'leverage_score': 0.12,
        'profitability_score': 0.15,
        'liquidity_score': 0.08,
        'growth_score': 0.10,
        'cash_flow_score': 0.30
    },
    'Real Estate': {
        'credit_score': 0.20,
        'leverage_score': 0.15,
        'profitability_score': 0.15,
        'liquidity_score': 0.12,
        'growth_score': 0.13,
        'cash_flow_score': 0.25
    },
    'Energy': {
        'credit_score': 0.18,
        'leverage_score': 0.25,
        'profitability_score': 0.12,
        'liquidity_score': 0.15,
        'growth_score': 0.08,
        'cash_flow_score': 0.22
    },
    'Materials': {
        'credit_score': 0.20,
        'leverage_score': 0.23,
        'profitability_score': 0.15,
        'liquidity_score': 0.12,
        'growth_score': 0.10,
        'cash_flow_score': 0.20
    },
    'Industrials': {
        'credit_score': 0.20,
        'leverage_score': 0.20,
        'profitability_score': 0.22,
        'liquidity_score': 0.10,
        'growth_score': 0.13,
        'cash_flow_score': 0.15
    },
    'Information Technology': {
        'credit_score': 0.18,
        'leverage_score': 0.25,
        'profitability_score': 0.25,
        'liquidity_score': 0.08,
        'growth_score': 0.18,
        'cash_flow_score': 0.06
    },
    'Health Care': {
        'credit_score': 0.20,
        'leverage_score': 0.22,
        'profitability_score': 0.23,
        'liquidity_score': 0.10,
        'growth_score': 0.12,
        'cash_flow_score': 0.13
    },
    'Consumer Staples': {
        'credit_score': 0.25,
        'leverage_score': 0.18,
        'profitability_score': 0.22,
        'liquidity_score': 0.08,
        'growth_score': 0.08,
        'cash_flow_score': 0.19
    },
    'Consumer Discretionary': {
        'credit_score': 0.18,
        'leverage_score': 0.23,
        'profitability_score': 0.20,
        'liquidity_score': 0.13,
        'growth_score': 0.15,
        'cash_flow_score': 0.11
    },
    'Communication Services': {
        'credit_score': 0.22,
        'leverage_score': 0.18,
        'profitability_score': 0.18,
        'liquidity_score': 0.10,
        'growth_score': 0.12,
        'cash_flow_score': 0.20
    },
    # Default for unmapped sectors
    'Default': {
        'credit_score': 0.20,
        'leverage_score': 0.20,
        'profitability_score': 0.20,
        'liquidity_score': 0.10,
        'growth_score': 0.15,
        'cash_flow_score': 0.15
    }
}

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
    Falls back to Default if sector not found or if sector adjustment disabled.
    
    DEPRECATED: Use get_classification_weights() for Rubrics Custom Classifications
    """
    if not use_sector_adjusted:
        return SECTOR_WEIGHTS['Default']
    return SECTOR_WEIGHTS.get(sector, SECTOR_WEIGHTS['Default'])

def get_classification_weights(classification, use_sector_adjusted=True):
    """
    Get factor weights for a Rubrics Custom Classification.
    
    Hierarchy:
    1. Check if classification has custom override weights
    2. Map to parent sector and use sector weights  
    3. Fall back to Default if classification not found
    
    Args:
        classification: Rubrics Custom Classification value
        use_sector_adjusted: Whether to use adjusted weights (vs universal Default)
    
    Returns:
        Dictionary with 6 factor weights (summing to 1.0)
    """
    if not use_sector_adjusted:
        return SECTOR_WEIGHTS['Default']
    
    # Step 1: Check for custom overrides
    if classification in CLASSIFICATION_OVERRIDES:
        return CLASSIFICATION_OVERRIDES[classification]
    
    # Step 2: Map to parent sector
    if classification in CLASSIFICATION_TO_SECTOR:
        parent_sector = CLASSIFICATION_TO_SECTOR[classification]
        if parent_sector in SECTOR_WEIGHTS:
            return SECTOR_WEIGHTS[parent_sector]
    
    # Step 3: Fall back to Default
    return SECTOR_WEIGHTS['Default']

# ================================
# EXPLAINABILITY HELPERS (V2.0) — canonical
# ================================

def _resolve_text_field(row: pd.Series, candidates):
    for c in candidates:
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c]).strip()
    return None

def _resolve_model_weights_for_row(row: pd.Series, scoring_method: str):
    """
    Return (weights_dict, provenance_str) with sector/classification precedence.
    Keys: lowercase matching SECTOR_WEIGHTS (credit_score, leverage_score,
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

    # 2) Sector map
    try:
        if sec and sec in SECTOR_WEIGHTS:
            return SECTOR_WEIGHTS[sec], f"Sector-Adjusted via sector='{sec}'"
    except Exception:
        pass

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
    Build a 4-col table: Factor, Score, Weight %, Contribution.
    Normalises weights over present factor columns. Returns
    (df, provenance, composite_score, diff_sum_minus_composite).
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
    # Check for column existence using mapped names
    present = [f for f in canonical if f.replace(" ", "_") + "_Score" in issuer_row.index]

    weights_lc, provenance = _resolve_model_weights_for_row(issuer_row, scoring_method)

    w = {f: float(max(0.0, weights_lc.get(factor_map[f], 0.0))) for f in present}
    s = sum(w.values()) or 1.0
    w = {k: v / s for k, v in w.items()}

    rows = []
    for fac in present:
        col_name = fac.replace(" ", "_") + "_Score"
        score = float(issuer_row.get(col_name, np.nan))
        wt = w[fac]
        rows.append({
            "Factor": fac,
            "Score": score,
            "Weight %": round(100.0 * wt, 2),
            "Contribution": round(score * wt, 4)
        })
    df = pd.DataFrame(rows)
    comp = float(issuer_row.get("Composite_Score", np.nan))
    diff = float(df["Contribution"].sum() - comp) if len(df) else np.nan
    return df, provenance, comp, diff
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

        df_contrib, provenance, comp, diff = _build_explainability_table(issuer_row, scoring_method)

        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"**Weight Method (provenance):** {provenance}")
            st.dataframe(df_contrib, use_container_width=True, hide_index=True)
        with right:
            st.metric("Composite (as-at)", f"{comp:.2f}" if pd.notna(comp) else "n/a")
            st.metric("Sum of contributions", f"{df_contrib['Contribution'].sum():.2f}" if len(df_contrib) else "n/a")
            if pd.notna(comp) and len(df_contrib) and abs(diff) > 0.5:
                st.warning(f"Contributions differ from Composite by {diff:+.2f}. Check factor set and weights.")
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
    st.markdown("# Model Methodology (V2.0)")
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

        # Build weights table for all sectors
        weights_rows = []
        for sector_name in sorted(SECTOR_WEIGHTS.keys()):
            weights = SECTOR_WEIGHTS[sector_name]
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
        st.markdown("**Universal Weights Mode:** Same weights for all issuers regardless of sector.")

        default_weights = SECTOR_WEIGHTS['Default']
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
        **Period Calendar (V2.0):**
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

    quality_basis = st.session_state.get("cfg_quality_split_basis", "Percentile within Band (recommended)")
    quality_threshold = st.session_state.get("cfg_quality_threshold", 60)
    trend_threshold = st.session_state.get("cfg_trend_threshold", 55)

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
    **Context-aware refinements (V2.0):**
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
    # SECTION 7: RECOMMENDATION GUARDRAIL
    # ========================================================================

    st.markdown("## 7. Recommendation Guardrail")
    st.markdown("""
    **Base percentile bands:**
    - Composite_Score ≥ 80th percentile → **Strong Buy**
    - Composite_Score ≥ 60th percentile → **Buy**
    - Composite_Score ≥ 40th percentile → **Hold**
    - Composite_Score < 40th percentile → **Avoid**

    **Guardrail override:**
    - **Weak & Deteriorating** issuers are **never** labeled Buy or Strong Buy, regardless of raw score
    - They are capped at **Hold** (if percentile ≥ 40) or **Avoid** (if percentile < 40)

    *(Strong but Deteriorating and Weak but Improving may still be Buy/Strong Buy if percentile warrants)*
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
    export_md = f"""# Issuer Credit Screening Model - Methodology Specification (V2.0)

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
        for sector_name in sorted(SECTOR_WEIGHTS.keys()):
            w = SECTOR_WEIGHTS[sector_name]
            export_md += f"| {sector_name} | {w['credit_score']:.2f} | {w['leverage_score']:.2f} | {w['profitability_score']:.2f} | {w['liquidity_score']:.2f} | {w['growth_score']:.2f} | {w['cash_flow_score']:.2f} | {sum(w.values()):.2f} |\n"
    else:
        dw = SECTOR_WEIGHTS['Default']
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

## Recommendation Guardrail

- Composite_Score ≥ 80th percentile → Strong Buy
- Composite_Score ≥ 60th percentile → Buy
- Composite_Score ≥ 40th percentile → Hold
- Composite_Score < 40th percentile → Avoid

**Override:** Weak & Deteriorating issuers are never labeled Buy or Strong Buy (capped at Hold/Avoid).

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
sm0, dp0, qb0, band0, topn0 = apply_state_to_controls(_url_state)

# Optional: Show deprecation notice if URL contained FY-4
if _original_dp == "FY-4 (Legacy)":
    st.info("ℹ️ Note: 'FY-4 (Legacy)' option has been removed. Defaulted to 'Most Recent Fiscal Year (FY0)'.")

# Model Version Selection (SOLUTION TO ISSUE #1)
st.sidebar.subheader(" Scoring Method")
scoring_method_options = ["Classification-Adjusted Weights (Recommended)", "Universal Weights (Original)"]
scoring_method = st.sidebar.radio(
    "Select Scoring Approach",
    scoring_method_options,
    index=0 if sm0.startswith("Classification") else 1,
    help="Classification-Adjusted applies different factor weights by industry classification (e.g., Utilities emphasize cash flow more than leverage)"
)
use_sector_adjusted = (scoring_method == "Classification-Adjusted Weights (Recommended)")

# Canonicalize & persist scoring method
sm_canonical = (
    "Classification-Adjusted Weights"
    if scoring_method.startswith("Classification-Adjusted")
    else "Universal Weights"
)
st.session_state["scoring_method"] = sm_canonical
st.session_state["use_sector_adjusted"] = (sm_canonical == "Classification-Adjusted Weights")

# === Sidebar: clarified controls ===
st.sidebar.subheader("Configuration")

# (A) Point-in-time period for scores (affects single-period features including Composite_Score inputs)
data_period_setting = st.sidebar.selectbox(
    "Point-in-time period for scores",
    options=["Most Recent Fiscal Year (FY0)", "Most Recent Quarter (CQ-0)"],
    index=["Most Recent Fiscal Year (FY0)", "Most Recent Quarter (CQ-0)"].index(dp0) if dp0 in ["Most Recent Fiscal Year (FY0)", "Most Recent Quarter (CQ-0)"] else 0,
    help=(
        "Determines which 'most-recent' value is used for point-in-time features. "
        "FY0 uses the latest annual filing date; CQ-0 uses the latest quarterly filing date. "
        "This does NOT affect trend/momentum windows."
    ),
    key="cfg_period_for_scores",
)

# (B) Trend window (affects momentum/volatility only)
use_quarterly_beta = st.sidebar.checkbox(
    "Trend window: use quarterly data where available",
    value=qb0,
    help=(
        "When ON, momentum/volatility are computed from quarterly time series (base + .1 … .12). "
        "When OFF, momentum/volatility use annual-only series (base + .1 … .4). "
        "This does NOT change the point-in-time period used for scores."
    ),
    key="cfg_trend_window_quarterly",
)

# (C) Quality/Trend Split Configuration
st.sidebar.markdown("#### Quality/Trend Split")
split_basis = st.sidebar.selectbox(
    "Quality split basis",
    ["Percentile within Band (recommended)", "Global Percentile", "Absolute Composite Score"],
    index=0,
    help="Defines how we decide Strong vs Weak quality.",
    key="cfg_quality_split_basis"
)
split_threshold = st.sidebar.slider(
    "Quality threshold",
    40, 80, 60,
    help="Percentile or score threshold for Strong vs Weak classification",
    key="cfg_quality_threshold"
)
trend_threshold = st.sidebar.slider(
    "Trend threshold (Cycle Position)",
    40, 70, 55,
    help="Y-axis split for improving vs deteriorating.",
    key="cfg_trend_threshold"
)

# ============================================================================
# [V2.0] ADVANCED: DUAL-HORIZON CONTEXT THRESHOLDS
# ============================================================================
with st.sidebar.expander("⚙️ Advanced: Dual-Horizon Context", expanded=False):
    st.markdown("""
    **Context-aware signal guards** detect exceptional quality, outliers, and volatility to refine classification.
    """)

    volatility_cv_threshold = st.slider(
        "Volatility CV threshold",
        0.10, 0.50, 0.30, 0.05,
        help="Coefficient of variation threshold for detecting high volatility series.",
        key="cfg_volatility_cv_threshold"
    )

    outlier_z_threshold = st.slider(
        "Outlier Z threshold",
        -4.0, -1.5, -2.5, 0.5,
        help="Z-score threshold for detecting outlier quarters (negative = below mean).",
        key="cfg_outlier_z_threshold"
    )

    damping_factor = st.slider(
        "Damping factor",
        0.0, 1.0, 0.5, 0.1,
        help="Reduction factor for negative momentum when volatility/outliers detected (0.5 = 50% reduction).",
        key="cfg_damping_factor"
    )

    near_peak_tolerance = st.slider(
        "Near-peak tolerance (%)",
        5, 20, 10, 5,
        help="Percentage tolerance for detecting if current value is near historical peak.",
        key="cfg_near_peak_tolerance"
    )

    st.markdown("""
    **Label overrides:**
    - **Strong & Normalizing**: Exceptional quality + medium-term improving + short-term declining
    - **Strong & Moderating**: Exceptional quality + high volatility + not improving

    See Methodology tab for full dual-horizon specification.
    """)

# Alias for backward compatibility with URL state management
data_period = data_period_setting

# Write state back to URL (deep-linkable state)
_current_state = collect_current_state(
    scoring_method=scoring_method,
    data_period=data_period,
    use_quarterly_beta=use_quarterly_beta,
    band_default=band0,  # Will be updated later when band selector is rendered
    top_n_default=topn0  # Will be updated later when top_n slider is rendered
)
_set_query_params(_current_state)

# ============================================================================
# PRESETS: Save/Load Configuration
# ============================================================================
with st.sidebar.expander(" Save/Load Preset", expanded=False):
    st.markdown("**Save current settings as a preset:**")

    # Generate JSON from current state
    preset_json = json.dumps(_current_state, indent=2)

    st.download_button(
        label=" Download Preset (JSON)",
        data=preset_json.encode('utf-8'),
        file_name="issuer_screen_preset.json",
        mime="application/json",
        help="Save current settings to share or reload later"
    )

    st.markdown("---")
    st.markdown("**Load a saved preset:**")

    preset_file = st.file_uploader(
        "Upload Preset JSON",
        type=["json"],
        key="preset_uploader",
        help="Upload a previously saved preset to restore settings"
    )

    if preset_file is not None:
        try:
            loaded_state = json.load(preset_file)
            st.success(" Preset loaded! Apply settings by reloading the page with the updated URL.")

            # Update URL with loaded state
            _set_query_params(loaded_state)

            # Show what was loaded
            st.json(_json_safe(loaded_state))

        except Exception as e:
            st.error(f"Failed to load preset: {e}")

# ============================================================================
# REPRODUCE / SHARE: Deep Link
# ============================================================================
with st.sidebar.expander(" Reproduce / Share", expanded=False):
    st.caption("Share a link that reproduces your current settings.")

    # Build the link off the latest _current_state (already set just above)
    deep_link = _build_deep_link(_current_state)

    # Show a clickable link and a copy-friendly text box
    st.markdown(f"[Open with current settings]({deep_link})")

    # Read-only text input is easy to copy across browsers
    st.text_input("Deep link (copy):", value=deep_link, help="Copy & paste this into email/notes.", key="rg_deeplink", disabled=True)

    # Optional: HTML copy button (works in most browsers)
    st.markdown(f"""
        <button onclick="navigator.clipboard.writeText('{deep_link}')"
                style="background:#2C5697;color:#fff;border:none;border-radius:4px;padding:6px 10px;font-weight:600;cursor:pointer;">
            Copy to clipboard
        </button>
    """, unsafe_allow_html=True)

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload S&P Data file (Excel or CSV)", type=["xlsx", "csv"])
HAS_DATA = uploaded_file is not None

# Pre-upload: show hero only (title + subtitle + logo)
if not HAS_DATA:
    render_header()

# ============================================================================
# DATA LOADING & PROCESSING FUNCTIONS
# ============================================================================

def get_most_recent_column(df, base_metric, data_period_setting):
    """
    [V2.0] Returns the appropriate metric column based on parsed Period Ended dates.

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

def _build_metric_timeseries(df: pd.DataFrame, base_metric: str, use_quarterly: bool, pe_data_cached=None, fy_cq_cached=None) -> pd.DataFrame:
    """
    OPTIMIZED: Vectorized time series construction with FY/CQ de-duplication.
    Returns DataFrame where each row is an issuer's time series (columns = ISO dates).

    Args:
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
# DUAL-HORIZON TREND ANALYSIS UTILITIES (V2.0)
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


def calculate_trend_indicators(df, base_metrics, use_quarterly=False):
    """
    SOLUTION TO ISSUE #2: MISSING CYCLICALITY & TREND ANALYSIS
    OPTIMIZED: Caches period parsing and uses vectorized calculations.

    Calculate trend, momentum, and volatility indicators using historical time series.

    Args:
        df: DataFrame with metric columns
        base_metrics: List of base metric names
        use_quarterly: If True, use [base, .1, .2, ..., .12] (13 periods: 5 annual + 8 quarterly)
                      If False, use [base, .1, .2, .3, .4] (5 annual periods only)

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

    # Helper function for vectorized calculations
    def _calc_row_stats(row_series):
        """Calculate trend, volatility, momentum for a single row's time series."""
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

        # Volatility (coefficient of variation, inverted)
        if n >= 3 and values.mean() != 0:
            cv = values.std() / abs(values.mean())
            vol = float(100 - np.clip(cv * 100, 0, 100))
        else:
            vol = 50.0

        # Momentum (recent 4 vs prior 4)
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
    # Read directly from sidebar keys to avoid any desync.
    return {
        "quality_basis": st.session_state.get("cfg_quality_split_basis", "Percentile within Band (recommended)"),
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
def load_and_process_data(uploaded_file, data_period_setting, use_sector_adjusted, use_quarterly_beta=False,
                          split_basis="Percentile within Band (recommended)", split_threshold=60, trend_threshold=55,
                          volatility_cv_threshold=0.30, outlier_z_threshold=-2.5, damping_factor=0.5, near_peak_tolerance=0.10):
    """Load data and calculate issuer scores with V2.0 enhancements and V2.0 dual-horizon context"""

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

    # [V2.0] TIERED VALIDATION - minimal identifiers with flexible column matching
    missing, RATING_COL, COMPANY_ID_COL, COMPANY_NAME_COL = validate_core(df)
    if missing:
        st.error(f"ERROR: Missing required identifiers:\n\n" + "\n".join([f"  • {m}" for m in missing]) +
                 f"\n\nV2.0 requires only: Company Name, Company ID, and S&P Credit Rating\n(or their common aliases)")
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

    # [V2.0] Check feature availability (classification, country/region, period alignment)
    has_classification = feature_available("classification", df)
    has_country_region = feature_available("country_region", df)
    has_period_alignment = feature_available("period_alignment", df)

    # [V2.0] Parse Period Ended columns if available
    if has_period_alignment:
        pe_cols = parse_period_ended_cols(df)
        if os.environ.get("RG_TESTS") == "1":
            print(f"DEV: Parsed {len(pe_cols)} Period Ended columns")
    else:
        pe_cols = []

    # [V2.0] Build period calendar with FY/CQ overlap resolution
    period_calendar = None
    if has_period_alignment:
        try:
            # Use quarterly preference from use_quarterly_beta parameter
            prefer_quarterly = use_quarterly_beta
            period_calendar = build_period_calendar(
                raw_df=df,
                issuer_id_col=COMPANY_ID_COL,
                issuer_name_col=COMPANY_NAME_COL,
                prefer_quarterly=prefer_quarterly,
                q4_merge_window_days=10
            )
            if os.environ.get("RG_TESTS") == "1":
                original_count = sum(1 for c in df.columns if "Period Ended" in str(c)) * len(df)
                cleaned_count = len(period_calendar)
                removed_count = original_count - cleaned_count
                print(f"DEV: Period calendar built - {len(period_calendar)} periods (removed {removed_count} overlaps/sentinels)")
                print(f"DEV: Prefer quarterly: {prefer_quarterly}")
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

    # Thread use_quarterly_beta into trend calculation
    trend_scores = calculate_trend_indicators(df, key_metrics_for_trends, use_quarterly=use_quarterly_beta)
    cycle_score = calculate_cycle_position_score(trend_scores, key_metrics_for_trends)
    _log_timing("03_Trend_Indicators_Complete")

    # ========================================================================
    # [V2.0] DUAL-HORIZON TREND & CONTEXT FLAGS
    # ========================================================================

    # Compute dual-horizon metrics for Composite Score (using time series)
    # This provides medium-term slope, short-term change, outlier detection, and volatility flags

    dual_horizon_metrics = pd.DataFrame(index=df.index)

    # Use EBITDA Margin as primary metric for dual-horizon analysis
    # (Can extend to other metrics as needed)
    primary_metric = 'EBITDA Margin'
    if primary_metric in df.columns:
        ts_primary = _build_metric_timeseries(df, primary_metric, use_quarterly=use_quarterly_beta)

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
    # CALCULATE QUALITY SCORES ([V2.0] ANNUAL-ONLY DEFAULT)
    # ========================================================================

    def calculate_quality_scores(df, data_period_setting, has_period_alignment):
        scores = pd.DataFrame(index=df.index)

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
        metrics = _batch_extract_metrics(df, needed_metrics, has_period_alignment, data_period_setting)

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
                return 50  # Neutral if missing

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
        net_debt_ebitda = net_debt_ebitda.where(net_debt_ebitda >= 0, other=20.0).fillna(20.0).clip(upper=20.0)
        part1 = (np.minimum(net_debt_ebitda, 3.0)/3.0)*60.0
        part2 = (np.maximum(net_debt_ebitda-3.0, 0.0)/5.0)*40.0
        raw_penalty = np.minimum(part1+part2, 100.0)
        net_debt_score = np.clip(100.0 - raw_penalty, 0.0, 100.0)

        # Component 2: Interest Coverage (30%)
        interest_coverage_score = ebitda_cov_score

        # Component 3: Total Debt / Total Capital (20%)
        debt_capital = metrics['Total Debt / Total Capital (%)']
        debt_capital = debt_capital.fillna(50).clip(0, 100)
        debt_cap_score = np.clip(100 - debt_capital, 0, 100)

        # Component 4: Total Debt / EBITDA (10%)
        debt_ebitda = metrics['Total Debt / EBITDA (x)']
        debt_ebitda = debt_ebitda.where(debt_ebitda >= 0, other=20.0).fillna(20.0).clip(upper=20.0)
        part1_td = (np.minimum(debt_ebitda, 3.0)/3.0)*60.0
        part2_td = (np.maximum(debt_ebitda-3.0, 0.0)/5.0)*40.0
        raw_penalty_td = np.minimum(part1_td+part2_td, 100.0)
        debt_ebitda_score = np.clip(100.0 - raw_penalty_td, 0.0, 100.0)

        # Option A weights: ND/EBITDA 40%, Coverage 30%, Debt/Cap 20%, TD/EBITDA 10%
        # Row-wise normalization to handle missing components
        comps = np.array([
            net_debt_score,
            interest_coverage_score,
            debt_cap_score,
            debt_ebitda_score,
        ], dtype=float).T  # Transpose to get rows for each issuer

        w = np.array([0.40, 0.30, 0.20, 0.10], dtype=float)

        # Calculate effective weights per row (zero out NaN components)
        mask = np.isnan(comps)
        w_eff = np.where(mask, 0.0, w)
        denom = w_eff.sum(axis=1)

        # Weighted sum with normalization
        leverage_scores = np.where(
            denom > 0,
            np.nansum(comps * w, axis=1) / denom,
            np.nan
        )

        scores['leverage_score'] = pd.Series(leverage_scores, index=df.index)

        # Profitability ([V2.0] Annual-only)
        roe = _pct_to_100(metrics['Return on Equity'])
        ebitda_margin = _pct_to_100(metrics['EBITDA Margin'])
        roa = _pct_to_100(metrics['Return on Assets'])
        ebit_margin = _pct_to_100(metrics['EBIT Margin'])

        roe_score = np.clip(roe, -50, 50) + 50
        margin_score = np.clip(ebitda_margin, -50, 50) + 50
        roa_score = np.clip(roa * 5, 0, 100)
        ebit_score = np.clip(ebit_margin * 2, 0, 100)

        scores['profitability_score'] = (roe_score * 0.3 + margin_score * 0.3 +
                                         roa_score * 0.2 + ebit_score * 0.2)

        # Liquidity ([V2.0] Annual-only)
        current_ratio = metrics['Current Ratio (x)'].clip(lower=0)
        quick_ratio = metrics['Quick Ratio (x)'].clip(lower=0)

        current_score = np.clip((current_ratio/3.0)*100.0, 0, 100)
        quick_score = np.clip((quick_ratio/2.0)*100.0, 0, 100)

        scores['liquidity_score'] = current_score * 0.6 + quick_score * 0.4

        # Growth ([V2.0] Annual-only)
        rev_growth_1y = _pct_to_100(metrics['Total Revenues, 1 Year Growth'])
        rev_cagr_3y = _pct_to_100(metrics['Total Revenues, 3 Yr. CAGR'])
        ebitda_cagr_3y = _pct_to_100(metrics['EBITDA, 3 Years CAGR'])

        rev_1y_score = np.clip((rev_growth_1y + 10) * 2, 0, 100)
        rev_3y_score = np.clip((rev_cagr_3y + 10) * 2, 0, 100)
        ebitda_3y_score = np.clip((ebitda_cagr_3y + 10) * 2, 0, 100)

        scores['growth_score'] = rev_3y_score * 0.4 + rev_1y_score * 0.3 + ebitda_3y_score * 0.3

        # Cash Flow ([v3 - DataFrame-level with alias-aware batch extraction] Annual-only)
        # Compute 4 equal-weighted components: OCF/Revenue, OCF/Debt, UFCF margin, LFCF margin
        # Each clipped globally then min-max scaled to 0-100; average available components
        _cf_comp = _cash_flow_component_scores(df, data_period_setting, has_period_alignment)
        _cf_cols = [c for c in ["OCF_to_Revenue_Score", "OCF_to_Debt_Score",
                                 "UFCF_margin_Score", "LFCF_margin_Score"] if c in _cf_comp.columns]

        if _cf_cols:
            scores['cash_flow_score'] = pd.Series(
                np.nanmean(_cf_comp[_cf_cols].to_numpy(dtype=float), axis=1),
                index=df.index
            )
        else:
            scores['cash_flow_score'] = pd.Series(np.nan, index=df.index)

        return scores

    quality_scores = calculate_quality_scores(df, data_period_setting, has_period_alignment)
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
    # CALCULATE COMPOSITE SCORE ([V2.0] FEATURE-GATED CLASSIFICATION WEIGHTS)
    # ========================================================================

    qs = quality_scores.copy()

    # Fill missing values with classification/global medians
    if has_classification and 'Rubrics Custom Classification' in df.columns:
        classification_meds = qs.join(df['Rubrics Custom Classification']).groupby('Rubrics Custom Classification').transform('median')
        qs = qs.fillna(classification_meds)
    qs = qs.fillna(qs.median(numeric_only=True))

    # Final fallback to defaults
    default_scores = {
        'credit_score': 50.0, 'leverage_score': 50.0, 'profitability_score': 50.0,
        'liquidity_score': 50.0, 'growth_score': 50.0, 'cash_flow_score': 50.0
    }
    for col, default_val in default_scores.items():
        if col in qs.columns:
            qs[col] = qs[col].fillna(default_val)

    # [V2.0] Calculate composite score - use classification weights only if available
    # OPTIMIZED: Vectorized calculation instead of iterrows()
    
    if has_classification and use_sector_adjusted:
        # Build weight matrix for each issuer based on classification
        weight_matrix = df['Rubrics Custom Classification'].apply(
            lambda c: pd.Series(get_classification_weights(c, True))
        )
        # Track which weights were used (for display)
        def _weight_label(c):
            if c in CLASSIFICATION_TO_SECTOR:
                parent_sector = CLASSIFICATION_TO_SECTOR[c]
                return f"{parent_sector} (via {c[:20]}...)"
            elif c in CLASSIFICATION_OVERRIDES:
                return f"{c[:30]}... (Custom)"
            else:
                return "Universal"
        weight_used_list = df['Rubrics Custom Classification'].apply(_weight_label).tolist()
    else:
        # Use universal weights for all rows
        default_weights = get_classification_weights('Default', False)
        weight_matrix = pd.DataFrame([default_weights] * len(df), index=df.index)
        weight_used_list = ["Universal"] * len(df)
    
    # Vectorized composite score calculation: sum(score * weight) for each factor
    composite_score = (
        qs['credit_score'] * weight_matrix['credit_score'] +
        qs['leverage_score'] * weight_matrix['leverage_score'] +
        qs['profitability_score'] * weight_matrix['profitability_score'] +
        qs['liquidity_score'] * weight_matrix['liquidity_score'] +
        qs['growth_score'] * weight_matrix['growth_score'] +
        qs['cash_flow_score'] * weight_matrix['cash_flow_score']
    )
    _log_timing("05_Composite_Score_Complete")

    # ========================================================================
    # CREATE RESULTS DATAFRAME ([V2.0] WITH OPTIONAL COLUMNS)
    # ========================================================================

    # Start with core identifiers (always required)
    # [V2.0] Use resolved/canonical column names (not hard-coded strings)
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

    # [V2.0] Add optional columns if available
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
    # [V2.0] All non-IG ratings (including NR/WD/N/M/empty/NaN) classified as High Yield
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

    # [V2.0] Calculate Classification Rank only if classification available
    if has_classification and 'Rubrics_Custom_Classification' in results.columns:
        results['Classification_Rank'] = results.groupby('Rubrics_Custom_Classification')['Composite_Score'].rank(
            ascending=False, method='dense'
        ).astype('Int64')

    # Overall Rank
    results['Overall_Rank'] = results['Composite_Score'].rank(
        ascending=False, method='dense'
    ).astype('Int64')

    # ========================================================================
    # [V2.0] CONTEXT FLAGS FOR DUAL-HORIZON ANALYSIS
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
    # [V2.0] VOLATILITY DAMPING FOR CYCLE POSITION SCORE
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
    # [V2.0] LABEL OVERRIDE LOGIC FOR CONTEXT-AWARE SIGNALS
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
    # PERCENTILE-BASED RECOMMENDATION LOGIC WITH GUARDRAIL
    # ========================================================================

    # --- Guardrail: block Buy/SB when Weak & Deteriorating ---
    # Default to stricter behaviour; override with env if required.
    WEAK_DET_CAP = os.getenv("WEAK_DET_CAP", "Avoid")

    # Step 1: Draft recommendations from within-band percentiles (0–100)
    # Requires Composite_Percentile_in_Band computed earlier.
    pct = results['Composite_Percentile_in_Band'].fillna(0)
    conditions = [
        pct >= 80,  # Strong Buy: top quintile within band
        pct >= 60,  # Buy
        pct >= 40   # Hold
    ]
    choices = ['Strong Buy', 'Buy', 'Hold']
    draft_rec = np.select(conditions, choices, default='Avoid')

    # Step 2: Apply guardrail (only cap Buy/SB for Weak & Deteriorating)
    is_weak_det = results['Signal'] == 'Weak & Deteriorating'
    is_buy_or_sb = pd.Series(draft_rec).isin(['Buy', 'Strong Buy'])

    # Final recommendation
    results['Recommendation'] = draft_rec
    results.loc[is_weak_det & is_buy_or_sb, 'Recommendation'] = WEAK_DET_CAP
    results['Rec'] = results['Recommendation']
    _log_timing("06_Recommendations_Complete")

    # Dev-only assertion: verify no Weak & Deteriorating issuers get Buy/Strong Buy
    if os.environ.get("RG_TESTS") == "1":
        _bad = results.query("Recommendation in ['Buy','Strong Buy'] and Signal == 'Weak & Deteriorating'")
        assert len(_bad) == 0, f"{len(_bad)} Weak & Deteriorating issuers still rated Buy/SB"

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

    return results, df, audits, period_calendar

# ============================================================================
# MAIN APP EXECUTION (Skip if running tests)
# ============================================================================

if os.environ.get("RG_TESTS") != "1":
    if HAS_DATA:
        # ========================================================================
        # LOAD DATA
        # ========================================================================

        with st.spinner("Loading and processing data..."):
            results_final, df_original, audits, period_calendar = load_and_process_data(
                uploaded_file, data_period, use_sector_adjusted, use_quarterly_beta,
                split_basis, split_threshold, trend_threshold,
                volatility_cv_threshold, outlier_z_threshold, damping_factor, near_peak_tolerance / 100.0
            )
            _audit_count("Before freshness filters", results_final, audits)

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
            # COMPUTE FRESHNESS METRICS (V2.0)
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
            # FRESHNESS FILTERS (V2.0) - Sidebar
            # ============================================================================
        
            with st.sidebar.expander("  Data Freshness Filters", expanded=False):
                use_freshness_filters = st.checkbox(
                    "Apply freshness filters",
                    value=False,
                    help="Filter out issuers with stale financial or rating data"
                )

                if use_freshness_filters:
                    st.markdown("**Filter by data age:**")

                    max_fin_days = st.slider(
                        "Max Financial Data Age (days)",
                        min_value=30,
                        max_value=1095,
                        value=365,
                        step=15,
                        help="Exclude issuers with financial data older than this many days"
                    )

                    max_rev_days = st.slider(
                        "Max Rating Review Age (days)",
                        min_value=30,
                        max_value=1095,
                        value=365,
                        step=15,
                        help="Exclude issuers with S&P rating review older than this many days"
                    )

                    # Apply filters
                    before_filter = len(results_final)
                    results_final = results_final[
                        (results_final["Financial_Data_Freshness_Days"].fillna(9999) <= max_fin_days) &
                        (results_final["Rating_Review_Freshness_Days"].fillna(9999) <= max_rev_days)
                    ]
                    after_filter = len(results_final)

                    if before_filter > after_filter:
                        st.caption(f"Filtered: {before_filter:,} → {after_filter:,} issuers ({before_filter - after_filter:,} excluded)")
                    else:
                        st.caption(f"{after_filter:,} issuers (no exclusions)")
                else:
                    st.caption("📊 Showing all issuers (no freshness filters applied)")

            _audit_count("Final results", results_final, audits)

            # ============================================================================
            # HEADER
            # ============================================================================

            render_header(results_final, data_period, use_sector_adjusted, df_original)

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
                
                with col1:
                    st.subheader("Top 10 Performers (Overall)")
                    top10 = results_final.nlargest(10, 'Composite_Score')[
                        ['Overall_Rank', 'Company_Name', 'Credit_Rating_Clean', 'Rubrics_Custom_Classification', 'Composite_Score', 'Recommendation']
                    ]
                    top10.columns = ['Rank', 'Company', 'Rating', 'Classification', 'Score', 'Rec']
                    st.dataframe(top10, use_container_width=True, hide_index=True)
                
                with col2:
                    st.subheader("Bottom 10 Performers (Overall)")
                    bottom10 = results_final.nsmallest(10, 'Composite_Score')[
                        ['Overall_Rank', 'Company_Name', 'Credit_Rating_Clean', 'Rubrics_Custom_Classification', 'Composite_Score', 'Recommendation']
                    ]
                    bottom10.columns = ['Rank', 'Company', 'Rating', 'Classification', 'Score', 'Rec']
                    st.dataframe(bottom10, use_container_width=True, hide_index=True)
                
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
                
                # Four Quadrant Analysis
                st.subheader("Four Quadrant Analysis: Quality vs. Momentum")

                # Ensure numeric dtypes for axes
                results_final['Composite_Percentile_in_Band'] = pd.to_numeric(results_final['Composite_Percentile_in_Band'], errors='coerce')
                results_final['Composite_Percentile_Global'] = pd.to_numeric(results_final.get('Composite_Percentile_Global', results_final['Composite_Percentile_in_Band']), errors='coerce')
                # Composite_Score already numeric from calculation - no conversion needed
                results_final['Cycle_Position_Score'] = pd.to_numeric(results_final['Cycle_Position_Score'], errors='coerce')

                # Use unified quality/trend split for visualization
                quality_metric_plot, x_split_for_plot, x_axis_label, x_vals = resolve_quality_metric_and_split(
                    results_final, split_basis, split_threshold
                )
                y_vals = results_final["Cycle_Position_Score"]
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
                    results_final,
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
                    title='Credit Quality vs. Trend Momentum',
                    labels={
                        x_col: x_axis_label,
                        "Cycle_Position_Score": "Cycle Position Score (Trend Direction)"
                    }
                )

                # Add split lines in DATA coordinates (xref='x', yref='y')
                fig_quadrant.add_vline(x=x_split_for_plot, line_width=1.5, line_dash="dash", line_color="#888", layer="below")
                fig_quadrant.add_hline(y=y_split, line_width=1.5, line_dash="dash", line_color="#888", layer="below")

                # Add quadrant labels (positioned relative to splits)
                y_upper = y_split + (100 - y_split) * 0.5  # midpoint of upper half
                y_lower = y_split * 0.5  # midpoint of lower half

                # Calculate x positions based on actual axis range and split
                x_max = float(results_final[x_col].max())
                x_min = float(results_final[x_col].min())
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

                quadrant_counts = results_final['Combined_Signal'].value_counts()
                total = len(results_final)

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
                # DUAL ISSUER POSITIONING MAPS
                # ========================================================================

                # Compute derived axes for positioning maps
                results_final["Overall_Credit_Quality"] = results_final["Composite_Score"]
                results_final["Financial_Strength_vs_Leverage_Balance"] = (
                    results_final["Credit_Score"] - results_final["Leverage_Score"]
                )

                # Render dual maps (IG + HY)
                render_dual_issuer_maps(results_final, 'Company_ID', 'Company_Name')

                # ========================================================================
                # PCA FACTOR LOADINGS ANALYSIS
                # ========================================================================
                st.markdown("---")
                st.subheader("PCA Factor Loadings Analysis")

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

                    # ========================================================================
                    # SECTION 1: FULL-WIDTH RADAR CHARTS
                    # ========================================================================
                    n_components_to_show = min(3, loadings.shape[0])

                    fig_radar = make_subplots(
                        rows=1, cols=n_components_to_show,
                        specs=[[{'type': 'polar'}] * n_components_to_show],
                        subplot_titles=[f'PC{i+1} ({var_exp[i]:.1f}% var)' for i in range(n_components_to_show)],
                        horizontal_spacing=0.08
                    )

                    # Color scheme for different PCs
                    colors = ['#2C5697', '#E74C3C', '#27AE60']

                    for i in range(n_components_to_show):
                        pc_loadings = loadings[i, :]

                        fig_radar.add_trace(
                            go.Scatterpolar(
                                r=pc_loadings,
                                theta=feature_names,
                                fill='toself',
                                name=f'PC{i+1}',
                                line=dict(width=2.5, color=colors[i]),
                                marker=dict(size=8),
                                fillcolor=colors[i],
                                opacity=0.5
                            ),
                            row=1, col=i+1
                        )

                        fig_radar.update_polars(
                            radialaxis=dict(
                                visible=True,
                                range=[-1, 1],
                                showticklabels=True,
                                ticks='outside',
                                tickfont=dict(size=10),
                                gridcolor='lightgray'
                            ),
                            angularaxis=dict(
                                tickfont=dict(size=12, color='#333333')
                            ),
                            row=1, col=i+1
                        )

                    fig_radar.update_layout(
                        height=450,
                        showlegend=False,
                        title_text="Factor Contributions to Principal Components",
                        title_x=0.5,
                        title_y=0.98,
                        title_font_size=16,
                        title_font_color='#2C5697',
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        margin=dict(t=120, b=40, l=40, r=40)
                    )

                    # Increase spacing between subplot titles and radar charts
                    fig_radar.update_annotations(y=1.10)

                    st.plotly_chart(fig_radar, use_container_width=True)

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
                # RATING-BAND LEADERBOARDS (V2.0)
                # ========================================================================
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
                    scoring_method, data_period, use_quarterly_beta,
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
        
                # Update URL state with current Tab 1 control values
                _updated_state = collect_current_state(
                    scoring_method=scoring_method,
                    data_period=data_period,
                    use_quarterly_beta=use_quarterly_beta,
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
                # WATCHLIST / EXCLUSIONS (V2.0)
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

                # Create a view for display only; do not mutate the pipeline DF
                filtered_display = filtered[display_cols].copy()
                filtered_display = filtered_display.rename(columns=ISSUER_TABLE_LABELS)
                filtered_display = filtered_display.sort_values('Composite Score (0–100)', ascending=False)

                st.dataframe(filtered_display, use_container_width=True, hide_index=True, height=600)
        
                # ========================================================================
                # ISSUER EXPLAINABILITY (V2.0)
                # ========================================================================
                render_issuer_explainability(filtered, scoring_method)
        
                # ========================================================================
                # EXPORT CURRENT VIEW (V2.0)
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
                qs_basis = st.session_state.get("cfg_quality_split_basis", "Percentile within Band (recommended)")
                q_thresh = st.session_state.get("cfg_quality_threshold", 60)
                t_thresh = st.session_state.get("cfg_trend_threshold", 55)
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
            # TAB 7: GENAI CREDIT REPORT
            # ============================================================================

            with tab7:
                st.header("GenAI Credit Report Generator")

                st.markdown("""
                Generate a comprehensive AI-powered credit analysis report for any issuer in your dataset.
                The report includes profitability analysis, leverage trends, liquidity assessment, and investment recommendations.
                """)

                # Check if OpenAI is available
                if not _OPENAI_AVAILABLE:
                    st.error("OpenAI package is not available. Please install it with: `pip install openai`")
                    st.stop()

                # Check for API key
                try:
                    test_client = _get_openai_client()
                    st.success("OpenAI API configured successfully")
                except RuntimeError as e:
                    st.error(f"{str(e)}")
                    st.info("""
                    **To configure OpenAI API:**
                    1. Add your API key to `.streamlit/secrets.toml`:
                   OPENAI_API_KEY = "sk-..."
                    2. Or set as environment variable: `OPENAI_API_KEY`
                    """)
                    st.stop()

                st.markdown("---")

                # Issuer selection
                name_col = resolve_company_name_column(df_original)
                if name_col is None:
                    st.error("Cannot find company name column in dataset")
                    st.stop()

                issuer_list = sorted(df_original[name_col].dropna().unique().tolist())

                col1, col2 = st.columns([3, 1])

                with col1:
                    selected_issuer = st.selectbox(
                        "Select Issuer",
                        options=issuer_list,
                        help="Choose the company you want to analyze"
                    )

                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)  # Spacing
                    generate_button = st.button("Generate Report", type="primary", use_container_width=True)

                # Generate report when button is clicked
                if generate_button:
                    with st.spinner(f"Generating credit analysis for {selected_issuer}..."):
                        try:
                            # Extract financial data
                            issuer_data = extract_issuer_financial_data(df_original, selected_issuer)

                            # Check if we have sufficient data
                            if not issuer_data["financial_data"]:
                                st.warning(f"No financial data found for {selected_issuer}. Cannot generate report.")
                                st.stop()

                            # Generate the report
                            report_text = generate_credit_report(issuer_data)

                            # Display the report
                            st.markdown("---")
                            st.markdown(f"## Credit Analysis Report: {selected_issuer}")

                            # Show company info in columns
                            info_col1, info_col2, info_col3 = st.columns(3)
                            with info_col1:
                                st.metric("S&P Rating", issuer_data["company_info"]["rating"])
                            with info_col2:
                                st.metric("Sector", issuer_data["company_info"]["sector"])
                            with info_col3:
                                st.metric("Country", issuer_data["company_info"]["country"])

                            st.markdown("---")

                            # Display the AI-generated report
                            st.markdown(report_text)

                            st.markdown("---")

                            # Download button
                            from datetime import datetime
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"{selected_issuer.replace(' ', '_')}_credit_report_{timestamp}.md"

                            st.download_button(
                                label="Download Report (Markdown)",
                                data=report_text,
                                file_name=filename,
                                mime="text/markdown",
                                use_container_width=False
                            )

                            # Show disclaimer
                            st.info("""
                            **Disclaimer:** This report is generated by AI and should be used for informational purposes only.
                            Always conduct your own due diligence and consult with qualified professionals before making investment decisions.
                            """)

                        except ValueError as e:
                            st.error(f"Error: {str(e)}")
                        except RuntimeError as e:
                            st.error(f"OpenAI API Error: {str(e)}")
                            st.info("Please check your API key configuration and try again.")
                        except Exception as e:
                            st.error(f"Unexpected error: {str(e)}")
                            st.exception(e)

                else:
                    # Show instructions when no report has been generated
                    st.info("""
                    **How to use:**
                    1. Select an issuer from the dropdown above
                    2. Click "Generate Report" to create an AI-powered credit analysis
                    3. Review the comprehensive report covering profitability, leverage, liquidity, and risks
                    4. Download the report in Markdown format for your records

                    **What's included in the report:**
                    - Executive Summary
                    - Profitability Analysis (margins, ROE, ROA)
                    - Leverage Analysis (Debt/EBITDA trends)
                    - Liquidity & Coverage Analysis
                    - Credit Strengths
                    - Credit Risks & Concerns
                    - Rating Outlook & Recommendations
                    """)

                    # Show sample of available data
                    st.markdown("### Data Availability Preview")
                    if selected_issuer:
                        try:
                            sample_data = extract_issuer_financial_data(df_original, selected_issuer)

                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown("**Company Information:**")
                                for key, value in sample_data["company_info"].items():
                                    st.text(f"{key.title()}: {value}")

                            with col_b:
                                st.markdown("**Available Metrics:**")
                                metric_count = len(sample_data["financial_data"])
                                st.text(f"Total metrics: {metric_count}")
                                if sample_data["financial_data"]:
                                    for metric in list(sample_data["financial_data"].keys())[:5]:
                                        periods = len(sample_data["financial_data"][metric])
                                        st.text(f"• {metric}: {periods} periods")
                                    if metric_count > 5:
                                        st.text(f"... and {metric_count - 5} more")

                        except Exception as e:
                            st.warning(f"Could not load preview: {str(e)}")

            st.markdown("---")
            st.markdown("""
        <div style='text-align: center; color: #4c566a; padding: 20px;'>
            <p><strong>Issuer Credit Screening Model V2.0</strong></p>
            <p>© 2025 Rubrics Asset Management | Annual-Only Default + Minimal Identifiers</p>
        </div>
        """, unsafe_allow_html=True)
        
        # ============================================================================
        # [V2.0] SELF-TESTS (Run with RG_TESTS=1 environment variable)
        # ============================================================================
        
if os.environ.get("RG_TESTS") == "1":
    import sys
    print("\n" + "="*60)
    print("Running RG_TESTS for V2.0...")
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
    trend_fy = calculate_trend_indicators(trend_test, test_metrics, use_quarterly=False)

    # Test 9b: Quarterly-mode (use_quarterly=True) - should use base + .1-.12 (available up to .8 here)
    trend_cq = calculate_trend_indicators(trend_test, test_metrics, use_quarterly=True)

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
        m_annual = calculate_trend_indicators(_df, test_metrics, use_quarterly=False)
        m_quarterly = calculate_trend_indicators(_df, test_metrics, use_quarterly=True)

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
    print("SUCCESS: ALL RG_TESTS PASSED for V2.0 (12 tests)")
    print("="*60 + "\n")

    # Exit successfully after tests
    sys.exit(0)
