"""
Shared utility functions for credit analytics application.

This module contains helper functions and constants used by both app.py and multi_agent_credit.py
to avoid circular import issues.
"""

import pandas as pd
import numpy as np
import unicodedata


# ============================================================================
# COLUMN RESOLUTION UTILITIES
# ============================================================================

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


# ============================================================================
# METRIC ALIAS REGISTRY
# ============================================================================

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


def resolve_metric_column(df_like, canonical: str) -> str | None:
    """Resolve metric column name using aliases."""
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


# ============================================================================
# SECTOR WEIGHTS
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


def get_classification_weights(classification, use_sector_adjusted=True, calibrated_weights=None):
    """
    Get factor weights for a Rubrics Custom Classification.

    Hierarchy:
    1. Check if classification has custom override weights
    2. Map to parent sector and use sector weights
    3. Fall back to Default if classification not found

    Args:
        classification: Rubrics Custom Classification value
        use_sector_adjusted: Whether to use adjusted weights (vs universal Default)
        calibrated_weights: Optional dict of dynamically calibrated weights (V2.2.1)

    Returns:
        Dictionary with 6 factor weights (summing to 1.0)
    """
    # Use calibrated weights if provided, otherwise use static weights
    weights_to_use = calibrated_weights if calibrated_weights is not None else SECTOR_WEIGHTS

    if not use_sector_adjusted:
        return weights_to_use.get('Default', SECTOR_WEIGHTS['Default'])

    # Step 1: Check for custom overrides
    if classification in CLASSIFICATION_OVERRIDES:
        return CLASSIFICATION_OVERRIDES[classification]

    # Step 2: Map to parent sector
    if classification in CLASSIFICATION_TO_SECTOR:
        parent_sector = CLASSIFICATION_TO_SECTOR[classification]
        if parent_sector in weights_to_use:
            return weights_to_use[parent_sector]

    # Step 3: Fall back to Default
    return weights_to_use.get('Default', SECTOR_WEIGHTS['Default'])
