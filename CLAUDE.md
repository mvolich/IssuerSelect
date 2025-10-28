# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Issuer Credit Screening Model v2.3** - A Streamlit-based application for analyzing corporate credit quality using a 6-factor composite scoring system with sector-adjusted weights, trend analysis, and PCA-based cohort clustering.

## Running the Application

```bash
# Set environment for testing (disables Streamlit UI initialization)
set RG_TESTS=1

# Run application (normal mode)
streamlit run app.py

# Run embedded test suite
set RG_TESTS=1 && python app.py
```

## Architecture

### Single-File Structure
The entire application is contained in `app.py` (~4000 lines). This is intentional for deployment simplicity to Streamlit Cloud.

### Core Scoring System

**6 Factor Scores** (each 0-100):
1. `credit_score` - Based on S&P rating mapped to numeric scale
2. `leverage_score` - Net Debt / EBITDA (inverse scoring)
3. `profitability_score` - EBITDA Margin
4. `liquidity_score` - Current Ratio
5. `growth_score` - Revenue CAGR (multi-period trend)
6. `cash_flow_score` - Free Cash Flow / Total Debt

**Composite Score**: Weighted average of the 6 factors (0-100 scale).

### Weight Resolution System

Factor weights vary by issuer classification following this hierarchy:

1. **CLASSIFICATION_OVERRIDES** (app.py:1319) - Custom weights for specific classifications
2. **CLASSIFICATION_TO_SECTOR** (app.py:1267) - Maps 30+ classifications to 10 parent sectors
3. **SECTOR_WEIGHTS** (app.py:1171) - Sector-specific factor weights
4. **Default weights** - Universal fallback if no classification provided

Example: "Software and Services" → "Information Technology" sector → uses IT sector weights (high growth/cash flow emphasis, lower leverage weight).

### Period Classification System

Data columns use suffix notation:
- **Base column** (e.g., "EBITDA Margin") = most recent period
- **Historical periods** (e.g., "EBITDA Margin.1", ".2", ".3", ".4") = prior periods
- **Quarterly data** (optional): ".5" through ".12"

The system automatically classifies periods as FY (fiscal year) or CQ (calendar quarter) based on:
1. "Period Ended" column dates (preferred method)
2. Fallback heuristic: first 5 periods = FY, remainder = CQ

**Key functions**:
- `parse_period_ended_cols()` (app.py:825) - Parse and sort period columns by actual dates
- `get_metric_series_row()` (app.py:918) - Extract FY-only or CQ-only time series for a metric
- `get_most_recent_column()` (app.py:1667) - Resolve correct column based on data period setting

### Trend Analysis

Two modes controlled by `use_quarterly_beta`:
- **Annual mode** (False): Uses base + .1-.4 periods (5 FY periods)
- **Quarterly mode** (True): Uses base + .1-.12 periods (13 periods if available)

**Cycle Position Score** (app.py:1859): Composite of 3 trend metrics (Growth, Profitability, Leverage) weighted 40%/30%/30%.

### Data Validation

**Required columns** (flexible name matching via aliases):
- Company Name/ID (COMPANY_NAME_ALIASES, COMPANY_ID_ALIASES at app.py:94-115)
- S&P LT Issuer Credit Rating (RATING_ALIASES at app.py:86-93)

**Optional features** (gated by REQ_FOR at app.py:118):
- `classification`: Requires "Rubrics Custom Classification" column
- `country_region`: Requires "Country" and "Region" columns
- `period_alignment`: Requires "Period Ended" column

Functions `resolve_column()` and `validate_core()` (app.py:138-181) handle robust column name matching (case-insensitive, handles NBSP, extra spaces).

### PCA Visualization

`compute_pca_ig_hy()` (app.py:228) and `render_pca_scatter_ig_hy()` (app.py:265) create 2D PCA plots showing:
- Investment Grade (IG) vs High Yield (HY) clustering
- Uses 6 factor scores as input features
- RobustScaler normalization before PCA
- Subsamples to 2000 points max for performance

### AI Analysis (Optional)

Requires OpenAI API key in `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "sk-..."
```

Functions: `_get_openai_client()`, `_run_ai()` (app.py:635-751)

## Testing

Embedded test suite runs when `RG_TESTS=1` environment variable is set.

**Test coverage** (app.py:3708-4001):
1. Period classification with labeled dates
2. 1900 sentinel date handling
3. Classification weight validation (all sum to 1.0)
4. CQ exclusion from FY series
5. Annual value extraction for Interest Coverage
6. Rating group classification (IG vs HY)
7. Column alias resolution and canonicalization
8. Date-based period selection
9. Quarterly vs annual trend window selection
10. Period vs trend window control separation

Run tests:
```bash
set RG_TESTS=1 && python app.py
```

## State Management

**URL Query Parameters** (app.py:757):
- `scoring_method` - Universal or Classification-Adjusted
- `data_period` - FY0, CQ-0, or trailing periods
- `use_quarterly_beta` - Trend window mode
- `classification_filter`, `band_filter`, `region_filter`
- `min_score`, `max_score`

Functions:
- `collect_current_state()` (app.py:783) - Capture current UI state
- `apply_state_to_controls()` (app.py:795) - Restore state from dict
- `_build_deep_link()` (app.py:813) - Generate shareable URL

**Presets**: Saved in `st.session_state.presets` as JSON-serializable dicts.

## Key Data Flow

1. **Upload** → `load_and_process_data()` (app.py:1896)
2. **Column validation** → `validate_core()` (app.py:162)
3. **Period parsing** → `parse_period_ended_cols()` (app.py:825)
4. **Metric extraction** → `get_most_recent_column()` per user's data period setting
5. **Trend calculation** → `calculate_trend_indicators()` (app.py:1765)
6. **Cycle scoring** → `calculate_cycle_position_score()` (app.py:1859)
7. **Weight resolution** → `get_classification_weights()` (app.py:1342)
8. **Composite scoring** → Weighted average of 6 factors
9. **Rating band assignment** → `assign_rating_band()` (app.py:1388)
10. **Display** → Filters → PCA → Leaderboards → AI Analysis

## Code Style Notes

- **Testing mode**: Many sections check `os.environ.get("RG_TESTS")` to disable UI code during tests
- **Caching**: `@st.cache_data` on `load_and_process_data()` for performance
- **Auditing**: `_audit_count()` (app.py:1008) tracks row count at each processing stage
- **Diagnostics**: `diagnostics_summary()` (app.py:604) provides data health metrics

## Common Modifications

**Adding a new sector**:
1. Add entry to `SECTOR_WEIGHTS` (app.py:1171)
2. Update `CLASSIFICATION_TO_SECTOR` mapping if needed (app.py:1267)

**Adding a new factor**:
1. Extract metric in `load_and_process_data()`
2. Calculate score (0-100 scale)
3. Add to all sector weight dicts (must sum to 1.0)
4. Update composite score calculation
5. Update `_factor_score_columns()` for PCA (app.py:215)

**Modifying rating bands**:
- Edit `RATING_BANDS` dict (app.py:1378)
- Used by `assign_rating_band()` function

## Dependencies

Core:
- streamlit
- pandas
- numpy
- plotly (graph_objects, express)
- sklearn (RobustScaler, PCA)

Optional:
- openai (for AI analysis feature)
