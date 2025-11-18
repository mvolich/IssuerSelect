# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Streamlit-based credit analytics application** (V2.3) for evaluating fixed-income issuers using a 6-factor composite scoring model with trend analysis. The application processes financial data from Excel/CSV files and generates issuer rankings, visualizations, and AI-powered credit analysis.

**Main file:** `app.py` (~9,500 lines)

**Key V2.3 Features:**
- Unified period selection system (Latest Available vs Reference Aligned)
- Recommendation-based ranking (prioritizes Strong Buy > Buy > Hold > Avoid)
- Enhanced issuer explainability (shows original weights vs current calibration)
- Simplified UI (removed advanced debug controls)

## Running the Application

```bash
# Run the Streamlit app
streamlit run app.py

# Run tests (environment variable gates test execution)
set RG_TESTS=1
python app.py
```

The application expects Excel (.xlsx) or CSV files with financial data. It looks for a 'Pasted Values' sheet in Excel files by default, falling back to the first sheet if not found.

## Architecture Overview

### Core Pipeline (11 stages)

1. **Data Upload & Validation** - Validates core columns (Company ID, Name, S&P Rating)
2. **Period Parsing** - Extracts fiscal year (FY) and calendar quarter (CQ) dates from "Period Ended" columns
3. **Metric Extraction** - Pulls most recent values per user's data period setting
4. **Factor Scoring** - Transforms raw metrics into 0-100 scores for 6 factors
5. **Weight Resolution** - Applies Universal or Sector-Adjusted weights (stores original weights for explainability)
6. **Composite Score** - Weighted average → single 0-100 quality metric
7. **Trend Overlay** - Calculates Cycle Position Score from time-series momentum
8. **Signal Assignment** - Classifies into 4 quadrants (Strong/Weak × Improving/Deteriorating)
9. **Recommendation** - Percentile-based bands with guardrails (Strong Buy, Buy, Hold, Avoid)
10. **Overall Rank** - Recommendation-based ranking (Strong Buy first, then quality within tier)
11. **Visualization & Export** - Charts, leaderboards, AI analysis

### Six-Factor Model

The core scoring system evaluates issuers on:

1. **Credit Score** - S&P LT Issuer Rating (100%)
2. **Leverage Score** - Net Debt/EBITDA (40%), Interest Coverage (30%), Debt/Capital (20%), Total Debt/EBITDA (10%)
3. **Profitability Score** - EBITDA Margin, ROA, Net Margin
4. **Liquidity Score** - Current Ratio, Cash/Total Debt
5. **Growth Score** - Revenue CAGR, EBITDA growth (multi-period trend)
6. **Cash Flow Score** - OCF/Revenue, OCF/Debt, UFCF margin, LFCF margin

Each factor is scored 0-100. Higher is better (leverage is inverted).

### Weighting System

Two weighting modes:

- **Universal Weights** - Equal treatment across all sectors (Default: credit 20%, leverage 20%, profitability 20%, liquidity 10%, growth 15%, cash flow 15%)
- **Sector-Adjusted Weights** - Custom weights per GICS sector (e.g., Utilities emphasize cash flow 30%, credit 25%)

Dynamic calibration (V2.2.1) can calculate sector-specific weights from BBB-rated issuers to optimize classification accuracy.

### Period Selection System (V2.3)

**Unified Period Selection** replaces separate controls with two coherent modes:

1. **Latest Available (Maximum Currency)**
   - Uses most recent quarter (CQ-0) for quality scores
   - Uses quarterly data for trend analysis
   - Maximizes data freshness
   - Each issuer uses their latest available data

2. **Align to Reference Period**
   - Aligns all issuers to a common reference date
   - Enables true apples-to-apples comparison
   - User selects reference quarter with coverage indicators
   - Shows data coverage % for each available period

**Period Types:**
- **FY (Fiscal Year)** - Annual data (columns: base, .1, .2, .3, .4)
- **CQ (Calendar Quarter)** - Quarterly data (columns: base, .1-.12)

**Key Features:**
- **Auto Reference Date** - Recommends optimal reference date based on current date and reporting lags
- **FY/CQ Overlap Resolution** - Handles Q4 overlaps (CQ preferred when dates match within 10-day window)
- **Sentinel Date Handling** - Filters invalid dates (1900-01-01 used as missing data marker)
- **Coverage Indicators** - Shows % of issuers with data for each period

## Key Functions & Architecture

### Data Loading & Validation

- `load_and_process_data()` - Main entry point, cached with Streamlit (@st.cache_data)
- `validate_core()` - Validates required columns with flexible alias matching
- `parse_period_ended_cols()` - Extracts "Period Ended" columns
- `build_period_calendar()` - Creates FY/CQ calendar with overlap resolution

### Column Resolution System

Flexible column matching using aliases:
- `resolve_column()` - Generic alias resolver
- `resolve_rating_column()` - Finds S&P rating column
- `resolve_company_id_column()` - Finds company identifier
- `resolve_company_name_column()` - Finds company name
- `resolve_metric_column()` - Finds metric columns with numbered suffixes

### Metric Extraction

- `get_most_recent_column()` - Extracts single-period values (respects data_period_setting)
- `_build_metric_timeseries()` - Builds time series for trend analysis
- `_batch_extract_metrics()` - Bulk metric extraction with period alignment support
- `_metric_series_for_row()` - Row-level metric extraction with FY/CQ preference

### Scoring System

- `compute_raw_scores_v2()` - Main scoring function (generates all 6 factor scores)
- `_cf_components_dataframe()` - Cash flow component calculations
- `_cash_flow_component_scores()` - Cash flow factor scoring with clipping and scaling
- `assign_rating_band()` - Maps S&P ratings to IG/HY bands

### Trend Analysis

- `calculate_trend_indicators()` - Dual-horizon trend metrics (momentum, volatility, proximity to peak)
- `compute_dual_horizon_trends()` - Calculates short/long-term momentum
- `calculate_cycle_position_score()` - Converts trends into single 0-100 score
- `robust_slope()` - Median-based slope calculation (outlier-resistant)

### Weight Resolution

- `get_sector_weights()` - Returns weights for a sector
- `get_classification_weights()` - Returns weights for classification (supports calibrated weights)
- `calculate_calibrated_sector_weights()` - V2.2.1 dynamic calibration
- `_resolve_model_weights_for_row()` - Row-level weight resolution

### Signal Generation

- `build_buckets_v2()` - Assigns issuers to 4 quality-trend quadrants
- `_detect_signal()` - Determines signal based on thresholds
- `_compute_quality_metrics()` - Splits dataset into High/Low quality

### Ranking System (V2.3)

**Recommendation-Based Ranking** prioritizes actionability over pure quality:

- **Overall_Rank calculation** (Line ~7377-7393) - Uses recommendation priority first, then composite score
- **Recommendation priority mapping**: `{"Strong Buy": 4, "Buy": 3, "Hold": 2, "Avoid": 1}`
- **Sort logic**: Higher priority + higher score = lower rank number (Rank 1 = best)
- **Applied in**:
  - Dashboard "Top 10 Opportunities" (shows mostly Strong Buy/Buy)
  - Dashboard "Bottom 10 Risks" (shows mostly Avoid/Hold)
  - Issuer Search results table (Strong Buy at top)

**Benefits:**
- Top performers are actionable opportunities (not just high-quality deteriorating credits)
- Rankings align with investment decisions
- Incorporates both quality AND trend momentum

### Visualization

- `render_pca_scatter_ig_hy()` - PCA-based issuer map (IG/HY split)
- `compute_pca_ig_hy()` - Factor score PCA reduction
- `build_band_leaderboard()` - Top N issuers by rating band
- `render_issuer_explainability()` - Enhanced issuer-level breakdown with weight comparison (V2.3)
- `_build_explainability_table()` - Builds comparison table showing original vs current weights

### AI Analysis (Optional - requires OpenAI API key)

- `render_ai_analysis_chat()` - Interactive AI analyst interface
- `extract_issuer_financial_data()` - Extracts data for LLM prompt
- `build_credit_analysis_prompt()` - Generates structured credit analysis prompt
- `generate_credit_report()` - Produces professional credit report
- `assemble_issuer_context()` - Packages issuer data for AI
- `assemble_classification_context()` - Packages sector/classification data for AI

AI requires OpenAI API key in `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "sk-..."
```

### Reference Date System

- `get_reference_date()` - Auto-determines appropriate reference date based on current date and reporting lags
- `calculate_reference_date_coverage()` - Calculates % of issuers with data for each quarter
- `build_dynamic_period_labels()` - Generates user-friendly period labels (e.g., "Q4 2024 (87% coverage)")

### Enhanced Issuer Explainability (V2.3)

**Shows both original and current weights for full transparency:**

**Weight Storage** (Line ~7035-7041):
- Stores 6 weight columns during composite score calculation
- `Weight_Credit_Used`, `Weight_Leverage_Used`, `Weight_Profitability_Used`, `Weight_Liquidity_Used`, `Weight_Growth_Used`, `Weight_CashFlow_Used`
- Preserves original weights used in calculation for comparison

**Comparison Table** (Function: `_build_explainability_table`):
- **8 columns**: Factor, Score, Original Weight %, Current Weight %, Weight Change, Original Contrib, Current Contrib, Contrib Change
- Shows side-by-side comparison of weights used in calculation vs current calibration settings
- Calculates percentage change in weights and contribution impact

**Display Features** (Function: `render_issuer_explainability`):
- **Three-metric breakdown**: Stored Composite, Original Calculation, Current Calibration
- **Smart interpretation**:
  - ⚠️ Warning for significant impact (>5 points)
  - ℹ️ Info for moderate impact (1-5 points)
  - ✓ Success for minor impact (<1 point)
- **Educational captions** explaining how to read the comparison table
- **Graceful fallback** if original weights not stored

**Benefits:**
- Users see exactly how dynamic calibration affects each issuer
- Validates calculation correctness (original should match stored composite)
- Shows which factors benefit from sector-specific weighting
- Clear cause-and-effect between weight changes and score impact

## Testing

Tests are gated by `RG_TESTS` environment variable. Set to "1" to enable:

```bash
set RG_TESTS=1
python app.py
```

Tests validate:
- Period selection (FY vs CQ)
- Metric extraction with reference dates
- Sentinel date filtering
- Overlap de-duplication (FY/CQ)
- Trend window behavior
- LLM extractor FY/CQ classification

Tests run at module level (after function definitions, before Streamlit execution).

## Important Constants

```python
# Rating aliases (flexible column matching)
RATING_ALIASES = ["S&P LT Issuer Rating", "S&P Rating", ...]

# Company ID aliases
COMPANY_ID_ALIASES = ["Company ID", "Issuer ID", ...]

# Company name aliases
COMPANY_NAME_ALIASES = ["Company Name", "Issuer Name", ...]
```

## Caching Strategy

- `@st.cache_data` on `load_and_process_data()` - includes `_cache_buster` parameter to force recalculation when key parameters change
- Cache key includes: `use_quarterly_beta`, `align_to_reference`, `reference_date_override`, `use_dynamic_calibration`, `calibration_rating_band`

## File Structure

```
experiment/
├── app.py                    # Main application (~10k lines)
├── archive/
│   └── app.py               # Previous version
├── .streamlit/
│   └── secrets.toml         # OpenAI API key (not tracked)
└── __pycache__/
```

## Key UI Controls (Streamlit Sidebar)

### V2.3 Simplified Interface

**Core Settings:**
- **Scoring Method** - Universal vs Sector-Adjusted weights
- **Period Selection Mode** - Latest Available (Maximum Currency) vs Align to Reference Period
  - When "Align to Reference Period" selected: Reference Date dropdown with coverage indicators
  - Auto-recommends optimal reference date based on current date
- **Dynamic Calibration** - Optional BBB-based weight optimization (when Sector-Adjusted enabled)
- **Rating Filter** - All Issuers / Investment Grade / High Yield / Specific bands (AAA, AA, A, BBB, BB, B, CCC, Unrated)

**Fixed Parameters (V2.3):**
- Quality/Trend split thresholds: **60/55** (hardcoded at recommended defaults)
- Split basis: **Percentile within Band** (hardcoded)
- Dual-horizon context: **volatility_cv=0.30, outlier_z=-2.5, damping=0.5, near_peak_tolerance=10**

**Removed in V2.3** (UI simplification):
- Advanced Dual-Horizon Context expander
- Save/Load Preset controls
- Reproduce/Share configuration expander
- Data Freshness Filters
- DEBUG displays
- Quality/Trend threshold sliders
- Split basis dropdown
- V2.3 Parameters info box
- DIAGNOSTICS expander

## Column Naming Conventions

- Base metric: `"EBITDA Margin"`
- Historical periods: `"EBITDA Margin.1"`, `"EBITDA Margin.2"`, etc.
- Period dates: `"Period Ended"`, `"Period Ended.1"`, `"Period Ended.2"`, etc.

Numbering:
- `.1` = most recent period
- `.2`, `.3`, `.4` = earlier periods
- FY typically uses base + .1-.4 (5 annual periods)
- CQ typically uses base + .1-.12 (13 quarterly periods)

## Performance Considerations

- Main bottleneck: Period calendar construction and metric extraction
- File size: Designed for datasets with 100s-1000s of issuers
- Timing diagnostics embedded in `load_and_process_data()` (prints to console when RG_TESTS=1)

## Version History Notes

- **V2.3** - Current version (Major UI/UX improvements)
  - **Unified Period Selection**: Consolidated 4 separate controls into 2-mode system (Latest Available vs Reference Aligned)
  - **Recommendation-Based Ranking**: Overall_Rank now prioritizes Strong Buy > Buy > Hold > Avoid (instead of pure composite score)
  - **Enhanced Explainability**: Shows original weights vs current weights with 8-column comparison table
  - **UI Simplification**: Removed 9+ advanced/debug controls, hardcoded recommended defaults
  - **Fixed Recommendation Priorities**: Correctly maps Strong Buy, Buy, Hold, Avoid (removed incorrect "Sell")
  - **Dashboard Improvements**: "Top 10 Opportunities" and "Bottom 10 Risks" with Combined_Signal column

- **V2.2** - Period alignment, reference dates, and FY/CQ overlap resolution
- **V2.2.1** - Dynamic calibration for sector-specific weights
- Prior versions stored in `archive/`

## Important V2.3 Implementation Notes

### Recommendation System
- **Four recommendations only**: Strong Buy, Buy, Hold, Avoid (NO "Sell")
- **Priority mapping used in 3 locations**:
  1. Overall_Rank calculation (line ~7381)
  2. Dashboard Top/Bottom 10 (line ~7720)
  3. Issuer Search sorting (line ~8928)

### Period Selection
- **Two enums**: `PeriodSelectionMode` (LATEST_AVAILABLE, REFERENCE_ALIGNED) and `PeriodType` (ANNUAL, QUARTERLY)
- **Function signature change**: `load_and_process_data()` now uses `period_mode` instead of separate `data_period_setting`, `use_quarterly_beta`, `align_to_reference` parameters
- **Backward compatibility**: Legacy variables derived from `period_mode` for compatibility with existing code

### Enhanced Explainability
- **6 new columns** in results DataFrame: `Weight_*_Used` columns store original weights
- **Function signature change**: `_build_explainability_table()` returns 6 values instead of 4
- **Comparison display**: Shows impact of dynamic calibration with before/after weights
