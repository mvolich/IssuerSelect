# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Issuer Credit Screening Model V2.0** - A Streamlit-based application for analyzing corporate credit quality using a 6-factor composite scoring system with sector-adjusted weights, dual-horizon trend analysis, PCA-based cohort clustering, and AI-powered credit report generation.

## Running the Application

```bash
# Set environment for testing (disables Streamlit UI initialization)
set RG_TESTS=1

# Run application (normal mode)
streamlit run app.py

# Run embedded test suite
set RG_TESTS=1 && python app.py

# Run GenAI patch tests
python test_genai_patches.py
```

## Architecture

### Single-File Structure
The entire application is contained in `app.py` (~8100 lines). This is intentional for deployment simplicity to Streamlit Cloud.

### Tab Structure (7 tabs)

1. **Dashboard** - Model overview and key insights
2. **Issuer Search** - Search and filter issuers
3. **Rating Group Analysis** - IG vs HY cohort analysis
4. **Classification Analysis** - Sector-specific analysis
5. **Trend Analysis** - Dual-horizon trend metrics
6. **Methodology** - Complete model documentation
7. **GenAI Credit Report** - AI-powered credit analysis reports

### Core Scoring System

**6 Factor Scores** (each 0-100):
1. `credit_score` - Based on S&P rating mapped to numeric scale
2. `leverage_score` - Net Debt / EBITDA (inverse scoring)
3. `profitability_score` - EBITDA Margin
4. `liquidity_score` - Current Ratio
5. `growth_score` - Revenue CAGR (multi-period trend)
6. `cash_flow_score` - Free Cash Flow / Total Debt

**Raw Scores** (V2.0):
- `Raw_Quality_Score` - Composite of quality factors
- `Raw_Trend_Score` - Composite of trend indicators

**Composite Score**: Weighted average of the 6 factors (0-100 scale).

### Weight Resolution System

Factor weights vary by issuer classification following this hierarchy:

1. **CLASSIFICATION_OVERRIDES** - Custom weights for specific classifications
2. **CLASSIFICATION_TO_SECTOR** - Maps 30+ classifications to 10 parent sectors
3. **SECTOR_WEIGHTS** - Sector-specific factor weights
4. **Default weights** - Universal fallback if no classification provided

Example: "Software and Services" → "Information Technology" sector → uses IT sector weights (high growth/cash flow emphasis, lower leverage weight).

### Period Classification System

Data columns use suffix notation:
- **Base column** (e.g., "EBITDA Margin") = most recent period
- **Historical periods** (e.g., "EBITDA Margin.1", ".2", ".3", ".4") = prior periods
- **Quarterly data** (optional): ".5" through ".12"

The system automatically classifies periods as FY (fiscal year) or CQ (calendar quarter) based on:
1. **Period Ended column dates** (preferred method) - Uses `parse_period_ended_cols()` and `period_cols_by_kind()`
2. **Fallback heuristic**: Month-based classification (December → FY, other → CQ)

**Key features**:
- Handles non-December fiscal year ends (June 30, September 30, etc.)
- Filters 1900 sentinel dates automatically
- Robust parsing of Excel serial dates and malformed data
- Consistent classification across application and GenAI reports

**Key functions**:
- `parse_period_ended_cols()` - Parse and sort period columns by actual dates
- `period_cols_by_kind()` - Classify periods as FY or CQ based on date spacing
- `_metric_series_for_row()` - Extract time series with proper period classification
- `get_most_recent_column()` - Resolve correct column based on data period setting

### Trend Analysis (Dual-Horizon)

Two modes controlled by `use_quarterly_beta`:
- **Annual mode** (False): Uses base + .1-.4 periods (5 FY periods)
- **Quarterly mode** (True): Uses base + .1-.12 periods (13 periods if available)

**Dual-Horizon System** (V2.0):
- **Medium-term trend** (3-5 periods): Primary direction indicator
- **Short-term trend** (1-2 periods): Recent momentum indicator
- **Context-aware signals**: Combined analysis produces refined signals like "Strong & Normalizing", "Weak & Deteriorating"

### Data Validation

**Required columns** (flexible name matching via aliases):
- Company Name/ID (COMPANY_NAME_ALIASES, COMPANY_ID_ALIASES)
- S&P LT Issuer Credit Rating (RATING_ALIASES)

**Optional features** (gated by REQ_FOR):
- `classification`: Requires "Rubrics Custom Classification" column
- `country_region`: Requires "Country" and "Region" columns
- `period_alignment`: Requires "Period Ended" column

Functions `resolve_column()` and `validate_core()` handle robust column name matching (case-insensitive, handles NBSP, extra spaces).

### PCA Visualization

Creates 2D PCA plots showing:
- Investment Grade (IG) vs High Yield (HY) clustering
- Uses 6 factor scores as input features
- RobustScaler normalization before PCA
- Subsamples to 2000 points max for performance

### GenAI Credit Report (New in V2.0)

AI-powered credit analysis using OpenAI GPT-4 Turbo. Requires API key in `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "sk-..."
```

**Key Functions**:
- `extract_issuer_financial_data()` (app.py:774-891) - Extracts 11 financial metrics with full time series
- `build_credit_analysis_prompt()` (app.py:894-985) - Builds structured prompt for OpenAI
- `generate_credit_report()` (app.py:988-1021) - Calls GPT-4 Turbo API

**Metrics Extracted**:
1. EBITDA Margin
2. Total Debt / EBITDA (x)
3. Net Debt / EBITDA
4. EBITDA / Interest Expense (x)
5. Current Ratio (x)
6. Quick Ratio (x)
7. Return on Equity
8. Return on Assets
9. Total Revenues
10. Total Debt
11. Cash and Short-Term Investments

**Report Structure** (7 sections):
1. Executive Summary
2. Profitability Analysis
3. Leverage Analysis
4. Liquidity & Coverage Analysis
5. Credit Strengths (3-4 points)
6. Credit Risks & Concerns (3-4 points)
7. Rating Outlook & Recommendation

**Features**:
- Uses robust FY/CQ classification (handles non-December fiscal years)
- Automatic period type detection (FY vs CQ)
- Filters 1900 sentinel dates
- Data availability preview before generation
- Downloadable markdown reports
- Cost: ~$0.05-0.07 per report

### Metric Aliases System

**METRIC_ALIASES dictionary** (app.py:170-187) maps canonical metric names to common variations:
- Handles vendor data inconsistencies
- Case-insensitive matching
- Used throughout application for robust column resolution

**Helper Functions**:
- `resolve_metric_column()` - Find metric column using aliases
- `list_metric_columns()` - Get base + all suffixed columns for a metric
- `get_from_row()` - Row-level safe getter honoring aliases

## Testing

Embedded test suite runs when `RG_TESTS=1` environment variable is set.

**Test coverage** (12 tests total):
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
11. FY/CQ overlap de-duplication (CQ preferred)
12. Non-December FY classification in LLM extractor

**Additional Test Suites**:
- `test_patches.py` - Surgical patches verification (44 tests)
- `test_genai_patches.py` - GenAI FY/CQ classification tests (3 tests)

Run tests:
```bash
set RG_TESTS=1 && python app.py
python test_patches.py
python test_genai_patches.py
```

## State Management

**URL Query Parameters**:
- `scoring_method` - Universal or Classification-Adjusted
- `data_period` - FY0, CQ-0, or trailing periods
- `use_quarterly_beta` - Trend window mode
- `classification_filter`, `band_filter`, `region_filter`
- `min_score`, `max_score`

Functions:
- `collect_current_state()` - Capture current UI state
- `apply_state_to_controls()` - Restore state from dict
- `_build_deep_link()` - Generate shareable URL

**Presets**: Saved in `st.session_state.presets` as JSON-serializable dicts.

## Key Data Flow

1. **Upload** → `load_and_process_data()`
2. **Column validation** → `validate_core()`
3. **Period parsing** → `parse_period_ended_cols()`
4. **Period classification** → `period_cols_by_kind()` (FY vs CQ)
5. **Metric extraction** → `get_most_recent_column()` per user's data period setting
6. **Trend calculation** → Dual-horizon trend indicators (medium-term + short-term)
7. **Weight resolution** → `get_classification_weights()`
8. **Scoring** → Raw quality/trend scores + composite score
9. **Signal derivation** → Context-aware signals (V2.0)
10. **Rating band assignment** → `assign_rating_band()`
11. **Display** → Filters → PCA → Leaderboards → GenAI Reports

## React-Safe Table Rendering

**Migration to st.table** (V2.0):
- Replaced `st.dataframe()` with `st.table()` for diagnostics tables
- Eliminates React ChunkLoadError
- All data converted to strings before rendering
- Simplified data structures (counts vs complex strings)

**Key Locations**:
- Diagnostics table rendering
- Evidence table rendering
- Global data health display

## Code Style Notes

- **Testing mode**: Many sections check `os.environ.get("RG_TESTS")` to disable UI code during tests
- **Caching**: `@st.cache_data` on `load_and_process_data()` for performance
- **Version markers**: Code sections tagged with `[V2.0]` comments
- **No emojis**: Professional appearance, emojis removed in V2.0
- **Robust error handling**: Graceful fallbacks throughout

## Common Modifications

**Adding a new sector**:
1. Add entry to `SECTOR_WEIGHTS`
2. Update `CLASSIFICATION_TO_SECTOR` mapping if needed

**Adding a new factor**:
1. Extract metric in `load_and_process_data()`
2. Calculate score (0-100 scale)
3. Add to all sector weight dicts (must sum to 1.0)
4. Update composite score calculation
5. Update `_factor_score_columns()` for PCA

**Modifying rating bands**:
- Edit `RATING_BANDS` dict
- Used by `assign_rating_band()` function

**Adding metrics to GenAI reports**:
1. Add to `metrics_to_extract` list in `extract_issuer_financial_data()`
2. Add to `METRIC_ALIASES` if needed for column resolution
3. Update prompt in `build_credit_analysis_prompt()` if needed

## Important Functions Reference

**Period Classification**:
- `parse_period_ended_cols()` - Parse period dates
- `period_cols_by_kind()` - Classify FY vs CQ by date spacing
- `_find_period_cols()` - Map suffix to period column

**Metric Resolution**:
- `resolve_metric_column()` - Find metric using aliases
- `list_metric_columns()` - Get base + suffixed columns
- `_metric_series_for_row()` - Extract time series with period types

**GenAI Functions**:
- `extract_issuer_financial_data()` - Extract financial data for one issuer
- `build_credit_analysis_prompt()` - Build OpenAI prompt
- `generate_credit_report()` - Call GPT-4 Turbo API

**Data Validation**:
- `resolve_column()` - Generic column alias resolver
- `validate_core()` - Validate required columns
- `resolve_company_name_column()` - Find company name column
- `resolve_rating_column()` - Find rating column

**Diagnostics**:
- `generate_data_diagnostics_v2()` - Per-issuer data quality report
- `generate_global_data_health_v2()` - Dataset-wide health metrics
- `build_issuer_evidence_table()` - Evidence table for AI reports

## Dependencies

Core:
- streamlit
- pandas
- numpy
- plotly (graph_objects, express)
- sklearn (RobustScaler, PCA)
- dateutil (parser)

Optional:
- openai (for GenAI credit report feature)

## Performance Notes

- **File size**: ~8100 lines (single-file architecture)
- **Dataset handling**: Optimized for 100-5000 issuers
- **PCA**: Subsamples to 2000 points for performance
- **Caching**: Data loading and processing cached via `@st.cache_data`
- **GenAI reports**: ~5-15 seconds per report (OpenAI API latency)

## Recent Changes (V2.0)

1. **GenAI Credit Report Tab**
   - AI-powered credit analysis using GPT-4 Turbo
   - Replaces legacy AI Analysis tab
   - Professional 7-section reports
   - Downloadable markdown format

2. **Robust FY/CQ Classification**
   - Handles non-December fiscal year ends
   - Uses `parse_period_ended_cols()` for accuracy
   - Graceful fallback to month heuristic
   - Consistent across application

3. **React-Safe Rendering**
   - Migrated to `st.table()` from `st.dataframe()`
   - Simplified data structures
   - Eliminated ChunkLoadError issues

4. **Version Standardization**
   - All version references now "V2.0"
   - Updated header, comments, documentation

5. **Enhanced Testing**
   - 12 RG_TESTS (up from 11)
   - Added non-December FY classification test
   - Additional GenAI patch test suite

## Known Issues / Edge Cases

1. **Non-calendar fiscal years**: Now handled correctly via robust period classification
2. **1900 sentinel dates**: Automatically filtered out
3. **Excel serial dates**: Handled via `pd.to_datetime(..., errors="coerce")`
4. **Missing Period Ended columns**: Graceful fallback to suffix-based indexing
5. **React serialization**: Resolved via st.table migration and string conversion

## Documentation Files

- `CLAUDE.md` - This file
- `GENAI_CREDIT_REPORT_TAB.md` - Comprehensive GenAI feature documentation
- `GENAI_PATCHES_SUMMARY.md` - FY/CQ classification enhancement details
- `ST_TABLE_MIGRATION.md` - React-safe table rendering documentation
- Various other technical documentation files for specific patches
