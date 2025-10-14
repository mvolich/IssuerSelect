# Issuer Credit Screening Model

## Overview

ML-driven credit screening model for analyzing S&P issuer data, providing Investment Grade (IG) and High Yield (HY) analysis with automated scoring, ranking, and visualization.

## Features

### Core Functionality
- **Automated Credit Scoring**: 6-factor composite scoring system
- **IG/HY Segmentation**: Separate analysis for Investment Grade and High Yield issuers
- **Risk Categorization**: Strong Buy, Buy, Hold, Avoid recommendations
- **Visual Analytics**: PCA-based positioning maps for pattern recognition
- **AI Insights**: GPT-4 powered analysis and recommendations

### Scoring Methodology

The model uses a weighted composite score across six dimensions:

| Factor | Weight | Description |
|--------|--------|-------------|
| Credit Score | 20% | S&P credit rating converted to numeric scale |
| Leverage Score | 20% | Multi-metric: Net Debt/EBITDA, Total Debt/EBITDA, Debt/Capital |
| Profitability Score | 20% | Multi-metric: ROE, EBITDA Margin, ROA, EBIT Margin |
| Liquidity Score | 10% | Current ratio and Quick ratio |
| Growth Score | 15% | Multi-period: Revenue 1Y/3Y CAGR, EBITDA 3Y CAGR |
| Cash Flow Score | 15% | FCF/Debt ratio, FCF Margin, Cash Ops Coverage |

### Categorization Thresholds

**Investment Grade:**
- Strong Buy: ≥72
- Buy: 57-71
- Hold: 42-56
- Avoid: <42

**High Yield:**
- Strong Buy: ≥67
- Buy: 52-66
- Hold: 37-51
- Avoid: <37

## Installation

### Requirements
```bash
pip install streamlit pandas numpy plotly scikit-learn openai xlsxwriter openpyxl
```

### Required Python Version
- Python 3.8 or higher

## Usage

### Running the Application
```bash
streamlit run issuer_screening_app_final.py
```

### Data Input Requirements

The model expects an Excel file (.xlsx) or CSV file with the following columns:

**Required Core Columns:**
- `Company ID`
- `Company Name`
- `Ticker`
- `S&P Credit Rating`
- `Sector`
- `Industry`
- `Market Capitalization`

**Required Financial Metrics:**
- `Total Debt / EBITDA (x)`
- `Net Debt / EBITDA`
- `Total Debt`
- `Total Debt / Total Capital (%)`
- `Return on Equity`
- `Return on Assets`
- `EBITDA Margin`
- `EBIT Margin`
- `Current Ratio (x)`
- `Quick Ratio (x)`
- `Total Revenues, 1 Year Growth`
- `Total Revenues, 3 Yr. CAGR`
- `EBITDA, 3 Years CAGR`
- `Levered Free Cash Flow`
- `Levered Free Cash Flow Margin`
- `Cash from Ops. to Curr. Liab. (x)`

### Configuration

1. **OpenAI API Key**: 
   - Add to Streamlit secrets as `api_key`
   - Or enter manually in the sidebar

2. **File Upload**:
   - Use the sidebar file uploader
   - Accepts .xlsx or .csv formats
   - Excel files can use any sheet (defaults to first if "Pasted Values" not found)

## Key Features Explained

### 1. EDA Tab
- **Dataset Overview**: Key statistics and metrics
- **Composite Score Distribution**: Histograms with statistical analysis
- **Factor Score Analysis**: Box plots comparing IG vs HY
- **Correlation Analysis**: Heatmaps showing factor relationships
- **Sector Analysis**: Multiple visualizations showing sector performance
- **Credit Rating Distribution**: Breakdown by rating categories
- **Summary Statistics**: Detailed tables by investment category

### 2. Methodology Tab
- **6-Factor Scoring Engine**: Detailed explanation of each factor
- **Individual Factor Calculations**: Examples and formulas
- **IG vs HY Segmentation**: Different thresholds and rationale
- **Category Assignment**: Investment recommendation logic
- **Ranking Systems**: Percentile and peer comparisons

### 3. Top Rankings Tab
- **Top 20 Lists**: Best performers in IG and HY categories
- **Composite Scores**: Overall quality metrics
- **Industry Context**: Sector and industry information

### 4. Rating Group Analysis Tab
- **Group Distribution**: Breakdown by rating categories (AAA, AA, A, BBB, etc.)
- **Peer Comparison**: Rankings within rating groups
- **Top Performers**: Best issuers in each rating category

### 5. Detailed Data Tab
- **Filterable Tables**: Full dataset with all metrics
- **Export Functionality**: Download filtered results as CSV
- **Custom Filters**: By IG/HY, category, minimum score

### 6. Overview & Positioning Tab
- **Positioning Maps**: Visual representation of issuers in 2D space using PCA
- **Category Distributions**: Bar charts showing Strong Buy/Buy/Hold/Avoid breakdown
- **Color Coding**: Green (Strong Buy), Blue (Buy), Yellow (Hold), Red (Avoid)

### 7. AI Analysis Tab
- **Executive Summary**: GPT-4 generated market overview
- **Investment Recommendations**: Specific opportunities identified
- **Market Insights**: Trends and patterns analysis
- **Methodology Assessment**: Model validation and suggestions
- **Button-Activated**: Click to generate analysis (saves API costs)

## Data Processing Details

### Rating Normalization
The model handles various rating formats:
- Removes outlook indicators (NEG, POS, WATCH)
- Maps aliases (BBBM→BBB, BMNS→B, CCCC→CCC)
- Handles NR, N/A, WD as Unknown

### Percent Parsing
Automatically detects and converts:
- Percentage strings ("12.5%" → 12.5)
- Fractional values (0.125 → 12.5)
- Mixed formats within same column

### Missing Data Handling
Three-tier imputation strategy:
1. Sector × IG/HY medians
2. Overall sector medians
3. Global medians or default values

### Special Cases
- **Negative EBITDA**: Treated as very high leverage (score=0)
- **Extreme Values**: Capped to prevent outlier distortion
- **Missing Growth**: Uses sector/global medians

## Visualization Details

### PCA Positioning Maps
- **X-axis**: Overall credit quality (better →, Strong Buy on right)
- **Y-axis**: Financial strength vs leverage balance
- **Size**: Bubble size reflects composite score
- **Jitter**: Small random offset prevents overlap
- **6-Factor Analysis**: Based on all six scoring dimensions

### Interpretation Guide
- Companies close together have similar credit profiles across all six factors
- Green clusters (right side) indicate high-quality opportunities
- Red areas (left side) suggest higher risk or distressed situations
- Top 10 issuers highlighted with gold borders

## Version History

### v3.0 (Current)
- **Enhanced 6-factor scoring model**: Added Cash Flow Score (15% weight)
- **Multi-metric factor scoring**:
  - Leverage: Net Debt/EBITDA, Total Debt/EBITDA, Debt/Capital
  - Profitability: ROE, EBITDA Margin, ROA, EBIT Margin
  - Liquidity: Current Ratio, Quick Ratio
  - Growth: Revenue 1Y/3Y CAGR, EBITDA 3Y CAGR
- **Adjusted category thresholds**: Updated for 6-factor model
- **PCA orientation fix**: Composite Score correlation for proper axis alignment
- **New EDA tab**: Comprehensive statistical analysis and visualizations
- **Requirements update**: Python 3.12 compatible dependencies

### v2.2
- Fixed rating classification consistency
- Corrected PCA axis orientation
- Added business-friendly labels
- Improved percent parsing
- Fixed leverage scoring for negative EBITDA

### v2.1
- Added CSV support
- Enhanced error handling
- Improved diagnostics

### v2.0
- Initial production release
- Core scoring engine (5-factor)
- Visualization framework

## Troubleshooting

### Common Issues

1. **"Missing required columns" error**
   - Verify all required columns are present
   - Check for spelling/spacing variations
   - Ensure column headers are in first row

2. **No data processed**
   - Check credit ratings are in standard S&P format
   - Verify numeric columns contain valid numbers
   - Look for "None" or "N/A" text in numeric fields

3. **Unexpected rankings**
   - Review sidebar diagnostics for data quality
   - Check for high percentage of missing values
   - Verify leverage values (negative = distressed)

### Debug Information

Diagnostic information has been streamlined. Key metrics visible in:
- EDA tab: Dataset overview and statistics
- Detailed Data tab: Full data inspection and export
- Error messages: Clear feedback on missing columns or data issues

## Support

For issues or questions:
1. Check sidebar diagnostics
2. Review data quality metrics
3. Verify input format matches requirements
4. Ensure all dependencies are installed

## License

Proprietary - Rubrics Asset Management

## Authors

Developed by Rubrics Asset Management
ML-Driven Investment Grade & High Yield Analysis

---

*Note: This model is for investment research purposes. Always perform additional due diligence before making investment decisions.*
