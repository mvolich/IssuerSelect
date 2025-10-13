# Issuer Credit Screening Model

## Overview

ML-driven credit screening model for analyzing S&P issuer data, providing Investment Grade (IG) and High Yield (HY) analysis with automated scoring, ranking, and visualization.

## Features

### Core Functionality
- **Automated Credit Scoring**: 5-factor composite scoring system
- **IG/HY Segmentation**: Separate analysis for Investment Grade and High Yield issuers
- **Risk Categorization**: Strong Buy, Buy, Hold, Avoid recommendations
- **Visual Analytics**: PCA-based positioning maps for pattern recognition
- **AI Insights**: GPT-4 powered analysis and recommendations

### Scoring Methodology

The model uses a weighted composite score across five dimensions:

| Factor | Weight | Description |
|--------|--------|-------------|
| Credit Score | 25% | S&P credit rating converted to numeric scale |
| Leverage Score | 20% | Debt/EBITDA ratio (inverse scoring, lower is better) |
| Profitability Score | 25% | Combined ROE and EBITDA margin |
| Liquidity Score | 15% | Current ratio assessment |
| Growth Score | 15% | Revenue growth rate |

### Categorization Thresholds

**Investment Grade:**
- Strong Buy: ≥70
- Buy: 55-70
- Hold: 40-55
- Avoid: <40

**High Yield:**
- Strong Buy: ≥65
- Buy: 50-65
- Hold: 35-50
- Avoid: <35

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
streamlit run issuer_screening_app.py
```

### Data Input Requirements

The model expects an Excel file (.xlsx) or CSV file with the following columns:
- `Company ID`
- `Company Name`
- `Ticker`
- `S&P Credit Rating`
- `Sector`
- `Industry`
- `Market Capitalization`
- `Total Debt / EBITDA (x)`
- `Return on Equity`
- `EBITDA Margin`
- `Current Ratio (x)`
- `Total Revenues, 1 Year Growth`

### Configuration

1. **OpenAI API Key**: 
   - Add to Streamlit secrets as `api_key`
   - Or enter manually in the sidebar

2. **File Upload**:
   - Use the sidebar file uploader
   - Accepts .xlsx or .csv formats
   - Excel files can use any sheet (defaults to first if "Pasted Values" not found)

## Key Features Explained

### 1. Overview & Positioning Tab
- **Positioning Maps**: Visual representation of issuers in 2D space
- **Category Distributions**: Bar charts showing Strong Buy/Buy/Hold/Avoid breakdown
- **Color Coding**: Green (Strong Buy), Blue (Buy), Yellow (Hold), Red (Avoid)

### 2. Top Rankings Tab
- **Top 20 Lists**: Best performers in IG and HY categories
- **Composite Scores**: Overall quality metrics
- **Industry Context**: Sector and industry information

### 3. Rating Group Analysis Tab
- **Group Distribution**: Breakdown by rating categories (AAA, AA, A, BBB, etc.)
- **Peer Comparison**: Rankings within rating groups
- **Top Performers**: Best issuers in each rating category

### 4. Detailed Data Tab
- **Filterable Tables**: Full dataset with all metrics
- **Export Functionality**: Download filtered results as CSV
- **Custom Filters**: By IG/HY, category, minimum score

### 5. AI Analysis Tab
- **Executive Summary**: GPT-4 generated market overview
- **Investment Recommendations**: Specific opportunities identified
- **Market Insights**: Trends and patterns analysis
- **Methodology Assessment**: Model validation and suggestions

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
- **X-axis**: Overall credit quality (better →)
- **Y-axis**: Financial strength vs leverage balance
- **Size**: Bubble size reflects composite score
- **Jitter**: Small random offset prevents overlap

### Interpretation Guide
- Companies close together have similar credit profiles
- Green clusters indicate high-quality opportunities
- Red areas suggest higher risk or distressed situations

## Version History

### v2.2 (Current)
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
- Core scoring engine
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

The sidebar displays:
- Rows loaded and processed
- Unique credit ratings found
- Missing data percentages
- Duplicate vector warnings
- PCA variance explained

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
