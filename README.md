# Credit Analytics Application V2.3

A Streamlit-based credit analytics platform for evaluating fixed-income issuers using a 6-factor composite scoring model with trend analysis and AI-powered credit reports.

## Features

- **6-Factor Composite Scoring**: Credit, Leverage, Profitability, Liquidity, Growth, Cash Flow
- **Trend Analysis**: Dual-horizon momentum tracking with cycle position scoring
- **Dynamic Calibration**: Sector-specific weight optimization based on BBB-rated issuers
- **AI Credit Reports**: Comprehensive credit analysis using Claude Sonnet 4 or GPT-4o
- **Interactive Visualizations**: PCA scatter plots, leaderboards, factor breakdowns
- **Recommendation Engine**: Strong Buy/Buy/Hold/Avoid classifications

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up API keys (optional - for AI reports):

Create `.streamlit/secrets.toml`:
```toml
CLAUDE_API_KEY = "your-claude-api-key"
# OR
OPENAI_API_KEY = "your-openai-api-key"
```

## Running the Application

```bash
streamlit run app.py
```

## Data Format

Upload Excel (.xlsx) or CSV files with financial data. Required columns:
- Company ID / Issuer ID
- Company Name / Issuer Name
- S&P LT Issuer Rating / S&P Rating

The app supports flexible column naming and will automatically detect standard financial metrics.

## Key Tabs

1. **Dashboard**: Top/bottom performers, signal distribution
2. **Issuer Search**: Detailed issuer-level analysis
3. **Scatter Plot**: PCA visualization of credit quality
4. **Leaderboard**: Ranked issuers by rating band
5. **Methodology**: Model documentation
6. **Data Export**: Download results and diagnostics
7. **AI Credit Report**: AI-generated comprehensive credit analysis

## Version

**V2.3** - Current version with unified period selection, recommendation-based ranking, and enhanced explainability

## Testing

```bash
set RG_TESTS=1
python app.py
```

## License

Proprietary - Internal use only
