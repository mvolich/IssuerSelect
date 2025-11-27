\# Issuer Credit Screening Model V5.0



A Streamlit-based credit analytics platform for evaluating corporate bond issuers using a 5-factor composite scoring model with trend analysis, comprehensive diagnostics, and AI-powered credit reports.



\## Overview



The application processes Capital IQ financial data for ~2,000 corporate issuers, calculating quality scores based on five financial factors and generating credit rankings, classifications, and investment recommendations.



\## Features



\### Core Analytics

\- \*\*5-Factor Composite Scoring\*\*: Credit, Leverage, Profitability, Liquidity, Cash Flow

\- \*\*Trend Analysis\*\*: Dual-horizon momentum tracking with cycle position scoring (0-100)

\- \*\*Signal Classification\*\*: Strong \& Improving, Weak \& Deteriorating, and 4 other states

\- \*\*Recommendation Engine\*\*: Strong Buy / Buy / Hold / Avoid with rating guardrails



\### Scoring Methodology

\- \*\*Quality Score (80%)\*\*: Weighted combination of 5 factors with 15+ component metrics

\- \*\*Trend Score (20%)\*\*: Direction, momentum, and volatility across key metrics

\- \*\*Composite Score\*\*: Final 0-100 score combining quality and trend



\### Weight Configuration

\- \*\*Universal Weights (Default)\*\*: Consistent weights across all sectors

&nbsp; - Credit: 20%, Leverage: 25%, Profitability: 20%, Liquidity: 10%, Cash Flow: 25%

\- \*\*Dynamic Calibration (Optional)\*\*: Sector-specific optimization (use with caution)



\### Diagnostic System

\- \*\*10-Stage Pipeline Transparency\*\*: Complete calculation trace from raw data to final recommendation

\- \*\*Export Formats\*\*: JSON, CSV, Excel with full component breakdowns

\- \*\*Data Quality Monitoring\*\*: Freshness indicators, completeness tracking



\### AI Credit Reports

\- \*\*Structured Output\*\*: Executive summary, key drivers, risks, investment thesis



\## Installation



\### Requirements

\- Python 3.9+

\- Dependencies listed in `requirements.txt`



\### Setup



1\. Install dependencies:

```bash

pip install -r requirements.txt

```



2\. (Optional) Configure API keys for AI reports in `.streamlit/secrets.toml`:

```toml

CLAUDE\_API\_KEY = "your-claude-api-key"

\# OR

OPENAI\_API\_KEY = "your-openai-api-key"

```



\## Running the Application



```bash

streamlit run app.py

```



The application will be available at `http://localhost:8501`



\## Data Format



\### Input File

Upload Excel (.xlsx) or CSV files with Capital IQ financial data.



\### Required Columns

| Column | Description |

|--------|-------------|

| Company ID / Issuer ID | Unique identifier |

| Company Name | Issuer name |

| S\&P Credit Rating | Credit rating (AAA to D) |

| Rubrics Custom Classification | Sector classification |



\### Financial Metrics

The application automatically detects standard Capital IQ column names including:

\- Balance sheet: Total Assets, Total Debt, Cash \& Short-term Investments

\- Income: Revenue, EBITDA, Net Income, Interest Expense

\- Cash flow: Operating Cash Flow, Free Cash Flow

\- Ratios: Current Ratio, Quick Ratio, Debt/EBITDA, Interest Coverage



\### Multi-Period Support

\- Supports up to 13 periods of historical data (FY and LTM)

\- Automatic period detection and alignment

\- Configurable: Latest Available vs Reference Aligned modes



\## Application Tabs



| Tab | Description |

|-----|-------------|

| \*\*Dashboard\*\* | Model overview, Top 10 opportunities, Bottom 10 risks, quadrant analysis |

| \*\*Issuer Search\*\* | Individual issuer deep-dive with factor breakdown |

| \*\*Rating Group Analysis\*\* | Performance analysis by rating band |

| \*\*Classification Analysis\*\* | Sector-level analytics and comparisons |

| \*\*Trend Analysis\*\* | Cyclicality heatmaps, improving/deteriorating trends |

| \*\*GenAI Credit Report\*\* | AI-generated comprehensive credit analysis |

| \*\*Diagnostics\*\* | Full calculation transparency with 10-stage pipeline trace |



\## Data Exclusions



The following entity types are excluded from rankings as corporate credit metrics don't apply:



\- \*\*Government/Municipal Entities\*\*: Provinces, cities, councils, municipalities

\- \*\*Sovereign Issuers\*\*: National governments

\- \*\*Financing Subsidiaries\*\*: Treasury vehicles with parent guarantees (flagged but included)



\## Configuration Options



\### Sidebar Settings



| Setting | Default | Description |

|---------|---------|-------------|

| Data Period | LTM0 | Latest available vs specific reference date |

| Period Priority | LTM First | Prefer LTM over FY data |

| Dynamic Calibration | OFF | Universal weights (recommended) |

| Trend Analysis | FY Only | Annual data for trend calculations |



\### Signal Thresholds

\- Quality Split: 60th percentile within rating band

\- Trend Threshold: 55 (cycle position score)



\## Testing



Run the built-in test suite:



```bash

\# Windows

set RG\_TESTS=1 \&\& python app.py



\# Linux/Mac

RG\_TESTS=1 python app.py

```



Tests cover:

\- Period selection logic

\- Trend calculations

\- Factor scoring

\- Signal classification

\- Data export functionality



\## Architecture



```

app.py (18,270 lines)

├── Data Loading \& Validation

├── Period Selection Engine

├── 5-Factor Scoring Pipeline

│   ├── Credit Score (3 components)

│   ├── Leverage Score (3 components)

│   ├── Profitability Score (3 components)

│   ├── Liquidity Score (3 components)

│   └── Cash Flow Score (4 components)

├── Trend Analysis Engine

├── Composite Score Calculation

├── Ranking \& Classification

├── Diagnostic Data Capture

├── UI Rendering (7 tabs)

└── AI Report Generation

```



\## Version History



| Version | Key Changes |

|---------|-------------|

| V5.0 | 5-factor model, universal weights default, enhanced diagnostics, government entity exclusions |

| V2.3 | Unified period selection, recommendation-based ranking |

| V2.2 | Dynamic weight calibration, sector adjustments |

| V2.0 | Trend analysis, cycle position scoring |

| V1.0 | Initial 6-factor model |



\## Known Limitations



1\. \*\*Financing Subsidiaries\*\*: Entities like "Mitsui \& Co. (U.S.A.), Inc." have metrics that reflect treasury functions, not operating quality. Ratings reflect parent guarantee.



2\. \*\*Data Freshness\*\*: Model accuracy depends on input data currency. Stale data (>180 days) is flagged.



3\. \*\*Sector Coverage\*\*: Dynamic calibration requires minimum 5 issuers per sector for meaningful results.



\## Support



For issues or questions, contact the development team.



\## License



Proprietary - Internal use only



---



\*Last updated: November 2025\*

