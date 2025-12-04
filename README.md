# Issuer Select Project

A comprehensive credit risk dashboard application built with Streamlit that analyzes financial data for corporate issuers to calculate credit quality scores, trend indicators, and generate AI-powered credit reports.

## Overview

The Issuer Select application processes Capital IQ Excel spreadsheets containing multi-period time series data to:

- Calculate composite credit quality scores using a weighted 5-factor model
- Analyze trends across key financial metrics
- Generate signal classifications (Strong/Weak × Improving/Deteriorating)
- Produce AI-powered credit analysis reports
- Provide peer comparison and ranking capabilities

## Key Features

- **Credit Scoring Engine**: 5-factor quality assessment (Credit, Leverage, Profitability, Liquidity, Cash Flow)
- **Trend Analysis**: Direction, momentum, and volatility scoring across 8 time periods
- **Signal Classification**: 4-quadrant matrix combining quality and trend assessments
- **Peer Comparison**: Sector and rating-band relative rankings
- **GenAI Reports**: Comprehensive credit analysis using Claude AI
- **Interactive Diagnostics**: Full calculation transparency with audit trail
- **Export Capabilities**: Excel, CSV, and PDF report generation

## Data Requirements

### Input File Format

The application expects a Capital IQ Excel export with:

- **364 columns** across 8 time periods (5 Fiscal Year + 3 LTM)
- **~2,000 corporate issuers** with financial data
- **Monetary values in Thousands** (per CIQ setting `Mag=Thousands`)

### Required Columns

| Category | Key Columns |
|----------|-------------|
| Identification | Company ID, Company Name, Ticker, Country, Sector |
| Classification | Rubrics Custom Classification, S&P Credit Rating |
| Leverage | Total Debt / EBITDA (x), Net Debt / EBITDA, Total Debt/Equity (%) |
| Profitability | EBITDA Margin, Return on Equity, Return on Assets |
| Liquidity | Current Ratio (x), Quick Ratio (x), Cash & Short-term Investments |
| Coverage | EBITDA / Interest Expense (x) |
| Cash Flow | Levered Free Cash Flow, Cash from Ops., Unlevered Free Cash Flow |

## Scoring Methodology

### Composite Score Formula

```
Composite Score = (Quality Score × 0.80) + (Trend Score × 0.20)
```

### Quality Score Components

| Factor | Weight | Key Metrics |
|--------|--------|-------------|
| Credit | 20% | S&P Rating mapping |
| Leverage | 25% | Debt/EBITDA, Net Debt/EBITDA, Debt/Capital |
| Profitability | 20% | EBITDA Margin, ROE, ROA |
| Liquidity | 10% | Current Ratio, Quick Ratio, Cash Position |
| Cash Flow | 25% | FCF Margins, CFO/Debt, Cash Conversion |

### Trend Score Components

| Component | Weight | Description |
|-----------|--------|-------------|
| Direction | 50% | Slope of metric over time |
| Volatility | 30% | Consistency/stability (inverted) |
| Momentum | 20% | Recent vs. prior period comparison |

## Signal Classification

| Signal | Quality | Trend | Recommendation |
|--------|---------|-------|----------------|
| Strong & Improving | ≥55 | ≥55 | Strong Buy |
| Strong & Deteriorating | ≥55 | <55 | Hold |
| Weak & Improving | <55 | ≥55 | Speculative Buy |
| Weak & Deteriorating | <55 | <55 | Avoid |

## Installation

```bash
# Clone repository
git clone [repository-url]
cd issuer-select

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

## Dependencies

- Python 3.9+
- Streamlit
- Pandas / NumPy
- Plotly
- OpenPyXL
- Anthropic (for GenAI features)


## Architecture Principles

1. **Single Source of Truth**: All metrics defined in `METRIC_REGISTRY`
2. **Calculate Once, Display Everywhere**: Diagnostics read pre-computed results
3. **Raw Data Integrity**: Input spreadsheet values flow through unchanged
4. **Calculation Transparency**: Every score traceable to source values

## License

Proprietary - Internal Use Only
