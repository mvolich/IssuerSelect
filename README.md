# Issuer Credit Screening Model - Streamlit App

## 🚀 Deployment Instructions

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at share.streamlit.io)
- OpenAI API key

### Step 1: Push to GitHub

1. Create a new repository on GitHub (e.g., `issuer-screening-app`)

2. Upload these files to your repository:
   - `issuer_screening_app.py` (the main application)
   - `requirements.txt` (dependencies)
   - `README.md` (this file)

3. Optionally, add a logo:
   - Create an `assets/` folder
   - Add `rubrics_logo.png` or `rubrics_logo.svg` to this folder

### Step 2: Configure Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository and branch
5. Set main file path: `issuer_screening_app.py`
6. Click "Advanced settings"

### Step 3: Add Secrets

In the Advanced settings, add your OpenAI API key to secrets:

```toml
api_key = "sk-your-openai-api-key-here"
```

### Step 4: Deploy

1. Click "Deploy!"
2. Wait for deployment (usually 2-3 minutes)
3. Your app will be live at: `https://[your-app-name].streamlit.app`

---

## 📊 Using the App

### Upload Data

The app requires an Excel file with the S&P issuer data in the format:
- Sheet name: "Pasted Values"
- Required columns:
  - Company ID
  - Company Name
  - Ticker
  - S&P Credit Rating
  - Sector
  - Industry
  - Market Capitalization
  - Total Debt / EBITDA (x)
  - Return on Equity
  - EBITDA Margin
  - Current Ratio (x)
  - Total Revenues, 1 Year Growth

### Features

**Tab 1: Overview & Positioning**
- Investment Grade positioning map (PCA visualization)
- High Yield positioning map
- Category distributions

**Tab 2: Top Rankings**
- Top 20 IG and HY issuers
- Side-by-side comparison
- Interactive bar charts

**Tab 3: Rating Group Analysis**
- Distribution by rating group (AAA, AA, A, BBB, BB, B)
- Top performers within each rating tier
- Peer group comparison

**Tab 4: Detailed Data**
- Filterable issuer database
- Export to CSV
- All component scores visible

**Tab 5: AI Analysis**
- Executive summary (GPT-4)
- Investment recommendations
- Market insights & trends
- Methodology assessment

---

## 🔧 Customization

### Colors

The app uses Rubrics brand colors defined in CSS:
- `--rb-blue: #001E4F` (primary)
- `--rb-mblue: #2C5697` (medium blue)
- `--rb-lblue: #7BA4DB` (light blue)
- `--rb-grey: #D8D7DF` (grey)
- `--rb-orange: #CF4520` (accent)

Category colors:
- Strong Buy: `#00C851` (green)
- Buy: `#33b5e5` (blue)
- Hold: `#ffbb33` (orange)
- Avoid: `#ff4444` (red)

### Scoring Weights

To adjust the composite score weights, modify lines 180-186:

```python
weights = {
    'credit_score': 0.25,        # Credit rating weight
    'leverage_score': 0.20,      # Debt/EBITDA weight
    'profitability_score': 0.25, # ROE & margins weight
    'liquidity_score': 0.15,     # Current ratio weight
    'growth_score': 0.15         # Revenue growth weight
}
```

### Category Thresholds

To adjust what qualifies as Strong Buy/Buy/Hold:

**Investment Grade (lines 279-283):**
```python
ig_results['Category'] = pd.cut(
    ig_results['Composite_Score'],
    bins=[0, 40, 55, 70, 100],  # Adjust these thresholds
    labels=['Avoid', 'Hold', 'Buy', 'Strong Buy']
)
```

**High Yield (lines 285-289):**
```python
hy_results['Category'] = pd.cut(
    hy_results['Composite_Score'],
    bins=[0, 35, 50, 65, 100],  # Adjust these thresholds
    labels=['Avoid', 'Hold', 'Buy', 'Strong Buy']
)
```

---

## 🤖 AI Analysis Notes

The AI analysis requires an OpenAI API key with GPT-4 access. The app makes 4 API calls:

1. **Executive Summary** (~1,500 tokens)
2. **Investment Recommendations** (~2,000 tokens)
3. **Market Insights** (~2,000 tokens)
4. **Methodology Assessment** (~1,500 tokens)

**Total estimated cost per analysis:** ~$0.15-0.25 USD

To reduce costs:
- Change model from `gpt-4` to `gpt-3.5-turbo` (90% cost reduction)
- Reduce `max_tokens` parameters
- Cache results using `@st.cache_data`

---

## 📈 Model Methodology

### Composite Score Calculation

Each issuer receives a score from 0-100 based on 5 components:

1. **Credit Score (25%)**: S&P rating converted to numeric (AAA=21, D=0)
2. **Leverage Score (20%)**: Inverse of Debt/EBITDA ratio
3. **Profitability Score (25%)**: Average of ROE and EBITDA margin
4. **Liquidity Score (15%)**: Based on current ratio
5. **Growth Score (15%)**: Revenue growth rate

### Investment Grade vs High Yield

- **IG**: BBB- and above (separate ranking)
- **HY**: BB+ and below (separate ranking)
- Different category thresholds reflect different risk profiles

### Rating Groups

Issuers are grouped into 6 tiers for peer comparison:
- Group 1: AAA
- Group 2: AA+/AA/AA-
- Group 3: A+/A/A-
- Group 4: BBB+/BBB/BBB-
- Group 5: BB+/BB/BB-
- Group 6: B+/B/B-

### PCA Visualization

- 2-component PCA on normalized scores
- Captures ~60-70% of variance
- Enables visual relative positioning
- Bubble size = composite score

---

## 🔒 Security Notes

- Never commit your OpenAI API key to GitHub
- Always use Streamlit secrets for API keys
- Keep your secrets.toml file in .gitignore
- Rotate API keys periodically

---

## 📞 Support

For issues or questions about the model:
1. Check this README
2. Review the code comments
3. Test with sample data first

For Streamlit deployment issues:
- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Community Forum](https://discuss.streamlit.io)

---

## 📝 License

This application is proprietary to Rubrics Asset Management.

---

*Last Updated: October 2025*
