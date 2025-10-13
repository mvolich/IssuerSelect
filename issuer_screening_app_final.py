import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
import base64
import os
import hashlib
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Issuer Credit Screening Model",
    layout="wide",
    page_icon="https://rubricsam.com/wp-content/uploads/2021/01/cropped-rubrics-logo-tight.png",
)

def inject_brand_css():
    st.markdown("""
    <style>
      :root{
        --rb-blue:#001E4F; --rb-mblue:#2C5697; --rb-lblue:#7BA4DB;
        --rb-grey:#D8D7DF; --rb-orange:#CF4520;
      }

      /* Global font + color (exclude div, span, .stTabs from global color override) */
      html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
      [data-testid="stSidebarContent"], [data-testid="stMarkdownContainer"],
      h1, h2, h3, h4, h5, h6, p, label, input, textarea, select,
      .stText, .stDataFrame, .stMetric {
        font-family: Arial, Helvetica, sans-serif !important;
        color: var(--rb-blue) !important;
      }
      [data-testid="stDataFrame"] * {
        font-family: Arial, Helvetica, sans-serif !important;
        color: var(--rb-blue) !important;
      }

      html, body, .stApp { background:#f8f9fa; }
      header[data-testid="stHeader"] { background: transparent !important; }

      /* Header layout shared by RG & ROAM */
      .rb-header { display:flex; align-items:flex-start; justify-content:space-between;
        gap:12px; margin: 0 0 12px 0; }
      .rb-title h1 { font-size:2.6rem; color:var(--rb-blue); font-weight:700; margin:0; }
      .rb-sub { color:#4c566a; font-weight:600; margin-top:.25rem; }
      .rb-logo img { height:48px; display:block; }
      @media (max-width:1200px){ .rb-title h1{ font-size:2.2rem; } .rb-logo img{ height:42px; } }

      /* Tabs */
      .stTabs [data-baseweb="tab-list"]{ gap:12px; border-bottom:none; }
      .stTabs [data-baseweb="tab"]{
        background:var(--rb-grey); border-radius:4px 4px 0 0;
        font-weight:600; min-width:180px; text-align:center; padding:8px 16px;
      }
      /* Unselected tabs - blue text */
      .stTabs [data-baseweb="tab"]:not([aria-selected="true"]) {
        color: var(--rb-blue) !important;
      }
      /* Selected tabs - white text on element and all children */
      .stTabs [aria-selected="true"],
      .stTabs [aria-selected="true"] * {
        background:var(--rb-mblue)!important;
        color: #ffffff !important;
        border-bottom:3px solid var(--rb-orange)!important;
      }

      /* Buttons - force white text on button and all children */
      .stButton > button,
      .stButton > button * {
        background:var(--rb-mblue);
        color: #ffffff !important;
        border:none;
        border-radius:4px;
        padding:8px 16px;
        font-weight:600;
      }
      .stButton > button:hover,
      .stButton > button:hover * {
        background:var(--rb-blue);
        color: #ffffff !important;
      }
      .stDownloadButton > button,
      .stDownloadButton > button * {
        background:var(--rb-mblue);
        color: #ffffff !important;
        border:none;
        border-radius:4px;
        padding:8px 16px;
        font-weight:600;
      }
      .stDownloadButton > button:hover,
      .stDownloadButton > button:hover * {
        background:var(--rb-blue);
        color: #ffffff !important;
      }

      /* Inputs/selects (keep light, legible) */
      div[data-baseweb="select"] > div {
        background:#fff !important; border:1px solid var(--rb-grey);
        border-radius:4px; color:#000;
      }
      div[data-baseweb="select"] > div:hover { border-color: var(--rb-mblue); }

      /* Sidebar hygiene */
      [data-testid="stSidebar"] { min-height:100vh!important; height:auto!important; overflow-y:visible!important; }
      [data-testid="stSidebarContent"] { min-height:100vh!important; height:auto!important; }
    </style>
    """, unsafe_allow_html=True)

def _logo_src():
    # Prefer a local asset if present
    for p in ("assets/rubrics_logo.png", "assets/rubrics_logo.svg", "rubrics_logo.png"):
        if os.path.exists(p):
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
                ext = "svg+xml" if p.endswith(".svg") else "png"
                return f"data:image/{ext};base64,{b64}"
    # Fallback to hosted
    return "https://rubricsam.com/wp-content/uploads/2021/01/cropped-rubrics-logo-tight.png"

def render_brand_header(title="Issuer Credit Screening Model", subtitle=None, href="https://rubricsam.com"):
    src = _logo_src()
    st.markdown(f"""
    <div class="rb-header">
      <div class="rb-title">
        <h1>{title}</h1>
        {f'<div class="rb-sub">{subtitle}</div>' if subtitle else ''}
      </div>
      <a class="rb-logo" href="{href}" target="_blank" rel="noopener">
        <img src="{src}" alt="Rubrics"/>
      </a>
    </div>
    """, unsafe_allow_html=True)

def apply_rubrics_plot_fonts(fig):
    # DO NOT change any trace colors. Fonts/background only.
    fig.update_layout(
        font=dict(family="Arial, Helvetica, sans-serif", size=13, color="#001E4F"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        title=dict(font=dict(size=16))
    )
    return fig

inject_brand_css()

render_brand_header(
    title="Issuer Credit Screening Model",
    subtitle="ML-Driven Investment Grade & High Yield Analysis"
)

# --- Sidebar for API keys and file upload ---
st.sidebar.markdown(
    f'<div style="padding:4px 0 12px 0;"><img src="{_logo_src()}" alt="Rubrics" style="width:100%; max-width:180px;"></div>',
    unsafe_allow_html=True
)
st.sidebar.header("Configuration")

# OpenAI API Key
try:
    openai_api_key = st.secrets["api_key"]
    st.sidebar.success("OpenAI API key loaded from secrets.")
except (KeyError, TypeError):
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload S&P Data file (Excel or CSV)", type=["xlsx", "csv"])

if not uploaded_file:
    st.warning("Please upload the S&P issuer data file (Excel or CSV) to proceed.")
    st.stop()

# --- Load and Process Data ---
@st.cache_data(show_spinner=False)
def load_and_process_data(uploaded_file):
    """Load data and calculate issuer scores"""
    
    # Load data - handle both Excel and CSV
    file_name = uploaded_file.name.lower()
    if file_name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith('.xlsx'):
        # Try to read the Excel file with flexible sheet name handling
        try:
            # First try "Pasted Values" sheet
            df = pd.read_excel(uploaded_file, sheet_name='Pasted Values')
        except (ValueError, KeyError):
            # If "Pasted Values" doesn't exist, try the first sheet
            df = pd.read_excel(uploaded_file, sheet_name=0)
            st.info("Note: Using first sheet (no 'Pasted Values' sheet found)")
        
        # Only skip first row if it's a duplicated header row (same as columns)
        if len(df) > 0:
            first_row = df.iloc[0].astype(str).str.strip().str.lower().tolist()
            cols_norm = df.columns.astype(str).str.strip().str.lower().tolist()
            has_header_row = first_row == cols_norm
            if has_header_row:
                df = df.iloc[1:].reset_index(drop=True)
    else:
        st.error("Unsupported file format. Please upload .xlsx or .csv file.")
        st.stop()
    
    # Debug: Show initial shape and columns
    # st.sidebar.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    required = [
        'Company ID','Company Name','Ticker','S&P Credit Rating','Sector','Industry',
        'Market Capitalization','Total Debt / EBITDA (x)','Return on Equity','EBITDA Margin',
        'Current Ratio (x)','Total Revenues, 1 Year Growth',
        'Levered Free Cash Flow','Levered Free Cash Flow Margin','Cash from Ops. to Curr. Liab. (x)',
        'Net Debt / EBITDA','Total Debt','Total Debt / Total Capital (%)',
        'Total Revenues, 3 Yr. CAGR','EBITDA, 3 Years CAGR',
        'Return on Assets','EBIT Margin','Quick Ratio (x)'
    ]
    
    # Normalize column names to handle spacing variations and unicode
    df.columns = df.columns.str.strip().str.replace('\u00a0',' ')
    
    # Handle case where CSV split "Total Revenues, 1 Year Growth" into two columns
    if 'Total Revenues' in df.columns and '1 Year Growth' in df.columns and 'Total Revenues, 1 Year Growth' not in df.columns:
        # Use the growth column values and preserve originals
        df['Total Revenues, 1 Year Growth'] = df['1 Year Growth']
        st.info("Note: Merged 'Total Revenues' + '1 Year Growth' -> 'Total Revenues, 1 Year Growth' (sources preserved)")
    
    # Check for alternative column name variations
    column_aliases = {
        'Total Revenues,1 Year Growth': 'Total Revenues, 1 Year Growth',
        'Total Revenues,  1 Year Growth': 'Total Revenues, 1 Year Growth',
        'Total Revenues 1 Year Growth': 'Total Revenues, 1 Year Growth'
    }
    
    for old_name, new_name in column_aliases.items():
        if old_name in df.columns and new_name not in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)
    
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.info(f"Available columns in your file: {list(df.columns)}")
        st.stop()
    
    # Debug: Filter out rows with missing critical data
    initial_count = len(df)
    
    # Remove rows where critical columns are null/empty
    df = df.dropna(subset=['Company ID', 'Company Name', 'S&P Credit Rating'])
    
    # Filter out rows where credit rating is blank or just whitespace
    df = df[df['S&P Credit Rating'].astype(str).str.strip() != '']
    
    rows_after_cleaning = len(df)
    # if rows_after_cleaning < initial_count:
    #     st.sidebar.warning(f"Removed {initial_count - rows_after_cleaning} rows with missing/invalid data")
    
    if len(df) == 0:
        st.error("No valid data rows found after cleaning. Please check your data file.")
        st.info("Ensure that Company ID, Company Name, and S&P Credit Rating columns have valid data.")
        st.stop()
    
    # st.sidebar.success(f"Processing {len(df)} valid rows")
    
    # Calculate quality scores
    def calculate_quality_scores(df):
        scores = pd.DataFrame(index=df.index)
        
        def _pct_to_100(s):
            # normalize strings and strip common decorations
            if isinstance(s, pd.Series):
                s = s.replace(['None','none','N/A','n/a','#N/A'], np.nan)
                s = s.astype(str).str.replace('%','',regex=False).str.replace(',','',regex=False).str.strip()
            s = pd.to_numeric(s, errors='coerce')
            # Heuristic scale detection: if majority <=1 in |value|, treat as fractions
            valid = s.dropna().abs()
            frac_like = (valid <= 1).mean() > 0.6 if len(valid) > 0 else False
            if frac_like:
                s = s * 100.0
            # Bound extreme garbage
            return s.clip(lower=-1000, upper=1000)
        
        # Clean rating function to normalize ratings
        def _clean_rating(x):
            x = str(x).upper().strip()
            # Drop outlook/watch/parentheticals and trailing notes
            x = x.replace('NOT RATED','NR').replace('N/R','NR').replace('N\\M','N/M')
            x = x.split('(')[0].strip()
            x = x.replace(' ','').replace('*','')
            # Map known funky aliases
            alias = {'BBBM':'BBB','BMNS':'B','CCCC':'CCC'}
            return alias.get(x, x)
        
        # Credit rating numeric
        rating_map = {
            'AAA': 21, 'AA+': 20, 'AA': 19, 'AA-': 18,
            'A+': 17, 'A': 16, 'A-': 15,
            'BBB+': 14, 'BBB': 13, 'BBB-': 12,
            'BB+': 11, 'BB': 10, 'BB-': 9,
            'B+': 8, 'B': 7, 'B-': 6,
            'CCC+': 5, 'CCC': 4, 'CCC-': 3,
            'CC': 2, 'C': 1, 'D': 0, 'SD': 0, 'NR': np.nan, 'N/A': np.nan, 'NA': np.nan, 'N/M': np.nan, 'WD': np.nan, '': np.nan
        }
        cr = df['S&P Credit Rating'].map(_clean_rating)
        
        # Debug: Show unique ratings not in map
        unique_ratings = cr.unique()
        unmapped_ratings = [r for r in unique_ratings if r not in rating_map and r != 'NAN']
        # if unmapped_ratings:
        #     st.sidebar.warning(f"Unknown ratings found: {unmapped_ratings[:5]}")

        # Debug: Show how many unique credit ratings we have
        unique_credit_count = cr.nunique()
        # st.sidebar.info(f"Unique credit ratings in data: {unique_credit_count}")
        
        scores['credit_score'] = cr.map(rating_map)
        # Rescale 0-21 to 0-100 so weights are comparable
        scores['credit_score'] = scores['credit_score'] * (100.0/21.0)
        
        # Debug: Check uniqueness in raw financial metrics
        # st.sidebar.write("**Raw Data Unique Values:**")
        # raw_metrics = ['Total Debt / EBITDA (x)', 'Return on Equity', 'EBITDA Margin',
        #               'Current Ratio (x)', 'Total Revenues, 1 Year Growth']
        # for metric in raw_metrics:
        #     if metric in df.columns:
        #         unique_vals = df[metric].nunique()
        #         total_vals = len(df[metric])
        #         st.sidebar.write(f"- {metric[:20]}: {unique_vals}/{total_vals} unique")
        
        # Leverage (lower is better) - Multi-metric approach
        # Net Debt/EBITDA (40% weight)
        net_debt_ebitda_raw = df['Net Debt / EBITDA']
        if isinstance(net_debt_ebitda_raw, pd.Series):
            net_debt_ebitda_raw = net_debt_ebitda_raw.replace(['None', 'none', 'N/A', 'n/a', '#N/A'], np.nan)
        net_debt_ebitda = pd.to_numeric(net_debt_ebitda_raw, errors='coerce')
        net_debt_ebitda = net_debt_ebitda.where(net_debt_ebitda >= 0, other=20.0).fillna(20.0).clip(upper=20.0)
        part1 = (np.minimum(net_debt_ebitda, 3.0)/3.0)*60.0
        part2 = (np.maximum(net_debt_ebitda-3.0, 0.0)/5.0)*40.0
        raw_penalty = np.minimum(part1+part2, 100.0)
        net_debt_score = np.clip(100.0 - raw_penalty, 0.0, 100.0)

        # Total Debt/EBITDA (30% weight)
        debt_ebitda_raw = df['Total Debt / EBITDA (x)']
        if isinstance(debt_ebitda_raw, pd.Series):
            debt_ebitda_raw = debt_ebitda_raw.replace(['None', 'none', 'N/A', 'n/a', '#N/A'], np.nan)
        debt_ebitda = pd.to_numeric(debt_ebitda_raw, errors='coerce')
        debt_ebitda = debt_ebitda.where(debt_ebitda >= 0, other=20.0).fillna(20.0).clip(upper=20.0)
        part1_td = (np.minimum(debt_ebitda, 3.0)/3.0)*60.0
        part2_td = (np.maximum(debt_ebitda-3.0, 0.0)/5.0)*40.0
        raw_penalty_td = np.minimum(part1_td+part2_td, 100.0)
        debt_ebitda_score = np.clip(100.0 - raw_penalty_td, 0.0, 100.0)

        # Debt/Capital (30% weight) - inverse scoring
        debt_capital_raw = df['Total Debt / Total Capital (%)']
        if isinstance(debt_capital_raw, pd.Series):
            debt_capital_raw = debt_capital_raw.replace(['None', 'none', 'N/A', 'n/a', '#N/A'], np.nan)
        debt_capital = pd.to_numeric(debt_capital_raw, errors='coerce').fillna(50).clip(0, 100)
        debt_cap_score = np.clip(100 - debt_capital, 0, 100)

        # Combined leverage score
        scores['leverage_score'] = net_debt_score * 0.4 + debt_ebitda_score * 0.3 + debt_cap_score * 0.3
        
        # Profitability - Enhanced with efficiency metrics
        roe = _pct_to_100(df['Return on Equity'])
        ebitda_margin = _pct_to_100(df['EBITDA Margin'])
        roa = _pct_to_100(df['Return on Assets'])
        ebit_margin = _pct_to_100(df['EBIT Margin'])

        # Each metric capped at reasonable ranges
        roe_score = np.clip(roe, -50, 50) + 50  # Convert to 0-100
        margin_score = np.clip(ebitda_margin, -50, 50) + 50
        roa_score = np.clip(roa * 5, 0, 100)  # ROA typically lower, scale up
        ebit_score = np.clip(ebit_margin * 2, 0, 100)

        scores['profitability_score'] = (roe_score * 0.3 + margin_score * 0.3 +
                                         roa_score * 0.2 + ebit_score * 0.2)
        
        # Liquidity - Dual-metric approach
        current_raw = df['Current Ratio (x)']
        if isinstance(current_raw, pd.Series):
            current_raw = current_raw.replace(['None', 'none', 'N/A', 'n/a', '#N/A'], np.nan)
        current_ratio = pd.to_numeric(current_raw, errors='coerce').clip(lower=0)

        quick_raw = df['Quick Ratio (x)']
        if isinstance(quick_raw, pd.Series):
            quick_raw = quick_raw.replace(['None', 'none', 'N/A', 'n/a', '#N/A'], np.nan)
        quick_ratio = pd.to_numeric(quick_raw, errors='coerce').clip(lower=0)

        # Current ratio: linear up to 3x -> 100
        current_score = np.clip((current_ratio/3.0)*100.0, 0, 100)
        # Quick ratio: linear up to 2x -> 100 (typically lower)
        quick_score = np.clip((quick_ratio/2.0)*100.0, 0, 100)

        scores['liquidity_score'] = current_score * 0.6 + quick_score * 0.4
        
        # Growth - Multi-period approach
        rev_growth_1y = _pct_to_100(df['Total Revenues, 1 Year Growth'])
        rev_cagr_3y = _pct_to_100(df['Total Revenues, 3 Yr. CAGR'])
        ebitda_cagr_3y = _pct_to_100(df['EBITDA, 3 Years CAGR'])

        # Score each component (moderate growth preferred)
        rev_1y_score = np.clip((rev_growth_1y + 10) * 2, 0, 100)
        rev_3y_score = np.clip((rev_cagr_3y + 10) * 2, 0, 100)
        ebitda_3y_score = np.clip((ebitda_cagr_3y + 10) * 2, 0, 100)

        scores['growth_score'] = rev_3y_score * 0.4 + rev_1y_score * 0.3 + ebitda_3y_score * 0.3

        # Cash Flow Score (new 6th factor)
        fcf_raw = df['Levered Free Cash Flow']
        if isinstance(fcf_raw, pd.Series):
            fcf_raw = fcf_raw.replace(['None', 'none', 'N/A', 'n/a', '#N/A'], np.nan)
        fcf = pd.to_numeric(fcf_raw, errors='coerce')

        total_debt_raw = df['Total Debt']
        if isinstance(total_debt_raw, pd.Series):
            total_debt_raw = total_debt_raw.replace(['None', 'none', 'N/A', 'n/a', '#N/A'], np.nan)
        total_debt = pd.to_numeric(total_debt_raw, errors='coerce')

        fcf_margin = _pct_to_100(df['Levered Free Cash Flow Margin'])

        cash_ops_ratio_raw = df['Cash from Ops. to Curr. Liab. (x)']
        if isinstance(cash_ops_ratio_raw, pd.Series):
            cash_ops_ratio_raw = cash_ops_ratio_raw.replace(['None', 'none', 'N/A', 'n/a', '#N/A'], np.nan)
        cash_ops_ratio = pd.to_numeric(cash_ops_ratio_raw, errors='coerce')

        # FCF/Debt ratio (0-100 scale, cap at 50% = 100 score)
        fcf_debt_ratio = (fcf / total_debt).clip(upper=0.5) * 200
        fcf_debt_score = fcf_debt_ratio.fillna(0).clip(0, 100)

        # Cash ops coverage (0-100 scale, 1.0x = 100 score)
        cash_ops_score = (cash_ops_ratio * 100).clip(0, 100)

        # Combine: 50% FCF/Debt, 30% FCF Margin, 20% Cash Ops Coverage
        scores['cash_flow_score'] = (fcf_debt_score * 0.5 + fcf_margin * 0.3 + cash_ops_score * 0.2)

        # Debug: Show count of valid scores
        # if scores['growth_score'].isna().all():
        #     st.sidebar.warning("WARNING: All growth scores are missing - using defaults")

        return scores
    
    quality_scores = calculate_quality_scores(df)
    
    # Reuse the same cleaning logic outside for cohorting (duplication OK/minimal)
    def _clean_rating_outer(x):
        x = str(x).upper().strip()
        x = x.replace('NOT RATED','NR').replace('N/R','NR').replace('N\\M','N/M')
        x = x.split('(')[0].strip().replace(' ','').replace('*','')
        return {'BBBM':'BBB','BMNS':'B','CCCC':'CCC'}.get(x, x)
    df['_Credit_Rating_Clean'] = df['S&P Credit Rating'].map(_clean_rating_outer)
    
    # Calculate composite score (6-factor model)
    weights = {
        'credit_score': 0.20,
        'leverage_score': 0.20,
        'profitability_score': 0.20,
        'liquidity_score': 0.10,
        'growth_score': 0.15,
        'cash_flow_score': 0.15
    }
    
    qs = quality_scores.copy()
    
    # Fill missing values more aggressively
    # First try sector medians
    if 'Sector' in df.columns:
        sector_meds = qs.join(df['Sector']).groupby('Sector').transform('median')
        qs = qs.fillna(sector_meds)
    
    # Then use overall medians
    qs = qs.fillna(qs.median(numeric_only=True))
    
    # Finally, for any remaining NaNs, use default values
    default_scores = {
        'credit_score': 50.0,  # Middle of the range
        'leverage_score': 50.0,
        'profitability_score': 50.0,
        'liquidity_score': 50.0,
        'growth_score': 50.0,
        'cash_flow_score': 50.0
    }
    for col, default_val in default_scores.items():
        if col in qs.columns:
            qs[col] = qs[col].fillna(default_val)
    
    composite_score = sum(qs[col] * weight for col, weight in weights.items())
    
    # Create results dataframe
    results = pd.DataFrame({
        'Company_ID': df['Company ID'],
        'Company_Name': df['Company Name'],
        'Ticker': df['Ticker'],
        'Credit_Rating': df['S&P Credit Rating'],
        'Credit_Rating_Clean': df['_Credit_Rating_Clean'],
        'Sector': df['Sector'],
        'Industry': df['Industry'],
        'Market_Cap': pd.to_numeric(df['Market Capitalization'], errors='coerce'),
        'Composite_Score': composite_score,
        'Credit_Score': quality_scores['credit_score'],
        'Leverage_Score': quality_scores['leverage_score'],
        'Profitability_Score': quality_scores['profitability_score'],
        'Liquidity_Score': quality_scores['liquidity_score'],
        'Growth_Score': quality_scores['growth_score'],
        'Cash_Flow_Score': quality_scores['cash_flow_score']
    })
    
    # Clean data
    # st.sidebar.info(f"Raw results: {len(results)} rows")
    results_clean = results.dropna(subset=['Composite_Score']).copy()
    # st.sidebar.info(f"After cleaning: {len(results_clean)} rows")
    
    # If no data survived cleaning, show diagnostic info
    if len(results_clean) == 0:
        st.error("WARNING: No valid data after processing. Diagnostic information:")
        
        # Show sample of problematic data
        st.subheader("Sample of Raw Data (first 5 rows):")
        st.dataframe(df.head())
        
        # Show which scores are missing
        st.subheader("Score Calculation Issues:")
        score_cols = ['credit_score', 'leverage_score', 'profitability_score', 'liquidity_score', 'growth_score', 'cash_flow_score']
        for col in score_cols:
            if col in quality_scores.columns:
                null_count = quality_scores[col].isna().sum()
                st.write(f"- {col}: {null_count}/{len(quality_scores)} missing values")
        
        # Show composite score issues
        st.write(f"- Composite scores calculated: {(~composite_score.isna()).sum()}/{len(composite_score)}")
        
        st.info("Common causes: Invalid numeric data in financial columns, or unusual credit rating formats")
        st.stop()
    
    # Define Investment Grade vs High Yield
    ig = set('AAA AA+ AA AA- A+ A A- BBB+ BBB BBB-'.split())
    hy = set('BB+ BB BB- B+ B B- CCC+ CCC CCC- CC C D SD'.split())
    # already cleaned; treat NR/NM/WD as Unknown  
    nr_aliases = {'NR','N/M','N/A','NA','WD','UNKNOWN'}
    cr = results_clean['Credit_Rating_Clean'].astype(str).str.strip().str.upper()
    results_clean['IG_HY'] = np.where(cr.isin(ig),'Investment Grade',
                               np.where(cr.isin(hy),'High Yield','Unknown'))
    
    # Define rating groups
    def assign_rating_group(rating):
        rating = str(rating).strip().upper()
        if rating == 'AAA':
            return 'Group 1: AAA'
        elif rating in ['AA+', 'AA', 'AA-']:
            return 'Group 2: AA'
        elif rating in ['A+', 'A', 'A-']:
            return 'Group 3: A'
        elif rating in ['BBB+', 'BBB', 'BBB-']:
            return 'Group 4: BBB'
        elif rating in ['BB+', 'BB', 'BB-']:
            return 'Group 5: BB'
        elif rating in ['B+', 'B', 'B-']:
            return 'Group 6: B'
        elif rating in ['CCC+', 'CCC', 'CCC-', 'CC', 'C', 'D']:
            return 'Group 7: CCC and below'
        else:
            return 'Unknown'
    
    results_clean['Rating_Group'] = results_clean['Credit_Rating_Clean'].apply(assign_rating_group)
    
    # Separate IG and HY
    ig_results = results_clean[results_clean['IG_HY'] == 'Investment Grade'].copy()
    hy_results = results_clean[results_clean['IG_HY'] == 'High Yield'].copy()
    
    # Rankings within IG and HY
    ig_results['IG_Overall_Rank'] = ig_results['Composite_Score'].rank(ascending=False, method='dense')
    hy_results['HY_Overall_Rank'] = hy_results['Composite_Score'].rank(ascending=False, method='dense')
    
    # Rankings within rating groups
    ig_results['Rating_Group_Rank'] = ig_results.groupby('Rating_Group')['Composite_Score'].rank(ascending=False, method='dense')
    hy_results['Rating_Group_Rank'] = hy_results.groupby('Rating_Group')['Composite_Score'].rank(ascending=False, method='dense')
    
    # Percentiles
    ig_results['IG_Percentile'] = ig_results['Composite_Score'].rank(pct=True) * 100
    hy_results['HY_Percentile'] = hy_results['Composite_Score'].rank(pct=True) * 100
    
    # Categories for each group (updated thresholds for 6-factor model)
    ig_results['Category'] = pd.cut(
        ig_results['Composite_Score'],
        bins=[0, 42, 57, 72, 100],
        include_lowest=True,
        labels=['Avoid', 'Hold', 'Buy', 'Strong Buy']
    )

    hy_results['Category'] = pd.cut(
        hy_results['Composite_Score'],
        bins=[0, 37, 52, 67, 100],
        include_lowest=True,
        labels=['Avoid', 'Hold', 'Buy', 'Strong Buy']
    )
    
    # Combine back
    unknown_results = results_clean[results_clean['IG_HY'] == 'Unknown'].copy()
    results_final = pd.concat([ig_results, hy_results, unknown_results], ignore_index=True)
    
    # PCA for visualization
    quality_cols = ['Credit_Score', 'Leverage_Score', 'Profitability_Score', 'Liquidity_Score', 'Growth_Score', 'Cash_Flow_Score']

    # IG PCA (only if we have enough data)
    if len(ig_results) > 10:
        ig_features = ig_results[quality_cols].fillna(ig_results[quality_cols].median())

        scaler_ig = RobustScaler()
        ig_scaled = scaler_ig.fit_transform(ig_features)
        pca_ig = PCA(n_components=2)
        ig_pca = pca_ig.fit_transform(ig_scaled)

        # Stabilize orientation: ensure higher composite scores are on the right
        pc1 = ig_pca[:, 0]
        # Use Composite_Score for correlation check (more reliable with 6 factors)
        if np.corrcoef(pc1, ig_results['Composite_Score'])[0,1] < 0:
            pc1 = -pc1
            ig_pca[:, 0] = -ig_pca[:, 0]  # Also flip the original array
        ig_results['PC1'] = pc1
        ig_results['PC2'] = ig_pca[:, 1]

        # Add small jitter to break up linear patterns from credit score discretization
        np.random.seed(42)  # For reproducibility
        jitter_strength = 0.03  # Small amount of noise
        ig_results['PC1'] = ig_results['PC1'] + np.random.normal(0, jitter_strength, len(ig_results))
        ig_results['PC2'] = ig_results['PC2'] + np.random.normal(0, jitter_strength, len(ig_results))
    else:
        ig_results['PC1'] = np.nan
        ig_results['PC2'] = np.nan

    # HY PCA (only if we have enough data)
    if len(hy_results) > 10:
        hy_features = hy_results[quality_cols].fillna(hy_results[quality_cols].median())

        scaler_hy = RobustScaler()
        hy_scaled = scaler_hy.fit_transform(hy_features)
        pca_hy = PCA(n_components=2)
        hy_pca = pca_hy.fit_transform(hy_scaled)

        # Stabilize orientation: ensure higher composite scores are on the right
        pc1 = hy_pca[:, 0]
        # Use Composite_Score for correlation check (more reliable with 6 factors)
        if np.corrcoef(pc1, hy_results['Composite_Score'])[0,1] < 0:
            pc1 = -pc1
            hy_pca[:, 0] = -hy_pca[:, 0]  # Also flip the original array
        hy_results['PC1'] = pc1
        hy_results['PC2'] = hy_pca[:, 1]

        # Add small jitter to break up linear patterns from credit score discretization
        np.random.seed(43)  # Different seed from IG for variety
        jitter_strength = 0.03  # Small amount of noise
        hy_results['PC1'] = hy_results['PC1'] + np.random.normal(0, jitter_strength, len(hy_results))
        hy_results['PC2'] = hy_results['PC2'] + np.random.normal(0, jitter_strength, len(hy_results))
    else:
        hy_results['PC1'] = np.nan
        hy_results['PC2'] = np.nan
    
    return results_final, ig_results, hy_results

# Load data
progress_bar = st.progress(0)
status_text = st.empty()

with st.spinner("Loading and processing data..."):
    status_text.text("Loading Excel file...")
    progress_bar.progress(20)

    # Simply pass the uploaded file directly - Streamlit handles caching properly
    results_final, ig_results, hy_results = load_and_process_data(uploaded_file)

    progress_bar.progress(100)
    status_text.text("Processing complete!")
    
# Clear progress indicators after a moment
import time
time.sleep(0.5)
progress_bar.empty()
status_text.empty()

st.success(f"Processed {len(results_final):,} issuers ({len(ig_results):,} IG, {len(hy_results):,} HY)")

# --- Summary Metrics ---
st.subheader("Universe Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Issuers", f"{len(results_final):,}")
with col2:
    ig_attractive = len(ig_results[ig_results['Category'].isin(['Strong Buy', 'Buy'])])
    ig_denom = max(len(ig_results), 1)
    st.metric("IG Attractive", f"{ig_attractive}", f"{ig_attractive/ig_denom*100:.1f}%")
with col3:
    hy_attractive = len(hy_results[hy_results['Category'].isin(['Strong Buy', 'Buy'])])
    hy_denom = max(len(hy_results), 1)
    st.metric("HY Attractive", f"{hy_attractive}", f"{hy_attractive/hy_denom*100:.1f}%")
with col4:
    total_attractive = ig_attractive + hy_attractive
    total_rated = len(ig_results) + len(hy_results)  # Only IG + HY, not Unknown
    total_denom = max(total_rated, 1)
    st.metric("Total Attractive", f"{total_attractive}", f"{total_attractive/total_denom*100:.1f}%")

# --- Tabs for different views ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "EDA",
    "Methodology",
    "Top Rankings",
    "Rating Group Analysis",
    "Detailed Data",
    "Overview & Positioning",
    "AI Analysis"
])

with tab6:
    st.header("Investment Grade vs High Yield Positioning")

    # Add explanation of the maps
    st.info("""
    **How to read these maps:**
    - Each dot represents one issuer, sized by composite score
    - Colors indicate our recommendation: Green = Strong Buy, Blue = Buy, Yellow = Hold, Red = Avoid
    - Position reflects relative credit quality and financial characteristics
    - Companies closer together have similar credit profiles
    - The axes represent mathematical combinations of all scoring factors
    """)
    
    # Create side-by-side positioning maps
    colors_map = {'Strong Buy': '#00C851', 'Buy': '#33b5e5', 'Hold': '#ffbb33', 'Avoid': '#ff4444'}
    
    # IG Positioning Map
    st.subheader("Investment Grade Issuer Map")
    
    if len(ig_results) > 0:
        fig_ig = go.Figure()
        
        for category in ['Avoid', 'Hold', 'Buy', 'Strong Buy']:
            mask = ig_results['Category'] == category
            data = ig_results[mask]
            fig_ig.add_trace(go.Scatter(
                x=data['PC1'], 
                y=data['PC2'],
                mode='markers',
                marker=dict(
                    size=np.clip(data['Composite_Score'], 20, 100)*0.4,
                color=colors_map[category],
                opacity=0.6,
                line=dict(width=0.5, color='white')
            ),
            name=f'{category} ({len(data)})',
            text=data.apply(lambda r: f"{r['Company_Name']} | {r['Credit_Rating']} | {r['Category']} | Score: {r['Composite_Score']:.1f}", axis=1),
            hovertemplate='%{text}<extra></extra>'
        ))
    
        # Highlight top 10
        top_10_ig = ig_results.nlargest(10, 'Composite_Score')
        fig_ig.add_trace(go.Scatter(
            x=top_10_ig['PC1'],
            y=top_10_ig['PC2'],
            mode='markers',
            marker=dict(
                size=20,
                color='rgba(255,215,0,0)',
                line=dict(width=3, color='gold')
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_ig.update_layout(
            title=f'Investment Grade Positioning Map (n={len(ig_results)})',
            xaxis_title='Overall Credit Quality (Better)',
            yaxis_title='Financial Strength vs Leverage Balance',
            hovermode='closest',
            height=500
        )
        st.plotly_chart(apply_rubrics_plot_fonts(fig_ig), use_container_width=True)
    else:
        st.info("No Investment Grade issuers in this dataset")
    
    # HY Positioning Map
    st.subheader("High Yield Issuer Map")
    
    if len(hy_results) > 0:
        fig_hy = go.Figure()
        
        for category in ['Avoid', 'Hold', 'Buy', 'Strong Buy']:
            mask = hy_results['Category'] == category
            data = hy_results[mask]
            fig_hy.add_trace(go.Scatter(
                x=data['PC1'],
                y=data['PC2'],
                mode='markers',
                marker=dict(
                    size=np.clip(data['Composite_Score'], 20, 100)*0.4,
                    color=colors_map[category],
                    opacity=0.6,
                    line=dict(width=0.5, color='white')
                ),
                name=f'{category} ({len(data)})',
                text=data.apply(lambda r: f"{r['Company_Name']} | {r['Credit_Rating']} | {r['Category']} | Score: {r['Composite_Score']:.1f}", axis=1),
                hovertemplate='%{text}<extra></extra>'
            ))
    
        # Highlight top 10
        top_10_hy = hy_results.nlargest(10, 'Composite_Score')
        fig_hy.add_trace(go.Scatter(
            x=top_10_hy['PC1'],
            y=top_10_hy['PC2'],
            mode='markers',
            marker=dict(
                size=20,
                color='rgba(255,215,0,0)',
                line=dict(width=3, color='gold')
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_hy.update_layout(
            title=f'High Yield Positioning Map (n={len(hy_results)})',
            xaxis_title='Overall Credit Quality (Better)',
            yaxis_title='Financial Strength vs Leverage Balance',
            hovermode='closest',
            height=500
        )
        st.plotly_chart(apply_rubrics_plot_fonts(fig_hy), use_container_width=True)
    else:
        st.info("No High Yield issuers in this dataset")
    
    # Category distributions
    st.subheader("Category Distributions")
    col1, col2 = st.columns(2)
    
    if len(ig_results) > 0:
        with col1:
            ig_cat_dist = ig_results['Category'].value_counts()
            fig_ig_cat = go.Figure(go.Bar(
                x=ig_cat_dist.values,
                y=ig_cat_dist.index,
                orientation='h',
            marker=dict(color=[colors_map[cat] for cat in ig_cat_dist.index]),
            text=[f"{val} ({(val/max(len(ig_results),1))*100:.1f}%)" for val in ig_cat_dist.values],
            textposition='auto'
        ))
            fig_ig_cat.update_layout(
                title='Investment Grade Categories',
                xaxis_title='Number of Issuers',
                height=300
            )
            st.plotly_chart(apply_rubrics_plot_fonts(fig_ig_cat), use_container_width=True)
    else:
        with col1:
            st.info("No IG issuers to display")
    
    if len(hy_results) > 0:
        with col2:
            hy_cat_dist = hy_results['Category'].value_counts()
            fig_hy_cat = go.Figure(go.Bar(
                x=hy_cat_dist.values,
                y=hy_cat_dist.index,
                orientation='h',
                marker=dict(color=[colors_map[cat] for cat in hy_cat_dist.index]),
                text=[f"{val} ({(val/max(len(hy_results),1))*100:.1f}%)" for val in hy_cat_dist.values],
                textposition='auto'
            ))
            fig_hy_cat.update_layout(
                title='High Yield Categories',
                xaxis_title='Number of Issuers',
                height=300
            )
            st.plotly_chart(apply_rubrics_plot_fonts(fig_hy_cat), use_container_width=True)
    else:
        with col2:
            st.info("No HY issuers to display")

with tab3:
    st.header("Top Ranked Issuers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 20 Investment Grade")
        top_20_ig = ig_results.nlargest(20, 'Composite_Score')[
            ['IG_Overall_Rank', 'Company_Name', 'Credit_Rating', 'Industry', 'Composite_Score', 'Category']
        ].copy()
        top_20_ig['Composite_Score'] = top_20_ig['Composite_Score'].round(1)
        st.dataframe(top_20_ig, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Top 20 High Yield")
        top_20_hy = hy_results.nlargest(20, 'Composite_Score')[
            ['HY_Overall_Rank', 'Company_Name', 'Credit_Rating', 'Industry', 'Composite_Score', 'Category']
        ].copy()
        top_20_hy['Composite_Score'] = top_20_hy['Composite_Score'].round(1)
        st.dataframe(top_20_hy, use_container_width=True, hide_index=True)
    
    # Top performers bar charts
    st.subheader("Top Performers Comparison")
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Top 15 Investment Grade", "Top 15 High Yield")
    )
    
    top_15_ig = ig_results.nlargest(15, 'Composite_Score')
    fig.add_trace(
        go.Bar(
            y=top_15_ig['Company_Name'].str[:30],
            x=top_15_ig['Composite_Score'],
            orientation='h',
            marker=dict(color=[colors_map[cat] for cat in top_15_ig['Category']]),
            text=top_15_ig['Composite_Score'].round(1),
            textposition='auto',
            showlegend=False
        ),
        row=1, col=1
    )
    
    top_15_hy = hy_results.nlargest(15, 'Composite_Score')
    fig.add_trace(
        go.Bar(
            y=top_15_hy['Company_Name'].str[:30],
            x=top_15_hy['Composite_Score'],
            orientation='h',
            marker=dict(color=[colors_map[cat] for cat in top_15_hy['Category']]),
            text=top_15_hy['Composite_Score'].round(1),
            textposition='auto',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Composite Score", row=1, col=1)
    fig.update_xaxes(title_text="Composite Score", row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_layout(height=600)
    
    st.plotly_chart(apply_rubrics_plot_fonts(fig), use_container_width=True)

with tab4:
    st.header("Rating Group Analysis")
    
    rating_groups = [
        ('Group 1: AAA', 'AAA'),
        ('Group 2: AA', 'AA+/AA/AA-'),
        ('Group 3: A', 'A+/A/A-'),
        ('Group 4: BBB', 'BBB+/BBB/BBB-'),
        ('Group 5: BB', 'BB+/BB/BB-'),
        ('Group 6: B', 'B+/B/B-'),
        ('Group 7: CCC and below', 'CCC+/CCC/CCC-/CC/C/D')
    ]
    
    # Rating group distribution
    st.subheader("Issuer Distribution by Rating Group")
    rating_dist = results_final.groupby(['Rating_Group', 'IG_HY']).size().unstack(fill_value=0)
    
    fig = go.Figure()
    
    # Handle Investment Grade data - use array of zeros if column doesn't exist
    if 'Investment Grade' in rating_dist.columns:
        ig_values = rating_dist['Investment Grade']
    else:
        ig_values = [0] * len(rating_dist.index)
    
    fig.add_trace(go.Bar(
        name='Investment Grade',
        x=rating_dist.index,
        y=ig_values,
        marker_color='#4CAF50',
        text=ig_values,
        textposition='auto'
    ))
    
    # Handle High Yield data - use array of zeros if column doesn't exist
    if 'High Yield' in rating_dist.columns:
        hy_values = rating_dist['High Yield']
    else:
        hy_values = [0] * len(rating_dist.index)
    
    fig.add_trace(go.Bar(
        name='High Yield',
        x=rating_dist.index,
        y=hy_values,
        marker_color='#FF9800',
        text=hy_values,
        textposition='auto'
    ))
    fig.update_layout(
        barmode='group',
        title='Distribution by Rating Group',
        xaxis_title='Rating Group',
        yaxis_title='Number of Issuers',
        height=400
    )
    st.plotly_chart(apply_rubrics_plot_fonts(fig), use_container_width=True)
    
    # Top performers in each rating group
    st.subheader("Top Performers by Rating Group")
    
    selected_group = st.selectbox(
        "Select Rating Group",
        [g[0] for g in rating_groups]
    )
    
    group_data = results_final[results_final['Rating_Group'] == selected_group]
    if len(group_data) > 0:
        top_group = group_data.nlargest(20, 'Composite_Score')[
            ['Company_Name', 'Credit_Rating', 'Industry', 'Composite_Score', 
             'Category', 'Rating_Group_Rank']
        ].copy()
        top_group['Composite_Score'] = top_group['Composite_Score'].round(1)
        
        st.dataframe(top_group, use_container_width=True, hide_index=True)
        
        # Bar chart
        fig = go.Figure(go.Bar(
            y=top_group['Company_Name'].str[:40],
            x=top_group['Composite_Score'],
            orientation='h',
            marker=dict(color=[colors_map[cat] for cat in top_group['Category']]),
            text=top_group['Composite_Score'],
            textposition='auto'
        ))
        fig.update_layout(
            title=f'Top 20 in {selected_group}',
            xaxis_title='Composite Score',
            yaxis=dict(autorange="reversed"),
            height=600
        )
        st.plotly_chart(apply_rubrics_plot_fonts(fig), use_container_width=True)
    else:
        st.info(f"No issuers found in {selected_group}")

with tab5:
    st.header("Detailed Issuer Data")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_ig_hy = st.multiselect(
            "Filter by IG/HY",
            ['Investment Grade', 'High Yield', 'Unknown'],
            default=['Investment Grade', 'High Yield']
        )
    
    with col2:
        filter_category = st.multiselect(
            "Filter by Category",
            ['Strong Buy', 'Buy', 'Hold', 'Avoid'],
            default=['Strong Buy', 'Buy']
        )
    
    with col3:
        min_score = st.slider(
            "Minimum Composite Score",
            0, 100, 50
        )
    
    # Apply filters
    filtered_data = results_final[
        (results_final['IG_HY'].isin(filter_ig_hy)) &
        (results_final['Category'].isin(filter_category)) &
        (results_final['Composite_Score'] >= min_score)
    ].sort_values('Composite_Score', ascending=False)
    
    st.write(f"Showing {len(filtered_data)} issuers")
    
    # Display data
    display_cols = [
        'Company_Name', 'Ticker', 'Credit_Rating', 'Rating_Group', 'IG_HY',
        'Composite_Score', 'Credit_Score', 'Leverage_Score', 
        'Profitability_Score', 'Liquidity_Score', 'Growth_Score',
        'Category', 'Industry', 'Sector'
    ]
    
    st.dataframe(
        filtered_data[display_cols].round(1),
        use_container_width=True,
        hide_index=True
    )
    
    # Download button
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="issuer_screening_filtered.csv",
        mime="text/csv"
    )

with tab1:
    st.header("Exploratory Data Analysis (EDA)")

    st.markdown("""
    Comprehensive statistical analysis and visualizations of the issuer dataset.
    """)

    # Section 1: Dataset Overview
    st.subheader("1. Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Issuers", f"{len(results_final):,}")
    with col2:
        st.metric("Investment Grade", f"{len(ig_results):,}")
    with col3:
        st.metric("High Yield", f"{len(hy_results):,}")
    with col4:
        unknown_count = len(results_final[results_final['IG_HY'] == 'Unknown'])
        st.metric("Unknown/Unrated", f"{unknown_count:,}")

    # Section 2: Score Distribution Analysis
    st.subheader("2. Composite Score Distribution")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Investment Grade Score Distribution**")
        if len(ig_results) > 0:
            fig_ig_dist = go.Figure()
            fig_ig_dist.add_trace(go.Histogram(
                x=ig_results['Composite_Score'],
                nbinsx=20,
                marker_color='#2C5697',
                name='IG Scores'
            ))
            fig_ig_dist.update_layout(
                xaxis_title='Composite Score',
                yaxis_title='Count',
                showlegend=False,
                height=350
            )
            st.plotly_chart(apply_rubrics_plot_fonts(fig_ig_dist), use_container_width=True)

            # Statistics
            st.markdown(f"""
            - **Mean:** {ig_results['Composite_Score'].mean():.2f}
            - **Median:** {ig_results['Composite_Score'].median():.2f}
            - **Std Dev:** {ig_results['Composite_Score'].std():.2f}
            - **Min:** {ig_results['Composite_Score'].min():.2f}
            - **Max:** {ig_results['Composite_Score'].max():.2f}
            """)
        else:
            st.info("No Investment Grade data available")

    with col2:
        st.markdown("**High Yield Score Distribution**")
        if len(hy_results) > 0:
            fig_hy_dist = go.Figure()
            fig_hy_dist.add_trace(go.Histogram(
                x=hy_results['Composite_Score'],
                nbinsx=20,
                marker_color='#CF4520',
                name='HY Scores'
            ))
            fig_hy_dist.update_layout(
                xaxis_title='Composite Score',
                yaxis_title='Count',
                showlegend=False,
                height=350
            )
            st.plotly_chart(apply_rubrics_plot_fonts(fig_hy_dist), use_container_width=True)

            # Statistics
            st.markdown(f"""
            - **Mean:** {hy_results['Composite_Score'].mean():.2f}
            - **Median:** {hy_results['Composite_Score'].median():.2f}
            - **Std Dev:** {hy_results['Composite_Score'].std():.2f}
            - **Min:** {hy_results['Composite_Score'].min():.2f}
            - **Max:** {hy_results['Composite_Score'].max():.2f}
            """)
        else:
            st.info("No High Yield data available")

    # Section 3: Factor Score Analysis
    st.subheader("3. Individual Factor Score Analysis")

    factor_cols = ['Credit_Score', 'Leverage_Score', 'Profitability_Score', 'Liquidity_Score', 'Growth_Score']
    factor_names = ['Credit', 'Leverage', 'Profitability', 'Liquidity', 'Growth']

    # Box plots for factor comparison
    st.markdown("**Factor Score Comparison (IG vs HY)**")

    fig_factors = make_subplots(
        rows=2, cols=3,
        subplot_titles=factor_names,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    for idx, (col, name) in enumerate(zip(factor_cols, factor_names)):
        row = idx // 3 + 1
        col_pos = idx % 3 + 1

        if len(ig_results) > 0:
            fig_factors.add_trace(
                go.Box(y=ig_results[col], name='IG', marker_color='#2C5697', showlegend=(idx==0)),
                row=row, col=col_pos
            )

        if len(hy_results) > 0:
            fig_factors.add_trace(
                go.Box(y=hy_results[col], name='HY', marker_color='#CF4520', showlegend=(idx==0)),
                row=row, col=col_pos
            )

    fig_factors.update_yaxes(title_text="Score", range=[0, 100])
    fig_factors.update_layout(height=600, showlegend=True)

    st.plotly_chart(apply_rubrics_plot_fonts(fig_factors), use_container_width=True)

    # Section 4: Correlation Analysis
    st.subheader("4. Factor Correlation Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Investment Grade Correlation Matrix**")
        if len(ig_results) > 5:
            corr_ig = ig_results[factor_cols].corr()

            fig_corr_ig = go.Figure(data=go.Heatmap(
                z=corr_ig.values,
                x=factor_names,
                y=factor_names,
                colorscale='RdBu',
                zmid=0,
                text=corr_ig.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            fig_corr_ig.update_layout(height=400)
            st.plotly_chart(apply_rubrics_plot_fonts(fig_corr_ig), use_container_width=True)
        else:
            st.info("Insufficient data for correlation analysis")

    with col2:
        st.markdown("**High Yield Correlation Matrix**")
        if len(hy_results) > 5:
            corr_hy = hy_results[factor_cols].corr()

            fig_corr_hy = go.Figure(data=go.Heatmap(
                z=corr_hy.values,
                x=factor_names,
                y=factor_names,
                colorscale='RdBu',
                zmid=0,
                text=corr_hy.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            fig_corr_hy.update_layout(height=400)
            st.plotly_chart(apply_rubrics_plot_fonts(fig_corr_hy), use_container_width=True)
        else:
            st.info("Insufficient data for correlation analysis")

    # Section 5: Sector Analysis
    st.subheader("5. Sector Analysis")

    if 'Sector' in results_final.columns:
        # Top sectors by count
        st.markdown("**Issuer Distribution by Sector**")

        sector_counts = results_final.groupby(['Sector', 'IG_HY']).size().unstack(fill_value=0)

        fig_sector = go.Figure()

        if 'Investment Grade' in sector_counts.columns:
            fig_sector.add_trace(go.Bar(
                name='Investment Grade',
                x=sector_counts.index,
                y=sector_counts['Investment Grade'],
                marker_color='#2C5697'
            ))

        if 'High Yield' in sector_counts.columns:
            fig_sector.add_trace(go.Bar(
                name='High Yield',
                x=sector_counts.index,
                y=sector_counts['High Yield'],
                marker_color='#CF4520'
            ))

        fig_sector.update_layout(
            barmode='stack',
            xaxis_title='Sector',
            yaxis_title='Number of Issuers',
            height=400,
            xaxis={'categoryorder': 'total descending'}
        )
        st.plotly_chart(apply_rubrics_plot_fonts(fig_sector), use_container_width=True)

        # Average scores by sector
        st.markdown("**Average Composite Score by Sector**")

        sector_scores = results_final.groupby('Sector')['Composite_Score'].agg(['mean', 'count']).reset_index()
        sector_scores = sector_scores.sort_values('mean', ascending=False)

        fig_sector_scores = go.Figure(go.Bar(
            x=sector_scores['mean'],
            y=sector_scores['Sector'],
            orientation='h',
            marker_color='#2C5697',
            text=sector_scores['mean'].round(1),
            textposition='auto',
            customdata=sector_scores['count'],
            hovertemplate='<b>%{y}</b><br>Avg Score: %{x:.1f}<br>Count: %{customdata}<extra></extra>'
        ))

        fig_sector_scores.update_layout(
            xaxis_title='Average Composite Score',
            yaxis_title='Sector',
            height=max(400, len(sector_scores) * 25),
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(apply_rubrics_plot_fonts(fig_sector_scores), use_container_width=True)

        # Sector breakdown by category
        st.markdown("**Sector Breakdown by Investment Category**")

        sector_category = results_final.groupby(['Sector', 'Category']).size().unstack(fill_value=0)

        # Reorder columns to match color scheme
        category_order = ['Strong Buy', 'Buy', 'Hold', 'Avoid']
        colors_map = {'Strong Buy': '#00C851', 'Buy': '#33b5e5', 'Hold': '#ffbb33', 'Avoid': '#ff4444'}

        fig_sector_cat = go.Figure()

        for cat in category_order:
            if cat in sector_category.columns:
                fig_sector_cat.add_trace(go.Bar(
                    name=cat,
                    x=sector_category.index,
                    y=sector_category[cat],
                    marker_color=colors_map[cat]
                ))

        fig_sector_cat.update_layout(
            barmode='stack',
            xaxis_title='Sector',
            yaxis_title='Number of Issuers',
            height=450,
            xaxis={'categoryorder': 'total descending'},
            legend=dict(title='Category')
        )
        st.plotly_chart(apply_rubrics_plot_fonts(fig_sector_cat), use_container_width=True)

        # Factor scores by sector - heatmap
        st.markdown("**Average Factor Scores by Sector**")

        sector_factor_avg = results_final.groupby('Sector')[factor_cols].mean()

        fig_sector_heatmap = go.Figure(data=go.Heatmap(
            z=sector_factor_avg.values,
            x=factor_names,
            y=sector_factor_avg.index,
            colorscale='Blues',
            text=sector_factor_avg.values.round(1),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Score")
        ))

        fig_sector_heatmap.update_layout(
            xaxis_title='Factor',
            yaxis_title='Sector',
            height=max(400, len(sector_factor_avg) * 30)
        )
        st.plotly_chart(apply_rubrics_plot_fonts(fig_sector_heatmap), use_container_width=True)

        # Sector comparison - box plots for selected sectors
        st.markdown("**Sector Composite Score Comparison**")

        top_sectors = results_final['Sector'].value_counts().head(10).index.tolist()

        fig_sector_box = go.Figure()

        for sector in top_sectors:
            sector_data = results_final[results_final['Sector'] == sector]
            fig_sector_box.add_trace(go.Box(
                y=sector_data['Composite_Score'],
                name=sector,
                marker_color='#2C5697'
            ))

        fig_sector_box.update_layout(
            xaxis_title='Sector',
            yaxis_title='Composite Score',
            height=500,
            showlegend=False
        )
        st.plotly_chart(apply_rubrics_plot_fonts(fig_sector_box), use_container_width=True)

        # Individual factor comparison by sector
        st.markdown("**Factor Score Distribution by Sector (Top 6 Sectors)**")

        top_6_sectors = results_final['Sector'].value_counts().head(6).index.tolist()

        fig_sector_factors = make_subplots(
            rows=2, cols=3,
            subplot_titles=factor_names,
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        for idx, (col, name) in enumerate(zip(factor_cols, factor_names)):
            row = idx // 3 + 1
            col_pos = idx % 3 + 1

            for sector in top_6_sectors:
                sector_data = results_final[results_final['Sector'] == sector]
                fig_sector_factors.add_trace(
                    go.Box(y=sector_data[col], name=sector, showlegend=(idx==0)),
                    row=row, col=col_pos
                )

        fig_sector_factors.update_yaxes(title_text="Score", range=[0, 100])
        fig_sector_factors.update_layout(height=700, showlegend=True, legend=dict(title='Sector'))

        st.plotly_chart(apply_rubrics_plot_fonts(fig_sector_factors), use_container_width=True)

        # Sector performance metrics table
        st.markdown("**Sector Performance Metrics**")

        sector_metrics = results_final.groupby('Sector').agg({
            'Composite_Score': ['count', 'mean', 'std', 'min', 'max'],
            'Credit_Score': 'mean',
            'Leverage_Score': 'mean',
            'Profitability_Score': 'mean',
            'Liquidity_Score': 'mean',
            'Growth_Score': 'mean'
        }).round(2)

        sector_metrics.columns = ['_'.join(col).strip() for col in sector_metrics.columns.values]
        sector_metrics = sector_metrics.reset_index()
        sector_metrics = sector_metrics.sort_values('Composite_Score_mean', ascending=False)

        st.dataframe(sector_metrics, use_container_width=True, hide_index=True)

    # Section 6: Rating Distribution
    st.subheader("6. Credit Rating Distribution")

    rating_dist = results_final['Credit_Rating_Clean'].value_counts().head(15)

    fig_rating = go.Figure(go.Bar(
        x=rating_dist.index,
        y=rating_dist.values,
        marker_color='#2C5697',
        text=rating_dist.values,
        textposition='auto'
    ))

    fig_rating.update_layout(
        title='Top 15 Credit Ratings by Frequency',
        xaxis_title='Credit Rating',
        yaxis_title='Number of Issuers',
        height=400
    )
    st.plotly_chart(apply_rubrics_plot_fonts(fig_rating), use_container_width=True)

    # Section 7: Summary Statistics Table
    st.subheader("7. Summary Statistics by Category")

    summary_stats = results_final.groupby(['IG_HY', 'Category']).agg({
        'Composite_Score': ['count', 'mean', 'std', 'min', 'max'],
        'Credit_Score': 'mean',
        'Leverage_Score': 'mean',
        'Profitability_Score': 'mean',
        'Liquidity_Score': 'mean',
        'Growth_Score': 'mean'
    }).round(2)

    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
    summary_stats = summary_stats.reset_index()

    st.dataframe(summary_stats, use_container_width=True, hide_index=True)

with tab2:
    st.header("Scoring Methodology & Ranking System")

    st.markdown("""
    This tab explains how the model scores issuers and assigns investment categories.
    """)

    # Section 1: Composite Scoring Engine
    st.subheader("1. Six-Factor Composite Scoring Engine")

    st.markdown("""
    The model calculates a composite score (0-100) for each issuer by combining six financial dimensions.
    Each factor is normalized to a 0-100 scale, then weighted to create the final composite score.
    """)

    # Create a visual table for the weights
    weights_df = pd.DataFrame({
        'Factor': ['Credit Score', 'Leverage Score', 'Profitability Score', 'Liquidity Score', 'Growth Score', 'Cash Flow Score'],
        'Weight': ['20%', '20%', '20%', '10%', '15%', '15%'],
        'Input Metrics': [
            'S&P Credit Rating',
            'Net Debt/EBITDA, Debt/EBITDA, Debt/Capital',
            'ROE, EBITDA Margin, ROA, EBIT Margin',
            'Current Ratio, Quick Ratio',
            'Revenue 1Y/3Y CAGR, EBITDA 3Y CAGR',
            'FCF/Debt, FCF Margin, Cash Ops Coverage'
        ],
        'Interpretation': [
            'Higher credit rating = higher score',
            'Lower leverage = higher score',
            'Higher profitability = higher score',
            'Higher liquidity = higher score',
            'Moderate growth = higher score',
            'Stronger cash generation = higher score'
        ]
    })

    st.dataframe(weights_df, use_container_width=True, hide_index=True)

    st.markdown("""
    **Composite Score Formula:**
    ```
    Composite Score = (Credit Score x 0.20) + (Leverage Score x 0.20) +
                      (Profitability Score x 0.20) + (Liquidity Score x 0.10) +
                      (Growth Score x 0.15) + (Cash Flow Score x 0.15)
    ```
    """)

    # Section 2: Individual Factor Calculations
    st.subheader("2. Individual Factor Calculations")

    with st.expander("Credit Score (20% weight)"):
        st.markdown("""
        **Input:** S&P Credit Rating

        **Calculation:**
        - Each S&P rating is mapped to a numeric value (AAA=21, AA+=20, ... D=0)
        - Scaled to 0-100 range: `(rating_value / 21) x 100`
        - Handles rating variations: removes outlook indicators (NEG, POS, WATCH)
        - Maps aliases: BBBM->BBB, BMNS->B, CCCC->CCC

        **Example:**
        - AAA -> 21/21 x 100 = **100.0**
        - A -> 16/21 x 100 = **76.2**
        - BB -> 10/21 x 100 = **47.6**
        """)

    with st.expander("Leverage Score (20% weight)"):
        st.markdown("""
        **Inputs:** Net Debt/EBITDA (40%), Total Debt/EBITDA (30%), Debt/Total Capital % (30%)

        **Calculation (inverse scoring - lower leverage is better):**
        - **Net Debt/EBITDA:** Piecewise penalty (0-3x: 60pts, 3-8x: 40pts), Score = 100 - penalty
        - **Total Debt/EBITDA:** Same penalty structure as Net Debt/EBITDA
        - **Debt/Capital %:** Inverse linear, Score = 100 - Debt%
        - Combined: `0.4 x NetDebt + 0.3 x TotalDebt + 0.3 x DebtCap`

        **Example:**
        - Net Debt 2x, Total Debt 2.5x, Debt/Cap 40% -> **62.5**
        - Net Debt 4x, Total Debt 5x, Debt/Cap 60% -> **30.0**
        """)

    with st.expander("Profitability Score (20% weight)"):
        st.markdown("""
        **Inputs:** ROE (30%), EBITDA Margin (30%), ROA (20%), EBIT Margin (20%)

        **Calculation:**
        - **ROE:** Clipped to +/-50%, converted to 0-100 scale: `(ROE + 50)`
        - **EBITDA Margin:** Same as ROE
        - **ROA:** Scaled up 5x (typically lower), clipped to 0-100
        - **EBIT Margin:** Scaled up 2x, clipped to 0-100
        - Combined: `0.3xROE + 0.3xMargin + 0.2xROA + 0.2xEBIT`

        **Example:**
        - ROE=20%, EBITDA=25%, ROA=8%, EBIT=15% -> **71.0**
        - ROE=10%, EBITDA=15%, ROA=5%, EBIT=10% -> **57.5**
        """)

    with st.expander("Liquidity Score (10% weight)"):
        st.markdown("""
        **Inputs:** Current Ratio (60%), Quick Ratio (40%)

        **Calculation:**
        - **Current Ratio:** Linear scaling up to 3.0x: `(Current / 3.0) x 100`, capped at 100
        - **Quick Ratio:** Linear scaling up to 2.0x: `(Quick / 2.0) x 100`, capped at 100
        - Combined: `0.6 x Current + 0.4 x Quick`

        **Example:**
        - Current=2.0x, Quick=1.2x -> (66.7 x 0.6 + 60 x 0.4) = **64.0**
        - Current=3.0x, Quick=2.0x -> **100.0**
        """)

    with st.expander("Growth Score (15% weight)"):
        st.markdown("""
        **Inputs:** Revenue 3Y CAGR (40%), Revenue 1Y Growth (30%), EBITDA 3Y CAGR (30%)

        **Calculation:**
        - Each metric: `(Growth% + 10) x 2`, capped at 0-100
        - Rewards moderate sustained growth over short-term volatility
        - Combined: `0.4 x Rev3Y + 0.3 x Rev1Y + 0.3 x EBITDA3Y`

        **Example:**
        - Rev3Y=8%, Rev1Y=10%, EBITDA3Y=12% -> **48.0**
        - Rev3Y=15%, Rev1Y=20%, EBITDA3Y=18% -> **66.0**
        """)

    with st.expander("Cash Flow Score (15% weight)"):
        st.markdown("""
        **Inputs:** FCF/Debt Ratio (50%), FCF Margin % (30%), Cash Ops to Current Liab (20%)

        **Calculation:**
        - **FCF/Debt:** `(FCF / Total Debt)`, capped at 50% = 100 score, scaled by 200
        - **FCF Margin:** Percentage, scaled to 0-100
        - **Cash Ops Coverage:** `(Cash from Ops / Current Liabilities) x 100`, capped at 100
        - Combined: `0.5 x FCF/Debt + 0.3 x Margin + 0.2 x CashOps`

        **Example:**
        - FCF/Debt=20%, Margin=15%, CashOps=80% -> **71.0**
        - FCF/Debt=5%, Margin=8%, CashOps=50% -> **30.4**
        """)

    # Section 3: IG vs HY Segmentation
    st.subheader("3. Investment Grade vs High Yield Segmentation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Investment Grade Ratings:**")
        ig_ratings = pd.DataFrame({
            'Rating': ['AAA', 'AA+, AA, AA-', 'A+, A, A-', 'BBB+, BBB, BBB-'],
            'Category': ['Group 1: AAA', 'Group 2: AA', 'Group 3: A', 'Group 4: BBB']
        })
        st.dataframe(ig_ratings, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**High Yield Ratings:**")
        hy_ratings = pd.DataFrame({
            'Rating': ['BB+, BB, BB-', 'B+, B, B-', 'CCC+, CCC, CCC-, CC, C, D'],
            'Category': ['Group 5: BB', 'Group 6: B', 'Group 7: CCC and below']
        })
        st.dataframe(hy_ratings, use_container_width=True, hide_index=True)

    st.info("""
    **Why Separate IG and HY?** Investment Grade and High Yield issuers have different risk profiles
    and investor expectations. The model uses different category thresholds for each segment to
    provide appropriate recommendations within each universe.
    """)

    # Section 4: Category Assignment
    st.subheader("4. Investment Category Assignment")

    st.markdown("""
    Based on the composite score, each issuer is assigned to one of four investment categories.
    The thresholds differ between Investment Grade and High Yield to reflect their different risk profiles.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Investment Grade Thresholds:**")
        ig_thresholds = pd.DataFrame({
            'Category': ['Strong Buy', 'Buy', 'Hold', 'Avoid'],
            'Score Range': ['>= 72', '57 - 71', '42 - 56', '< 42'],
            'Color': ['Green', 'Blue', 'Yellow', 'Red'],
            'Interpretation': [
                'Top-tier quality, highest conviction',
                'Attractive quality, good fundamentals',
                'Adequate quality, monitor closely',
                'Weak quality, elevated risk'
            ]
        })
        st.dataframe(ig_thresholds, use_container_width=True, hide_index=True)

        # Current IG distribution
        if len(ig_results) > 0:
            st.markdown("**Current IG Distribution:**")
            ig_cat_counts = ig_results['Category'].value_counts()
            for cat in ['Strong Buy', 'Buy', 'Hold', 'Avoid']:
                count = ig_cat_counts.get(cat, 0)
                pct = (count / len(ig_results) * 100) if len(ig_results) > 0 else 0
                st.metric(cat, f"{count} issuers", f"{pct:.1f}%")

    with col2:
        st.markdown("**High Yield Thresholds:**")
        hy_thresholds = pd.DataFrame({
            'Category': ['Strong Buy', 'Buy', 'Hold', 'Avoid'],
            'Score Range': ['>= 67', '52 - 66', '37 - 51', '< 37'],
            'Color': ['Green', 'Blue', 'Yellow', 'Red'],
            'Interpretation': [
                'Best-in-class HY, strong fundamentals',
                'Solid HY, favorable risk-reward',
                'Average HY, standard risk',
                'Distressed, significant concerns'
            ]
        })
        st.dataframe(hy_thresholds, use_container_width=True, hide_index=True)

        # Current HY distribution
        if len(hy_results) > 0:
            st.markdown("**Current HY Distribution:**")
            hy_cat_counts = hy_results['Category'].value_counts()
            for cat in ['Strong Buy', 'Buy', 'Hold', 'Avoid']:
                count = hy_cat_counts.get(cat, 0)
                pct = (count / len(hy_results) * 100) if len(hy_results) > 0 else 0
                st.metric(cat, f"{count} issuers", f"{pct:.1f}%")

    # Section 5: Ranking Systems
    st.subheader("5. Multiple Ranking Perspectives")

    st.markdown("""
    The model provides three complementary ranking systems to help identify opportunities:
    """)

    ranking_types = pd.DataFrame({
        'Ranking Type': ['Overall Rank', 'Rating Group Rank', 'Percentile'],
        'Description': [
            'Rank within entire IG or HY universe (1 = best)',
            'Rank within specific rating group (e.g., among all BBB-rated issuers)',
            'Percentile position (0-100, higher = better relative position)'
        ],
        'Use Case': [
            'Find absolute best opportunities across all ratings',
            'Compare issuers with similar credit ratings (peer comparison)',
            'Understand relative positioning vs. the full universe'
        ]
    })

    st.dataframe(ranking_types, use_container_width=True, hide_index=True)

    # Section 6: Missing Data Handling
    st.subheader("6. Missing Data Imputation Strategy")

    st.markdown("""
    When financial data is missing, the model uses a three-tier imputation strategy:
    """)

    st.markdown("""
    1. **Sector x IG/HY Median:** Use median value from same sector and same IG/HY classification
    2. **Sector Median:** If (1) unavailable, use overall sector median
    3. **Global Median/Default:** If (1) and (2) unavailable, use overall median or default value of 50.0

    This ensures all issuers receive scores while preserving sector and rating-specific characteristics.
    """)

    # Show current dataset's missing data stats if available
    if 'quality_cols' in locals():
        st.markdown("**Current Dataset Missing Data:**")
        missing_summary = []
        for col in ['Credit_Score', 'Leverage_Score', 'Profitability_Score', 'Liquidity_Score', 'Growth_Score']:
            if col in results_final.columns:
                missing_count = results_final[col].isna().sum()
                total_count = len(results_final)
                missing_pct = (missing_count / total_count * 100) if total_count > 0 else 0
                missing_summary.append({
                    'Factor': col.replace('_', ' '),
                    'Missing Values': missing_count,
                    'Total Values': total_count,
                    'Missing %': f"{missing_pct:.1f}%"
                })
        if missing_summary:
            st.dataframe(pd.DataFrame(missing_summary), use_container_width=True, hide_index=True)

    # Section 7: Visualization
    st.subheader("7. PCA Positioning Maps")

    st.markdown("""
    **What are the positioning maps?**

    The 2D scatter plots in the "Overview & Positioning" tab use Principal Component Analysis (PCA)
    to visualize the five-dimensional credit quality space in two dimensions.

    - **X-axis (PC1):** Overall Credit Quality - companies further right generally have better composite characteristics
    - **Y-axis (PC2):** Financial Strength vs Leverage Balance - captures the tradeoff between profitability/liquidity and leverage
    - **Bubble Size:** Proportional to composite score
    - **Color:** Investment category (Strong Buy, Buy, Hold, Avoid)

    **How to interpret:**
    - Companies close together have similar credit profiles across all six factors
    - Green clusters indicate high-quality opportunities
    - Red areas suggest higher risk or distressed situations
    - Top 10 issuers highlighted with gold borders
    """)

    # Section 8: Key Assumptions
    st.subheader("8. Key Model Assumptions & Limitations")

    st.markdown("""
    **Assumptions:**
    - Equal importance given to credit quality and profitability (25% each)
    - Leverage is critical but slightly less weighted (20%)
    - Liquidity and growth are supporting factors (15% each)
    - Lower leverage is always better (inverse scoring)
    - Moderate growth is preferred over extreme growth (caps in place)

    **Limitations:**
    - Model is backward-looking (based on historical financials)
    - Does not incorporate market pricing, spread levels, or valuation
    - Sector-specific nuances may not be fully captured
    - Missing data imputation may mask company-specific issues
    - Should be used as a screening tool, not a final investment decision
    """)

    st.warning("""
    **Important:** This model is designed for initial screening and relative ranking.
    Always perform additional due diligence, fundamental analysis, and consider market conditions
    before making investment decisions.
    """)

with tab7:
    st.header("AI-Powered Model Analysis")

    if not openai_api_key:
        st.warning("Please add your OpenAI API key to Streamlit Cloud secrets to enable AI analysis.")
    else:
        st.markdown("""
        Generate AI-powered insights using GPT-4 to analyze your issuer screening results.
        This will produce:
        - Executive Summary
        - Investment Recommendations
        - Market Insights & Trends
        - Methodology Assessment
        """)

        st.info("Note: This will make 4 API calls to OpenAI and may take 30-60 seconds to complete.")

        # Button to trigger analysis
        if st.button("Generate AI Analysis", type="primary", use_container_width=True):
            # Prepare summary data for AI
            summary_stats = f"""
ISSUER SCREENING MODEL SUMMARY

UNIVERSE OVERVIEW:
- Total Issuers Analyzed: {len(results_final):,}
- Investment Grade: {len(ig_results):,} ({len(ig_results)/max(len(results_final),1)*100:.1f}%)
- High Yield: {len(hy_results):,} ({len(hy_results)/max(len(results_final),1)*100:.1f}%)

INVESTMENT GRADE BREAKDOWN:
- Strong Buy: {len(ig_results[ig_results['Category']=='Strong Buy'])} ({len(ig_results[ig_results['Category']=='Strong Buy'])/max(len(ig_results),1)*100:.1f}%)
- Buy: {len(ig_results[ig_results['Category']=='Buy'])} ({len(ig_results[ig_results['Category']=='Buy'])/max(len(ig_results),1)*100:.1f}%)
- Hold: {len(ig_results[ig_results['Category']=='Hold'])} ({len(ig_results[ig_results['Category']=='Hold'])/max(len(ig_results),1)*100:.1f}%)
- Avoid: {len(ig_results[ig_results['Category']=='Avoid'])} ({len(ig_results[ig_results['Category']=='Avoid'])/max(len(ig_results),1)*100:.1f}%)
- Average Score: {ig_results['Composite_Score'].mean() if len(ig_results) > 0 else 0:.1f}

HIGH YIELD BREAKDOWN:
- Strong Buy: {len(hy_results[hy_results['Category']=='Strong Buy'])} ({len(hy_results[hy_results['Category']=='Strong Buy'])/max(len(hy_results),1)*100:.1f}%)
- Buy: {len(hy_results[hy_results['Category']=='Buy'])} ({len(hy_results[hy_results['Category']=='Buy'])/max(len(hy_results),1)*100:.1f}%)
- Hold: {len(hy_results[hy_results['Category']=='Hold'])} ({len(hy_results[hy_results['Category']=='Hold'])/max(len(hy_results),1)*100:.1f}%)
- Avoid: {len(hy_results[hy_results['Category']=='Avoid'])} ({len(hy_results[hy_results['Category']=='Avoid'])/max(len(hy_results),1)*100:.1f}%)
- Average Score: {hy_results['Composite_Score'].mean() if len(hy_results) > 0 else 0:.1f}

TOP 5 INVESTMENT GRADE ISSUERS:
{ig_results.nlargest(min(5, len(ig_results)), 'Composite_Score')[['Company_Name', 'Credit_Rating', 'Composite_Score', 'Industry']].to_string(index=False) if len(ig_results) > 0 else 'No Investment Grade issuers'}

TOP 5 HIGH YIELD ISSUERS:
{hy_results.nlargest(min(5, len(hy_results)), 'Composite_Score')[['Company_Name', 'Credit_Rating', 'Composite_Score', 'Industry']].to_string(index=False) if len(hy_results) > 0 else 'No High Yield issuers'}

RATING GROUP DISTRIBUTION:
{results_final.groupby('Rating_Group').size().to_string()}

METHODOLOGY:
The model uses unsupervised machine learning to score issuers across 5 dimensions:
1. Credit Rating Score (25%): S&P credit rating converted to numeric scale
2. Leverage Score (20%): Based on Debt/EBITDA ratios (lower is better)
3. Profitability Score (25%): Combined ROE and EBITDA margins
4. Liquidity Score (15%): Current ratio assessment
5. Growth Score (15%): Revenue and EBITDA growth rates

Composite scores range from 0-100, with separate thresholds for IG and HY:
- IG: Strong Buy >=70, Buy >=55, Hold >=40
- HY: Strong Buy >=65, Buy >=50, Hold >=35
"""

            # Analysis sections
            st.subheader("1. Executive Summary")
            with st.spinner("Generating executive summary..."):
                try:
                    client = OpenAI(api_key=openai_api_key)
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a senior credit analyst providing concise, actionable insights on fixed income markets."},
                            {"role": "user", "content": f"""Based on this issuer screening analysis, provide a concise executive summary (3-4 paragraphs) covering:
1. Overall market landscape (IG vs HY opportunities)
2. Key findings and standout performers
3. Notable trends by rating group
4. Investment implications

Data:
{summary_stats}"""}
                        ],
                        temperature=0.3,
                        max_tokens=1500
                    )
                    st.markdown(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"Error generating summary: {e}")

            st.subheader("2. Investment Recommendations")
            with st.spinner("Generating investment recommendations..."):
                try:
                    client = OpenAI(api_key=openai_api_key)
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a portfolio manager specializing in credit analysis and fixed income investing."},
                            {"role": "user", "content": f"""Based on this issuer screening analysis, provide specific investment recommendations:

1. Top 3 Investment Grade opportunities and why
2. Top 3 High Yield opportunities and why
3. Sectors to overweight/underweight
4. Risk considerations
5. Portfolio construction suggestions

Be specific and actionable.

Data:
{summary_stats}"""}
                        ],
                        temperature=0.3,
                        max_tokens=2000
                    )
                    st.markdown(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")

            st.subheader("3. Market Insights & Trends")
            with st.spinner("Analyzing market trends..."):
                try:
                    client = OpenAI(api_key=openai_api_key)
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a credit strategist analyzing fixed income markets."},
                            {"role": "user", "content": f"""Analyze the following credit market data and provide insights on:

1. Credit quality distribution - what does it tell us about the market?
2. IG vs HY comparison - relative value perspectives
3. Rating group analysis - opportunities by rating tier
4. Industry/sector patterns - where is credit quality concentrated?
5. Risk-return considerations across the universe

Data:
{summary_stats}"""}
                        ],
                        temperature=0.4,
                        max_tokens=2000
                    )
                    st.markdown(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"Error generating insights: {e}")

            st.subheader("4. Methodology Assessment")
            with st.spinner("Evaluating methodology..."):
                try:
                    client = OpenAI(api_key=openai_api_key)
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a quantitative credit analyst evaluating screening methodologies."},
                            {"role": "user", "content": f"""Evaluate this credit screening methodology:

METHODOLOGY:
- 6-factor composite score: Credit Rating (20%), Leverage (20%), Profitability (20%), Liquidity (10%), Growth (15%), Cash Flow (15%)
- Separate IG and HY analysis with different thresholds
- Unsupervised ML approach using PCA for visualization
- Rating group peer comparison

Provide:
1. Strengths of this approach
2. Potential limitations or blind spots
3. Suggestions for enhancement
4. How to best use these results in practice

Data:
{summary_stats}"""}
                        ],
                        temperature=0.3,
                        max_tokens=1500
                    )
                    st.markdown(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"Error generating assessment: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #4c566a;'>
    <p><strong>Issuer Credit Screening Model</strong> | Powered by Machine Learning</p>
    <p style='font-size: 0.9em;'>Methodology: 6-factor composite scoring with separate IG/HY analysis and rating group rankings</p>
</div>
""", unsafe_allow_html=True)
