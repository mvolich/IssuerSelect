import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai
import base64
import os
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
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
      
      /* Global font + color */
      html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
      [data-testid="stSidebarContent"], [data-testid="stMarkdownContainer"],
      h1, h2, h3, h4, h5, h6, p, div, span, label, input, textarea, select, button,
      .stText, .stDataFrame, .stMetric, .stTabs, .stButton, .stDownloadButton {
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
        color:var(--rb-blue); font-weight:600; min-width:180px; text-align:center; padding:8px 16px;
      }
      .stTabs [aria-selected="true"]{
        background:var(--rb-mblue)!important; color:#fff!important;
        border-bottom:3px solid var(--rb-orange)!important;
      }

      /* Buttons */
      .stButton > button, .stDownloadButton > button {
        background:var(--rb-mblue); color:#fff; border:none; border-radius:4px;
        padding:8px 16px; font-weight:600;
      }
      .stButton > button:hover, .stDownloadButton > button:hover { background:var(--rb-blue); }

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
uploaded_file = st.sidebar.file_uploader("Upload S&P Data Excel file", type=["xlsx"])

if not uploaded_file:
    st.warning("Please upload the S&P issuer data Excel file to proceed.")
    st.stop()

# --- Load and Process Data ---
@st.cache_data
def load_and_process_data(uploaded_file):
    """Load data and calculate issuer scores"""
    
    # Load data
    df = pd.read_excel(uploaded_file, sheet_name='Pasted Values')
    df = df.iloc[1:].reset_index(drop=True)
    
    # Calculate quality scores
    def calculate_quality_scores(df):
        scores = pd.DataFrame(index=df.index)
        
        # Credit rating numeric
        rating_map = {
            'AAA': 21, 'AA+': 20, 'AA': 19, 'AA-': 18,
            'A+': 17, 'A': 16, 'A-': 15,
            'BBB+': 14, 'BBB': 13, 'BBB-': 12,
            'BB+': 11, 'BB': 10, 'BB-': 9,
            'B+': 8, 'B': 7, 'B-': 6,
            'CCC+': 5, 'CCC': 4, 'CCC-': 3,
            'CC': 2, 'C': 1, 'D': 0, 'NR': np.nan
        }
        scores['credit_score'] = df['S&P Credit Rating'].map(rating_map)
        
        # Leverage (lower is better)
        debt_ebitda = pd.to_numeric(df['Total Debt / EBITDA (x)'], errors='coerce')
        scores['leverage_score'] = 100 - np.clip((debt_ebitda / 5) * 100, 0, 100)
        
        # Profitability
        roe = pd.to_numeric(df['Return on Equity'], errors='coerce')
        margin = pd.to_numeric(df['EBITDA Margin'], errors='coerce')
        scores['profitability_score'] = (np.clip(roe * 5, 0, 100) + np.clip(margin * 5, 0, 100)) / 2
        
        # Liquidity
        current = pd.to_numeric(df['Current Ratio (x)'], errors='coerce')
        scores['liquidity_score'] = np.clip(current * 50, 0, 100)
        
        # Growth
        rev_growth = pd.to_numeric(df['Total Revenues, 1 Year Growth'], errors='coerce')
        scores['growth_score'] = np.clip((rev_growth + 10) * 5, 0, 100)
        
        return scores
    
    quality_scores = calculate_quality_scores(df)
    
    # Calculate composite score
    weights = {
        'credit_score': 0.25,
        'leverage_score': 0.20,
        'profitability_score': 0.25,
        'liquidity_score': 0.15,
        'growth_score': 0.15
    }
    
    composite_score = sum(quality_scores[col].fillna(50) * weight for col, weight in weights.items())
    
    # Create results dataframe
    results = pd.DataFrame({
        'Company_ID': df['Company ID'],
        'Company_Name': df['Company Name'],
        'Ticker': df['Ticker'],
        'Credit_Rating': df['S&P Credit Rating'],
        'Sector': df['Sector'],
        'Industry': df['Industry'],
        'Market_Cap': pd.to_numeric(df['Market Capitalization'], errors='coerce'),
        'Composite_Score': composite_score,
        'Credit_Score': quality_scores['credit_score'],
        'Leverage_Score': quality_scores['leverage_score'],
        'Profitability_Score': quality_scores['profitability_score'],
        'Liquidity_Score': quality_scores['liquidity_score'],
        'Growth_Score': quality_scores['growth_score']
    })
    
    # Clean data
    results_clean = results.dropna(subset=['Composite_Score', 'Credit_Rating'])
    
    # Define Investment Grade vs High Yield
    investment_grade_ratings = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-']
    high_yield_ratings = ['BB+', 'BB', 'BB-', 'B+', 'B', 'B-', 'CCC+', 'CCC', 'CCC-', 'CC', 'C', 'D']
    
    results_clean['IG_HY'] = results_clean['Credit_Rating'].apply(
        lambda x: 'Investment Grade' if x in investment_grade_ratings else 'High Yield' if x in high_yield_ratings else 'Unknown'
    )
    
    # Filter out Unknown
    results_clean = results_clean[results_clean['IG_HY'].isin(['Investment Grade', 'High Yield'])]
    
    # Define rating groups
    def assign_rating_group(rating):
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
    
    results_clean['Rating_Group'] = results_clean['Credit_Rating'].apply(assign_rating_group)
    
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
    
    # Categories for each group
    ig_results['Category'] = pd.cut(
        ig_results['Composite_Score'],
        bins=[0, 40, 55, 70, 100],
        labels=['Avoid', 'Hold', 'Buy', 'Strong Buy']
    )
    
    hy_results['Category'] = pd.cut(
        hy_results['Composite_Score'],
        bins=[0, 35, 50, 65, 100],
        labels=['Avoid', 'Hold', 'Buy', 'Strong Buy']
    )
    
    # Combine back
    results_final = pd.concat([ig_results, hy_results], ignore_index=True)
    
    # PCA for visualization
    quality_cols = ['Credit_Score', 'Leverage_Score', 'Profitability_Score', 'Liquidity_Score', 'Growth_Score']
    
    # IG PCA
    ig_features = ig_results[quality_cols].fillna(ig_results[quality_cols].median())
    scaler_ig = RobustScaler()
    ig_scaled = scaler_ig.fit_transform(ig_features)
    pca_ig = PCA(n_components=2)
    ig_pca = pca_ig.fit_transform(ig_scaled)
    ig_results['PC1'] = ig_pca[:, 0]
    ig_results['PC2'] = ig_pca[:, 1]
    
    # HY PCA
    hy_features = hy_results[quality_cols].fillna(hy_results[quality_cols].median())
    scaler_hy = RobustScaler()
    hy_scaled = scaler_hy.fit_transform(hy_features)
    pca_hy = PCA(n_components=2)
    hy_pca = pca_hy.fit_transform(hy_scaled)
    hy_results['PC1'] = hy_pca[:, 0]
    hy_results['PC2'] = hy_pca[:, 1]
    
    return results_final, ig_results, hy_results

# Load data
with st.spinner("Loading and processing data..."):
    results_final, ig_results, hy_results = load_and_process_data(uploaded_file)

st.success(f"✓ Processed {len(results_final)} issuers ({len(ig_results)} IG, {len(hy_results)} HY)")

# --- Summary Metrics ---
st.subheader("📊 Universe Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Issuers", f"{len(results_final):,}")
with col2:
    ig_attractive = len(ig_results[ig_results['Category'].isin(['Strong Buy', 'Buy'])])
    st.metric("IG Attractive", f"{ig_attractive}", f"{ig_attractive/len(ig_results)*100:.1f}%")
with col3:
    hy_attractive = len(hy_results[hy_results['Category'].isin(['Strong Buy', 'Buy'])])
    st.metric("HY Attractive", f"{hy_attractive}", f"{hy_attractive/len(hy_results)*100:.1f}%")
with col4:
    total_attractive = ig_attractive + hy_attractive
    st.metric("Total Attractive", f"{total_attractive}", f"{total_attractive/len(results_final)*100:.1f}%")

# --- Tabs for different views ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview & Positioning", 
    "🏆 Top Rankings", 
    "📋 Rating Group Analysis",
    "📊 Detailed Data",
    "🤖 AI Analysis"
])

with tab1:
    st.header("Investment Grade vs High Yield Positioning")
    
    # Create side-by-side positioning maps
    colors_map = {'Strong Buy': '#00C851', 'Buy': '#33b5e5', 'Hold': '#ffbb33', 'Avoid': '#ff4444'}
    
    # IG Positioning Map
    st.subheader("Investment Grade Issuer Map")
    fig_ig = go.Figure()
    
    for category in ['Avoid', 'Hold', 'Buy', 'Strong Buy']:
        mask = ig_results['Category'] == category
        data = ig_results[mask]
        fig_ig.add_trace(go.Scatter(
            x=data['PC1'], 
            y=data['PC2'],
            mode='markers',
            marker=dict(
                size=data['Composite_Score']*0.4,
                color=colors_map[category],
                opacity=0.6,
                line=dict(width=0.5, color='white')
            ),
            name=f'{category} ({len(data)})',
            text=data['Company_Name'],
            hovertemplate='<b>%{text}</b><br>Score: %{marker.size:.1f}<extra></extra>'
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
        xaxis_title='Quality Dimension →',
        yaxis_title='Risk/Stability Dimension →',
        hovermode='closest',
        height=500
    )
    st.plotly_chart(apply_rubrics_plot_fonts(fig_ig), use_container_width=True)
    
    # HY Positioning Map
    st.subheader("High Yield Issuer Map")
    fig_hy = go.Figure()
    
    for category in ['Avoid', 'Hold', 'Buy', 'Strong Buy']:
        mask = hy_results['Category'] == category
        data = hy_results[mask]
        fig_hy.add_trace(go.Scatter(
            x=data['PC1'],
            y=data['PC2'],
            mode='markers',
            marker=dict(
                size=data['Composite_Score']*0.4,
                color=colors_map[category],
                opacity=0.6,
                line=dict(width=0.5, color='white')
            ),
            name=f'{category} ({len(data)})',
            text=data['Company_Name'],
            hovertemplate='<b>%{text}</b><br>Score: %{marker.size:.1f}<extra></extra>'
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
        xaxis_title='Quality Dimension →',
        yaxis_title='Risk/Stability Dimension →',
        hovermode='closest',
        height=500
    )
    st.plotly_chart(apply_rubrics_plot_fonts(fig_hy), use_container_width=True)
    
    # Category distributions
    st.subheader("Category Distributions")
    col1, col2 = st.columns(2)
    
    with col1:
        ig_cat_dist = ig_results['Category'].value_counts()
        fig_ig_cat = go.Figure(go.Bar(
            x=ig_cat_dist.values,
            y=ig_cat_dist.index,
            orientation='h',
            marker=dict(color=[colors_map[cat] for cat in ig_cat_dist.index]),
            text=[f"{val} ({val/len(ig_results)*100:.1f}%)" for val in ig_cat_dist.values],
            textposition='auto'
        ))
        fig_ig_cat.update_layout(
            title='Investment Grade Categories',
            xaxis_title='Number of Issuers',
            height=300
        )
        st.plotly_chart(apply_rubrics_plot_fonts(fig_ig_cat), use_container_width=True)
    
    with col2:
        hy_cat_dist = hy_results['Category'].value_counts()
        fig_hy_cat = go.Figure(go.Bar(
            x=hy_cat_dist.values,
            y=hy_cat_dist.index,
            orientation='h',
            marker=dict(color=[colors_map[cat] for cat in hy_cat_dist.index]),
            text=[f"{val} ({val/len(hy_results)*100:.1f}%)" for val in hy_cat_dist.values],
            textposition='auto'
        ))
        fig_hy_cat.update_layout(
            title='High Yield Categories',
            xaxis_title='Number of Issuers',
            height=300
        )
        st.plotly_chart(apply_rubrics_plot_fonts(fig_hy_cat), use_container_width=True)

with tab2:
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

with tab3:
    st.header("Rating Group Analysis")
    
    rating_groups = [
        ('Group 1: AAA', 'AAA'),
        ('Group 2: AA', 'AA+/AA/AA-'),
        ('Group 3: A', 'A+/A/A-'),
        ('Group 4: BBB', 'BBB+/BBB/BBB-'),
        ('Group 5: BB', 'BB+/BB/BB-'),
        ('Group 6: B', 'B+/B/B-')
    ]
    
    # Rating group distribution
    st.subheader("Issuer Distribution by Rating Group")
    rating_dist = results_final.groupby(['Rating_Group', 'IG_HY']).size().unstack(fill_value=0)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Investment Grade',
        x=rating_dist.index,
        y=rating_dist.get('Investment Grade', 0),
        marker_color='#4CAF50',
        text=rating_dist.get('Investment Grade', 0),
        textposition='auto'
    ))
    fig.add_trace(go.Bar(
        name='High Yield',
        x=rating_dist.index,
        y=rating_dist.get('High Yield', 0),
        marker_color='#FF9800',
        text=rating_dist.get('High Yield', 0),
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

with tab4:
    st.header("Detailed Issuer Data")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_ig_hy = st.multiselect(
            "Filter by IG/HY",
            ['Investment Grade', 'High Yield'],
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
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="issuer_screening_filtered.csv",
        mime="text/csv"
    )

with tab5:
    st.header("🤖 AI-Powered Model Analysis")
    
    if not openai_api_key:
        st.warning("Please add your OpenAI API key to Streamlit Cloud secrets to enable AI analysis.")
    else:
        openai.api_key = openai_api_key
        
        # Prepare summary data for AI
        summary_stats = f"""
ISSUER SCREENING MODEL SUMMARY

UNIVERSE OVERVIEW:
- Total Issuers Analyzed: {len(results_final):,}
- Investment Grade: {len(ig_results):,} ({len(ig_results)/len(results_final)*100:.1f}%)
- High Yield: {len(hy_results):,} ({len(hy_results)/len(results_final)*100:.1f}%)

INVESTMENT GRADE BREAKDOWN:
- Strong Buy: {len(ig_results[ig_results['Category']=='Strong Buy'])} ({len(ig_results[ig_results['Category']=='Strong Buy'])/len(ig_results)*100:.1f}%)
- Buy: {len(ig_results[ig_results['Category']=='Buy'])} ({len(ig_results[ig_results['Category']=='Buy'])/len(ig_results)*100:.1f}%)
- Hold: {len(ig_results[ig_results['Category']=='Hold'])} ({len(ig_results[ig_results['Category']=='Hold'])/len(ig_results)*100:.1f}%)
- Avoid: {len(ig_results[ig_results['Category']=='Avoid'])} ({len(ig_results[ig_results['Category']=='Avoid'])/len(ig_results)*100:.1f}%)
- Average Score: {ig_results['Composite_Score'].mean():.1f}

HIGH YIELD BREAKDOWN:
- Strong Buy: {len(hy_results[hy_results['Category']=='Strong Buy'])} ({len(hy_results[hy_results['Category']=='Strong Buy'])/len(hy_results)*100:.1f}%)
- Buy: {len(hy_results[hy_results['Category']=='Buy'])} ({len(hy_results[hy_results['Category']=='Buy'])/len(hy_results)*100:.1f}%)
- Hold: {len(hy_results[hy_results['Category']=='Hold'])} ({len(hy_results[hy_results['Category']=='Hold'])/len(hy_results)*100:.1f}%)
- Avoid: {len(hy_results[hy_results['Category']=='Avoid'])} ({len(hy_results[hy_results['Category']=='Avoid'])/len(hy_results)*100:.1f}%)
- Average Score: {hy_results['Composite_Score'].mean():.1f}

TOP 5 INVESTMENT GRADE ISSUERS:
{ig_results.nlargest(5, 'Composite_Score')[['Company_Name', 'Credit_Rating', 'Composite_Score', 'Industry']].to_string(index=False)}

TOP 5 HIGH YIELD ISSUERS:
{hy_results.nlargest(5, 'Composite_Score')[['Company_Name', 'Credit_Rating', 'Composite_Score', 'Industry']].to_string(index=False)}

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
- IG: Strong Buy ≥70, Buy ≥55, Hold ≥40
- HY: Strong Buy ≥65, Buy ≥50, Hold ≥35
"""
        
        # Analysis sections
        st.subheader("1. Executive Summary")
        with st.spinner("Generating executive summary..."):
            try:
                response = openai.chat.completions.create(
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
                response = openai.chat.completions.create(
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
                response = openai.chat.completions.create(
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
                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a quantitative credit analyst evaluating screening methodologies."},
                        {"role": "user", "content": f"""Evaluate this credit screening methodology:

METHODOLOGY:
- 5-factor composite score: Credit Rating (25%), Leverage (20%), Profitability (25%), Liquidity (15%), Growth (15%)
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
    <p style='font-size: 0.9em;'>Methodology: 5-factor composite scoring with separate IG/HY analysis and rating group rankings</p>
</div>
""", unsafe_allow_html=True)
