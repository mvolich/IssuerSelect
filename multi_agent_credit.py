"""
Multi-Agent Credit Report Generator (Sector-Aware)
Replaces GenAI approach with 5 specialist agents + supervisor
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI


# ============================================================================
# IMPORT APP.PY FUNCTIONS
# ============================================================================

from app import (
    get_from_row,
    resolve_metric_column,
    list_metric_columns,
    get_classification_weights,
    CLASSIFICATION_TO_SECTOR
)


# ============================================================================
# SECTOR CONTEXT HELPERS
# ============================================================================

def get_metric_median(df: pd.DataFrame, metric_name: str) -> Optional[float]:
    """Extract median value for a metric."""
    col = resolve_metric_column(df, metric_name)
    if col:
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(values) > 0:
            return float(values.median())
    return None


def calculate_sector_medians(df: pd.DataFrame, classification: str) -> Dict[str, float]:
    """Calculate median metrics for companies in same sector."""
    sector = CLASSIFICATION_TO_SECTOR.get(classification)
    if not sector:
        return {}

    sector_classifications = [k for k, v in CLASSIFICATION_TO_SECTOR.items() if v == sector]
    class_col = None
    for col in ['Rubrics Custom Classification', 'Rubrics_Custom_Classification']:
        if col in df.columns:
            class_col = col
            break

    if not class_col:
        return {}

    sector_df = df[df[class_col].isin(sector_classifications)]

    if len(sector_df) < 5:
        return {}

    metrics = {
        "EBITDA Margin": get_metric_median(sector_df, "EBITDA Margin"),
        "ROE": get_metric_median(sector_df, "Return on Equity"),
        "ROA": get_metric_median(sector_df, "Return on Assets"),
        "Net Debt/EBITDA": get_metric_median(sector_df, "Net Debt / EBITDA"),
        "Interest Coverage": get_metric_median(sector_df, "EBITDA / Interest Expense (x)"),
        "Current Ratio": get_metric_median(sector_df, "Current Ratio (x)"),
        "Quick Ratio": get_metric_median(sector_df, "Quick Ratio (x)"),
    }

    return {k: v for k, v in metrics.items() if v is not None}


def calculate_cohort_medians(df: pd.DataFrame, rating_group: str) -> Dict[str, float]:
    """Calculate medians for IG or HY cohort."""
    ig_ratings = ['AAA','AA+','AA','AA-','A+','A','A-','BBB+','BBB','BBB-']
    hy_ratings = ['BB+','BB','BB-','B+','B','B-','CCC+','CCC','CCC-','CC','C']

    ratings = ig_ratings if rating_group == 'IG' else hy_ratings

    rating_col = None
    for col in ['S&P LT Issuer Credit Rating', 'Credit_Rating_Clean', 'Rating']:
        if col in df.columns:
            rating_col = col
            break

    if not rating_col:
        return {}

    cohort_df = df[df[rating_col].isin(ratings)]

    if len(cohort_df) < 10:
        return {}

    metrics = {
        "EBITDA Margin": get_metric_median(cohort_df, "EBITDA Margin"),
        "ROE": get_metric_median(cohort_df, "Return on Equity"),
        "ROA": get_metric_median(cohort_df, "Return on Assets"),
        "Net Debt/EBITDA": get_metric_median(cohort_df, "Net Debt / EBITDA"),
        "Current Ratio": get_metric_median(cohort_df, "Current Ratio (x)"),
    }

    return {k: v for k, v in metrics.items() if v is not None}


def prepare_sector_context(
    row: pd.Series,
    df: pd.DataFrame,
    classification: str,
    use_sector_adjusted: bool,
    calibrated_weights: Optional[Dict] = None,
    rating_band: str = 'BBB'
) -> Dict[str, Any]:
    """Prepare sector-aware context for agents."""

    sector = CLASSIFICATION_TO_SECTOR.get(classification, 'Default')

    weights = get_classification_weights(
        classification,
        use_sector_adjusted,
        calibrated_weights
    )

    if calibrated_weights:
        weight_source = f"Dynamic Calibration ({rating_band} cohort)"
    elif use_sector_adjusted and sector != 'Default':
        weight_source = f"Sector-Adjusted ({sector})"
    else:
        weight_source = "Universal Default"

    sector_medians = calculate_sector_medians(df, classification)
    ig_medians = calculate_cohort_medians(df, 'IG')
    hy_medians = calculate_cohort_medians(df, 'HY')

    return {
        "classification": classification,
        "sector": sector,
        "weights": weights,
        "weight_source": weight_source,
        "sector_medians": sector_medians,
        "ig_medians": ig_medians,
        "hy_medians": hy_medians
    }


# ============================================================================
# DATA EXTRACTION
# ============================================================================

def extract_metric_time_series(
    row: pd.Series,
    df: pd.DataFrame,
    metric_name: str,
    max_periods: int = 5
) -> Dict[str, Any]:
    """Extract time series for a metric."""
    current_value = get_from_row(row, metric_name, df)

    metric_cols = list_metric_columns(df, metric_name)

    time_series = []
    for i, col in enumerate(metric_cols[:max_periods]):
        value = row.get(col)
        if pd.notna(value):
            period_label = "Current" if i == 0 else f"T-{i}"
            time_series.append({
                "period": period_label,
                "value": float(value)
            })

    return {
        "metric_name": metric_name,
        "current_value": float(current_value) if pd.notna(current_value) else None,
        "time_series": time_series,
        "data_available": pd.notna(current_value)
    }


# ============================================================================
# AGENT TOOLS
# ============================================================================

async def get_profitability_data(ctx, company_name: str, row_data: dict) -> str:
    """Extract profitability metrics with sector context."""
    metrics = ["EBITDA Margin", "Return on Equity", "Return on Assets", "EBIT Margin"]

    sector_context = prepare_sector_context(
        row=row_data["row"],
        df=row_data["df"],
        classification=row_data["classification"],
        use_sector_adjusted=row_data["use_sector_adjusted"],
        calibrated_weights=row_data.get("calibrated_weights"),
        rating_band=row_data.get("rating_band", "BBB")
    )

    result = {
        "company": company_name,
        "factor_score": row_data["profitability_score"],
        "sector_context": sector_context,
        "metrics": {}
    }

    for metric in metrics:
        result["metrics"][metric] = extract_metric_time_series(
            row_data["row"],
            row_data["df"],
            metric
        )

    state = await ctx.get("state")
    state["profitability_data"] = result
    await ctx.set("state", state)

    return f"Profitability data extracted. Score: {row_data['profitability_score']:.1f}/100, Sector: {sector_context['sector']}"


async def get_leverage_data(ctx, company_name: str, row_data: dict) -> str:
    """Extract leverage metrics with sector context."""
    metrics = [
        "Net Debt / EBITDA",
        "EBITDA / Interest Expense (x)",
        "Total Debt / Total Capital (%)",
        "Total Debt / EBITDA (x)"
    ]

    sector_context = prepare_sector_context(
        row=row_data["row"],
        df=row_data["df"],
        classification=row_data["classification"],
        use_sector_adjusted=row_data["use_sector_adjusted"],
        calibrated_weights=row_data.get("calibrated_weights"),
        rating_band=row_data.get("rating_band", "BBB")
    )

    result = {
        "company": company_name,
        "factor_score": row_data["leverage_score"],
        "sector_context": sector_context,
        "metrics": {}
    }

    for metric in metrics:
        result["metrics"][metric] = extract_metric_time_series(
            row_data["row"],
            row_data["df"],
            metric
        )

    state = await ctx.get("state")
    state["leverage_data"] = result
    await ctx.set("state", state)

    return f"Leverage data extracted. Score: {row_data['leverage_score']:.1f}/100"


async def get_liquidity_data(ctx, company_name: str, row_data: dict) -> str:
    """Extract liquidity metrics with sector context."""
    metrics = ["Current Ratio (x)", "Quick Ratio (x)"]

    sector_context = prepare_sector_context(
        row=row_data["row"],
        df=row_data["df"],
        classification=row_data["classification"],
        use_sector_adjusted=row_data["use_sector_adjusted"],
        calibrated_weights=row_data.get("calibrated_weights"),
        rating_band=row_data.get("rating_band", "BBB")
    )

    result = {
        "company": company_name,
        "factor_score": row_data["liquidity_score"],
        "sector_context": sector_context,
        "metrics": {}
    }

    for metric in metrics:
        result["metrics"][metric] = extract_metric_time_series(
            row_data["row"],
            row_data["df"],
            metric
        )

    state = await ctx.get("state")
    state["liquidity_data"] = result
    await ctx.set("state", state)

    return f"Liquidity data extracted. Score: {row_data['liquidity_score']:.1f}/100"


async def get_cash_flow_data(ctx, company_name: str, row_data: dict) -> str:
    """Extract cash flow metrics."""
    metrics = [
        "Cash from Ops.",
        "Total Revenues",
        "Total Debt",
        "Levered Free Cash Flow"
    ]

    result = {
        "company": company_name,
        "factor_score": row_data["cash_flow_score"],
        "metrics": {}
    }

    for metric in metrics:
        result["metrics"][metric] = extract_metric_time_series(
            row_data["row"],
            row_data["df"],
            metric
        )

    state = await ctx.get("state")
    state["cash_flow_data"] = result
    await ctx.set("state", state)

    return f"Cash flow data extracted. Score: {row_data['cash_flow_score']:.1f}/100"


async def get_growth_data(ctx, company_name: str, row_data: dict) -> str:
    """Extract growth metrics."""
    metrics = [
        "Total Revenues, 3 Yr. CAGR",
        "Total Revenues, 1 Year Growth",
        "EBITDA, 3 Years CAGR"
    ]

    result = {
        "company": company_name,
        "factor_score": row_data["growth_score"],
        "metrics": {}
    }

    for metric in metrics:
        result["metrics"][metric] = extract_metric_time_series(
            row_data["row"],
            row_data["df"],
            metric
        )

    state = await ctx.get("state")
    state["growth_data"] = result
    await ctx.set("state", state)

    return f"Growth data extracted. Score: {row_data['growth_score']:.1f}/100"


async def compile_final_report(ctx) -> str:
    """Signal all analyses collected."""
    state = await ctx.get("state")

    sections = {
        "profitability": state.get("profitability_analysis"),
        "leverage": state.get("leverage_analysis"),
        "liquidity": state.get("liquidity_analysis"),
        "cash_flow": state.get("cash_flow_analysis"),
        "growth": state.get("growth_analysis")
    }

    complete = sum(1 for s in sections.values() if s)
    return f"Collected {complete}/5 specialist analyses"


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

def create_profitability_agent(llm):
    return FunctionAgent(
        name="ProfitabilityAgent",
        description="Analyze profitability: EBITDA Margin, ROE, ROA, EBIT Margin",
        system_prompt="""You are a profitability analyst.

**Task:**
1. Call get_profitability_data
2. Analyze each metric vs sector medians (primary comparison)
3. Note sector-specific norms (e.g., "Tech EBITDA margins typically 25-35%")
4. Explain factor weight in context of sector (e.g., "Profitability weighted 25% for this sector vs 20% default")
5. Describe historical trends with specific values

**Output Format:**
### EBITDA Margin (Weight: X%)
- Current: Y%
- Sector Context: [Sector] companies typically achieve Z% margins. This company is [above/below/in-line].
- Historical: [trend with values]
- Assessment: [strength/weakness vs sector norms]

[Repeat for ROE, ROA, EBIT Margin]

**Overall Profitability:** [3-4 sentences on sector-relative performance]

Hand off to LeverageAgent when complete.""",
        llm=llm,
        tools=[get_profitability_data],
        can_handoff_to=["LeverageAgent"]
    )


def create_leverage_agent(llm):
    return FunctionAgent(
        name="LeverageAgent",
        description="Analyze leverage: Net Debt/EBITDA, Interest Coverage, Debt/Capital, Total Debt/EBITDA",
        system_prompt="""You are a leverage analyst.

**Task:**
1. Call get_leverage_data
2. Analyze each metric vs sector medians
3. Note if sector has different leverage tolerance (e.g., "Utilities typically 4-5x due to regulated cash flows")
4. Describe historical trends

**Output Format:**
### Net Debt / EBITDA (Weight: X%)
- Current: Yx
- Sector Context: [comparison to sector median]
- Historical: [trend]
- Assessment: [strength/weakness]

[Repeat for Interest Coverage, Debt/Capital, Total Debt/EBITDA]

**Overall Leverage:** [3-4 sentences]

Hand off to LiquidityAgent.""",
        llm=llm,
        tools=[get_leverage_data],
        can_handoff_to=["LiquidityAgent"]
    )


def create_liquidity_agent(llm):
    return FunctionAgent(
        name="LiquidityAgent",
        description="Analyze liquidity: Current Ratio, Quick Ratio",
        system_prompt="""You are a liquidity analyst.

**Task:**
1. Call get_liquidity_data
2. Analyze each metric vs sector medians
3. Describe historical trends

**Output Format:**
### Current Ratio (Weight: X%)
- Current: Yx
- Sector Context: [comparison]
- Historical: [trend]
- Assessment: [strength/weakness]

[Repeat for Quick Ratio]

**Overall Liquidity:** [3-4 sentences]

Hand off to CashFlowAgent.""",
        llm=llm,
        tools=[get_liquidity_data],
        can_handoff_to=["CashFlowAgent"]
    )


def create_cash_flow_agent(llm):
    return FunctionAgent(
        name="CashFlowAgent",
        description="Analyze cash flow quality",
        system_prompt="""You are a cash flow analyst.

**Task:**
1. Call get_cash_flow_data
2. Calculate ratios: OCF/Revenue, OCF/Debt, LFCF Margin
3. Analyze trends in cash generation

**Output Format:**
### Cash Flow Quality
- OCF/Revenue: [if calculable from OCF and Revenue]
- OCF/Debt: [if calculable]
- LFCF Margin: [if calculable]
- Historical: [trends]
- Assessment: [quality of cash generation]

**Overall Cash Flow:** [3-4 sentences]

Hand off to GrowthAgent.""",
        llm=llm,
        tools=[get_cash_flow_data],
        can_handoff_to=["GrowthAgent"]
    )


def create_growth_agent(llm):
    return FunctionAgent(
        name="GrowthAgent",
        description="Analyze growth: Revenue CAGR 3Y, Revenue Growth 1Y, EBITDA CAGR 3Y",
        system_prompt="""You are a growth analyst.

**Task:**
1. Call get_growth_data
2. Analyze growth trajectory
3. Note consistency and quality

**Output Format:**
### Growth Profile
- Revenue CAGR 3Y: X%
- Revenue Growth 1Y: Y%
- EBITDA CAGR 3Y: Z%
- Assessment: [consistency, quality]

**Overall Growth:** [3-4 sentences]

Hand off to SupervisorAgent.""",
        llm=llm,
        tools=[get_growth_data],
        can_handoff_to=["SupervisorAgent"]
    )


def create_supervisor_agent(llm):
    return FunctionAgent(
        name="SupervisorAgent",
        description="Synthesize all analyses into 8-section credit report",
        system_prompt="""You are the Chief Credit Officer. Synthesize all specialist analyses into comprehensive report.

**Task:**
1. Call compile_final_report
2. Extract 3-4 credit strengths from specialist outputs
3. Extract 3-4 credit risks from specialist outputs
4. Provide rating outlook

**Required Format:**

# Credit Analysis: {company_name}
**S&P Rating:** {rating} | **Composite Score:** {composite_score}/100 | **Band:** {rating_band}

## 1. Executive Summary
[3-4 sentences: overall profile, key themes]

## 2. Profitability Analysis
[ProfitabilityAgent output verbatim]

## 3. Leverage Analysis
[LeverageAgent output verbatim]

## 4. Liquidity Analysis
[LiquidityAgent output verbatim]

## 5. Cash Flow & Growth Analysis
[CashFlowAgent + GrowthAgent outputs combined]

## 6. Credit Strengths
- **[Strength 1]:** [1-2 sentences with metrics]
- **[Strength 2]:** [1-2 sentences with metrics]
- **[Strength 3]:** [1-2 sentences with metrics]

## 7. Credit Risks & Concerns
- **[Risk 1]:** [1-2 sentences with metrics]
- **[Risk 2]:** [1-2 sentences with metrics]
- **[Risk 3]:** [1-2 sentences with metrics]

## 8. Rating Outlook & Recommendation
[3-4 sentences on trajectory, catalysts]

**Output ONLY the markdown report above.**""",
        llm=llm,
        tools=[compile_final_report],
        can_handoff_to=[]
    )


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def generate_multiagent_credit_report(
    row: pd.Series,
    df: pd.DataFrame,
    composite_score: float,
    factor_scores: Dict[str, float],
    rating_band: str,
    company_name: str,
    rating: str,
    classification: str = "Unknown",
    use_sector_adjusted: bool = True,
    calibrated_weights: Optional[Dict] = None,
    api_key: str = None,
    claude_key: str = None
) -> str:
    """Generate multi-agent credit report."""

    if claude_key:
        llm = Anthropic(model="claude-sonnet-4-20250514", api_key=claude_key)
    elif api_key:
        llm = OpenAI(model="gpt-4o", api_key=api_key)
    else:
        raise ValueError("Must provide claude_key or api_key")

    profitability_agent = create_profitability_agent(llm)
    leverage_agent = create_leverage_agent(llm)
    liquidity_agent = create_liquidity_agent(llm)
    cash_flow_agent = create_cash_flow_agent(llm)
    growth_agent = create_growth_agent(llm)
    supervisor_agent = create_supervisor_agent(llm)

    workflow = AgentWorkflow(
        agents=[
            profitability_agent,
            leverage_agent,
            liquidity_agent,
            cash_flow_agent,
            growth_agent,
            supervisor_agent
        ],
        root_agent="ProfitabilityAgent",
        initial_state={
            "company_name": company_name,
            "rating": rating,
            "composite_score": composite_score,
            "rating_band": rating_band,
            "classification": classification,
            "factor_scores": factor_scores,
            "use_sector_adjusted": use_sector_adjusted,
            "calibrated_weights": calibrated_weights,

            "profitability_data": None,
            "leverage_data": None,
            "liquidity_data": None,
            "cash_flow_data": None,
            "growth_data": None,

            "profitability_analysis": None,
            "leverage_analysis": None,
            "liquidity_analysis": None,
            "cash_flow_analysis": None,
            "growth_analysis": None,

            "final_report": None,

            "row_data": {
                "row": row,
                "df": df,
                "classification": classification,
                "use_sector_adjusted": use_sector_adjusted,
                "calibrated_weights": calibrated_weights,
                "rating_band": rating_band,
                "profitability_score": factor_scores.get("profitability_score", 50),
                "leverage_score": factor_scores.get("leverage_score", 50),
                "liquidity_score": factor_scores.get("liquidity_score", 50),
                "cash_flow_score": factor_scores.get("cash_flow_score", 50),
                "growth_score": factor_scores.get("growth_score", 50)
            }
        }
    )

    user_msg = f"""Generate comprehensive credit analysis for {company_name}.

Context:
- S&P Rating: {rating}
- Composite Score: {composite_score:.1f}/100
- Rating Band: {rating_band}
- Classification: {classification}

Factor Scores:
- Credit: {factor_scores.get('credit_score', 50):.1f}/100
- Leverage: {factor_scores.get('leverage_score', 50):.1f}/100
- Profitability: {factor_scores.get('profitability_score', 50):.1f}/100
- Liquidity: {factor_scores.get('liquidity_score', 50):.1f}/100
- Growth: {factor_scores.get('growth_score', 50):.1f}/100
- Cash Flow: {factor_scores.get('cash_flow_score', 50):.1f}/100

Begin analysis."""

    import asyncio
    from llama_index.core.workflow import Context
    from llama_index.core.agent.workflow import AgentStream

    ctx = Context(workflow)
    handler = workflow.run(user_msg=user_msg, ctx=ctx)

    async def run_workflow():
        async for event in handler.stream_events():
            if isinstance(event, AgentStream):
                print(event.delta, end="", flush=True)
        return await handler

    result = asyncio.run(run_workflow())

    if hasattr(result, 'response') and hasattr(result.response, 'content'):
        return result.response.content
    elif isinstance(result, str):
        return result
    else:
        return f"# Error\n\nFailed to generate report."
