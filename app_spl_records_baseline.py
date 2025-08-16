import streamlit as st
import pandas as pd
import altair as alt
import time
import json
import re
from datetime import datetime
from snowflake.snowpark.context import get_active_session

st.set_page_config(
    page_title="demandcraft_–_generative_demand_forecasting_intelligence",
    page_icon="https://i.imgur.com/vAoVPLQ.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

solution_name = '''Solution 2: DemandCraft – Generative Demand Forecasting Intelligence'''
solution_name_clean = '''demandcraft_–_generative_demand_forecasting_intelligence'''
table_name = '''SPL_RECORDS'''
table_description = '''Consolidated table containing demand planning, sales, market intelligence, and economic data for AI-powered demand forecasting across multiple time horizons'''
solution_content = '''Solution 2: DemandCraft – Generative Demand Forecasting Intelligence**

**Primary Business Challenge:**
Traditional demand forecasting methods achieve only 60-70% accuracy, leading to $1.1 trillion in excess inventory globally and frequent stockouts that damage customer relationships. Companies struggle to incorporate external market signals, seasonal variations, and promotional impacts into accurate demand predictions.

**Key Features:**
• Multi-horizon demand forecasting from daily to annual timeframes
• Automated external factor integration (economic indicators, social trends, competitor actions)
• Promotional impact modeling with ROI optimization recommendations
• New product launch demand prediction using similar product analogies
• Collaborative planning interface with sales and marketing teams

**Data Sources:**
• Demand Planning Systems: Oracle Demantra, SAP Integrated Business Planning, Kinaxis RapidResponse
• Point of Sale (POS): NCR, Diebold Nixdorf, Toshiba Commerce Solutions
• Customer Relationship Management (CRM): Salesforce, Microsoft Dynamics CRM, HubSpot
• Market Intelligence: Nielsen, IRI, Euromonitor International
• Economic Data: Federal Reserve Economic Data (FRED), Bloomberg Terminal, Refinitiv Eikon

**Competitive Advantage:**
DemandCraft leverages generative AI to create synthetic demand scenarios and test forecasting models against thousands of simulated market conditions. Unlike traditional statistical models, it generates human-readable explanations for forecast changes and automatically adapts to new market patterns without manual model retraining.

**Key Stakeholders:**
• Primary: VP of Demand Planning, Chief Supply Chain Officer, Sales Operations Directors
• Secondary: Category Managers, Inventory Planners, Marketing Analytics Teams
• Top C-Level Executive: Chief Revenue Officer (CRO)

**Technical Approach:**
Employs transformer-based time series models enhanced with retrieval-augmented generation (RAG) to incorporate real-time market intelligence. Generative adversarial networks create synthetic demand scenarios for stress testing, while large language models translate complex forecasting insights into actionable business recommendations.

**Expected Business Results:**

• 4,800,000 units of excess inventory reduction annually
**40,000,000 units average inventory × 15% excess rate × 80% reduction = 4,800,000 units reduced/year**

• $ 14,400,000 in inventory carrying cost savings annually
**$ 120,000,000 inventory value × 15% carrying cost rate × 80% excess reduction = $ 14,400,000 savings/year**

• 960 fewer stockout incidents per year
**4,800 annual stockout events × 20% reduction = 960 fewer stockouts/year**

• $ 9,600,000 in additional revenue from improved availability
**960 prevented stockouts × $ 10,000 average lost revenue per stockout = $ 9,600,000 additional revenue/year**

**Success Metrics:**
• Forecast accuracy improvement (Mean Absolute Percentage Error reduction)
• Inventory turnover ratio enhancement...'''

# Display logo and title inline
st.markdown(f'''
<div style="display:flex; align-items:center; margin-bottom:15px">
    <img src="https://i.imgur.com/vAoVPLQ.png" width="50" style="margin-right:15px">
    <div>
        <h1 style="font-size:2.2rem; margin:0; padding:0">{solution_name_clean.replace('_', ' ').title()}</h1>
        <p style="font-size:1.1rem; color:gray; margin:0; padding:0">Fivetran and Cortex-powered Streamlit in Snowflake data application</p>
    </div>
</div>
''', unsafe_allow_html=True)

# Define available models as strings
MODELS = [
    "llama3.1-8b", "claude-4-sonnet", "claude-3-7-sonnet", "snowflake-llama-3.3-70b", "mistral-large2", "llama3.1-70b", "llama4-maverick", "llama4-scout", "claude-3-5-sonnet", "snowflake-llama3.1-405b", "deepseek-r1"
]

if 'insights_history' not in st.session_state:
    st.session_state.insights_history = []

if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}

try:
    session = get_active_session()
except Exception as e:
    st.error(f"❌ Error connecting to Snowflake: {str(e)}")
    st.stop()

def query_snowflake(query):
    try:
        return session.sql(query).to_pandas()
    except Exception as e:
        st.error(f"Query failed: {str(e)}")
        return pd.DataFrame()

def load_data():
    query = f"SELECT * FROM {table_name} LIMIT 1000"
    df = query_snowflake(query)
    df.columns = [col.lower() for col in df.columns]
    return df

def call_cortex_model(prompt, model_name):
    try:
        cortex_query = "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS response"
        response = session.sql(cortex_query, params=[model_name, prompt]).collect()[0][0]
        return response
    except Exception as e:
        st.error(f"❌ Cortex error: {str(e)}")
        return None

def generate_insights(data, focus_area, model_name):
    data_summary = f"Table: {table_name}\n"
    data_summary += f"Description: {table_description}\n"
    data_summary += f"Records analyzed: {len(data)}\n"

    # Calculate basic statistics for numeric columns
    numeric_stats = {}
    key_metrics = ["forecast_horizon_days", "baseline_demand_forecast", "adjusted_demand_forecast", "actual_sales_units", "current_inventory_level", "promotion_discount_percent", "market_share_percent", "category_growth_rate", "competitor_price_index", "economic_indicator_gdp", "consumer_confidence_index", "seasonal_index", "forecast_accuracy_mape", "sales_rep_adjustment"]
    for col in key_metrics:
        if col in data.columns:
            numeric_stats[col] = {
                "mean": data[col].mean(),
                "min": data[col].min(),
                "max": data[col].max(),
                "std": data[col].std()
            }
            data_summary += f"- {col} (avg: {data[col].mean():.2f}, min: {data[col].min():.2f}, max: {data[col].max():.2f})\n"

    # Get top values for categorical columns
    categorical_stats = {}
    categorical_options = ["product_sku", "location_code"]
    for cat_col in categorical_options:
        if cat_col in data.columns:
            top = data[cat_col].value_counts().head(3)
            categorical_stats[cat_col] = top.to_dict()
            data_summary += f"\nTop {cat_col} values:\n" + "\n".join(f"- {k}: {v}" for k, v in top.items())

    # Calculate correlations if enough numeric columns available
    correlation_info = ""
    if len(key_metrics) >= 2:
        try:
            correlations = data[key_metrics].corr()
            # Get the top 3 strongest correlations (absolute value)
            corr_pairs = []
            for i in range(len(correlations.columns)):
                for j in range(i+1, len(correlations.columns)):
                    col1 = correlations.columns[i]
                    col2 = correlations.columns[j]
                    corr_value = correlations.iloc[i, j]
                    corr_pairs.append((col1, col2, abs(corr_value), corr_value))

            # Sort by absolute correlation value
            corr_pairs.sort(key=lambda x: x[2], reverse=True)

            # Add top correlations to the summary
            if corr_pairs:
                correlation_info = "Top correlations between metrics:\n"
                for col1, col2, _, corr_value in corr_pairs[:3]:
                    correlation_info += f"- {col1} and {col2}: r = {corr_value:.2f}\n"
        except:
            correlation_info = "Could not calculate correlations between metrics.\n"

    # Define specific instructions for each focus area
    focus_area_instructions = {
        "Overall Performance": """
        For the Overall Performance analysis:
        1. Provide a comprehensive analysis of the current state of the solution using the available data
        2. Identify the most significant patterns and relationships in the data
        3. Highlight 3-5 key metrics that best indicate overall performance
        4. Discuss both strengths and areas for improvement
        5. Include 3-5 actionable insights based on the data

        Structure your response with these sections:
        - Key Insights (5 specific insights with supporting data)
        - Performance Trends (3-4 significant trends)
        - Recommendations (3-5 data-backed recommendations)
        - Action Items (3-5 concrete next steps)
        """,

        "Optimization Opportunities": """
        For the Optimization Opportunities analysis:
        1. Focus specifically on areas where performance can be improved
        2. Identify inefficiencies, bottlenecks, or underperforming aspects
        3. Analyze correlations to discover cause-and-effect relationships
        4. Prioritize optimization opportunities based on potential impact
        5. Suggest specific technical or process improvements

        Structure your response with these sections:
        - Optimization Priorities (3-5 areas with highest optimization potential)
        - Impact Analysis (quantified benefits of addressing each opportunity)
        - Implementation Strategy (specific steps to implement each optimization)
        - Technical Recommendations (specific technical changes needed)
        - Risk Assessment (potential challenges and how to mitigate them)
        """,

        "Financial Impact": """
        For the Financial Impact analysis:
        1. Focus on cost-benefit analysis and ROI of the solution
        2. Quantify financial impacts in specific dollar amounts or percentages
        3. Identify cost savings opportunities
        4. Analyze resource allocation efficiency
        5. Project future financial outcomes based on current trends

        Structure your response with these sections:
        - Cost Analysis (breakdown of costs and potential savings)
        - Revenue Impact (how the solution affects revenue generation)
        - ROI Calculation (specific calculations showing return on investment)
        - Cost Reduction Opportunities (specific areas to reduce costs)
        - Financial Forecasting (projections based on the data)
        """,

        "Strategic Recommendations": """
        For the Strategic Recommendations analysis:
        1. Focus on long-term strategic implications rather than tactical fixes
        2. Identify competitive advantages that can be leveraged
        3. Suggest new directions or expansions based on the data
        4. Connect recommendations to broader business goals
        5. Provide a strategic roadmap with prioritized initiatives

        Structure your response with these sections:
        - Strategic Context (how this solution fits into broader business strategy)
        - Competitive Advantage Analysis (how to maximize competitive edge)
        - Strategic Priorities (3-5 high-impact strategic initiatives)
        - Long-term Vision (how to evolve the solution over 1-3 years)
        - Implementation Roadmap (sequenced steps for strategic execution)
        """
    }

    # Get the specific instructions for the selected focus area
    selected_focus_instructions = focus_area_instructions.get(focus_area, "")

    prompt = f'''
    You are an expert data analyst specializing in {focus_area.lower()} analysis.

    SOLUTION CONTEXT:
    {solution_name}

    {solution_content}

    DATA SUMMARY:
    {data_summary}

    {correlation_info}

    ANALYSIS INSTRUCTIONS:
    {selected_focus_instructions}

    IMPORTANT GUIDELINES:
    - Base all insights directly on the data provided
    - Use specific metrics and numbers from the data in your analysis
    - Maintain a professional, analytical tone
    - Be concise but thorough in your analysis
    - Focus specifically on {focus_area} as defined in the instructions
    - Ensure your response is unique and tailored to this specific focus area
    - Include a mix of observations, analysis, and actionable recommendations
    - Use bullet points and clear section headers for readability
    '''

    return call_cortex_model(prompt, model_name)

data = load_data()
if data.empty:
    st.error("No data found.")
    st.stop()

categorical_cols = [col for col in ["product_sku", "location_code"] if col in data.columns]
numeric_cols = [col for col in ["forecast_horizon_days", "baseline_demand_forecast", "adjusted_demand_forecast", "actual_sales_units", "current_inventory_level", "promotion_discount_percent", "market_share_percent", "category_growth_rate", "competitor_price_index", "economic_indicator_gdp", "consumer_confidence_index", "seasonal_index", "forecast_accuracy_mape", "sales_rep_adjustment"] if col in data.columns]
date_cols = [col for col in ["forecast_date"] if col in data.columns]

sample_cols = data.columns.tolist()
numeric_candidates = [col for col in sample_cols if data[col].dtype in ['float64', 'int64'] and 'id' not in col.lower()]
date_candidates = [col for col in sample_cols if 'date' in col.lower() or 'timestamp' in col.lower()]
cat_candidates = [col for col in sample_cols if data[col].dtype == 'object' and data[col].nunique() < 1000]

# Three tabs - added Insights History as a separate tab
tabs = st.tabs(["✨ AI Insights", "📁 Insights History", "🔍 Data Explorer"])

# AI Insights tab
with tabs[0]:
    st.subheader("✨ AI-Powered Insights")
    focus_area = st.radio("Focus Area", [
        "Overall Performance", 
        "Optimization Opportunities", 
        "Financial Impact", 
        "Strategic Recommendations"
    ])
    selected_model = st.selectbox("Cortex Model", MODELS, index=0)

    if st.button("Generate Insights"):
        with st.spinner("Generating with Snowflake Cortex..."):
            insights = generate_insights(data, focus_area, selected_model)
            if insights:
                st.markdown(insights)
                timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                st.session_state.insights_history.append({
                    "timestamp": timestamp,
                    "focus": focus_area,
                    "insights": insights
                })
                st.download_button("Download Insights", insights, file_name=f"{solution_name.replace(' ', '_').lower()}_insights.md")
            else:
                st.error("No insights returned.")

# New Insights History tab
with tabs[1]:
    st.subheader("📁 Insights History")
    if st.session_state.insights_history:
        for i, item in enumerate(reversed(st.session_state.insights_history)):
            with st.expander(f"{item['timestamp']} - {item['focus']}", expanded=False):
                st.markdown(item["insights"])
    else:
        st.info("No insights generated yet. Go to the AI Insights tab to generate some insights.")

# Data Explorer tab
with tabs[2]:
    st.subheader("🔍 Data Explorer")
    rows_per_page = st.slider("Rows per page", 5, 50, 10)
    page = st.number_input("Page", min_value=1, value=1)
    start = (page - 1) * rows_per_page
    end = min(start + rows_per_page, len(data))
    st.dataframe(data.iloc[start:end], use_container_width=True)
    st.caption(f"Showing rows {start + 1}–{end} of {len(data)}")