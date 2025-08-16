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

# Add CSS for agent progress styling
st.markdown("""
<style>
.agent-current {
    background-color: #e3f2fd;
    border-left: 4px solid #2196f3;
    padding: 10px;
    margin: 5px 0;
    border-radius: 5px;
    font-weight: 500;
}

.agent-completed {
    background-color: #e8f5e8;
    border-left: 4px solid #4caf50;
    padding: 8px;
    margin: 3px 0;
    border-radius: 5px;
    font-size: 0.9em;
    color: #2e7d32;
}

.agent-container {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    border: 1px solid #e0e0e0;
}

.agent-status-active {
    color: #4CAF50;
    font-weight: bold;
    font-size: 1.1em;
}

.agent-button-container {
    display: flex;
    gap: 10px;
    align-items: center;
    margin: 10px 0;
}

.agent-report-header {
    background-color: #f8f9fa;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
    border-left: 4px solid #2196F3;
}
</style>
""", unsafe_allow_html=True)

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
    <img src="https://i.imgur.com/Og6gFnB.png" width="100" style="margin-right:15px">
    <div>
        <h1 style="font-size:2.2rem; margin:0; padding:0">{solution_name_clean.replace('_', ' ').title()}</h1>
        <p style="font-size:1.1rem; color:gray; margin:0; padding:0">Fivetran and Cortex-powered Streamlit in Snowflake data application for Supply Chain Demand Forecasting</p>
    </div>
</div>
''', unsafe_allow_html=True)

# Define available models as strings
MODELS = [
    "openai-gpt-oss-120b", "openai-gpt-4.1", "openai-gpt-5", "openai-gpt-5-mini", "openai-gpt-5-nano", "openai-gpt-5-chat", "claude-4-sonnet", "claude-3-7-sonnet", "claude-3-5-sonnet", "llama3.1-8b", "llama3.1-70b", "llama4-maverick", "llama4-scout", "llama3.2-1b", "snowflake-llama-3.1-405b", "snowflake-llama-3.3-70b", "mistral-large2", "mistral-7b", "deepseek-r1", "snowflake-arctic", "reka-flash", "jamba-instruct", "gemma-7b"
]

if 'insights_history' not in st.session_state:
    st.session_state.insights_history = []

if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}

# Initialize completed steps for each focus area
focus_areas = ["Overall Performance", "Optimization Opportunities", "Financial Impact", "Strategic Recommendations"]
for area in focus_areas:
    if f'{area.lower().replace(" ", "_")}_completed_steps' not in st.session_state:
        st.session_state[f'{area.lower().replace(" ", "_")}_completed_steps'] = []

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

def get_focus_area_info(focus_area):
    """Get business challenge and agent solution for each focus area"""
    
    focus_info = {
        "Overall Performance": {
            "challenge": "VP of Demand Planning and Chief Supply Chain Officers manually review hundreds of demand forecasts, inventory levels, and sales performance metrics daily, spending 4+ hours analyzing forecast accuracy, promotional impacts, and seasonal variations to identify critical supply chain inefficiencies and demand planning optimization opportunities.",
            "solution": "Autonomous demand forecasting workflow that analyzes forecast data, inventory metrics, sales performance, and market intelligence to generate automated demand summaries, identify forecasting bottlenecks, and produce prioritized supply chain insights with adaptive demand planning recommendations."
        },
        "Optimization Opportunities": {
            "challenge": "Sales Operations Directors spend 5+ hours daily manually identifying inefficiencies in demand planning strategies, inventory optimization criteria, and promotional impact modeling across multiple product categories and geographic locations.",
            "solution": "AI-powered demand planning optimization analysis that automatically detects forecast accuracy gaps, inventory management inefficiencies, and promotional strategy improvements with specific implementation recommendations for Oracle Demantra, SAP IBP, and Kinaxis RapidResponse system integration."
        },
        "Financial Impact": {
            "challenge": "Chief Revenue Officers manually calculate complex ROI metrics across demand planning activities and inventory management performance, requiring 4+ hours of cost modeling to assess supply chain efficiency and inventory optimization across the product portfolio.",
            "solution": "Automated supply chain financial analysis that calculates comprehensive demand planning ROI, identifies inventory carrying cost reduction opportunities across product categories, and projects forecast accuracy benefits with detailed supply chain cost forecasting."
        },
        "Strategic Recommendations": {
            "challenge": "Chief Supply Chain Officers spend hours manually analyzing digital transformation opportunities and developing strategic demand planning technology roadmaps for forecasting advancement and adaptive planning implementation across product portfolios.",
            "solution": "Strategic demand planning intelligence workflow that analyzes competitive advantages against traditional statistical forecasting processes, identifies AI and adaptive forecasting integration opportunities, and creates prioritized digital supply chain transformation roadmaps."
        }
    }
    
    return focus_info.get(focus_area, {"challenge": "", "solution": ""})

def generate_insights_with_agent_workflow(data, focus_area, model_name, progress_placeholder=None):
    """Generate insights using AI agent workflow - Supply Chain Demand Forecasting focused version"""
    
    try:
        # FIRST: Generate the actual insights (behind the scenes)
        insights = generate_insights(data, focus_area, model_name)
        
        # THEN: Prepare for animation
        session_key = f'{focus_area.lower().replace(" ", "_")}_completed_steps'
        st.session_state[session_key] = []
        
        def update_progress(step_name, progress_percent, details, results):
            """Update progress display with completed steps"""
            if progress_placeholder:
                with progress_placeholder.container():
                    # Progress bar
                    st.progress(progress_percent / 100)
                    st.write(f"**{step_name} ({progress_percent}%)**")
                    
                    if details:
                        st.markdown(f'<div class="agent-current">{details}</div>', unsafe_allow_html=True)
                    
                    if results:
                        st.session_state[session_key].append((step_name, results))
                        
                    # Display completed steps
                    for completed_step, completed_result in st.session_state[session_key]:
                        st.markdown(f'<div class="agent-completed">✅ {completed_step}: {completed_result}</div>', unsafe_allow_html=True)
        
        # Calculate real data for enhanced context
        total_records = len(data)
        key_metrics = ["forecast_accuracy_mape", "current_inventory_level", "baseline_demand_forecast", "actual_sales_units"]
        available_metrics = [col for col in key_metrics if col in data.columns]
        
        # Calculate enhanced supply chain data insights
        avg_forecast_accuracy = data['forecast_accuracy_mape'].mean() if 'forecast_accuracy_mape' in data.columns else 0
        avg_inventory_level = data['current_inventory_level'].mean() if 'current_inventory_level' in data.columns else 0
        product_skus = len(data['product_sku'].unique()) if 'product_sku' in data.columns else 0
        locations = len(data['location_code'].unique()) if 'location_code' in data.columns else 0
        avg_baseline_forecast = data['baseline_demand_forecast'].mean() if 'baseline_demand_forecast' in data.columns else 0
        stockout_rate = data['stockout_indicator'].mean() if 'stockout_indicator' in data.columns else 0
        
        # Define enhanced agent workflows for each focus area
        if focus_area == "Overall Performance":
            steps = [
                ("Demand Forecasting Data Initialization", 15, f"Loading comprehensive demand planning dataset with enhanced validation across {total_records} forecast records and {product_skus} product SKUs", f"Connected to {len(available_metrics)} demand metrics across {len(data.columns)} total supply chain data dimensions"),
                ("Forecast Accuracy Assessment", 35, f"Advanced calculation of demand planning indicators with forecast analysis (avg MAPE: {avg_forecast_accuracy:.3f})", f"Computed supply chain metrics: {avg_forecast_accuracy:.3f} forecast accuracy, {avg_inventory_level:,.0f} avg inventory level, {stockout_rate:.1%} stockout rate"),
                ("Demand Planning Pattern Recognition", 55, f"Sophisticated identification of demand patterns with promotional correlation analysis across {locations} locations", f"Detected significant patterns in {len(data['promotional_activity_flag'].unique()) if 'promotional_activity_flag' in data.columns else 'N/A'} promotional categories with inventory correlation analysis completed"),
                ("AI Demand Planning Intelligence Processing", 75, f"Processing comprehensive supply chain data through {model_name} with advanced reasoning for demand optimization insights", f"Enhanced AI analysis of demand forecasting effectiveness across {total_records} forecast records completed"),
                ("Supply Chain Performance Report Compilation", 100, f"Professional demand planning analysis with evidence-based recommendations and actionable forecasting insights ready", f"Comprehensive supply chain performance report with {len(available_metrics)} demand metrics analysis and inventory optimization recommendations generated")
            ]
            
        elif focus_area == "Optimization Opportunities":
            promotional_coverage = data['promotional_activity_flag'].mean() if 'promotional_activity_flag' in data.columns else 0
            forecast_efficiency = (1 - avg_forecast_accuracy) * 100 if avg_forecast_accuracy > 0 else 0
            
            steps = [
                ("Demand Planning Optimization Data Preparation", 12, f"Advanced loading of demand forecasting data with enhanced validation across {total_records} records for forecasting improvement identification", f"Prepared {product_skus} product SKUs, {locations} locations for optimization analysis with {promotional_coverage:.1%} promotional coverage rate"),
                ("Forecast Accuracy Inefficiency Detection", 28, f"Sophisticated analysis of demand planning strategies and inventory performance with evidence-based inefficiency identification", f"Identified optimization opportunities across {product_skus} product categories with demand forecasting and inventory management gaps"),
                ("Supply Chain Correlation Analysis", 45, f"Enhanced examination of relationships between product categories, promotional activities, and forecast accuracy rates", f"Analyzed correlations between demand characteristics and inventory outcomes across {total_records} supply chain records"),
                ("ERP Integration Optimization", 65, f"Comprehensive evaluation of demand planning integration with existing Oracle Demantra, SAP IBP, and Kinaxis RapidResponse systems", f"Assessed integration opportunities across {len(data.columns)} data points and demand planning system optimization needs"),
                ("AI Demand Planning Intelligence", 85, f"Generating advanced supply chain optimization recommendations using {model_name} with demand planning reasoning and implementation strategies", f"AI-powered demand planning optimization strategy across {product_skus} product categories and forecast accuracy improvements completed"),
                ("Demand Planning Strategy Finalization", 100, f"Professional supply chain optimization report with prioritized implementation roadmap and inventory impact analysis ready", f"Comprehensive optimization strategy with {len(available_metrics)} performance improvement areas and demand planning implementation plan generated")
            ]
            
        elif focus_area == "Financial Impact":
            total_inventory_value = avg_inventory_level * product_skus * 10 if avg_inventory_level > 0 else 0
            carrying_cost_savings = total_inventory_value * 0.15 * 0.8 if total_inventory_value > 0 else 0
            
            steps = [
                ("Supply Chain Financial Data Integration", 15, f"Advanced loading of demand planning financial data and inventory cost metrics with enhanced validation across {total_records} forecast records", f"Integrated supply chain financial data: {avg_forecast_accuracy:.3f} avg MAPE, {stockout_rate:.1%} stockout rate across {locations} locations"),
                ("Demand Planning Cost-Benefit Calculation", 30, f"Sophisticated ROI metrics calculation with inventory analysis and demand planning efficiency cost savings", f"Computed comprehensive cost analysis: inventory expenses, stockout costs, and ${carrying_cost_savings:,.0f} estimated inventory optimization potential"),
                ("Inventory Management Impact Assessment", 50, f"Enhanced analysis of supply chain revenue impact with inventory turnover metrics and demand correlation analysis", f"Assessed supply chain implications: {stockout_rate:.1%} stockout rate with {locations} locations requiring inventory optimization"),
                ("Demand Planning Resource Efficiency Analysis", 70, f"Comprehensive evaluation of resource allocation efficiency across demand planning activities with forecast lifecycle cost optimization", f"Analyzed resource efficiency: {product_skus} product categories with inventory carrying cost reduction opportunities identified"),
                ("AI Supply Chain Financial Modeling", 90, f"Advanced demand planning financial projections and supply chain ROI calculations using {model_name} with comprehensive inventory cost-benefit analysis", f"Enhanced financial impact analysis and forecasting across {len(available_metrics)} supply chain cost metrics completed"),
                ("Supply Chain Economics Report Generation", 100, f"Professional supply chain financial impact analysis with detailed demand planning ROI calculations and inventory cost forecasting ready", f"Comprehensive supply chain financial report with ${carrying_cost_savings:,.0f} cost optimization analysis and demand planning efficiency strategy generated")
            ]
            
        elif focus_area == "Strategic Recommendations":
            planning_efficiency_score = forecast_efficiency * 10 if forecast_efficiency > 0 else 0
            
            steps = [
                ("Supply Chain Technology Assessment", 15, f"Advanced loading of demand planning digital context with competitive positioning analysis across {total_records} forecast records and {product_skus} product SKUs", f"Analyzed supply chain technology landscape: {product_skus} product categories, {locations} locations, comprehensive demand planning digitization assessment completed"),
                ("Demand Planning Competitive Advantage Analysis", 30, f"Sophisticated evaluation of competitive positioning against traditional statistical forecasting with AI-powered demand optimization effectiveness", f"Assessed competitive advantages: {planning_efficiency_score:.1f}% planning efficiency, {avg_forecast_accuracy:.3f} MAPE vs industry benchmarks"),
                ("Advanced Supply Chain Technology Integration", 50, f"Enhanced analysis of integration opportunities with IoT sensors, real-time POS data, and AI-powered demand sensing across {len(data.columns)} supply chain data dimensions", f"Identified strategic technology integration: real-time demand sensing, adaptive forecasting algorithms, automated inventory optimization opportunities"),
                ("Digital Supply Chain Strategy Development", 70, f"Comprehensive development of prioritized digital transformation roadmap with evidence-based supply chain technology adoption strategies", f"Created sequenced implementation plan across {product_skus} product categories with advanced demand planning technology integration opportunities"),
                ("AI Supply Chain Strategic Processing", 85, f"Advanced demand planning strategic recommendations using {model_name} with long-term competitive positioning and supply chain technology analysis", f"Enhanced strategic analysis with supply chain competitive positioning and digital transformation roadmap completed"),
                ("Digital Demand Planning Report Generation", 100, f"Professional digital supply chain transformation roadmap with competitive analysis and demand planning technology implementation plan ready for CRO executive review", f"Comprehensive strategic report with {product_skus}-category implementation plan and supply chain competitive advantage analysis generated")
            ]
        
        # NOW: Animate the progress with pre-calculated results
        for step_name, progress_percent, details, results in steps:
            update_progress(step_name, progress_percent, details, results)
            time.sleep(1.2)
        
        return insights
        
    except Exception as e:
        if progress_placeholder:
            progress_placeholder.error(f"❌ Enhanced Agent Analysis failed: {str(e)}")
        return f"Enhanced Agent Analysis failed: {str(e)}"

def generate_insights(data, focus_area, model_name):
    data_summary = f"Table: {table_name}\n"
    data_summary += f"Description: {table_description}\n"
    data_summary += f"Records analyzed: {len(data)}\n"

    # Calculate basic statistics for numeric columns only - exclude ID columns
    numeric_stats = {}
    # Only include actual numeric metrics from supply chain dataset
    key_metrics = ["forecast_horizon_days", "baseline_demand_forecast", "adjusted_demand_forecast", 
                   "actual_sales_units", "current_inventory_level", "promotion_discount_percent", 
                   "market_share_percent", "category_growth_rate", "competitor_price_index", 
                   "economic_indicator_gdp", "consumer_confidence_index", "seasonal_index", 
                   "forecast_accuracy_mape", "sales_rep_adjustment"]
    
    # Filter to only columns that exist and are actually numeric
    available_metrics = []
    for col in key_metrics:
        if col in data.columns:
            try:
                # Test if the column is actually numeric by trying to calculate mean
                test_mean = pd.to_numeric(data[col], errors='coerce').mean()
                if not pd.isna(test_mean):
                    available_metrics.append(col)
            except:
                # Skip columns that can't be converted to numeric
                continue
    
    for col in available_metrics:
        try:
            numeric_data = pd.to_numeric(data[col], errors='coerce')
            numeric_stats[col] = {
                "mean": numeric_data.mean(),
                "min": numeric_data.min(),
                "max": numeric_data.max(),
                "std": numeric_data.std()
            }
            data_summary += f"- {col} (avg: {numeric_data.mean():.2f}, min: {numeric_data.min():.2f}, max: {numeric_data.max():.2f})\n"
        except Exception as e:
            # Skip columns that cause errors
            continue

    # Get top values for categorical columns
    categorical_stats = {}
    categorical_options = ["product_sku", "location_code"]
    for cat_col in categorical_options:
        if cat_col in data.columns:
            try:
                top = data[cat_col].value_counts().head(3)
                categorical_stats[cat_col] = top.to_dict()
                data_summary += f"\nTop {cat_col} values:\n" + "\n".join(f"- {k}: {v}" for k, v in top.items())
            except:
                # Skip columns that cause errors
                continue

    # Calculate correlations if enough numeric columns available
    correlation_info = ""
    if len(available_metrics) >= 2:
        try:
            # Create a dataframe with only the numeric columns
            numeric_df = data[available_metrics].apply(pd.to_numeric, errors='coerce')
            correlations = numeric_df.corr()
            
            # Get the top 3 strongest correlations (absolute value)
            corr_pairs = []
            for i in range(len(correlations.columns)):
                for j in range(i+1, len(correlations.columns)):
                    col1 = correlations.columns[i]
                    col2 = correlations.columns[j]
                    corr_value = correlations.iloc[i, j]
                    if not pd.isna(corr_value):
                        corr_pairs.append((col1, col2, abs(corr_value), corr_value))

            # Sort by absolute correlation value
            corr_pairs.sort(key=lambda x: x[2], reverse=True)

            # Add top correlations to the summary
            if corr_pairs:
                correlation_info = "Top correlations between metrics:\n"
                for col1, col2, _, corr_value in corr_pairs[:3]:
                    correlation_info += f"- {col1} and {col2}: r = {corr_value:.2f}\n"
        except Exception as e:
            correlation_info = "Could not calculate correlations between metrics.\n"

    # Define specific instructions for each focus area
    focus_area_instructions = {
        "Overall Performance": """
        For the Overall Performance analysis of DemandCraft:
        1. Provide a comprehensive analysis of the demand forecasting and supply chain optimization system using forecast accuracy, inventory levels, and sales performance metrics
        2. Identify significant patterns in product performance, location-based demand, promotional impacts, and market dynamics across supply chain operations
        3. Highlight 3-5 key demand forecasting metrics that best indicate forecasting effectiveness (MAPE, baseline vs adjusted forecasts, actual sales alignment)
        4. Discuss both strengths and areas for improvement in the AI-powered demand forecasting process
        5. Include 3-5 actionable insights for improving supply chain planning based on the demand planning data
        
        Structure your response with these supply chain focused sections:
        - Demand Forecasting Insights (5 specific insights with supporting forecast accuracy and inventory data)
        - Supply Chain Performance Trends (3-4 significant trends in forecast accuracy and inventory management)
        - Inventory Optimization Recommendations (3-5 data-backed recommendations for improving demand planning operations)
        - Implementation Steps (3-5 concrete next steps for supply chain and demand planning teams)
        """,
        
        "Optimization Opportunities": """
        For the Optimization Opportunities analysis of DemandCraft:
        1. Focus specifically on areas where demand forecasting accuracy, inventory optimization, and supply chain efficiency can be improved
        2. Identify inefficiencies in demand planning, promotional impact modeling, and inventory management across supply chain operations
        3. Analyze correlations between forecast horizons, promotional activities, market conditions, and forecast accuracy
        4. Prioritize optimization opportunities based on potential impact on inventory reduction and forecast accuracy improvement
        5. Suggest specific technical or process improvements for integration with existing demand planning systems (Oracle Demantra, SAP IBP)
        
        Structure your response with these supply chain focused sections:
        - Supply Chain Optimization Priorities (3-5 areas with highest inventory reduction and forecast accuracy improvement potential)
        - Demand Planning Impact Analysis (quantified benefits of addressing each opportunity in terms of MAPE reduction and inventory optimization)
        - ERP Integration Strategy (specific steps for supply chain teams to implement each optimization)
        - System Integration Recommendations (specific technical changes needed for seamless integration with Oracle, SAP, and Kinaxis systems)
        - Supply Chain Risk Assessment (potential challenges for planners and operations teams and how to mitigate them)
        """,
        
        "Financial Impact": """
        For the Financial Impact analysis of DemandCraft:
        1. Focus on cost-benefit analysis and ROI in supply chain terms (inventory carrying costs vs. forecast accuracy improvements)
        2. Quantify financial impacts through inventory reduction, stockout prevention, and promotional ROI optimization
        3. Identify cost savings opportunities across different product categories and geographic locations
        4. Analyze resource allocation efficiency across different forecasting horizons and promotional activities
        5. Project future financial outcomes based on improved demand forecasting accuracy and reduced excess inventory
        
        Structure your response with these supply chain focused sections:
        - Inventory Cost Analysis (breakdown of carrying costs and potential savings by product category and location)
        - Demand Planning Efficiency Impact (how improved forecasting affects inventory costs and revenue)
        - Supply Chain ROI Calculation (specific calculations showing return on investment in terms of inventory reduction and stockout prevention)
        - Cost Reduction Opportunities (specific areas to reduce inventory carrying costs and improve promotional ROI)
        - Financial Forecasting (projections based on improved demand planning efficiency metrics)
        """,
        
        "Strategic Recommendations": """
        For the Strategic Recommendations analysis of DemandCraft:
        1. Focus on long-term strategic implications for digital transformation in supply chain demand planning
        2. Identify competitive advantages against traditional statistical forecasting approaches
        3. Suggest new directions for AI integration with emerging supply chain technologies (e.g., IoT sensors, real-time POS data)
        4. Connect recommendations to broader supply chain goals of reducing costs and improving customer service levels
        5. Provide a digital supply chain roadmap with prioritized initiatives
        
        Structure your response with these supply chain focused sections:
        - Digital Supply Chain Context (how DemandCraft fits into broader digital transformation in supply chain management)
        - Competitive Advantage Analysis (how to maximize efficiency advantages compared to traditional demand planning)
        - Supply Chain Technology Strategic Priorities (3-5 high-impact strategic initiatives for improving demand planning operations)
        - Advanced Forecasting Technology Integration Vision (how to evolve DemandCraft with IoT and real-time data over 1-3 years)
        - Supply Chain Transformation Roadmap (sequenced steps for expanding to predictive analytics and autonomous demand planning)
        """
    }

    # Get the specific instructions for the selected focus area
    selected_focus_instructions = focus_area_instructions.get(focus_area, "")

    prompt = f'''
    You are an expert data analyst specializing in {focus_area.lower()} analysis for supply chain demand forecasting and inventory optimization.

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
    - Frame all insights in the context of supply chain demand forecasting and inventory optimization
    '''

    return call_cortex_model(prompt, model_name)

def create_metrics_charts(data):
    """Create metric visualizations for the supply chain demand forecasting data"""
    charts = []
    
    # Forecast Accuracy Distribution
    if 'forecast_accuracy_mape' in data.columns:
        accuracy_chart = alt.Chart(data).mark_bar().encode(
            alt.X('forecast_accuracy_mape:Q', bin=alt.Bin(maxbins=15), title='Forecast Accuracy MAPE'),
            alt.Y('count()', title='Number of Records'),
            color=alt.value('#1f77b4')
        ).properties(
            title='Forecast Accuracy Distribution (MAPE)',
            width=380,
            height=340
        )
        charts.append(('Forecast Accuracy Distribution', accuracy_chart))
    
    # Inventory Levels by Location
    if 'current_inventory_level' in data.columns and 'location_code' in data.columns:
        inventory_chart = alt.Chart(data).mark_boxplot().encode(
            alt.X('location_code:N', title='Location Code'),
            alt.Y('current_inventory_level:Q', title='Current Inventory Level'),
            color=alt.Color('location_code:N', legend=None)
        ).properties(
            title='Inventory Levels by Location',
            width=380,
            height=340
        )
        charts.append(('Inventory Levels by Location', inventory_chart))
    
    # Forecast vs Actual Sales Comparison
    if 'baseline_demand_forecast' in data.columns and 'actual_sales_units' in data.columns:
        forecast_actual_chart = alt.Chart(data).mark_point(size=60, opacity=0.7).encode(
            alt.X('baseline_demand_forecast:Q', title='Baseline Demand Forecast'),
            alt.Y('actual_sales_units:Q', title='Actual Sales Units'),
            color=alt.value('#ff7f0e'),
            tooltip=['baseline_demand_forecast:Q', 'actual_sales_units:Q', 'product_sku:N']
        ).properties(
            title='Forecast vs Actual Sales',
            width=380,
            height=340
        )
        charts.append(('Forecast vs Actual Sales', forecast_actual_chart))
    
    # Promotional Impact Analysis
    if 'promotional_activity_flag' in data.columns and 'promotion_discount_percent' in data.columns:
        promo_chart = alt.Chart(data[data['promotional_activity_flag'] == True]).mark_bar().encode(
            alt.X('promotion_discount_percent:Q', bin=alt.Bin(maxbins=10), title='Promotion Discount %'),
            alt.Y('count()', title='Number of Promotions'),
            color=alt.value('#2ca02c')
        ).properties(
            title='Promotional Discount Distribution',
            width=380,
            height=340
        )
        charts.append(('Promotional Discount Distribution', promo_chart))
    
    # Market Share by Product Category
    if 'market_share_percent' in data.columns and 'product_sku' in data.columns:
        # Group by first 3 characters of SKU as product category
        data_copy = data.copy()
        data_copy['product_category'] = data_copy['product_sku'].str[:3]
        
        market_share_chart = alt.Chart(data_copy).mark_bar().encode(
            alt.X('product_category:N', title='Product Category'),
            alt.Y('mean(market_share_percent):Q', title='Average Market Share %'),
            color=alt.Color('product_category:N', legend=None),
            tooltip=['product_category:N', 'mean(market_share_percent):Q']
        ).properties(
            title='Average Market Share by Product Category',
            width=380,
            height=340
        )
        charts.append(('Market Share by Product Category', market_share_chart))
    
    # Stockout Frequency Analysis
    if 'stockout_indicator' in data.columns and 'location_code' in data.columns:
        stockout_data = data.groupby('location_code')['stockout_indicator'].agg(['sum', 'count']).reset_index()
        stockout_data['stockout_rate'] = stockout_data['sum'] / stockout_data['count']
        stockout_data = stockout_data[stockout_data['count'] >= 5]  # Only locations with 5+ records
        
        if not stockout_data.empty:
            stockout_chart = alt.Chart(stockout_data).mark_bar().encode(
                alt.X('location_code:O', title='Location Code', sort='-y'),
                alt.Y('stockout_rate:Q', title='Stockout Rate'),
                color=alt.Color('stockout_rate:Q', title='Stockout Rate', scale=alt.Scale(scheme='reds')),
                tooltip=['location_code:O', alt.Tooltip('stockout_rate:Q', format='.3f')]
            ).properties(
                title='Stockout Rate by Location',
                width=380,
                height=340
            )
            charts.append(('Stockout Rate by Location', stockout_chart))
    
    # Economic Indicators vs Forecast Accuracy
    if 'economic_indicator_gdp' in data.columns and 'forecast_accuracy_mape' in data.columns:
        econ_chart = alt.Chart(data).mark_point(size=60, opacity=0.7).encode(
            alt.X('economic_indicator_gdp:Q', title='GDP Indicator'),
            alt.Y('forecast_accuracy_mape:Q', title='Forecast Accuracy MAPE'),
            color=alt.value('#9467bd'),
            tooltip=['economic_indicator_gdp:Q', 'forecast_accuracy_mape:Q']
        ).properties(
            title='Economic Indicators vs Forecast Accuracy',
            width=380,
            height=340
        )
        charts.append(('Economic Indicators vs Forecast Accuracy', econ_chart))
    
    # Seasonal Index Trends
    if 'seasonal_index' in data.columns and 'forecast_date' in data.columns:
        try:
            # Convert forecast_date to datetime if possible
            data_copy = data.copy()
            data_copy['forecast_date'] = pd.to_datetime(data_copy['forecast_date'], errors='coerce')
            data_copy = data_copy.dropna(subset=['forecast_date'])
            
            if not data_copy.empty:
                seasonal_chart = alt.Chart(data_copy).mark_line(point=True).encode(
                    alt.X('forecast_date:T', title='Forecast Date'),
                    alt.Y('mean(seasonal_index):Q', title='Average Seasonal Index'),
                    color=alt.value('#d62728')
                ).properties(
                    title='Seasonal Index Trends',
                    width=380,
                    height=340
                )
                charts.append(('Seasonal Index Trends', seasonal_chart))
        except:
            # If date conversion fails, create alternative chart
            if 'product_sku' in data.columns:
                seasonal_product_chart = alt.Chart(data).mark_bar().encode(
                    alt.X('product_sku:N', title='Product SKU'),
                    alt.Y('mean(seasonal_index):Q', title='Average Seasonal Index'),
                    color=alt.Color('mean(seasonal_index):Q', scale=alt.Scale(scheme='viridis'))
                ).properties(
                    title='Seasonal Index by Product',
                    width=380,
                    height=340
                ).transform_window(
                    rank='rank(mean_seasonal_index)',
                    sort=[alt.SortField('mean_seasonal_index', order='descending')]
                ).transform_filter(
                    alt.datum.rank <= 10
                )
                charts.append(('Seasonal Index by Product (Top 10)', seasonal_product_chart))
    
    return charts

data = load_data()
if data.empty:
    st.error("No data found.")
    st.stop()

    # Define column categories for analysis
categorical_cols = [col for col in ["product_sku", "location_code"] if col in data.columns]
numeric_cols = [col for col in ["forecast_horizon_days", "baseline_demand_forecast", "adjusted_demand_forecast", "actual_sales_units", "current_inventory_level", "promotion_discount_percent", "market_share_percent", "category_growth_rate", "competitor_price_index", "economic_indicator_gdp", "consumer_confidence_index", "seasonal_index", "forecast_accuracy_mape", "sales_rep_adjustment"] if col in data.columns]
date_cols = [col for col in ["forecast_date"] if col in data.columns]

sample_cols = data.columns.tolist()
numeric_candidates = [col for col in sample_cols if data[col].dtype in ['float64', 'int64'] and 'id' not in col.lower()]
date_candidates = [col for col in sample_cols if 'date' in col.lower() or 'timestamp' in col.lower()]
cat_candidates = [col for col in sample_cols if data[col].dtype == 'object' and data[col].nunique() < 1000]


# Four tabs - Metrics tab first, then AI Insights
tabs = st.tabs(["📊 Metrics", "✨ AI Insights", "📁 Insights History", "🔍 Data Explorer"])

# Metrics tab (now first)
with tabs[0]:
    st.subheader("📊 Key Performance Metrics")
    
    # Display key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'forecast_accuracy_mape' in data.columns:
            avg_mape = data['forecast_accuracy_mape'].mean()
            # If MAPE is stored as percentage (24.319), divide by 100
            avg_mape_percent = avg_mape / 100 if avg_mape > 1 else avg_mape
            st.metric("Avg Forecast Accuracy (MAPE)", f"{avg_mape_percent:.1%}", delta=f"{(0.15 - avg_mape_percent):.1%} vs target")
    
    with col2:
        if 'current_inventory_level' in data.columns:
            avg_inventory = data['current_inventory_level'].mean()
            st.metric("Avg Inventory Level", f"{avg_inventory:,.0f} units", delta=f"{((avg_inventory - 10000) / 1000):.1f}k vs baseline")
    
    with col3:
        # REPLACE WITH THIS 👇
        if 'market_share_percent' in data.columns:
            avg_market_share = data['market_share_percent'].mean()
            st.metric("Avg Market Share", f"{avg_market_share:.1f}%", delta=f"{(avg_market_share - 25):.1f}% vs target")
    
    with col4:
        if 'promotional_activity_flag' in data.columns:
            promo_coverage = data['promotional_activity_flag'].mean()
            st.metric("Promotional Coverage", f"{promo_coverage:.1%}")
    
    st.markdown("---")
    
    # Create and display charts
    charts = create_metrics_charts(data)
    
    # ---- Title clipping fix (Altair) ----
    # 1) TitleParams with offset to push the title down
    # 2) Extra top padding so the title never clips in Snowflake Streamlit
    def _fixed_title(text: str) -> alt.TitleParams:
        return alt.TitleParams(
            text=text,
            fontSize=16,
            fontWeight='bold',
            anchor='start',
            offset=14  # key: moves the title downward so it isn't cut off
        )

    _PAD = {"top": 28, "left": 6, "right": 6, "bottom": 6}  # key: explicit headroom
    
    charts_fixed = []
    if charts:
        for item in charts:
            try:
                t, ch = item
            except Exception:
                t, ch = "", item  # fallback if helper returns a bare chart
            ch = ch.properties(title=_fixed_title(t or ""), padding=_PAD)
            ch = ch.configure_title(anchor='start')
            charts_fixed.append((t, ch))
    
    if charts_fixed:
        st.subheader("📈 Performance Visualizations")
        
        # Display charts in a 2-column grid, ensuring all charts are shown
        num_charts = len(charts_fixed)
        for i in range(0, num_charts, 2):
            cols = st.columns(2)
            
            # Left column chart
            if i < num_charts:
                _, chart_obj = charts_fixed[i]
                with cols[0]:
                    st.altair_chart(chart_obj, use_container_width=True)
            
            # Right column chart
            if i + 1 < num_charts:
                _, chart_obj = charts_fixed[i + 1]
                with cols[1]:
                    st.altair_chart(chart_obj, use_container_width=True)
        
        # Display chart count for debugging
        st.caption(f"Displaying {num_charts} performance charts")
    else:
        st.info("No suitable data found for creating visualizations.")
    
    # Enhanced Summary statistics table
    st.subheader("📈 Summary Statistics")
    if numeric_candidates:
        # Create enhanced summary statistics
        summary_stats = data[numeric_candidates].describe()
        
        # Transpose for better readability and add formatting
        summary_df = summary_stats.T.round(3)
        
        # Add meaningful column names and formatting
        summary_df.columns = ['Count', 'Mean', 'Std Dev', 'Min', '25%', '50% (Median)', '75%', 'Max']
        
        # Create two columns for better organization
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Key Demand Planning Metrics**")
            key_metrics = ['forecast_accuracy_mape', 'current_inventory_level', 'baseline_demand_forecast', 'actual_sales_units']
            key_metrics_present = [m for m in key_metrics if m in summary_df.index]
            
            if key_metrics_present:
                key_stats_df = summary_df.loc[key_metrics_present]
                
                # Create a more readable format
                for metric in key_stats_df.index:
                    mean_val = key_stats_df.loc[metric, 'Mean']
                    min_val = key_stats_df.loc[metric, 'Min']
                    max_val = key_stats_df.loc[metric, 'Max']
                    
                    # Format based on metric type
                    if 'mape' in metric.lower():
                        st.metric(
                            label=metric.replace('_', ' ').title(),
                            value=f"{mean_val:.3f}",
                            help=f"Range: {min_val:.3f} - {max_val:.3f}"
                        )
                    elif 'level' in metric.lower() or 'forecast' in metric.lower() or 'sales' in metric.lower():
                        st.metric(
                            label=metric.replace('_', ' ').title(),
                            value=f"{mean_val:,.0f} units",
                            help=f"Range: {min_val:,.0f} - {max_val:,.0f} units"
                        )
                    else:
                        st.metric(
                            label=metric.replace('_', ' ').title(),
                            value=f"{mean_val:.2f}",
                            help=f"Range: {min_val:.2f} - {max_val:.2f}"
                        )
        
        with col2:
            st.markdown("**📊 Supply Chain Insights**")
            
            # Calculate and display key insights
            insights = []
            
            if 'forecast_accuracy_mape' in summary_df.index:
                mape_mean = summary_df.loc['forecast_accuracy_mape', 'Mean']
                mape_std = summary_df.loc['forecast_accuracy_mape', 'Std Dev']
                insights.append(f"• **Forecast Accuracy Variability**: {mape_std:.3f} (σ)")
                
                if mape_mean < 0.1:
                    insights.append(f"• **Excellent forecast accuracy** (MAPE: {mape_mean:.3f})")
                elif mape_mean < 0.2:
                    insights.append(f"• **Good forecast accuracy** (MAPE: {mape_mean:.3f})")
                else:
                    insights.append(f"• **⚠️ Forecast accuracy needs improvement** (MAPE: {mape_mean:.3f})")
            
            if 'current_inventory_level' in summary_df.index:
                inv_q75 = summary_df.loc['current_inventory_level', '75%']
                inv_q25 = summary_df.loc['current_inventory_level', '25%']
                inv_iqr = inv_q75 - inv_q25
                insights.append(f"• **Inventory Level IQR**: {inv_iqr:,.0f} units")
            
            if 'baseline_demand_forecast' in summary_df.index and 'actual_sales_units' in summary_df.index:
                forecast_median = summary_df.loc['baseline_demand_forecast', '50% (Median)']
                actual_median = summary_df.loc['actual_sales_units', '50% (Median)']
                insights.append(f"• **Forecast vs Actual (Median)**: {forecast_median:,.0f} vs {actual_median:,.0f}")
            
            if 'stockout_indicator' in data.columns:
                stockout_rate = data['stockout_indicator'].mean()
                insights.append(f"• **Stockout Rate**: {stockout_rate:.1%}")
                if stockout_rate > 0.1:
                    insights.append(f"• **⚠️ High stockout frequency**: {stockout_rate:.1%}")
            
            # Add categorical insights
            if 'product_sku' in data.columns:
                unique_products = data['product_sku'].nunique()
                insights.append(f"• **Product Portfolio**: {unique_products} unique SKUs")
            
            if 'location_code' in data.columns:
                unique_locations = data['location_code'].nunique()
                insights.append(f"• **Geographic Coverage**: {unique_locations} locations")
            
            for insight in insights:
                st.markdown(insight)
        
        # Full detailed table (collapsible)
        with st.expander("📋 Detailed Statistics Table", expanded=False):
            st.dataframe(
                summary_df.style.format({
                    'Count': '{:.0f}',
                    'Mean': '{:.3f}',
                    'Std Dev': '{:.3f}',
                    'Min': '{:.3f}',
                    '25%': '{:.3f}',
                    '50% (Median)': '{:.3f}',
                    '75%': '{:.3f}',
                    'Max': '{:.3f}'
                }),
                use_container_width=True
            )

# AI Insights tab with Agent Workflows
with tabs[1]:
    st.subheader("✨ AI-Powered Insights with Agent Workflows")
    st.markdown("**Experience behind-the-scenes AI agent processing for each supply chain demand forecasting analysis focus area**")
    
    focus_area = st.radio("Focus Area", [
        "Overall Performance", 
        "Optimization Opportunities", 
        "Financial Impact", 
        "Strategic Recommendations"
    ])
    
    # Show business challenge and solution
    focus_info = get_focus_area_info(focus_area)
    if focus_info["challenge"]:
        st.markdown("#### Business Challenge")
        st.info(focus_info["challenge"])
        st.markdown("#### Agent Solution")
        st.success(focus_info["solution"])
    
    st.markdown("**Select Snowflake Cortex Model for Analysis:**")
    selected_model = st.selectbox("", MODELS, index=0, label_visibility="collapsed")

    # Agent control buttons and status
    col1, col2, col3 = st.columns([2, 1, 1])
    
    agent_running_key = f"{focus_area}_agent_running"
    if agent_running_key not in st.session_state:
        st.session_state[agent_running_key] = False
    
    with col1:
        if st.button("🚀 Start Agent"):
            st.session_state[agent_running_key] = True
            st.rerun()
    
    with col2:
        if st.button("⏹ Stop Agent"):
            st.session_state[agent_running_key] = False
            st.rerun()
    
    with col3:
        st.markdown("**Status**")
        if st.session_state[agent_running_key]:
            st.markdown('<div class="agent-status-active">✅ Active</div>', unsafe_allow_html=True)
        else:
            st.markdown("⏸ Ready")

    # Progress placeholder
    progress_placeholder = st.empty()
    
    # Run agent if active
    if st.session_state[agent_running_key]:
        with st.spinner("Agent Running..."):
            insights = generate_insights_with_agent_workflow(data, focus_area, selected_model, progress_placeholder)
            
            if insights:
                # Show completion message
                st.success(f"🎉 {focus_area} Agent completed with real supply chain demand forecasting data analysis!")
                
                # Show report in expandable section
                with st.expander(f"📋 Generated {focus_area} Report (Real Supply Chain Data)", expanded=True):
                    st.markdown(f"""
                    <div class="agent-report-header">
                        <strong>{focus_area} Report - AI-Generated Supply Chain Demand Forecasting Analysis</strong><br>
                        <small>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</small><br>
                        <small>Data Source: Live Snowflake Supply Chain Demand Planning Analysis</small><br>
                        <small>AI Model: {selected_model}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(insights)
                
                # Save to history
                timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                st.session_state.insights_history.append({
                    "timestamp": timestamp,
                    "focus": focus_area,
                    "insights": insights,
                    "model": selected_model
                })
                
                # Stop the agent after completion
                st.session_state[agent_running_key] = False

# Insights History tab (now third)
with tabs[2]:
    st.subheader("📁 Insights History")
    if st.session_state.insights_history:
        for i, item in enumerate(reversed(st.session_state.insights_history)):
            with st.expander(f"{item['timestamp']} - {item['focus']} ({item['model']})", expanded=False):
                st.markdown(item["insights"])
    else:
        st.info("No insights generated yet. Go to the AI Insights tab to generate some insights.")

# Data Explorer tab (now fourth)
with tabs[3]:
    st.subheader("🔍 Data Explorer")
    rows_per_page = st.slider("Rows per page", 5, 50, 10)
    page = st.number_input("Page", min_value=1, value=1)
    start = (page - 1) * rows_per_page
    end = min(start + rows_per_page, len(data))
    st.dataframe(data.iloc[start:end], use_container_width=True)
    st.caption(f"Showing rows {start + 1}–{end} of {len(data)}")