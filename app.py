import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import Prophet, with fallback if not available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    try:
        from fbprophet import Prophet
        PROPHET_AVAILABLE = True
    except ImportError:
        PROPHET_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Marketing Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
def load_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Main container styling */
    .main > div {
        padding-top: 1rem;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Header styling */
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.15);
    }
    
    .dashboard-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .dashboard-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Metric card styling */
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2d3748;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #4a5568;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    .metric-icon {
        font-size: 2rem;
        float: right;
        opacity: 0.7;
    }
    
    /* Insights styling */
    .insight-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .insight-title {
        color: #2d3748;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .insight-list {
        list-style: none;
        padding-left: 0;
    }
    
    .insight-list li {
        color: #4a5568;
        margin: 0.5rem 0;
        padding-left: 1.5rem;
        position: relative;
    }
    
    .insight-list li::before {
        content: '💡';
        position: absolute;
        left: 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Data loading and preprocessing
@st.cache_data
def load_data():
    """Load and preprocess marketing data"""
    try:
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 12, 31)
        
        # Create a full date range for the base data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Initialize an empty list to store dataframes
        all_marketing_data = []

        # List of channels
        channels = ['Facebook', 'Google', 'TikTok']
        campaigns = {
            'Facebook': ['Brand Awareness', 'Lead Gen', 'Conversion', 'Retargeting'],
            'Google': ['Search Brand', 'Search Generic', 'Display', 'Shopping'],
            'TikTok': ['Hashtag Challenge', 'In-Feed', 'Brand Takeover', 'Branded Effects']
        }
        tactics = {
            'Facebook': ['Video', 'Carousel', 'Single Image', 'Collection'],
            'Google': ['Text Ad', 'Display Ad', 'Video Ad', 'Shopping Ad'],
            'TikTok': ['Video Ad', 'Spark Ad', 'Collection Ad', 'Dynamic Ad']
        }
        states = ['CA', 'NY', 'TX', 'FL', 'IL']

        for channel in channels:
            # Create a daily dataframe for each channel
            channel_df = pd.DataFrame({'date': dates})
            channel_df['channel'] = channel
            
            # Generate random metrics
            channel_df['impressions'] = np.random.randint(5000, 100000, len(dates))
            channel_df['clicks'] = (channel_df['impressions'] * np.random.uniform(0.01, 0.05, len(dates))).astype(int)
            channel_df['spend'] = np.random.uniform(100, 5000, len(dates))
            channel_df['attributed_revenue'] = channel_df['spend'] * np.random.uniform(1.5, 4.0, len(dates))
            channel_df['campaign'] = np.random.choice(campaigns[channel], len(dates))
            channel_df['state'] = np.random.choice(states, len(dates))
            channel_df['tactic'] = np.random.choice(tactics[channel], len(dates))
            
            all_marketing_data.append(channel_df)

        # Combine all marketing data
        marketing_data = pd.concat(all_marketing_data, ignore_index=True)
        marketing_data['date'] = pd.to_datetime(marketing_data['date'])
        
        # Business data
        business_data = pd.DataFrame({
            'date': dates,
            'orders': np.random.randint(50, 500, len(dates)),
            'new_orders': np.random.randint(10, 200, len(dates)),
            'new_customers': np.random.randint(5, 150, len(dates)),
            'total_revenue': np.random.uniform(5000, 50000, len(dates)),
            'gross_profit': np.random.uniform(2000, 25000, len(dates)),
            'COGS': np.random.uniform(1000, 15000, len(dates))
        })
        business_data['date'] = pd.to_datetime(business_data['date'])

        # Calculate additional metrics with safe division
        marketing_data['ctr'] = (marketing_data['clicks'] / marketing_data['impressions'].replace(0, 1)) * 100
        marketing_data['cpc'] = marketing_data['spend'] / marketing_data['clicks'].replace(0, 1)
        marketing_data['roas'] = marketing_data['attributed_revenue'] / marketing_data['spend'].replace(0, 1)
        marketing_data['cpa'] = marketing_data['spend'] / (marketing_data['attributed_revenue'] / 100).replace(0, 1)
        
        # Clean any infinite values
        marketing_data = marketing_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        return marketing_data, business_data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def create_metric_card(title, value, icon, delta=None):
    """Create a styled metric card"""
    delta_html = ""
    if delta is not None:
        delta_color = "green" if delta > 0 else "red"
        delta_symbol = "↑" if delta > 0 else "↓"
        delta_html = f'<div style="color: {delta_color}; font-size: 0.8rem;">{delta_symbol} {abs(delta):.1f}%</div>'
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def apply_filters(df, channels, tactics, states, date_range):
    """Apply filters to dataframe"""
    if df.empty:
        return df
        
    filtered_df = df.copy()
    
    if channels:
        filtered_df = filtered_df[filtered_df['channel'].isin(channels)]
    if tactics:
        filtered_df = filtered_df[filtered_df['tactic'].isin(tactics)]
    if states:
        filtered_df = filtered_df[filtered_df['state'].isin(states)]
    
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['date'] >= pd.Timestamp(date_range[0])) & 
            (filtered_df['date'] <= pd.Timestamp(date_range[1]))
        ]
    
    return filtered_df

def create_trends_tab(marketing_data, business_data):
    """Create trends analysis tab"""
    st.subheader("📈 Performance Trends")
    
    if marketing_data.empty or business_data.empty:
        st.warning("No data available for trends analysis.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Marketing trends
        try:
            daily_marketing = marketing_data.groupby('date').agg({
                'spend': 'sum',
                'attributed_revenue': 'sum'
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_marketing['date'],
                y=daily_marketing['spend'],
                name='Spend',
                line=dict(color='#667eea', width=3),
                hovertemplate='<b>Date:</b> %{x}<br><b>Spend:</b> $%{y:,.0f}<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=daily_marketing['date'],
                y=daily_marketing['attributed_revenue'],
                name='Attributed Revenue',
                line=dict(color='#764ba2', width=3),
                hovertemplate='<b>Date:</b> %{x}<br><b>Revenue:</b> $%{y:,.0f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Marketing Performance Over Time',
                xaxis_title='Date',
                yaxis_title='Amount ($)',
                hovermode='x unified',
                showlegend=True,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating marketing trends chart: {str(e)}")
    
    with col2:
        # Business trends
        try:
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig2.add_trace(
                go.Scatter(
                    x=business_data['date'],
                    y=business_data['total_revenue'],
                    name='Total Revenue',
                    line=dict(color='#764ba2', width=3),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Revenue:</b> $%{y:,.0f}<extra></extra>'
                ),
                secondary_y=False,
            )
            
            fig2.add_trace(
                go.Scatter(
                    x=business_data['date'],
                    y=business_data['orders'],
                    name='Orders',
                    line=dict(color='#f6d365', width=3),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Orders:</b> %{y:,.0f}<extra></extra>'
                ),
                secondary_y=True,
            )
            
            fig2.update_xaxes(title_text="Date")
            fig2.update_yaxes(title_text="Revenue ($)", secondary_y=False)
            fig2.update_yaxes(title_text="Orders", secondary_y=True)
            
            fig2.update_layout(
                title='Business Performance Over Time',
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating business trends chart: {str(e)}")

def create_channels_tab(marketing_data):
    """Create channels analysis tab"""
    st.subheader("📊 Channel Performance")
    
    if marketing_data.empty:
        st.warning("No data available for channel analysis.")
        return
    
    try:
        # Channel performance table
        channel_performance = marketing_data.groupby('channel').agg({
            'impressions': 'sum',
            'clicks': 'sum',
            'spend': 'sum',
            'attributed_revenue': 'sum',
            'roas': 'mean'
        }).reset_index()
        
        channel_performance['ctr'] = (channel_performance['clicks'] / channel_performance['impressions']) * 100
        channel_performance['cpc'] = channel_performance['spend'] / channel_performance['clicks']
        
        # Format for display
        display_df = channel_performance.copy()
        for col in ['spend', 'attributed_revenue', 'cpc']:
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")
        for col in ['impressions', 'clicks']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:,}")
        for col in ['ctr', 'roas']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(display_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Channel spend pie chart
            fig_pie = px.pie(
                channel_performance,
                values='spend',
                names='channel',
                title='Spend Distribution by Channel',
                color_discrete_sequence=['#667eea', '#764ba2', '#f093fb']
            )
            fig_pie.update_traces(
                hovertemplate='<b>%{label}</b><br>Spend: $%{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # ROAS by channel
            fig_roas = px.bar(
                channel_performance,
                x='channel',
                y='roas',
                title='ROAS by Channel',
                color='roas',
                color_continuous_scale='Viridis'
            )
            fig_roas.update_traces(
                hovertemplate='<b>%{x}</b><br>ROAS: %{y:.2f}<extra></extra>'
            )
            fig_roas.update_layout(showlegend=False)
            st.plotly_chart(fig_roas, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error in channels analysis: {str(e)}")

def create_campaigns_tab(marketing_data):
    """Create campaigns analysis tab"""
    st.subheader("🎯 Campaign Performance")
    
    if marketing_data.empty:
        st.warning("No data available for campaign analysis.")
        return
    
    try:
        # Campaign leaderboard
        campaign_performance = marketing_data.groupby(['campaign', 'channel']).agg({
            'spend': 'sum',
            'attributed_revenue': 'sum',
            'impressions': 'sum',
            'clicks': 'sum'
        }).reset_index()
        
        campaign_performance['roas'] = campaign_performance['attributed_revenue'] / campaign_performance['spend']
        campaign_performance['orders_estimate'] = campaign_performance['attributed_revenue'] / 50  # Assuming $50 AOV
        
        # Top campaigns by revenue
        top_campaigns = campaign_performance.nlargest(10, 'attributed_revenue')
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Scatter plot: Spend vs Revenue
            fig_scatter = px.scatter(
                campaign_performance,
                x='spend',
                y='attributed_revenue',
                size='orders_estimate',
                color='channel',
                hover_data=['campaign', 'roas'],
                title='Campaign Performance: Spend vs Revenue (Bubble size = Estimated Orders)',
                color_discrete_sequence=['#667eea', '#764ba2', '#f093fb']
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            st.markdown("### 🏆 Top Campaigns by Revenue")
            for i, row in top_campaigns.head(5).iterrows():
                st.markdown(f"""
                <div class="metric-card" style="margin: 0.5rem 0; padding: 1rem;">
                    <strong>{row['campaign']}</strong> ({row['channel']})<br>
                    <span style="color: #667eea;">Revenue: ${row['attributed_revenue']:,.0f}</span><br>
                    <span style="color: #764ba2;">ROAS: {row['roas']:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error in campaigns analysis: {str(e)}")

def create_business_tab(business_data):
    """Create business metrics tab"""
    st.subheader("💼 Business Metrics")
    
    if business_data.empty:
        st.warning("No data available for business analysis.")
        return
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            # Orders vs New Customers (last 30 days)
            recent_data = business_data.tail(30)
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Bar(
                    x=recent_data['date'],
                    y=recent_data['orders'],
                    name='Total Orders',
                    marker_color='#667eea',
                    hovertemplate='<b>Date:</b> %{x}<br><b>Orders:</b> %{y}<extra></extra>'
                ),
                secondary_y=False,
            )
            
            fig.add_trace(
                go.Scatter(
                    x=recent_data['date'],
                    y=recent_data['new_customers'],
                    name='New Customers',
                    line=dict(color='#f093fb', width=3),
                    hovertemplate='<b>Date:</b> %{x}<br><b>New Customers:</b> %{y}<extra></extra>'
                ),
                secondary_y=True,
            )
            
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Orders", secondary_y=False)
            fig.update_yaxes(title_text="New Customers", secondary_y=True)
            
            fig.update_layout(
                title_text="Orders vs New Customers (Last 30 Days)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Revenue breakdown
            fig_revenue = go.Figure(data=[
                go.Bar(
                    x=recent_data['date'],
                    y=recent_data['total_revenue'],
                    name='Total Revenue',
                    marker_color='#764ba2',
                    hovertemplate='<b>Date:</b> %{x}<br><b>Revenue:</b> $%{y:,.0f}<extra></extra>'
                ),
                go.Bar(
                    x=recent_data['date'],
                    y=recent_data['gross_profit'],
                    name='Gross Profit',
                    marker_color='#f6d365',
                    hovertemplate='<b>Date:</b> %{x}<br><b>Profit:</b> $%{y:,.0f}<extra></extra>'
                )
            ])
            
            fig_revenue.update_layout(
                title='Revenue vs Gross Profit (Last 30 Days)',
                xaxis_title='Date',
                yaxis_title='Amount ($)',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_revenue, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error in business analysis: {str(e)}")

def create_forecasting_tab(marketing_data, business_data):
    """Create forecasting tab with Prophet"""
    st.subheader("🔮 Forecasting")
    
    if not PROPHET_AVAILABLE:
        st.warning("Prophet library is not available. Install with: pip install prophet")
        return
    
    if marketing_data.empty or business_data.empty:
        st.warning("Insufficient data for forecasting.")
        return
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            # Forecast Attributed Revenue
            st.markdown("#### Attributed Revenue Forecast")
            
            # Prepare data for Prophet
            revenue_data = marketing_data.groupby('date')['attributed_revenue'].sum().reset_index()
            revenue_data.columns = ['ds', 'y']
            
            if len(revenue_data) < 10:
                st.warning("Need at least 10 data points for forecasting.")
                return
            
            # Create and fit Prophet model
            model_revenue = Prophet(
                daily_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model_revenue.fit(revenue_data)
            
            # Make future dataframe
            future_revenue = model_revenue.make_future_dataframe(periods=30)
            forecast_revenue = model_revenue.predict(future_revenue)
            
            # Plot forecast
            fig_revenue = go.Figure()
            
            # Historical data
            fig_revenue.add_trace(go.Scatter(
                x=revenue_data['ds'],
                y=revenue_data['y'],
                name='Actual',
                line=dict(color='#667eea', width=2),
                hovertemplate='<b>Date:</b> %{x}<br><b>Revenue:</b> $%{y:,.0f}<extra></extra>'
            ))
            
            # Forecast
            forecast_future = forecast_revenue.tail(30)
            fig_revenue.add_trace(go.Scatter(
                x=forecast_future['ds'],
                y=forecast_future['yhat'],
                name='Forecast',
                line=dict(color='#f093fb', width=2, dash='dash'),
                hovertemplate='<b>Date:</b> %{x}<br><b>Forecast:</b> $%{y:,.0f}<extra></extra>'
            ))
            
            fig_revenue.update_layout(
                title='30-Day Revenue Forecast',
                xaxis_title='Date',
                yaxis_title='Revenue ($)',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        with col2:
            # Forecast Orders
            st.markdown("#### Orders Forecast")
            
            # Prepare data for Prophet
            orders_data = business_data[['date', 'orders']].copy()
            orders_data.columns = ['ds', 'y']
            
            # Create and fit Prophet model
            model_orders = Prophet(
                daily_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model_orders.fit(orders_data)
            
            # Make future dataframe
            future_orders = model_orders.make_future_dataframe(periods=30)
            forecast_orders = model_orders.predict(future_orders)
            
            # Plot forecast
            fig_orders = go.Figure()
            
            # Historical data
            fig_orders.add_trace(go.Scatter(
                x=orders_data['ds'],
                y=orders_data['y'],
                name='Actual',
                line=dict(color='#764ba2', width=2),
                hovertemplate='<b>Date:</b> %{x}<br><b>Orders:</b> %{y:,.0f}<extra></extra>'
            ))
            
            # Forecast
            forecast_future_orders = forecast_orders.tail(30)
            fig_orders.add_trace(go.Scatter(
                x=forecast_future_orders['ds'],
                y=forecast_future_orders['yhat'],
                name='Forecast',
                line=dict(color='#f6d365', width=2, dash='dash'),
                hovertemplate='<b>Date:</b> %{x}<br><b>Forecast:</b> %{y:,.0f}<extra></extra>'
            ))
            
            fig_orders.update_layout(
                title='30-Day Orders Forecast',
                xaxis_title='Date',
                yaxis_title='Orders',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_orders, use_container_width=True)
        
        # Forecast insights
        st.markdown("#### 📊 Forecast Insights")
        
        col3, col4 = st.columns(2)
        
        with col3:
            current_avg_revenue = revenue_data.tail(30)['y'].mean()
            forecast_avg_revenue = forecast_future['yhat'].mean()
            revenue_change = ((forecast_avg_revenue - current_avg_revenue) / current_avg_revenue) * 100
            
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">Revenue Forecast</div>
                <div>Expected 30-day average: <strong>${forecast_avg_revenue:,.0f}</strong></div>
                <div>Change from current: <strong style="color: {'green' if revenue_change > 0 else 'red'}">
                    {'↑' if revenue_change > 0 else '↓'}{abs(revenue_change):.1f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            current_avg_orders = orders_data.tail(30)['y'].mean()
            forecast_avg_orders = forecast_future_orders['yhat'].mean()
            orders_change = ((forecast_avg_orders - current_avg_orders) / current_avg_orders) * 100
            
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">Orders Forecast</div>
                <div>Expected 30-day average: <strong>{forecast_avg_orders:,.0f}</strong></div>
                <div>Change from current: <strong style="color: {'green' if orders_change > 0 else 'red'}">
                    {'↑' if orders_change > 0 else '↓'}{abs(orders_change):.1f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error in forecasting: {str(e)}")

def generate_insights(marketing_data, business_data):
    """Generate automated insights and recommendations"""
    insights = []
    
    try:
        if marketing_data.empty:
            insights.append("No marketing data available for analysis.")
            return insights
        
        # Channel performance insights
        channel_performance = marketing_data.groupby('channel').agg({
            'roas': 'mean',
            'spend': 'sum',
            'attributed_revenue': 'sum'
        }).reset_index()
        
        if not channel_performance.empty:
            best_roas_channel = channel_performance.loc[channel_performance['roas'].idxmax()]
            worst_roas_channel = channel_performance.loc[channel_performance['roas'].idxmin()]
            
            insights.append(f"🎯 **{best_roas_channel['channel']}** has the highest ROAS at **{best_roas_channel['roas']:.2f}**.")
            insights.append(f"⚠ **{worst_roas_channel['channel']}** has the lowest ROAS at **{worst_roas_channel['roas']:.2f}**.")
            
            # Check if any channel has ROAS below 2.0 (considered poor)
            poor_performers = channel_performance[channel_performance['roas'] < 2.0]
            if not poor_performers.empty:
                for _, row in poor_performers.iterrows():
                    insights.append(f"🚨 **{row['channel']}** has a low ROAS of {row['roas']:.2f}. Consider optimizing campaigns or reallocating budget.")
        
        # Campaign performance insights
        campaign_performance = marketing_data.groupby(['campaign', 'channel']).agg({
            'roas': 'mean',
            'spend': 'sum'
        }).reset_index()
        
        if not campaign_performance.empty:
            top_campaign = campaign_performance.loc[campaign_performance['roas'].idxmax()]
            insights.append(f"🏆 **{top_campaign['campaign']}** on **{top_campaign['channel']}** is your best performing campaign with ROAS of **{top_campaign['roas']:.2f}**.")
        
        # Time-based insights
        marketing_data['month'] = marketing_data['date'].dt.to_period('M')
        monthly_performance = marketing_data.groupby('month').agg({
            'roas': 'mean',
            'spend': 'sum',
            'attributed_revenue': 'sum'
        }).reset_index()
        
        if len(monthly_performance) > 1:
            latest_month = monthly_performance.iloc[-1]
            previous_month = monthly_performance.iloc[-2]
            
            roas_change = ((latest_month['roas'] - previous_month['roas']) / previous_month['roas']) * 100
            if abs(roas_change) > 5:
                direction = "improved" if roas_change > 0 else "declined"
                insights.append(f"📈 ROAS has **{direction}** by **{abs(roas_change):.1f}%** compared to last month.")
        
        # Business metrics insights
        if not business_data.empty:
            # Calculate average order value
            business_data['aov'] = business_data['total_revenue'] / business_data['orders']
            recent_aov = business_data.tail(30)['aov'].mean()
            
            insights.append(f"💰 Current Average Order Value (AOV): **${recent_aov:.2f}**.")
            
            # Check customer acquisition trends
            recent_customers = business_data.tail(30)['new_customers'].sum()
            previous_customers = business_data.iloc[-60:-30]['new_customers'].sum()
            
            customer_change = ((recent_customers - previous_customers) / previous_customers) * 100 if previous_customers > 0 else 0
            if abs(customer_change) > 10:
                trend = "increasing" if customer_change > 0 else "decreasing"
                insights.append(f"👥 New customers are **{trend}** by **{abs(customer_change):.1f}%** compared to the previous period.")
        
        # Budget allocation insights
        total_spend = channel_performance['spend'].sum()
        for _, row in channel_performance.iterrows():
            spend_percentage = (row['spend'] / total_spend) * 100
            if spend_percentage > 50:
                insights.append(f"⚖ **{row['channel']}** accounts for **{spend_percentage:.1f}%** of total spend. Consider diversifying your budget across channels.")
        
        # Add general recommendations
        insights.append("💡 **Recommendation**: Test different ad creatives and messaging to improve CTR.")
        insights.append("💡 **Recommendation**: Consider increasing budget for top-performing campaigns.")
        insights.append("💡 **Recommendation**: Implement retargeting campaigns for users who didn't convert.")
        
    except Exception as e:
        insights.append(f"Error generating insights: {str(e)}")
    
    return insights

def create_geographical_tab(marketing_data):
    """Create geographical analysis tab"""
    st.subheader("🌎 Geographical Performance")
    
    if marketing_data.empty:
        st.warning("No data available for geographical analysis.")
        return
    
    try:
        # State performance
        state_performance = marketing_data.groupby('state').agg({
            'spend': 'sum',
            'attributed_revenue': 'sum',
            'impressions': 'sum',
            'clicks': 'sum'
        }).reset_index()
        
        state_performance['roas'] = state_performance['attributed_revenue'] / state_performance['spend']
        state_performance['ctr'] = (state_performance['clicks'] / state_performance['impressions']) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ROAS by state
            fig_roas = px.choropleth(
                state_performance,
                locations='state',
                locationmode='USA-states',
                color='roas',
                scope='usa',
                title='ROAS by State',
                color_continuous_scale='Viridis',
                hover_data=['state', 'roas', 'spend', 'attributed_revenue']
            )
            st.plotly_chart(fig_roas, use_container_width=True)
        
        with col2:
            # Spend by state
            fig_spend = px.choropleth(
                state_performance,
                locations='state',
                locationmode='USA-states',
                color='spend',
                scope='usa',
                title='Spend by State',
                color_continuous_scale='Blues',
                hover_data=['state', 'spend', 'attributed_revenue', 'roas']
            )
            st.plotly_chart(fig_spend, use_container_width=True)
        
        # Top performing states table
        st.markdown("### Top Performing States")
        display_state_df = state_performance.nlargest(10, 'roas')[['state', 'spend', 'attributed_revenue', 'roas', 'ctr']].copy()
        
        # Format for display
        for col in ['spend', 'attributed_revenue']:
            display_state_df[col] = display_state_df[col].apply(lambda x: f"${x:,.0f}")
        for col in ['roas', 'ctr']:
            display_state_df[col] = display_state_df[col].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(display_state_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error in geographical analysis: {str(e)}")

def main():
    """Main application function"""
    # Load CSS
    load_css()
    
    # Load data
    marketing_data, business_data = load_data()
    
    # Header
    st.markdown("""
    <div class="dashboard-header">
        <h1 class="dashboard-title">Marketing Intelligence Dashboard</h1>
        <p class="dashboard-subtitle">Analyze performance, optimize spend, and drive growth</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar filters
    st.sidebar.title("Filters")
    
    # Date range filter
    min_date = marketing_data['date'].min() if not marketing_data.empty else datetime(2023, 1, 1)
    max_date = marketing_data['date'].max() if not marketing_data.empty else datetime(2024, 12, 31)
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Channel filter
    channels = st.sidebar.multiselect(
        "Channels",
        options=marketing_data['channel'].unique() if not marketing_data.empty else [],
        default=marketing_data['channel'].unique() if not marketing_data.empty else []
    )
    
    # Tactic filter
    tactics = st.sidebar.multiselect(
        "Tactics",
        options=marketing_data['tactic'].unique() if not marketing_data.empty else [],
        default=marketing_data['tactic'].unique() if not marketing_data.empty else []
    )
    
    # State filter
    states = st.sidebar.multiselect(
        "States",
        options=marketing_data['state'].unique() if not marketing_data.empty else [],
        default=marketing_data['state'].unique() if not marketing_data.empty else []
    )
    
    # Apply filters
    filtered_marketing_data = apply_filters(marketing_data, channels, tactics, states, date_range)
    
    # Calculate summary metrics
    if not filtered_marketing_data.empty:
        total_spend = filtered_marketing_data['spend'].sum()
        total_revenue = filtered_marketing_data['attributed_revenue'].sum()
        total_impressions = filtered_marketing_data['impressions'].sum()
        total_clicks = filtered_marketing_data['clicks'].sum()
        
        roas = total_revenue / total_spend if total_spend > 0 else 0
        ctr = (total_clicks / total_impressions) * 100 if total_impressions > 0 else 0
        cpc = total_spend / total_clicks if total_clicks > 0 else 0
    else:
        total_spend = total_revenue = total_impressions = total_clicks = 0
        roas = ctr = cpc = 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card("Total Spend", f"${total_spend:,.0f}", "💰")
    
    with col2:
        create_metric_card("Attributed Revenue", f"${total_revenue:,.0f}", "💸")
    
    with col3:
        create_metric_card("ROAS", f"{roas:.2f}", "📊")
    
    with col4:
        create_metric_card("CTR", f"{ctr:.2f}%", "👆")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Trends", 
        "📊 Channels", 
        "🎯 Campaigns", 
        "💼 Business", 
        "🌎 Geographical",
        "🔮 Forecasting"
    ])
    
    with tab1:
        create_trends_tab(filtered_marketing_data, business_data)
    
    with tab2:
        create_channels_tab(filtered_marketing_data)
    
    with tab3:
        create_campaigns_tab(filtered_marketing_data)
    
    with tab4:
        create_business_tab(business_data)
    
    with tab5:
        create_geographical_tab(filtered_marketing_data)
    
    with tab6:
        create_forecasting_tab(filtered_marketing_data, business_data)
    
    # Generate and display insights
    st.markdown("---")
    st.subheader("💡 Insights & Recommendations")
    
    insights = generate_insights(filtered_marketing_data, business_data)
    
    if insights:
        st.markdown("<ul class='insight-list'>", unsafe_allow_html=True)
        for insight in insights:
            st.markdown(f"<li>{insight}</li>", unsafe_allow_html=True)
        st.markdown("</ul>", unsafe_allow_html=True)
    else:
        st.info("No insights available with current filters.")

if __name__ == "__main__":
    main()