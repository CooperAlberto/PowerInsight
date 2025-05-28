import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="⚡ PowerInsight - Electrical Usage Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .success-box {
        background: #d1edff;
        border-left: 4px solid #0084ff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

def validate_data(df):
    """Validate the uploaded data"""
    errors = []
    warnings = []
    
    required_columns = ['Start Datetime', 'Net Usage']
    for col in required_columns:
        if col not in df.columns:
            errors.append(f"Missing required column: '{col}'")
    
    if not errors:
        # Check for missing values
        if df['Start Datetime'].isnull().any():
            warnings.append("Some datetime values are missing")
        if df['Net Usage'].isnull().any():
            warnings.append("Some usage values are missing")
        
        # Check for reasonable usage values
        if (df['Net Usage'] < 0).any():
            warnings.append("Negative usage values detected")
        if (df['Net Usage'] > 50).any():  # Assuming residential usage
            warnings.append("Unusually high usage values detected (>50 kWh)")
    
    return errors, warnings

def clean_data(df):
    """Clean and prepare the data"""
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Convert datetime
    df['Start Datetime'] = pd.to_datetime(df['Start Datetime'], errors='coerce')
    
    # Remove rows with invalid datetime or usage
    df = df.dropna(subset=['Start Datetime', 'Net Usage'])
    
    # Ensure Net Usage is numeric
    df['Net Usage'] = pd.to_numeric(df['Net Usage'], errors='coerce')
    df = df.dropna(subset=['Net Usage'])
    
    # Extract time components
    df['Month'] = df['Start Datetime'].dt.month
    df['Day'] = df['Start Datetime'].dt.day
    df['Hour'] = df['Start Datetime'].dt.hour
    df['DayOfWeek'] = df['Start Datetime'].dt.day_name()
    df['Date'] = df['Start Datetime'].dt.date
    df['Season'] = df['Month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                   3: 'Spring', 4: 'Spring', 5: 'Spring',
                                   6: 'Summer', 7: 'Summer', 8: 'Summer',
                                   9: 'Fall', 10: 'Fall', 11: 'Fall'})
    
    return df.sort_values('Start Datetime')

def generate_insights(df, monthly_analysis):
    """Generate intelligent insights from the data"""
    insights = []
    
    # Peak usage month
    peak_month = max(monthly_analysis, key=lambda x: x['Total Usage'])
    low_month = min(monthly_analysis, key=lambda x: x['Total Usage'])
    
    insights.append(f"🔥 **Peak Usage**: {peak_month['Month']} had the highest consumption at {peak_month['Total Usage']:.1f} kWh")
    insights.append(f"🌟 **Lowest Usage**: {low_month['Month']} had the lowest consumption at {low_month['Total Usage']:.1f} kWh")
    
    # Day vs Night comparison
    total_day = sum([x['Average Daytime Usage'] for x in monthly_analysis])
    total_night = sum([x['Average Nighttime Usage'] for x in monthly_analysis])
    
    if total_day > total_night:
        insights.append(f"☀️ **Usage Pattern**: You use {((total_day/total_night - 1) * 100):.1f}% more electricity during the day")
    else:
        insights.append(f"🌙 **Usage Pattern**: You use {((total_night/total_day - 1) * 100):.1f}% more electricity at night")
    
    # Seasonal analysis
    seasonal_usage = df.groupby('Season')['Net Usage'].sum().to_dict()
    peak_season = max(seasonal_usage, key=seasonal_usage.get)
    insights.append(f"❄️🌞 **Seasonal Peak**: {peak_season} shows the highest energy consumption")
    
    # Weekly pattern
    weekend_usage = df[df['DayOfWeek'].isin(['Saturday', 'Sunday'])]['Net Usage'].mean()
    weekday_usage = df[~df['DayOfWeek'].isin(['Saturday', 'Sunday'])]['Net Usage'].mean()
    
    if weekend_usage > weekday_usage:
        insights.append(f"🏠 **Weekend Effect**: Usage is {((weekend_usage/weekday_usage - 1) * 100):.1f}% higher on weekends")
    else:
        insights.append(f"🏢 **Weekday Pattern**: Usage is {((weekday_usage/weekend_usage - 1) * 100):.1f}% higher on weekdays")
    
    return insights

def create_seasonal_chart(df):
    """Create seasonal usage comparison"""
    seasonal_data = df.groupby(['Season', 'Hour'])['Net Usage'].mean().reset_index()
    
    fig = px.line(
        seasonal_data, 
        x='Hour', 
        y='Net Usage', 
        color='Season',
        title="🌿 Seasonal Usage Patterns Throughout the Day",
        labels={'Net Usage': 'Average Usage (kWh)', 'Hour': 'Hour of Day'}
    )
    
    fig.update_layout(height=400)
    return fig

def convert_24_to_12_hour(hour_24):
    """Convert 24-hour format to 12-hour format for display"""
    if hour_24 == 0:
        return "12:00 AM"
    elif hour_24 < 12:
        return f"{hour_24}:00 AM"
    elif hour_24 == 12:
        return "12:00 PM"
    else:
        return f"{hour_24 - 12}:00 PM"

def convert_12_to_24_hour(hour_12, period):
    """Convert 12-hour format back to 24-hour format"""
    if period == "AM":
        if hour_12 == 12:
            return 0
        else:
            return hour_12
    else:  # PM
        if hour_12 == 12:
            return 12
        else:
            return hour_12 + 12

# Main App
st.markdown('<h1 class="main-header">⚡ PowerInsight</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Electrical Usage Analytics Dashboard</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your utility CSV file", 
        type=["csv"],
        help="Upload a CSV file with 'Start Datetime' and 'Net Usage' columns"
    )
    
    if uploaded_file:
        st.success("✅ File uploaded successfully!")
        
        st.header("⚙️ Analysis Settings")
        
        # Customizable day/night hours with 12-hour format
        st.subheader("Time Period Settings")
        
        # Day start time
        day_hour = st.selectbox("Day starts at:", 
                               options=list(range(1, 13)), 
                               index=5,  # Default to 6 AM
                               key="day_hour")
        day_period = st.selectbox("", options=["AM", "PM"], key="day_period")
        day_start_24 = convert_12_to_24_hour(day_hour, day_period)
        
        # Night start time  
        night_hour = st.selectbox("Night starts at:", 
                                 options=list(range(1, 13)), 
                                 index=5,  # Default to 6 PM
                                 key="night_hour")
        night_period = st.selectbox("", options=["AM", "PM"], 
                                   index=1,  # Default to PM
                                   key="night_period")
        night_start_24 = convert_12_to_24_hour(night_hour, night_period)
        
        st.info(f"Day: {convert_24_to_12_hour(day_start_24)} - {convert_24_to_12_hour(night_start_24)}")
        st.info(f"Night: {convert_24_to_12_hour(night_start_24)} - {convert_24_to_12_hour(day_start_24)}")

if uploaded_file is not None:
    try:
        # Load and validate data
        df = pd.read_csv(uploaded_file)
        
        with st.spinner("🔍 Validating data..."):
            errors, warnings = validate_data(df)
        
        if errors:
            st.error("❌ **Data Validation Errors:**")
            for error in errors:
                st.error(f"• {error}")
            st.stop()
        
        if warnings:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.warning("⚠️ **Data Quality Warnings:**")
            for warning in warnings:
                st.warning(f"• {warning}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Clean data
        with st.spinner("🧹 Cleaning and processing data..."):
            df = clean_data(df)
            
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.success(f"✅ Successfully processed {len(df):,} data points from {df['Start Datetime'].min().strftime('%Y-%m-%d')} to {df['Start Datetime'].max().strftime('%Y-%m-%d')}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data overview
        with st.expander("📊 Data Overview", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                st.metric("Date Range", f"{(df['Start Datetime'].max() - df['Start Datetime'].min()).days} days")
            with col3:
                st.metric("Total Usage", f"{df['Net Usage'].sum():.1f} kWh")
            with col4:
                st.metric("Avg Hourly Usage", f"{df['Net Usage'].mean():.2f} kWh")
            
            st.subheader("Sample Data")
            st.dataframe(df.head(10))
        
        # Monthly Analysis (Enhanced)
        months = ["January", "February", "March", "April", "May", "June", 
                 "July", "August", "September", "October", "November", "December"]
        monthly_analysis = []
        
        for month in range(1, 13):
            month_df = df[df['Month'] == month]
            
            if len(month_df) == 0:
                continue
                
            total_usage = month_df['Net Usage'].sum()
            num_days_in_month = len(month_df['Day'].unique())
            avg_24hr_usage = total_usage / num_days_in_month if num_days_in_month > 0 else 0
            
            # Use customizable day/night hours
            daytime_usage = month_df[(month_df['Hour'] >= day_start_24) & (month_df['Hour'] < night_start_24)]['Net Usage'].sum()
            nighttime_usage = total_usage - daytime_usage
            
            avg_daytime_usage = daytime_usage / num_days_in_month if num_days_in_month > 0 else 0
            avg_nighttime_usage = nighttime_usage / num_days_in_month if num_days_in_month > 0 else 0
            
            # Peak hour analysis
            hourly_avg = month_df.groupby('Hour')['Net Usage'].mean()
            peak_hour = hourly_avg.idxmax() if len(hourly_avg) > 0 else 0
            peak_usage = hourly_avg.max() if len(hourly_avg) > 0 else 0
            
            monthly_data = {
                'Month': months[month-1],
                'Total Usage': total_usage,
                'Average 24 Hour Usage': avg_24hr_usage,
                'Average Daytime Usage': avg_daytime_usage,
                'Average Nighttime Usage': avg_nighttime_usage,
                'Peak Hour': peak_hour,
                'Peak Hour Usage': peak_usage
            }
            
            monthly_analysis.append(monthly_data)
        
        # Generate insights
        insights = generate_insights(df, monthly_analysis)
        
        # Display insights
        st.header("🧠 Smart Insights")
        insight_cols = st.columns(2)
        for i, insight in enumerate(insights):
            with insight_cols[i % 2]:
                st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
        
        # Enhanced visualizations
        st.header("📈 Advanced Analytics")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Monthly Overview", "📅 Daily Patterns", "🌿 Seasonal Analysis"])
        
        with tab1:
            if monthly_analysis:
                monthly_df = pd.DataFrame(monthly_analysis)
                
                # Monthly usage bar chart
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Monthly Total Usage', 'Day vs Night Usage Comparison'),
                    specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
                )
                
                # Total usage bar chart
                fig.add_trace(
                    go.Bar(
                        x=monthly_df['Month'],
                        y=monthly_df['Total Usage'],
                        name='Total Usage',
                        marker_color='#667eea',
                        hovertemplate='<b>%{x}</b><br>Total Usage: %{y:.1f} kWh<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # Day vs Night comparison
                fig.add_trace(
                    go.Bar(
                        x=monthly_df['Month'],
                        y=monthly_df['Average Daytime Usage'],
                        name='Daytime',
                        marker_color='#ffa726',
                        hovertemplate='<b>%{x}</b><br>Daytime: %{y:.1f} kWh<extra></extra>'
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=monthly_df['Month'],
                        y=monthly_df['Average Nighttime Usage'],
                        name='Nighttime',
                        marker_color='#42a5f5',
                        hovertemplate='<b>%{x}</b><br>Nighttime: %{y:.1f} kWh<extra></extra>'
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(height=800, showlegend=True, title_text="Monthly Usage Analysis")
                fig.update_xaxes(title_text="Month", row=2, col=1)
                fig.update_yaxes(title_text="Usage (kWh)", row=1, col=1)
                fig.update_yaxes(title_text="Average Daily Usage (kWh)", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Monthly Summary (New Section)
                st.subheader("📊 Monthly Summary")
                st.dataframe(monthly_df.style.format({
                    'Total Usage': '{:.1f} kWh',
                    'Average 24 Hour Usage': '{:.2f} kWh',
                    'Average Daytime Usage': '{:.2f} kWh',
                    'Average Nighttime Usage': '{:.2f} kWh',
                    'Peak Hour Usage': '{:.2f} kWh'
                }), use_container_width=True)
                
                # Annual Summary
                st.subheader("📊 Annual Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                yearly_total = sum([x['Total Usage'] for x in monthly_analysis])
                yearly_24hr = sum([x['Average 24 Hour Usage'] for x in monthly_analysis]) / len(monthly_analysis)
                yearly_day = sum([x['Average Daytime Usage'] for x in monthly_analysis]) / len(monthly_analysis)
                yearly_night = sum([x['Average Nighttime Usage'] for x in monthly_analysis]) / len(monthly_analysis)
                
                with col1:
                    st.metric("Total Annual Usage", f"{yearly_total:.0f} kWh")
                with col2:
                    st.metric("Avg Daily Usage", f"{yearly_24hr:.1f} kWh")
                with col3:
                    st.metric("Avg Daytime Usage", f"{yearly_day:.1f} kWh")
                with col4:
                    st.metric("Avg Nighttime Usage", f"{yearly_night:.1f} kWh")
        
        with tab2:
            # Daily usage trends
            daily_usage = df.groupby('Date')['Net Usage'].sum().reset_index()
            daily_usage['Date'] = pd.to_datetime(daily_usage['Date'])
            
            fig = px.line(
                daily_usage, 
                x='Date', 
                y='Net Usage',
                title="📅 Daily Usage Trends Over Time"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Weekly patterns
            weekly_pattern = df.groupby('DayOfWeek')['Net Usage'].mean().reindex([
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ])
            
            fig = go.Figure(data=go.Bar(
                x=weekly_pattern.index,
                y=weekly_pattern.values,
                marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3', '#54a0ff']
            ))
            fig.update_layout(title="📊 Average Usage by Day of Week", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.plotly_chart(create_seasonal_chart(df), use_container_width=True)
            
            # Seasonal summary
            seasonal_summary = df.groupby('Season')['Net Usage'].agg(['sum', 'mean']).round(2)
            seasonal_summary.columns = ['Total Usage (kWh)', 'Average Usage (kWh)']
            st.subheader("Seasonal Usage Summary")
            st.dataframe(seasonal_summary)
        
        # Data export option
        st.header("📥 Export Results")
        if st.button("📊 Download Analysis Results"):
            # Create downloadable CSV
            export_df = pd.DataFrame(monthly_analysis)
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"usage_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"❌ **Error processing file**: {str(e)}")
        st.info("💡 **Tip**: Make sure your CSV file has 'Start Datetime' and 'Net Usage' columns with valid data.")

else:
    # Landing page when no file is uploaded
    st.markdown("""
    ## 🚀 Welcome to PowerInsight!
    
    Upload your electrical usage CSV file to get started with advanced analytics including:
    
    ### 📊 **Comprehensive Analysis**
    - Monthly, seasonal, and yearly usage patterns
    - Peak demand identification
    - Day vs night usage comparison with customizable time periods
    
    ### 📈 **Advanced Visualizations**
    - Seasonal trend analysis and daily usage graphs
    - Interactive monthly usage charts
    - Weekly pattern analysis
    
    ### 🧠 **Smart Insights**
    - Automated pattern recognition and recommendations
    - Comparative analysis across different time periods
    - Usage trend identification
    
    ### 📁 **File Requirements**
    Your CSV file should contain these columns:
    - `Start Datetime`: Timestamp of the reading
    - `Net Usage`: Energy usage in kWh
    
    ---
    
    **Ready to optimize your energy usage? Upload your file using the sidebar! 👈**
    """)
    
    # Sample data format
    with st.expander("📋 Sample Data Format"):
        sample_data = pd.DataFrame({
            'Start Datetime': ['2024-01-01 00:00:00', '2024-01-01 01:00:00', '2024-01-01 02:00:00'],
            'Net Usage': [1.25, 0.85, 0.65]
        })
        st.dataframe(sample_data)
