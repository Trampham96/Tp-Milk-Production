# Import required libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.model_selection import train_test_split

# 1.0 Title and Introduction
st.title("🐄 Milk Production Analysis Dashboard")
st.write(
    """
    This dashboard provides an interactive view of monthly milk production data with insights on moving averages, 
    trend analysis, and simple forecasting. Upload your data to start the analysis.
    """
)

# 2.0 File Upload Section
st.header("📁 Upload Milk Production Data")
uploaded_file = st.file_uploader(
    "Choose a CSV file", type="csv", help="The file should contain monthly milk production data."
)

if uploaded_file:
    # Load and preprocess the data
    data = pd.read_csv(uploaded_file, parse_dates=True)
    data.rename(columns={"Monthly milk production (pounds per cow)": "Production"}, inplace=True)
    data['Month'] = pd.date_range(start=data.index[0], periods=len(data), freq='MS')
    st.write("🔍 **Preview of the Data**")
    st.dataframe(data.head())

    # 3.0 Plotting Production Data
    st.header("📈 Milk Production Trends")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Month'], y=data['Production'], mode='lines+markers', name='Milk Production'))
    fig.update_layout(
        title='Monthly Milk Production',
        xaxis_title='Month',
        yaxis_title='Production (Pounds)',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

    # 4.0 Moving Averages
    st.header("📊 Moving Averages")
    data['SMA_12'] = data['Production'].rolling(window=12).mean()
    data['EMA_12'] = data['Production'].ewm(span=12, adjust=False).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Month'], y=data['Production'], mode='lines', name='Milk Production'))
    fig.add_trace(go.Scatter(x=data['Month'], y=data['SMA_12'], mode='lines', name='12-Month SMA', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=data['Month'], y=data['EMA_12'], mode='lines', name='12-Month EMA', line=dict(color='green')))
    fig.update_layout(
        title='Milk Production with Simple and Exponential Moving Averages',
        xaxis_title='Month',
        yaxis_title='Production (Pounds)',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

    # 5.0 Simple Exponential Smoothing Forecasting
    st.header("🔮 Simple Exponential Smoothing Forecast")
    # Train-test split
    train, test = train_test_split(data[['Production']], test_size=0.2, shuffle=False)
    model = SimpleExpSmoothing(train['Production']).fit(smoothing_level=0.1, optimized=False)
    forecast = model.forecast(steps=len(test))

    # Combine actual and forecast for visualization
    forecast_df = pd.DataFrame({
        'Month': test.index,
        'Actual': test['Production'].values,
        'Forecast': forecast
    }).reset_index(drop=True)

    # Plotting actual vs forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Month'], y=data['Production'], mode='lines', name='Actual Production'))
    fig.add_trace(go.Scatter(x=forecast_df['Month'], y=forecast_df['Forecast'], mode='lines', name='Forecast', line=dict(dash='dot')))
    fig.update_layout(
        title='Milk Production Forecasting',
        xaxis_title='Month',
        yaxis_title='Production (Pounds)',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

    # 6.0 Feedback Section
    st.header("💬 Feedback")
    feedback = st.text_area("Share your thoughts about the dashboard or suggestions for improvement.")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")

st.markdown("---")
st.write("This dashboard provides an interactive way to explore and analyze milk production data with trends, moving averages, and forecasting.")
