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
    Welcome to the Milk Production Analysis Dashboard!  
    Explore monthly milk production trends with moving averages, custom forecasts, and more.  
    Start by uploading your data.
    """
)

# 2.0 Data Upload Section
st.header("📁 Upload Milk Production Data")
uploaded_file = st.file_uploader(
    "Choose a CSV file", type="csv", help="Make sure your data contains monthly milk production columns."
)

if uploaded_file:
    # Load and preprocess the data
    data = pd.read_csv(uploaded_file, parse_dates=True)
    data.rename(columns={"Monthly milk production (pounds per cow)": "Production"}, inplace=True)
    st.write("🔍 **Data Preview**")
    st.dataframe(data.head())

    # 3.0 Plot Original Data
    st.header("📈 Milk Production Trends")
    fig = px.line(data, x=data.index, y='Production', title='Monthly Milk Production')
    st.plotly_chart(fig, use_container_width=True)

    # 4.0 Calculate and Plot Moving Averages
    st.header("📊 Moving Averages")
    data['SMA_12'] = data['Production'].rolling(window=12).mean()
    data['EMA_12'] = data['Production'].ewm(span=12, adjust=False).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Production'], mode='lines', name='Production'))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_12'], mode='lines', name='12-Month SMA', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_12'], mode='lines', name='12-Month EMA', line=dict(color='green')))
    fig.update_layout(title='Milk Production with Moving Averages', xaxis_title='Month', yaxis_title='Production (Pounds)')
    st.plotly_chart(fig)

    # 5.0 Custom Exponential Moving Average
    st.header("🔮 Custom Exponential Moving Average")
    data['Custom_EMA_0.6'] = data['Production'].ewm(alpha=0.6, adjust=False).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Production'], mode='lines', name='Milk Production'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Custom_EMA_0.6'], mode='lines', name='Custom EMA (α=0.6)', line=dict(color='purple')))
    fig.update_layout(title='Custom EMA with α=0.6', xaxis_title='Month', yaxis_title='Production (Pounds)')
    st.plotly_chart(fig)

    # 6.0 Train-Test Split and Forecasting
    st.header("📅 Train-Test Split and Forecasting")
    data.index = pd.date_range(start=data.index[0], periods=len(data), freq='MS')
    train, test = train_test_split(data[['Production']], test_size=0.2, shuffle=False)

    model = SimpleExpSmoothing(train['Production']).fit(smoothing_level=0.1, optimized=False)
    forecast = model.forecast(steps=len(test))

    forecast_df = pd.DataFrame({
        'Month': test.index,
        'Actual': test['Production'].values,
        'Forecast': forecast
    }).reset_index(drop=True)

    # Plotting actual vs forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['Production'], mode='lines', name='Training Data'))
    fig.add_trace(go.Scatter(x=test.index, y=test['Production'], mode='lines', name='Actual Test Data'))
    fig.add_trace(go.Scatter(x=test.index, y=forecast, mode='lines', name='Forecast', line=dict(dash='dot')))
    fig.update_layout(title='Milk Production Forecasting', xaxis_title='Month', yaxis_title='Production (Pounds)')
    st.plotly_chart(fig)

    # 7.0 Feedback Section
    st.header("💬 Feedback")
    feedback = st.text_area("Share your thoughts or suggestions:")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")

st.markdown("---")
st.write("This interactive dashboard allows you to explore and forecast milk production data with ease.")
