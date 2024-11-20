import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Title and Introduction
st.title("🐄 Milk Production Analysis Dashboard")
st.write(
    """
    Explore and analyze milk production data with moving averages and forecasts using interactive plots.
    """
)

# Data Upload Section
uploaded_file = st.file_uploader(
    "Upload your CSV file for Milk Production Data", type="csv"
)

if uploaded_file:
    # Load and preprocess data
    data = pd.read_csv(uploaded_file)
    data.rename(columns={"Monthly milk production (pounds per cow)": "Production"}, inplace=True)
    
    st.write("**Data Preview**")
    st.dataframe(data.head())

    # Adding moving averages
    data['SMA_12'] = data['Production'].rolling(window=12).mean()
    data['EMA_12'] = data['Production'].ewm(span=12, adjust=False).mean()
    data['Custom_EMA_0.6'] = data['Production'].ewm(alpha=0.6, adjust=False).mean()

    # Plot Original Production Data
    st.header("📈 Milk Production Trends")
    fig = px.line(data, x=data.index, y='Production', title='Monthly Milk Production', labels={'index': 'Month', 'Production': 'Milk Production (pounds)'})
    st.plotly_chart(fig)

    # Plot with Simple and Exponential Moving Averages
    st.header("📊 Moving Averages")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Production'], mode='lines', name='Milk Production', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_12'], mode='lines', name='12-Month SMA', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_12'], mode='lines', name='12-Month EMA', line=dict(color='green')))
    fig.update_layout(title='Milk Production with Moving Averages', xaxis_title='Month', yaxis_title='Milk Production (pounds)')
    st.plotly_chart(fig)

    # Custom EMA with alpha=0.6
    st.header("🔮 Custom Exponential Moving Average (α=0.6)")
