import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("Superstore.csv")
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
monthly_sales = df.groupby(['Year','Month'])['Sales'].sum().reset_index()

# Title
st.title("📊 Sales Forecasting Dashboard")

# KPIs
st.metric("Total Sales", round(df['Sales'].sum(),2))
st.metric("Total Profit", round(df['Profit'].sum(),2))

# Visualization
fig = px.line(monthly_sales, x='Month', y='Sales', color='Year', title="Monthly Sales Trend")
st.plotly_chart(fig)

# Forecast
X = monthly_sales[['Month']]
y = monthly_sales['Sales']
model = LinearRegression()
model.fit(X, y)
future_months = pd.DataFrame({'Month':[5,6,7,8,9,10,11,12]})
future_sales = model.predict(future_months)

forecast_df = pd.DataFrame({"Month":future_months['Month'], "Forecasted Sales":future_sales})
st.line_chart(forecast_df.set_index("Month"))
