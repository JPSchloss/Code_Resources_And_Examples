import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# Sample Data Generation
np.random.seed(0)
n = 100
dates = pd.date_range(datetime.today(), periods=n).tolist()
categories = np.random.choice(['Category A', 'Category B', 'Category C'], n)
values = np.random.rand(n) * 100
values2 = np.random.rand(n) * 100

data = pd.DataFrame({'Date': dates, 'Category': categories, 'Value': values, 'Value2': values2})

# Streamlit App
st.title('Streamlit App with Plotly Visualizations')

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a Chart", ('Bar Chart', 'Line Chart', 'Scatter Plot', 'Violin Plot'))

# Bar Chart
if page == 'Bar Chart':
    bar_chart = px.bar(data, x='Category', y='Value', color='Category')
    st.plotly_chart(bar_chart)

# Line Chart
elif page == 'Line Chart':
    line_chart = px.line(data, x='Date', y='Value2', color='Category')
    st.plotly_chart(line_chart)

# Scatter Plot
elif page == 'Scatter Plot':
    scatter_plot = px.scatter(data, x='Value', y='Value2', color='Category')
    st.plotly_chart(scatter_plot)

# Violin Plot
elif page == 'Violin Plot':
    violin_plot = px.violin(data, y='Value', color='Category', box=True, points="all")
    st.plotly_chart(violin_plot)
