import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.title("Plotly Test")

# Simple test data
df = pd.DataFrame({
    'x': [1, 2, 3, 4],
    'y': [10, 11, 12, 13]
})

# Test plotly express
fig1 = px.line(df, x='x', y='y', title='Test Line Chart')
st.plotly_chart(fig1)

# Test plotly graph objects  
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=[1, 2, 3], y=[4, 5, 6]))
st.plotly_chart(fig2)

st.success("Plotly is working!")
