import streamlit as st
import pandas as pd
import numpy as np

st.title("Test App")
st.write("If you can see this, Streamlit is working!")

# Test if pandas works
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
st.dataframe(df)

st.write("Basic packages are working!")
