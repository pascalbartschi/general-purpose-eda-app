import streamlit as st
from pathlib import Path
import pandas as pd

# Import tabs
from tabs import data_loading, eda, modeling

# Configure the page
st.set_page_config(
    page_title="General Purpose EDA App",
    page_icon="📊",
    layout="wide",
)

# Main app header
st.title("General Purpose EDA App")
st.markdown("**A complete solution for data exploration, analysis, and modeling**")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Data Loading and Filtering", "EDA", "Modeling"])

# Render appropriate content in each tab
with tab1:
    data_loading.app()

with tab2:
    eda.app()

with tab3:
    modeling.app()

# Footer
st.markdown("---")
st.caption("General Purpose EDA App © 2025")
