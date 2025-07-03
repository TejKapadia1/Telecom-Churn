import streamlit as st
from tabs import (
    data_visualization,
    classification,
    clustering,
    association_rules,
    regression_insights
)

st.set_page_config(page_title="Group PBL Dashboard", layout="wide")

st.title("Group PBL Dashboard")
st.sidebar.title("Navigation")

tabs = {
    "Data Visualization": data_visualization,
    "Classification": classification,
    "Clustering": clustering,
    "Association Rules": association_rules,
    "Regression Insights": regression_insights,
}

tab_selection = st.sidebar.radio("Go to:", list(tabs.keys()))
tabs[tab_selection].run()
