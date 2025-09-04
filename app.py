import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px


st.title("Sprint 1 Capstone")
st.subheader("A project made by Group 1")
st.markdown("A data science dashboard for analyzing credit card transactions and predicting fraudulent activity.")
st.warning("⚠️ **Disclaimer:** This project uses synthetic data for demonstration. It is not an actual bank dataset.")


# --- 1. APP CONFIGURATION & DATA LOADING ---
st.set_page_config(
    page_title="Adobo Bank Data Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLING ---
st.markdown("""
    <style>
    .stApp {
        background-color: #001f3f;
        color: black;
    }
    h1, h2, h3, h4, h5, p, label {
        color: white !important;
    }
    .card {
        background: rgba(255,255,255,0.05);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        font-size: 0.9rem;
        opacity: 0.7;
    }
    [data-testid="stSidebar"] {
        background-color: #800000 !important;
        color: white !important;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-baseweb="select"] {
        background-color: #800000 !important;
        border-radius: 8px;
    }
    [data-testid="baseButton-secondary"] {
        background-color: #ffce1b !important;
        color: white !important;
        border: none;
        border-radius: 8px;
    }
    [data-testid="baseButton-secondary"]:hover {
        background-color: #ff6666 !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """
    Loads the credit card transaction data from the CSV file,
    converts 'trans_datetime' to a datetime object, and
    extracts the year and month for filtering.
    """
    df = pd.read_csv('/Users/madeleinedg/Documents/Eskwelabs/Ongoing Projects/cc_clean.csv')
    df['trans_datetime'] = pd.to_datetime(df['trans_datetime'])
    df['trans_year'] = df['trans_datetime'].dt.year
    df['trans_month'] = df['trans_datetime'].dt.month
    return df

df = load_data()


# ------------ #









# -------------- #

# --- 2. SIDEBAR FOR FILTERING AND CONTROLS ---
st.sidebar.header("Please Filter Here:")
selected_year = st.sidebar.radio("Select Year", sorted(df['trans_year'].unique()))

# st.sidebar.markdown("---")
# st.sidebar.subheader("Menu")
# st.sidebar.button("Home")
# st.sidebar.button("Progress")

# --- MAIN PAGE ---
st.title("📊 Adobo Bank Credit Card Database")

# --- KPI SECTION ---
total_investment = df['amt'].sum()
most_frequent = df['amt'].mode()[0]
investment_average = df['amt'].mean()
investment_margin = df['amt'].max() - df['amt'].min()
# The 'Ratings' metric is not available in the provided data, so we will use a placeholder.
ratings = "N/A"

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f'<div class="card"><h4>Total Transactions Value</h4><p>${total_investment:,.2f}</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="card"><h4>Most Frequent Amount</h4><p>${most_frequent:,.2f}</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="card"><h4>Average Transaction</h4><p>${investment_average:,.2f}</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="card"><h4>Spending Range</h4><p>${investment_margin:,.2f}</p></div>', unsafe_allow_html=True)
with col5:
    st.markdown(f'<div class="card"><h4>Ratings</h4><p>{ratings}</p></div>', unsafe_allow_html=True)

# --- FILTERED DATA ---
filtered_df = df[df['trans_year'] == selected_year]

# --- 3. DATA OVERVIEW AND DEMOGRAPHICS ---
st.header("1. Data Overview")
st.subheader(f"Raw Data for {selected_year}")
st.dataframe(filtered_df.head())

# st.subheader("Key Statistics")
# st.write(filtered_df.describe())



# --- 4. SPENDING ANALYSIS AND VISUALIZATION ---
st.header("2. Spending and Transaction Analysis")
st.subheader("Top 10 Transaction Categories")
category_counts = filtered_df['category'].value_counts().head(10)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=category_counts.values, y=category_counts.index, ax=ax, palette='rocket')
ax.set_title(f"Top 10 Transaction Categories in {selected_year}")
ax.set_xlabel("Number of Transactions")
ax.set_ylabel("Category")
st.pyplot(fig)

# st.subheader("Transaction Amount Distribution")
# amount_distribution = filtered_df['amt']
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.histplot(amount_distribution, bins=50, kde=True, ax=ax, color='purple')
# ax.set_title("Distribution of Transaction Amounts")
# ax.set_xlabel("Amount ($)")
# st.pyplot(fig)

# --- 5. CUSTOMER SEGMENTATION ---
# st.header("3. Customer Segmentation")
# spending_threshold = st.slider("Define High Spender Threshold ($)", 0, int(filtered_df['amt'].max()), 500)
#
# high_spenders = filtered_df[filtered_df['amt'] > spending_threshold]
# low_spenders = filtered_df[filtered_df['amt'] <= spending_threshold]
#
# col1, col2 = st.columns(2)
#
# with col1:
#     st.subheader("High Spender Demographics")
#     st.write(f"Number of transactions above ${spending_threshold}: **{high_spenders.shape[0]}**")
#     if not high_spenders.empty:
#         st.write("Top 5 Cities for High Spenders:")
#         st.dataframe(high_spenders['city'].value_counts().head(5))
#
# with col2:
#     st.subheader("Low Spender Demographics")
#     st.write(f"Number of transactions at or below ${spending_threshold}: **{low_spenders.shape[0]}**")
#     if not low_spenders.empty:
#         st.write("Top 5 Cities for Low Spenders:")
#         st.dataframe(low_spenders['city'].value_counts().head(5))