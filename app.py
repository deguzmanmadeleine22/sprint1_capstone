import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px  # Added for more interactive charts

# --- 1. APP CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="Adobo Bank Data Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and styling
st.markdown("""
    <style>
    .stMainBlockContainer {
        max-width: 1100px !important;
        margin: auto;
    }

    
    .stApp {
        background-color: #001f3f;
        color: white;
    }
    h1, h2, h3, h4, h5, p, label, .stMarkdown, .stSelectbox label {
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
    
    </style>
""", unsafe_allow_html=True)


# --- 2. DATA LOADING & INITIAL PROCESSING ---
@st.cache_data
def load_data():
    """
    Loads the credit card transaction data, converts 'trans_datetime'
    and adds 'trans_year', 'trans_month', and 'classification' columns.
    """
    df = pd.read_csv('data/cc_clean.csv')
    df['trans_datetime'] = pd.to_datetime(df['trans_datetime'])
    df['trans_year'] = df['trans_datetime'].dt.year
    df['trans_month'] = df['trans_datetime'].dt.month_name()  # Using month names

    # Define and apply urbanization classification
    bins = [0, 50000, 200000, np.inf]
    labels = ["Rural", "Suburban", "Urban"]
    df['classification'] = pd.cut(df["city_pop"], bins=bins, labels=labels, right=False)

    return df


df = load_data()

# --- 3. SIDEBAR FOR FILTERING CONTROLS ---
st.sidebar.header("Filter Data")

# Year filter (select one)
selected_year = st.sidebar.radio("Select Year", sorted(df['trans_year'].unique()))

# Month filter (multiselect)
month_order = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

df['trans_month'] = pd.Categorical(df['trans_datetime'].dt.month_name(), categories=month_order, ordered=True)
all_months = df['trans_month'].unique().tolist()
selected_months = st.sidebar.multiselect("Select Months", options=all_months, default=all_months)

# Urbanization filter (multiselect)
all_classifications = df['classification'].unique().tolist()
selected_classifications = st.sidebar.multiselect(
    "Select Urbanization Level",
    options=all_classifications,
    default=all_classifications
)

# --- 4. MAIN DASHBOARD CONTENT ---
st.title("📊 Adobo Bank Data Dashboard")
st.markdown("A data science dashboard for analyzing credit card transactions and finding areas for improvement.")
st.warning("⚠️ **Disclaimer:** This project uses synthetic data for demonstration. It is not an actual bank dataset.")
st.markdown("---")

# --- 5. FILTERING LOGIC ---
if not selected_months or not selected_classifications:
    st.info('Please select at least one month and one urbanization level.')
    st.stop()

# Apply all filters to create the final filtered DataFrame
filtered_df = df[
    (df['trans_year'] == selected_year) &
    (df['trans_month'].isin(selected_months)) &
    (df['classification'].isin(selected_classifications))
    ].copy()

# --- 6. KEY PERFORMANCE INDICATORS (KPIs) ---
st.header("Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_transactions = len(filtered_df)
    st.markdown(f'<div class="card"><h4>Transactions</h4><p>{total_transactions:,}</p></div>', unsafe_allow_html=True)
with col2:
    total_value = filtered_df['amt'].sum()
    st.markdown(f'<div class="card"><h4>Total Value</h4><p>${total_value:,.2f}</p></div>', unsafe_allow_html=True)
with col3:
    avg_value = filtered_df['amt'].mean()
    st.markdown(f'<div class="card"><h4>Average Value</h4><p>${avg_value:,.2f}</p></div>', unsafe_allow_html=True)
with col4:
    unique_users = filtered_df['cc_num'].nunique()
    st.markdown(f'<div class="card"><h4>Unique Users</h4><p>{unique_users:,}</p></div>', unsafe_allow_html=True)
with col5:
    most_common_category = filtered_df['category'].mode()[0] if not filtered_df['category'].empty else "N/A"
    st.markdown(f'<div class="card"><h4>Top Category</h4><p>{most_common_category}</p></div>', unsafe_allow_html=True)
st.markdown("---")

# --- 7. RAW DATA DATABASE ---
st.subheader("Raw Dataset")
with st.expander("Click to view the raw data"):
    st.dataframe(filtered_df)
st.markdown("---")

# --- 8. VISUALIZATIONS ---
st.header("Transaction Analysis")

# --- Transaction Counts over Time (Line Chart) ---
st.subheader(f"Monthly Transactions in {selected_year}")
monthly_transactions = filtered_df.groupby('trans_month').size().reindex(month_order, fill_value=0)
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x=monthly_transactions.index, y=monthly_transactions.values, marker='o', color='#800000', ax=ax)
ax.set_title(f"Monthly Transactions ({selected_year})", color='black')
ax.set_xlabel("Month", color='black')
ax.set_ylabel("Number of Transactions", color='black')
ax.tick_params(axis='x', rotation=45, colors='black')
ax.tick_params(axis='y', colors='black')
st.pyplot(fig)

st.markdown("<br>", unsafe_allow_html=True)

# --- Top 10 Transaction Categories ---
st.subheader(f"Top 10 Categories in {selected_year}")
category_counts = filtered_df['category'].value_counts().head(10).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=category_counts.values, y=category_counts.index, ax=ax, palette='rocket')
ax.set_title("Top 10 Transaction Categories", color='black')
ax.set_xlabel("Number of Transactions", color='black')
ax.set_ylabel("Category", color='black')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')
st.pyplot(fig)

st.markdown("<br>", unsafe_allow_html=True)

# --- Transactions by Urbanization Level (Normalized Stacked Bar Chart) ---
st.subheader("Transactions by Urbanization Level (2020 vs 2021)")
# This chart should always compare both years, regardless of the filter
df_2020_2021 = df[df['trans_year'].isin([2020, 2021])].copy()
transactions_per_urbanization = df_2020_2021.groupby(['classification', 'trans_year']).size().reset_index(
    name='transaction_count')

# Pivot the data to prepare for a stacked chart
transactions_pivot = transactions_per_urbanization.pivot(
    index='classification',
    columns='trans_year',
    values='transaction_count'
).fillna(0)

# Normalize the data to show proportions (0 to 1)
transactions_normalized = transactions_pivot.copy()
for year in transactions_pivot.columns:
    total = transactions_pivot[year].sum()
    if total > 0:
        transactions_normalized[year] = transactions_normalized[year] / total

# Get a darker red gradient palette
gradient_colors = sns.color_palette("Reds_r", n_colors=len(transactions_normalized))

# Create and display the stacked bar chart using the new gradient
fig, ax = plt.subplots(figsize=(8, 5))
transactions_normalized.T.plot(kind='bar', stacked=True, ax=ax, color=gradient_colors)
ax.set_title('Proportion of Transactions per Urbanization Level (2020 vs 2021)', color='black')
ax.set_xlabel('Year', color='black')
ax.set_ylabel('Proportion of Transactions', color='black')
ax.set_xticklabels(transactions_normalized.columns, rotation=0, color='black')
ax.legend(title='Urbanization Level')
ax.tick_params(axis='y', colors='black')
plt.tight_layout()
st.pyplot(fig)
st.markdown("---")

# --- 9. FOOTER ---
st.markdown('<div class="footer">Adobo Bank Dashboard | Built with Streamlit</div>', unsafe_allow_html=True)
