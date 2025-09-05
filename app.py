import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px

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
        padding: 2rem;
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
     [data-testid="stSidebarHeader"] {
        display: none;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-baseweb="select"] {
        background-color: #800000 !important;
        border-radius: 8px;
    }
    [data-testid="stMetricValue"] {
        color: #ff4b4b;
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

# --- 3. SIDEBAR FOR FILTERING CONTROLS & NAVIGATION ---

st.sidebar.image("data/adobo_w.png")
page = st.sidebar.radio("Go to", ["Dashboard Overview", "Transaction Analysis", "Customer Spending Behavior"])

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
st.markdown("---")

st.title("📊 Dashboard")
st.warning("This report provides an in-depth analysis of the decline in customer spending at Adobo Bank in 2021 and presents a strategic solution for revenue recovery and customer retention.")

st.write(" ")


# ---------- INTRODUCTION -----------


def main():

    st.set_page_config(layout="centered", page_title="Adobo Bank Analysis")


    st.header("Why is Spending Down? 📉")
    st.write(" ")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Total Customer Spending Decline (2021 vs 2020)", value="-8.8%", delta_color="inverse")
            st.caption("A decline representing a loss of nearly $3 million.")
        with col2:
            st.metric(label="Biggest Financial Loss From", value="Baby Boomers", delta_color="inverse")
            st.caption("This group spent over $128,600 less.")

    st.markdown("In 2021, Adobo Bank experienced a widespread spending decline that affected every age group.")

    st.divider()

    with st.expander("Transaction vs. Average Spend Breakdown", expanded=False):
        st.subheader("Transaction vs. Average Spend")
        st.markdown("Interestingly, while the overall **number of transactions fell by 13.6%**, the **average amount spent per transaction grew by 3.48%**.")
        st.caption("This indicates a small, high-value group of customers drove the growth in average spending, highlighting their importance to the bank's revenue.")

    st.divider()

    st.header("The Solution: Customer Segmentation 👥")
    st.write(" ")

    st.markdown("""
    To address the decline in spending, the recommended strategy is **customer segmentation**. This approach involves grouping customers into specific categories to better understand their behavior and needs.

    The analysis identified three key segments:
    """)

    st.markdown("""
    - **Active and High-Value Spenders**: The bank's most valuable customers.
    - **Engaged Mid-Tier Spenders**: A solid group with moderate spending habits.
    - **At-Risk Spenders**: Customers who are decreasing their spending or becoming less active.
    """)

    st.divider()

    st.header("How Segmentation Can Help 💡")
    with st.container():
        col3, col4, col5 = st.columns(3)
        with col3:
            st.subheader("1. Proactively Identify At-Risk Customers")
            st.markdown("The bank can reach out to customers who are beginning to spend less, potentially preventing further churn.")
        with col4:
            st.subheader("2. Boost Customer Retention & Revenue")
            st.markdown("Targeted efforts can help retain existing customers and recover lost sales.")
        with col5:
            st.subheader("3. Tailor Marketing Campaigns")
            st.markdown("Different offers can be sent to each group. For example, the bank could offer **loyalty perks to high-value Baby Boomers** and send **digital promotions to re-engage Gen X and Silent Generation customers**.")

if __name__ == "__main__":
    main()





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


# --- 6. PAGE FUNCTIONS ---
def show_dashboard_overview():
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
        st.markdown(f'<div class="card"><h4>Avg Value</h4><p>${avg_value:,.2f}</p></div>', unsafe_allow_html=True)
    with col4:
        unique_users = filtered_df['cc_num'].nunique()
        st.markdown(f'<div class="card"><h4>Unique Users</h4><p>{unique_users:,}</p></div>', unsafe_allow_html=True)
    with col5:
        most_common_category = filtered_df['category'].mode()[0] if not filtered_df['category'].empty else "N/A"
        st.markdown(f'<div class="card"><h4>Top Category</h4><p>{most_common_category}</p></div>', unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("Raw Dataset")
    with st.expander("Click to view the raw data"):
        st.dataframe(filtered_df)
    st.markdown("---")


def show_transaction_analysis():
    st.header("Transaction Analysis")
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

    st.markdown("---")

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

    st.markdown("---")

    st.subheader("Transactions by Urbanization Level (2020 vs 2021)")
    df_2020_2021 = df[df['trans_year'].isin([2020, 2021])].copy()
    transactions_per_urbanization = df_2020_2021.groupby(['classification', 'trans_year']).size().reset_index(
        name='transaction_count')

    transactions_pivot = transactions_per_urbanization.pivot(
        index='classification',
        columns='trans_year',
        values='transaction_count'
    ).fillna(0)

    transactions_normalized = transactions_pivot.copy()
    for year in transactions_pivot.columns:
        total = transactions_pivot[year].sum()
        if total > 0:
            transactions_normalized[year] = transactions_normalized[year] / total

    gradient_colors = sns.color_palette("Reds_r", n_colors=len(transactions_normalized))

    fig, ax = plt.subplots(figsize=(8, 5))
    transactions_normalized.T.plot(kind='bar', stacked=True, ax=ax, color=gradient_colors)
    ax.set_title('Proportion of Transactions per Urbanization Level (2020 vs 2021)', color='black')
    ax.set_xlabel('Year', color='black')
    ax.set_ylabel('Proportion of Transactions', color='black')
    ax.set_xticklabels(transactions_normalized.columns, rotation=0, color='black')
    ax.legend(title='Urbanization Level', facecolor='white', edgecolor='white')
    ax.tick_params(axis='y', colors='black')
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("---")


def show_customer_spending():
    st.header("Customer Spending Behavior")

    st.subheader("Distribution of Mean Monetary Value per Account (2020 vs 2021)")
    df_acct_mean_overall = df.groupby(['acct_num', 'trans_year'])['amt'].mean().reset_index(name='mean_amt')
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x='trans_year', y='mean_amt', data=df_acct_mean_overall, ax=ax, color='red')
    ax.set_title('Distribution of Mean Monetary Value per Account', color='black')
    ax.set_xlabel('Transaction Year', color='black')
    ax.set_ylabel('Mean Monetary Value per Account', color='black')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Percentage Change in Mean Transactions and Monetary Value (2020 vs 2021)")
    df_acct_summary = df.groupby(['acct_num', 'trans_year']).agg(
        oa_mean_trans=('trans_num', 'count'),
        oa_mean_amt=('amt', 'mean')
    ).reset_index()

    percent_change_trans = ((df_acct_summary.loc[df_acct_summary['trans_year'] == 2021, 'oa_mean_trans'].mean() -
                             df_acct_summary.loc[df_acct_summary['trans_year'] == 2020, 'oa_mean_trans'].mean()) /
                            df_acct_summary.loc[df_acct_summary['trans_year'] == 2020, 'oa_mean_trans'].mean()) * 100

    percent_change_amt = ((df_acct_summary.loc[df_acct_summary['trans_year'] == 2021, 'oa_mean_amt'].mean() -
                           df_acct_summary.loc[df_acct_summary['trans_year'] == 2020, 'oa_mean_amt'].mean()) /
                          df_acct_summary.loc[df_acct_summary['trans_year'] == 2020, 'oa_mean_amt'].mean()) * 100

    plot_df = pd.DataFrame(
        {'Change': ['Avg Transactions', 'Avg Monetary Value'], 'Value': [percent_change_trans, percent_change_amt]})

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x='Change', y='Value', data=plot_df, ax=ax, palette='rocket')

    ax.bar_label(ax.containers[0], fmt='%.2f%%')
    ax.set_title('Percentage Change in Average Transactions and Monetary Value', color='black')
    ax.set_ylabel('Percentage Change (%)', color='black')
    ax.set_xlabel('')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(bottom=-30)
    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Top Spenders: Percentage Change in Average Transaction Amount by Category (2020 vs 2021)")

    df_acct_top_category = df.groupby(['acct_num', 'category', 'trans_year'])['amt'].mean().reset_index(name='mean_amt')
    df_acct_top_pivot = df_acct_top_category.pivot_table(
        index=['acct_num', 'category'],
        columns='trans_year',
        values='mean_amt'
    ).reset_index()

    df_acct_top_pivot['perc_amt_diff'] = (df_acct_top_pivot[2021] - df_acct_top_pivot[2020]) / df_acct_top_pivot[2020] * 100
    df_top_spenders = df_acct_top_pivot.sort_values(by='perc_amt_diff', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x='category',
        y='perc_amt_diff',
        data=df_top_spenders,
        ax=ax,
        palette='rocket'
    )
    ax.set_title('Top Spenders - Percentage Change in Average Amount Spent Per Transaction', color='black')
    ax.set_ylabel('Percent Change (%)', color='black')
    ax.set_xlabel('Category', color='black')
    ax.tick_params(axis='x', rotation=90, colors='black')
    ax.tick_params(axis='y', colors='black')
    st.pyplot(fig)


# --- 7. NAVIGATION LOGIC ---
if page == "Dashboard Overview":
    show_dashboard_overview()
elif page == "Transaction Analysis":
    show_transaction_analysis()
elif page == "Customer Spending Behavior":
    show_customer_spending()

# --- 8. FOOTER ---
st.markdown('<div class="footer">Adobo Bank Dashboard | Built with Streamlit</div>', unsafe_allow_html=True)