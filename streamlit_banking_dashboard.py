import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
from streamlit_lottie import st_lottie
from sklearn.preprocessing import MinMaxScaler

# Load Animations Function
def load_lottiefile(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# Load animations (small optimized icons only where needed)
lottie_user = load_lottiefile("lottie_files/Credit_card.json")
lottie_fraud = load_lottiefile("lottie_files/fraud.json")
lottie_loan = load_lottiefile("lottie_files/loan_approval.json")
lottie_risk = load_lottiefile("lottie_files/risk.json")

# Load Data
data = pd.read_csv("bank_data.csv")

# Streamlit Page Config
st.set_page_config(page_title="FinBank Risk & Fraud Analytics", layout="wide")

# Inject responsive CSS
st.markdown("""
    <style>
        @media (max-width: 1200px) {
            .css-1y0tads.e1f1d6gn3 {flex-direction: column;}
        }
        @media (max-width: 768px) {
            .css-1y0tads.e1f1d6gn3 {flex-direction: column;}
            .block-container {padding: 0.5rem;}
            .stSelectbox>div>div {font-size: 14px !important;}
            h2 {font-size: 24px !important;}
            h5 {font-size: 16px !important;}
        }
        body { background-color: #0f1117;}
        .stSidebar { background-color: #1c1f26; }
    </style>
""", unsafe_allow_html=True)

# Sidebar filters
st.sidebar.title("🔎 FinBank Filters")

account_types = ["All"] + list(data["Account_Type"].unique())
selected_account_type = st.sidebar.selectbox("Select Account Type", options=account_types)

loan_statuses = ["All"] + list(data["Loan_Status"].unique())
selected_loan_status = st.sidebar.selectbox("Select Loan Status", options=loan_statuses)

genders = ["All"] + list(data["Gender"].unique())
selected_gender = st.sidebar.selectbox("Select Gender", options=genders)

# Apply filters
filtered_data = data.copy()

if selected_account_type != "All":
    filtered_data = filtered_data[filtered_data["Account_Type"] == selected_account_type]

if selected_loan_status != "All":
    filtered_data = filtered_data[filtered_data["Loan_Status"] == selected_loan_status]

if selected_gender != "All":
    filtered_data = filtered_data[filtered_data["Gender"] == selected_gender]

# Calculate Risk Score
filtered_data['Loan_Status_Denied'] = np.where(filtered_data['Loan_Status'] == 'Denied', 1, 0)

scaler = MinMaxScaler()
filtered_data[['Loan_Amount_norm','Balance_norm','Txn_Amount_norm']] = scaler.fit_transform(
    filtered_data[['Loan_Amount','Balance','Total_Transaction_Amount']])

filtered_data['Risk_Score'] = (
    0.3 * filtered_data['Loan_Amount_norm'] +
    0.3 * (1 - filtered_data['Balance_norm']) +
    0.2 * filtered_data['Is_Fraud'] +
    0.1 * filtered_data['Txn_Amount_norm'] +
    0.1 * filtered_data['Loan_Status_Denied']
)

# Main Header
st.title("💼 FinBank Risk & Fraud Analytics")
st.markdown("---")

# Metrics section (adaptive layout)
with st.container():
    metric_cols = st.columns([1,1,1,1])
    with metric_cols[0]:
        st_lottie(lottie_user, height=80, key="user")
        st.markdown(f"<h5>Total Customers</h5><h2>{filtered_data.shape[0]}</h2>", unsafe_allow_html=True)
    with metric_cols[1]:
        st_lottie(lottie_loan, height=80, key="loan")
        st.markdown(f"<h5>Total Loan Amount</h5><h2>₹{filtered_data['Loan_Amount'].sum():,.0f}</h2>", unsafe_allow_html=True)
    with metric_cols[2]:
        st_lottie(lottie_fraud, height=80, key="fraud")
        st.markdown(f"<h5>Fraud Cases</h5><h2>{int(filtered_data['Is_Fraud'].sum())}</h2>", unsafe_allow_html=True)
    with metric_cols[3]:
        st_lottie(lottie_risk, height=80, key="risk")
        st.markdown(f"<h5>Avg Risk Score</h5><h2>{filtered_data['Risk_Score'].mean()*100:.2f}%</h2>", unsafe_allow_html=True)

# Customer Distribution
st.markdown("## 🧑‍🤝‍🧑 Customer Distribution")
col_a, col_b = st.columns(2)

with col_a:
    gender_count = filtered_data['Gender'].value_counts()
    color_map = {'Male': '#1976D2', 'Female': '#D32F2F'}
    fig_gender = px.pie(values=gender_count.values, names=gender_count.index, color=gender_count.index,
                        color_discrete_map=color_map, hole=0.5, title="Gender Distribution")
    st.plotly_chart(fig_gender, use_container_width=True)

with col_b:
    fraud_count = filtered_data['Is_Fraud'].value_counts()
    fraud_labels = ['Genuine', 'Fraud']
    fig_fraud = px.bar(x=fraud_labels, y=fraud_count.sort_index(), color=fraud_labels,
                       color_discrete_sequence=['#388E3C','#C62828'], title="Fraud vs Genuine Users")
    st.plotly_chart(fig_fraud, use_container_width=True)

# Loan Insights
st.markdown("## 🏦 Loan Insights")
col_c, col_d = st.columns(2)

with col_c:
    fig_loan = px.box(filtered_data, y="Loan_Amount", color="Loan_Status", title="Loan Amount Distribution")
    st.plotly_chart(fig_loan, use_container_width=True)

with col_d:
    avg_txn_by_fraud = filtered_data.groupby("Is_Fraud")["Total_Transaction_Amount"].mean().reset_index()
    avg_txn_by_fraud["Fraud Status"] = avg_txn_by_fraud["Is_Fraud"].map({0: "Genuine", 1: "Fraud"})
    fig_avg_txn = px.bar(avg_txn_by_fraud, x="Fraud Status", y="Total_Transaction_Amount",
                         color="Fraud Status", color_discrete_sequence=["#0288D1", "#FF9999"],
                         title="Avg Transaction Amount by Fraud Status")
    st.plotly_chart(fig_avg_txn, use_container_width=True)

# Risk Segmentation
st.markdown("## ⚠️ Customer Risk Segmentation")
filtered_data['Risk_Segment'] = filtered_data['Risk_Score'].apply(
    lambda score: 'High' if score >= 0.7 else ('Medium' if score >= 0.4 else 'Low'))
risk_counts = filtered_data['Risk_Segment'].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
fig_risk = px.bar(x=risk_counts.index, y=risk_counts.values,
                  color=risk_counts.index, color_discrete_sequence=["#2E7D32", "#FBC02D", "#B71C1C"],
                  title="Risk Level Distribution",
                  labels={"x": "Risk Level", "y": "Customer Count"})
st.plotly_chart(fig_risk, use_container_width=True)

# Account Type Loan Amount
st.markdown("## 💳 Avg Loan Amount by Account Type")
avg_loan_by_acc = filtered_data.groupby("Account_Type")["Loan_Amount"].mean().reset_index()
fig_avg_loan = px.bar(avg_loan_by_acc, x="Account_Type", y="Loan_Amount",
                      color="Account_Type", title="Avg Loan Amount by Account Type")
st.plotly_chart(fig_avg_loan, use_container_width=True)

# Loan vs Spending Behavior
st.markdown("## 🔬 Loan vs Spending Behavior")
fig_scatter = px.scatter(filtered_data, x="Loan_Amount", y="Total_Transaction_Amount",
                         color="Is_Fraud", color_discrete_map={0: "green", 1: "red"},
                         size_max=10, title="Loan Amount vs Transaction Amount (Fraud Highlight)")
fig_scatter.update_traces(marker=dict(size=4, opacity=0.7))
st.plotly_chart(fig_scatter, use_container_width=True)

# Top High-Risk Customers
st.markdown("## 🔥 Top 10 High-Risk Customers")
top_risky = filtered_data.sort_values("Loan_Amount", ascending=False).head(10)
st.dataframe(top_risky[["Customer_ID", "Customer_Name", "Loan_Amount", "Balance", "Is_Fraud"]])

# Footer
st.markdown("""<hr><center><small>🚀 Built by Sayali Chavan — FinTech SaaS 🔥</small></center>""", unsafe_allow_html=True)
