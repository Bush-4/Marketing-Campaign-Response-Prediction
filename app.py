import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Marketing Campaign Response Prediction",
    page_icon="📊",
    layout="wide"
)
st.title("Marketing Campaign Response Prediction")

st.write(
    "This application predicts whether a customer is likely "
    "to subscribe to a bank's term deposit."
)

model = joblib.load("marketing_model.pkl")
threshold = joblib.load("best_threshold.pkl")

st.header("1. Business Problem")

st.subheader("Problem Statement")

st.write(
    "Banks conduct marketing campaigns to encourage customers "
    "to subscribe to term deposits. However, contacting every "
    "customer can increase campaign costs and reduce efficiency."
)

st.subheader("Project Objective")

st.write(
    "The objective is to predict whether a customer is likely "
    "to subscribe to a term deposit using customer and campaign "
    "information available before the call."
)


st.header("2. Data Insights")

st.subheader("Dataset Overview")

df = pd.read_csv("bank-full.csv", sep=";")

st.write("The dataset contains information about customers contacted during bank marketing campaigns.")

col1, col2 = st.columns(2)

col1.metric("Total Records", f"{df.shape[0]:,}")
col2.metric("Total Features", df.shape[1] - 1)

st.subheader("Customer Response Distribution")

response_counts = df["y"].value_counts()

st.bar_chart(response_counts)

st.write("This chart shows the distribution of customers who subscribed and those who did not subscribe to the term deposit.")

st.subheader("Customer Response by Job")

job_response = pd.crosstab(df["job"], df["y"])

st.bar_chart(job_response)

st.write(
    "Compare subscription responses across job categories. "
    "This helps identify which customer groups may be more responsive "
    "to bank marketing campaigns."
)

st.header("3. Customer Prediction")

st.write(
    "Enter the customer's details below to predict "
    "whether they are likely to subscribe to a term deposit."
)

st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)

    job = st.selectbox(
        "Job",
        ["admin.", "blue-collar", "entrepreneur", "housemaid",
         "management", "retired", "self-employed", "services",
         "student", "technician", "unemployed", "unknown"]
    )

    marital = st.selectbox(
        "Marital Status",
        ["married", "single", "divorced"]
    )

with col2:
    education = st.selectbox(
        "Education",
        ["primary", "secondary", "tertiary", "unknown"]
    )

    default = st.selectbox(
        "Has Credit in Default?",
        ["yes", "no"]
    )

st.subheader("Financial Information")

col3, col4 = st.columns(2)

with col3:
    balance = st.number_input(
        "Average Yearly Balance", value=1000
    )

    housing = st.selectbox(
        "Housing Loan", ["yes", "no"]
    )

    loan = st.selectbox(
        "Personal Loan", ["yes", "no"]
    )

    contact = st.selectbox(
        "Contact Type", ["cellular", "telephone", "unknown"]
    )

    day = st.number_input(
        "Last Contact Day", min_value=1, max_value=31, value=15
    )

with col4:
    month = st.selectbox(
        "Last Contact Month",
        ["jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"]
    )

    campaign = st.number_input(
        "Number of Contacts During This Campaign",
        min_value=1, value=1
    )

    pdays = st.number_input(
        "Days Since Previous Contact",
        min_value=-1, value=-1
    )

    previous = st.number_input(
        "Number of Previous Contacts",
        min_value=0, value=0
    )

    poutcome = st.selectbox(
        "Previous Campaign Outcome",
        ["unknown", "failure", "other", "success"]
    )


st.subheader("Run Prediction")

input_data = {
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome
}

input_df = pd.DataFrame([input_data])

if st.button("Predict Customer Response", key="predict_customer_response"):

    probability = model.predict_proba(input_df)[0][
        list(model.classes_).index("yes")
    ]

    if probability >= threshold:
        st.success("Prediction: Customer is likely to subscribe!")
    else:
        st.warning("Prediction: Customer is unlikely to subscribe.")

    st.metric(
        "Subscription Probability",
        f"{probability * 100:.2f}%"
    )

    st.subheader("Business Recommendation")

    if probability >= threshold:
        st.write("The bank can prioritise this customer for follow-up.")
    else:
        st.write("The bank can consider a lower-priority follow-up.")

    st.progress(float(probability))