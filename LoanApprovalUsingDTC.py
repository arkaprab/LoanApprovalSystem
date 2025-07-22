import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Title
st.title("🏦 Loan Approval Prediction App")
st.caption("Built with Decision Tree Classifier | Streamlit + Scikit-learn")

# Dataset
data = pd.read_csv("credit_data")
df = pd.DataFrame(data)

# Show data
with st.expander("📊 View Sample Data"):
    st.dataframe(df)

# Features and Target
X = df[["CreditScore", "Income", "Amount", "EmploymentLength", "Married", "Education"]]
y = df["Approved"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.success(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Input Form
st.header("📝 Enter Your Loan Details Below:")
with st.form("loan_form"):
    col1, col2 = st.columns(2)
    with col1:
        creditscore = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
        income = st.number_input("Monthly Income (₹)", min_value=0, step=1000, value=50000)
        employmentlength = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)
    with col2:
        loanamount = st.number_input("Loan Amount (₹)", min_value=0, step=1000, value=15000)
        married = st.selectbox("Married?", options=["No", "Yes"])
        education = st.selectbox("Education Level", options=["High School", "Graduate", "Postgraduate"])
    
    submitted = st.form_submit_button("Predict Loan Approval")

# Predict
if submitted:
    married_val = 1 if married == "Yes" else 0
    education_val = ["High School", "Graduate", "Postgraduate"].index(education)
    
    input_data = [[creditscore, income, loanamount, employmentlength, married_val, education_val]]
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success("🎉 Loan Approved ✅")
    else:
        st.error("❌ Loan Not Approved")

    st.info(f"🧠 Confidence (Approved): {prediction_proba[1]*100:.2f}%")
