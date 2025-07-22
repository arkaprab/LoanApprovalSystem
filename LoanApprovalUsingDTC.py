import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Title
st.title("🏦 Loan Approval Prediction App")
st.caption("Built with Decision Tree Classifier | Streamlit + Scikit-learn")

# Dataset
data={  
    "CreditScore": [750, 600, 700, 800, 650, 720, 580, 820, 680, 750,
                    670, 590, 640, 780, 710, 630, 760, 690, 610, 740,
                    705, 665, 775, 615, 690, 720, 690, 600, 810, 665],
    "Income": [50000, 40000, 60000, 70000, 55000, 65000, 45000, 80000, 60000, 70000,
               48000, 43000, 52000, 75000, 62000, 41000, 67000, 59000, 42000, 66000,
               61000, 53000, 73000, 46000, 61000, 64000, 58000, 39000, 77000, 52000],
    "Amount": [10000, 20000, 15000, 25000, 12000, 18000, 22000, 30000, 15000, 20000,
               14000, 19000, 16000, 26000, 17000, 21000, 24000, 15500, 20500, 23000,
               18000, 13500, 25000, 17500, 16000, 22000, 15000, 19500, 29000, 18500],
    "EmploymentLength": [5, 3, 7, 10, 4, 6, 2, 12, 8, 9,
                         5, 1, 6, 11, 7, 3, 10, 4, 2, 8,
                         6, 3, 9, 2, 5, 7, 6, 1, 11, 5],
    "Married": [1, 0, 1, 1, 1, 1, 0, 1, 1, 1,
                0, 0, 1, 1, 1, 0, 1, 1, 0, 1,
                0, 0, 1, 0, 1, 1, 1, 0, 1, 1],
    "Education": [2, 0, 1, 2, 1, 1, 0, 2, 1, 2,
                  0, 1, 1, 2, 2, 0, 2, 1, 0, 2,
                  1, 0, 2, 0, 1, 1, 2, 0, 2, 1],
    "Approved": [1, 0, 1, 1, 1, 1, 0, 1, 1, 1,
                 0, 0, 1, 1, 1, 0, 1, 1, 0, 1,
                 1, 0, 1, 0, 1, 1, 1, 0, 1, 1]
}
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
