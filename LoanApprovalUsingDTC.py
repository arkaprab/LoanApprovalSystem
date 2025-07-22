import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 

# Title
st.title("🏦 Loan Approval Prediction App")

# Dataset
data = {
    "CreditScore":[750,600,700,800,650,720,580,820,680,750],
    "Income":[50000,40000,60000,70000,55000,65000,45000,80000,60000,70000],
    "Amount":[10000,20000,15000,25000,12000,18000,22000,30000,15000,20000],
    "EmploymentLength":[5,3,7,10,4,6,2,12,8,9],
    "Approved":[1,0,1,1,1,1,0,1,1,1]
}
df = pd.DataFrame(data)

# Show data (optional)
with st.expander("📊 View Sample Data"):
    st.dataframe(df)

# Features and Target
X = df[["CreditScore", "Income", "Amount", "EmploymentLength"]]
y = df["Approved"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Accuracy
predicted = classifier.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
st.success(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Sidebar Inputs
st.header("📝 Enter Your Loan Details Below:")

creditscore = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
income = st.number_input("Monthly Income (₹)", min_value=0, step=1000, value=50000)
loanamount = st.number_input("Loan Amount (₹)", min_value=0, step=1000, value=15000)
employmentlength = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)

# Prediction
if st.button("Predict Loan Approval"):
    input_data = [[creditscore, income, loanamount, employmentlength]]
    result = classifier.predict(input_data)
    if result[0] == 1:
        st.success("🎉 Loan Approved ✅")
    else:
        st.error("❌ Loan Not Approved")
