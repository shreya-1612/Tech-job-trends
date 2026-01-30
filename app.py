import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Tech Job Trends", layout="wide")

st.title("Tech Job Market Dashboard & Salary Predictor")

# Load datasets
top_skills = pd.read_csv("data/top_skills.csv")
top_companies = pd.read_csv("data/top_companies.csv")
top_locations = pd.read_csv("data/top_locations.csv")

# Load model
model = joblib.load("models/salary_model.pkl")


st.subheader("Top In-Demand Skills")
st.bar_chart(top_skills.set_index("skills"))


st.subheader("Top Hiring Locations")
st.dataframe(top_locations)


st.subheader("Top Hiring Companies")
st.dataframe(top_companies)


st.subheader("Salary Prediction")

months_exp = st.slider("Months of Experience", 0, 200, 36)

education = st.selectbox(
    "Education Level",
    ["bachelor degree", "master degree", "phd", "unknown"]
)

location = st.text_input("Location (e.g., San Francisco, CA)")
company = st.text_input("Company")

if st.button("Predict Salary"):
    input_df = pd.DataFrame([{
        "months_experience": months_exp,
        "education": education,
        "location": location,
        "company": company
    }])

    pred = model.predict(input_df)[0]
    st.success(f"Estimated Salary: ${int(pred):,}")
