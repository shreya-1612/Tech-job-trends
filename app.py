import streamlit as st
import pandas as pd
import joblib

# Page Config

st.set_page_config(page_title="Tech Job Trends", layout="wide")

st.title("Tech Job Market Dashboard & Salary Predictor")


# Load datasets

top_skills = pd.read_csv("data/top_skills.csv")
top_companies = pd.read_csv("data/top_companies.csv")
top_locations = pd.read_csv("data/top_locations.csv")


# Load trained model

model = joblib.load("models/salary_model.pkl")


# Dashboards

st.subheader("Top In-Demand Skills")
st.bar_chart(top_skills.set_index("skills"))

st.subheader("Top Hiring Locations")
st.dataframe(top_locations)

st.subheader("Top Hiring Companies")
st.dataframe(top_companies)


# Salary Prediction Section

st.subheader("Salary Prediction")

months_exp = st.slider("Months of Experience", 0, 200, 36)

education = st.selectbox(
    "Education Level",
    ["bachelor degree", "master degree", "phd", "unknown"]
)

location = st.text_input("Location (e.g., San Francisco, CA)")
company = st.text_input("Company")


# Skill Selection

TOP_SKILLS = [
    "python", "java", "sql", "excel", "aws",
    "machine learning", "c++", "javascript",
    "kubernetes", "azure"
]

st.subheader("Skills")

selected_skills = []

cols = st.columns(5)
for i, skill in enumerate(TOP_SKILLS):
    with cols[i % 5]:
        if st.checkbox(skill.title()):
            selected_skills.append(skill)


# Predict Button

if st.button("Predict Salary"):

    input_data = {
        "months_experience": months_exp,
        "education": education,
        "location": location,
        "company": company,
    }

    # add skill columns
    for skill in TOP_SKILLS:
        col = f"skill_{skill.replace(' ', '_')}"
        input_data[col] = int(skill in selected_skills)

    input_df = pd.DataFrame([input_data])

    pred = model.predict(input_df)[0]

    st.success(f"Estimated Salary: ${int(pred):,}")
