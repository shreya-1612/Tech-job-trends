Live demo: https://tech-job-trends-bgpqrqfruelkm3aiqbu3hb.streamlit.app

Overview

Tech Job Trends is a full-stack data analytics and machine learning project that analyzes real-world job-posting data to uncover:

 In-demand technical skills

* Top hiring locations

 * Leading employers

 * Education requirements

 * Experience patterns

 * Salary predictions

The project demonstrates an end-to-end analytics workflow — from raw data ingestion and NLP-based skill extraction to SQL analytics, interactive dashboards, and deployed machine-learning models.

A live interactive web dashboard allows users to explore trends and predict salaries for different job profiles.

 Tech Stack

* Programming: Python

* Data Processing: Pandas, NumPy

* Database: SQLite, SQL

* NLP: Regex, text preprocessing

* Machine Learning: Scikit-learn

* Visualization: Matplotlib, Streamlit

* Notebooks: Google Colab

* Version Control & Deployment: GitHub, Streamlit Cloud



 Phase 1–2 — Data Cleaning & NLP

* Ingested job-posting data from Kaggle

* Performed missing-value treatment and normalization

* Cleaned and standardized job descriptions

* Extracted technical skills using NLP techniques

* Generated skill-frequency metrics

* Created a reusable cleaned dataset for analytics

 Phase 3 — SQL Analytics & Visualization

* Loaded cleaned data into SQLite database

* Designed normalized relational schema

* Stored skills in a separate table

* Wrote SQL queries to analyze:

* Top hiring companies

* Most active locations

* Education distributions

* Experience requirements

* Skill demand trends

* Visualized insights using Python

* Exported aggregated tables for dashboards

 Phase 4 — Machine Learning & Salary Prediction

Engineered features from:

* Experience

* Education

* Location

* Company

* Technical skills

Trained multiple regression models:

* Random Forest

* Gradient Boosting

* Selected best model using RMSE and R² metrics

* Achieved R² ≈ 0.52 on unseen test data

* Built reproducible preprocessing pipelines using scikit-learn

* Serialized final model for deployment

 Phase 5 — Interactive Dashboard & Deployment

Built Streamlit web application for:

* Exploring hiring trends

* Viewing top skills, companies, and locations

* Predicting salaries

* Added skill-selection UI for personalized predictions

* Deployed publicly using Streamlit Cloud

* Handled production issues such as:

* Version mismatches

* Model serialization

* Input schema validation

* File-size optimization
