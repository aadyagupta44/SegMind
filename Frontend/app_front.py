# Import libraries
import streamlit as st
import requests
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.stats import skew

# FastAPI endpoint URL
FASTAPI_URL = "http://127.0.0.1:8000"

def main():
    st.set_page_config(page_title="Customer Segmentation Tool", page_icon="🧑‍💼", layout="centered")

    st.title("🧑‍💼 SegMind Customer Segmentation Tool")
    st.markdown("Enter customer details below to predict their segment and generate a marketing strategy.")

    # Initialize session state for input_data and strategy
    if "input_data" not in st.session_state:
        st.session_state.input_data = {
            "age": 35,
            "gender": "Male",
            "income": 90000,
            "spending_score": 85,
            "membership_years": 2,
            "purchase_frequency": 3,
            "preferred_category": "Electronics",
            "last_purchase_amount": 1200
        }
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "strategy" not in st.session_state:
        st.session_state.strategy = None

    # Customer input form
    with st.form("customer_form"):
        age = st.number_input("Age", min_value=18, max_value=100, value=st.session_state.input_data.get("age", 35))
        gender = st.selectbox("Gender", ["Male", "Female"], index=0 if st.session_state.input_data.get("gender", "Male") == "Male" else 1)
        income = st.number_input("Income ($)", min_value=0, max_value=1000000, value=st.session_state.input_data.get("income", 90000))
        spending_score = st.slider("Spending Score", min_value=0, max_value=100, value=st.session_state.input_data.get("spending_score", 85))
        membership_years = st.number_input("Membership Years", min_value=0, max_value=50, value=st.session_state.input_data.get("membership_years", 2))
        purchase_frequency = st.number_input("Purchase Frequency (per year)", min_value=0, max_value=365, value=st.session_state.input_data.get("purchase_frequency", 3))
        preferred_category = st.selectbox(
            "Preferred Category",
            ["Electronics", "Fashion", "Groceries", "Home", "Sports"],
            index=["Electronics", "Fashion", "Groceries", "Home", "Sports"].index(st.session_state.input_data.get("preferred_category", "Electronics"))
        )
        last_purchase_amount = st.number_input("Last Purchase Amount ($)", min_value=0, max_value=100000, value=st.session_state.input_data.get("last_purchase_amount", 1200))
        
        submitted = st.form_submit_button("Predict Segment")

        if submitted:
            st.session_state.input_data = {
                "age": age,
                "gender": gender,
                "income": income,
                "spending_score": spending_score,
                "membership_years": membership_years,
                "purchase_frequency": purchase_frequency,
                "preferred_category": preferred_category,
                "last_purchase_amount": last_purchase_amount
            }
            st.session_state.strategy = None  # Reset strategy on new prediction

            with st.spinner("Predicting segment and generating strategy..."):
                # Call /predict_segment
                response = requests.post(f"{FASTAPI_URL}/predict_segment", json=st.session_state.input_data)
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.prediction_result = result
                else:
                    st.session_state.prediction_result = None
                    st.error(f"Prediction failed: {response.text}")

                # Call /ai for strategy
                cust_info = "\n".join([f"{k}: {v}" for k, v in st.session_state.input_data.items()])
                try:
                    ai_response = requests.post(f"{FASTAPI_URL}/ai", json={"cust_info": cust_info})
                    if ai_response.status_code == 200:
                        data = ai_response.json()
                        if "strategy" in data:
                            st.session_state.strategy = data["strategy"]
                        elif "error" in data:
                            st.session_state.strategy = None
                            st.error(f"AI service error: {data['error']}")
                    else:
                        st.session_state.strategy = None
                        st.error(f"Failed to generate strategy: {ai_response.status_code} {ai_response.text}")
                except Exception as e:
                    st.session_state.strategy = None
                    st.error(f"Exception: {e}")

    # Display prediction result if available
    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        st.success(f"Predicted Segment: {result['segment_label']} - {result['segment_name']}")

    # Display strategy if available
    if st.session_state.strategy:
        st.markdown("### Generated Marketing Strategy")
        st.info(st.session_state.strategy)

if __name__ == "__main__":
    main()