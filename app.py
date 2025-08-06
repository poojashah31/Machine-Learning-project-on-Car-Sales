import streamlit as st
import pandas as pd
import joblib

# Load the trained Lasso model and RFE selector
lasso_model = joblib.load("lasso_model.pkl")
rfe_selector = joblib.load("rfe_selector.pkl")

# ✅ All features originally used before RFE
# These must match the training data feature names and order
all_features = [
   'automaker_Honda', 'automaker_Royal Enfield', 'vehicle_type_Passenger Vehicle', 'pv_sales', 'pv_premium_sales', 'pv_export_sales', 'tw_export_sales', 'repo_rate', 'inflation_rate', 'consumer_confidence_index', 'inventory_level', 'monsoon_index', 'gdp_growth', 'month', 'year'
]

# 🧠 Streamlit UI Setup
st.set_page_config(page_title="Car Sales Decline Prediction", layout="centered")
st.title("🚗 Car Sales Decline Predictor")
st.markdown("""
Predict whether car sales will decline based on various automotive, economic, and seasonal indicators.
Enter the values for the selected features below:
""")

# Input collection for all features
input_data = {}
for feature in all_features:
    if feature in ['month', 'year']:
        input_data[feature] = st.number_input(f"{feature}", min_value=1, step=1, format="%d")
    else:
        input_data[feature] = st.number_input(f"{feature}", value=0.0)

# Convert inputs to DataFrame
input_df = pd.DataFrame([input_data])

# Apply RFE to select required features
try:
    input_rfe = rfe_selector.transform(input_df)
except Exception as e:
    st.error(f"❌ Feature selection failed: {e}")
    st.stop()

# Predict and display result
if st.button("Predict"):
    try:
        prediction = lasso_model.predict(input_rfe)
        st.success(f"📉 Predicted Change in Car Sales: **{prediction[0]:.2f}%**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
