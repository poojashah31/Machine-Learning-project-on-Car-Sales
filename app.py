import streamlit as st
import pandas as pd
import joblib

# Load the trained Lasso model and RFE selector
lasso_model = joblib.load("lasso_model.pkl")
rfe_selector = joblib.load("rfe_selector.pkl")

# ✅ Use the full original feature list (before RFE)
# These should exactly match the training set column names used before RFE
all_features = [
    'automaker_Honda', 'automaker_Royal Enfield', 'automaker_Bajaj Auto',
    'automaker_Hero MotoCorp', 'automaker_Hyundai', 'automaker_Kia',
    'automaker_Mahindra', 'automaker_Maruti Suzuki', 'automaker_Tata',
    'automaker_TVS Motor', 'vehicle_type_Passenger Vehicle',
    'vehicle_type_Two Wheeler', 'pv_sales', 'pv_premium_sales',
    'pv_export_sales', 'tw_export_sales', 'repo_rate', 'inflation_rate',
    'consumer_confidence_index', 'inventory_level', 'monsoon_index',
    'gdp_growth', 'month', 'year'
]

# Streamlit UI
st.set_page_config(page_title="Car Sales Decline Prediction", layout="centered")
st.title("🚗 Car Sales Decline Predictor")

st.markdown("""
Predict the expected percentage change in car sales using automotive, economic, and seasonal indicators.
Fill in the inputs below:
""")

# Collect input values
input_data = {}
for feature in all_features:
    if feature in ['month', 'year']:
        input_data[feature] = st.number_input(f"{feature}", min_value=1, step=1, format="%d")
    else:
        input_data[feature] = st.number_input(f"{feature}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Apply RFE to select only the features used in training
try:
    input_rfe = rfe_selector.transform(input_df)
except Exception as e:
    st.error(f"❌ Feature selection failed: {e}")
    st.stop()

# Make prediction
if st.button("Predict"):
    try:
        prediction = lasso_model.predict(input_rfe)
        st.success(f"📉 Predicted Change in Car Sales: **{prediction[0]:.2f}%**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
