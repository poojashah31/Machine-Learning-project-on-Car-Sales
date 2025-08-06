import streamlit as st
import pandas as pd
import joblib

# Load the trained Lasso model and RFE selector
lasso_model = joblib.load("lasso_model.pkl")
rfe_selector = joblib.load("rfe_selector.pkl")

# ✅ Full original feature set used *before* RFE during training
# Copy these from X_train.columns (used before applying RFE)
all_features = [
    ['automaker_Bajaj Auto', 'automaker_Hero MotoCorp', 'automaker_Honda',
       'automaker_Hyundai', 'automaker_Kia', 'automaker_Mahindra',
       'automaker_Maruti Suzuki', 'automaker_Renault',
       'automaker_Royal Enfield', 'automaker_Suzuki Motorcycle',
       'automaker_TVS Motor', 'automaker_Tata Motors', 'automaker_Toyota',
       'automaker_Yamaha', 'region_Andaman & Nicobar', 'region_Andhra Pradesh',
       'region_Arunachal Pradesh', 'region_Assam', 'region_Bihar',
       'region_Chandigarh', 'region_Chhattisgarh',
       'region_Dadra & Nagar Haveli and Daman & Diu', 'region_Delhi',
       'region_Goa', 'region_Gujarat', 'region_Haryana',
       'region_Himachal Pradesh', 'region_Jammu & Kashmir', 'region_Jharkhand',
       'region_Karnataka', 'region_Kerala', 'region_Ladakh',
       'region_Lakshadweep', 'region_Madhya Pradesh', 'region_Maharashtra',
       'region_Manipur', 'region_Meghalaya', 'region_Mizoram',
       'region_Nagaland', 'region_Odisha', 'region_Puducherry',
       'region_Punjab', 'region_Rajasthan', 'region_Sikkim',
       'region_Tamil Nadu', 'region_Telangana', 'region_Tripura',
       'region_Uttar Pradesh', 'region_Uttarakhand', 'region_West Bengal',
       'vehicle_type_Passenger Vehicle', 'vehicle_type_Two-Wheeler',
       'pv_sales', 'tw_sales', 'pv_entry_level_sales', 'pv_premium_sales',
       'pv_export_sales', 'tw_export_sales', 'rural_demand_index', 'repo_rate',
       'inflation_rate', 'consumer_confidence_index', 'inventory_level',
       'monsoon_index', 'gdp_growth', 'month', 'year']
]

# Streamlit UI
st.set_page_config(page_title="Car Sales Decline Predictor", layout="centered")
st.title("🚗 Car Sales Decline Predictor")

st.markdown("""
Enter values for each feature below to predict the expected decline in car sales.
Use appropriate values for one-hot encoded automakers and vehicle type (e.g., 1 or 0).
""")

# Collect input values
input_data = {}
for feature in all_features:
    if feature in ['month', 'year']:
        input_data[feature] = st.number_input(f"{feature}", min_value=1, step=1, format="%d")
    elif feature.startswith("automaker_") or feature.startswith("vehicle_type_"):
        input_data[feature] = st.selectbox(f"{feature}", [0, 1], index=0)
    else:
        input_data[feature] = st.number_input(f"{feature}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Apply RFE to select required features
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
