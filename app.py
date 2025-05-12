import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load preprocessor and models
with open("model/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("model/flight_delay_model.pkl", "rb") as f:
    models = pickle.load(f)

# Define feature list (should match training features)
features = [
    'Day_Of_Week', 'Airline', 'Dep_Airport', 'DepTime_label',
    'Flight_Duration', 'Distance_type', 'Delay_Carrier', 'Delay_Weather', 'Delay_NAS',
    'Delay_Security', 'Delay_LastAircraft', 'Manufacturer', 'Aicraft_age',
    'tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'pres',
    'Aircraft_Age_Group'
]

# Load cleaned dataset for random data sampling
@st.cache_data
def load_cleaned_data():
    return pd.read_csv("data/processed/df_cleaned_ready_for_modeling.csv")

df = load_cleaned_data()

st.title("✈️ Flight Delay Predictor")
mode = st.radio("Choose Input Mode", ["Manual Input", "Impute Missing Data", "Generate Random Data"])

# Button to regenerate random input
if mode == "Generate Random Data" and st.button("🔁 Regenerate Random Input"):
    st.experimental_rerun()

input_data = {}

if mode == "Generate Random Data":
    sample_row = df.drop(columns=['Dep_Delay', 'Arr_Delay'], errors='ignore').sample(1).iloc[0]
    input_data = sample_row[features].to_dict()

for feature in features:
    default_value = input_data.get(feature, "")
    if feature in df.columns:
        unique_values = df[feature].dropna().unique()
        if df[feature].dtype == 'object' or len(unique_values) < 20:
            options = sorted(unique_values.tolist())
            input_data[feature] = st.selectbox(f"{feature}", options, index=options.index(default_value) if default_value in options else 0)
        else:
            val = st.text_input(f"{feature} (leave blank to impute)" if mode == "Impute Missing Data" else feature, value=str(default_value))
            try:
                input_data[feature] = None if val.strip() == "" else float(val)
            except ValueError:
                st.error(f"Invalid input for {feature}. Please enter a number.")
                st.stop()
    else:
        val = st.text_input(feature, value=str(default_value))
        input_data[feature] = None if val.strip() == "" else val

if st.button("Predict Delay"):
    input_df = pd.DataFrame([input_data])
    input_processed = preprocessor.transform(input_df)
    dep_pred = models['Dep_Delay'].predict(input_processed)[0]
    arr_pred = models['Arr_Delay'].predict(input_processed)[0]
    st.success(f"Estimated Departure Delay: {dep_pred:.1f} minutes")
    st.success(f"Estimated Arrival Delay: {arr_pred:.1f} minutes")

    # Show imputed values
    st.subheader("🧩 Final Input Used (after imputation)")
    input_df_imputed = pd.DataFrame(preprocessor.named_transformers_['num'].named_steps['imputer'].transform(
        input_df[numeric_features := df[features].select_dtypes(include=['number']).columns.tolist()]
    ), columns=numeric_features)

    input_df_combined = input_df.copy()
    input_df_combined[numeric_features] = input_df_imputed[numeric_features]
    st.dataframe(input_df_combined)