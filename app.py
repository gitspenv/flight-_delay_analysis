# Full Streamlit App with Regression (Pipeline-based) and Classification Tabs

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Define feature lists
regression_features = [
    'Day_Of_Week', 'Airline', 'Dep_Airport', 'DepTime_label',
    'Flight_Duration', 'Distance_type', 'Delay_Carrier', 'Delay_Weather', 'Delay_NAS',
    'Delay_Security', 'Delay_LastAircraft', 'Manufacturer', 'Aicraft_age',
    'tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'pres',
    'Aircraft_Age_Group'
]

classifier_features = [
    'FlightDate', 'Day_Of_Week', 'Airline', 'Dep_Airport', 'DepTime_label',
    'Arr_Airport', 'Arr_Delay', 'Flight_Duration', 'Distance_type', 'Manufacturer',
    'Aicraft_age', 'AIRPORT', 'tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir',
    'wspd', 'pres', 'Cancelled', 'Diverted', 'Aircraft_Age_Group'
]

valid_airlines = [
    "Endeavor Air", "American Airlines Inc.", "Alaska Airlines Inc.", "JetBlue Airways",
    "Delta Air Lines Inc", "Frontier Airlines Inc.", "Allegiant Air", "Hawaiian Airlines Inc.",
    "American Eagle Airlines Inc.", "Spirit Air Lines", "Southwest Airlines Co.",
    "Republic Airways", "PSA Airlines", "Skywest Airlines Inc.", "United Air Lines Inc."
]

st.title("✈️ Flight Delay Predictor")
tab1, tab2 = st.tabs(["Regression Prediction (Delay Minutes)", "Classification Prediction (Delayed/Not Delayed)"])

# --------------------------- REGRESSION TAB ---------------------------
with tab1:
    st.subheader("Predict Delay Minutes")

    try:
        with open("model/tuned_flight_delay_model.pkl", "rb") as f:
            regression_models = pickle.load(f)
    except FileNotFoundError:
        st.error("Regression model file not found.")
        st.stop()

    @st.cache_data
    def load_regression_data():
        return pd.read_csv("data/processed/df_cleaned_ready_for_modeling.csv")

    df_reg = load_regression_data()
    top_n = 50
    top_categories_reg = {
        'Dep_Airport': df_reg['Dep_Airport'].value_counts().nlargest(top_n).index.tolist(),
        'Airline': df_reg['Airline'].value_counts().nlargest(top_n).index.tolist(),
        'Manufacturer': df_reg['Manufacturer'].value_counts().nlargest(top_n).index.tolist(),
        'Distance_type': df_reg['Distance_type'].value_counts().nlargest(top_n).index.tolist(),
        'Aircraft_Age_Group': df_reg['Aircraft_Age_Group'].value_counts().nlargest(top_n).index.tolist(),
        'DepTime_label': ["Morning", "Afternoon", "Evening", "Night"]
    }

    regression_input_data = {}
    mode = st.radio("Choose Input Mode", ["Manual Input", "Impute Missing Data", "Generate Random Data"], key="regression_mode")

    if mode == "Generate Random Data" and st.button("🔁 Regenerate Random Input", key="regression_regenerate"):
        st.experimental_rerun()

    if mode == "Generate Random Data":
        sample_row = df_reg.drop(columns=['Dep_Delay', 'Arr_Delay', 'Delayed'], errors='ignore').sample(1).iloc[0]
        regression_input_data = {k: v for k, v in sample_row.items() if k in regression_features}

    regression_input_data['Day_Of_Week'] = st.selectbox("Day of Week", options=range(1, 8), index=int(regression_input_data.get('Day_Of_Week', 1)) - 1, key="reg_day_of_week")
    regression_input_data['Airline'] = st.selectbox("Airline", top_categories_reg['Airline'], index=top_categories_reg['Airline'].index(regression_input_data.get('Airline', top_categories_reg['Airline'][0])) if regression_input_data.get('Airline') in top_categories_reg['Airline'] else 0, key="reg_airline")
    regression_input_data['Dep_Airport'] = st.selectbox("Departure Airport", top_categories_reg['Dep_Airport'], index=top_categories_reg['Dep_Airport'].index(regression_input_data.get('Dep_Airport', top_categories_reg['Dep_Airport'][0])) if regression_input_data.get('Dep_Airport') in top_categories_reg['Dep_Airport'] else 0, key="reg_dep_airport")
    regression_input_data['DepTime_label'] = st.selectbox("Departure Time", top_categories_reg['DepTime_label'], index=top_categories_reg['DepTime_label'].index(regression_input_data.get('DepTime_label', "Morning")), key="reg_deptime")

    for feature in set(regression_features) - {'Day_Of_Week', 'Airline', 'Dep_Airport', 'DepTime_label'}:
        default_value = regression_input_data.get(feature, "")
        if feature in df_reg.columns:
            unique_values = df_reg[feature].dropna().unique()
            if df_reg[feature].dtype == 'object' or len(unique_values) < 20:
                options = top_categories_reg.get(feature, sorted(unique_values.tolist()))
                regression_input_data[feature] = st.selectbox(f"{feature}", options, index=options.index(default_value) if default_value in options else 0, key=f"reg_{feature}")
            else:
                val = st.text_input(f"{feature} (leave blank to impute)" if mode == "Impute Missing Data" else feature, value=str(default_value), key=f"reg_{feature}")
                try:
                    regression_input_data[feature] = None if val.strip() == "" else float(val)
                except ValueError:
                    st.error(f"Invalid input for {feature}. Please enter a number.")
                    st.stop()

    if st.button("Predict Delay Minutes", key="reg_predict"):
        regression_input_df = pd.DataFrame([regression_input_data], columns=regression_features)
        for col in ['Dep_Airport', 'Airline', 'Manufacturer', 'Distance_type', 'Aircraft_Age_Group', 'DepTime_label']:
            if col in regression_input_df.columns:
                regression_input_df[col] = regression_input_df[col].apply(lambda x: x if x in top_categories_reg[col] else "Other")

        try:
            dep_pred = regression_models['Dep_Delay'].predict(regression_input_df)[0]
            arr_pred = regression_models['Arr_Delay'].predict(regression_input_df)[0]

            st.subheader("Regression Prediction Results")
            st.success(f"Estimated Departure Delay: {dep_pred:.1f} minutes")
            st.success(f"Estimated Arrival Delay: {arr_pred:.1f} minutes")
            st.dataframe(regression_input_df)

        except Exception as e:
            st.error(f"Prediction error: {e}")

# --------------------------- CLASSIFICATION TAB ---------------------------
with tab2:
    st.subheader("Predict Delay Status")

    try:
        with open("model/flight_delay_classifier.pkl", "rb") as f:
            classifier_model = pickle.load(f)
    except FileNotFoundError:
        st.error("Classifier model file not found.")
        st.stop()

    @st.cache_data
    def load_classifier_data():
        return pd.read_csv("data/processed/df_cleaned_ready_for_modeling.csv")

    df_cls = load_classifier_data()
    top_n = 50
    top_categories = {
        'Dep_Airport': df_cls['Dep_Airport'].value_counts().nlargest(top_n).index.tolist(),
        'Arr_Airport': df_cls['Arr_Airport'].value_counts().nlargest(top_n).index.tolist(),
        'Airline': df_cls['Airline'].value_counts().nlargest(top_n).index.tolist()
    }

    classifier_input_data = {}
    mode = st.radio("Choose Input Mode", ["Manual Input", "Impute Missing Data", "Generate Random Data"], key="classifier_mode")

    if mode == "Generate Random Data" and st.button("🔁 Regenerate Random Input", key="classifier_regenerate"):
        st.experimental_rerun()

    if mode == "Generate Random Data":
        sample_row = df_cls.drop(columns=['Dep_Delay', 'Arr_Delay', 'Delayed'], errors='ignore').sample(1).iloc[0]
        classifier_input_data = {k: v for k, v in sample_row.items() if k in classifier_features}

    classifier_input_data['FlightDate'] = st.date_input("Flight Date", value=datetime.now()).strftime("%Y-%m-%d")
    classifier_input_data['Day_Of_Week'] = pd.to_datetime(classifier_input_data['FlightDate']).dayofweek + 1
    classifier_input_data['Airline'] = st.selectbox("Airline", valid_airlines, index=valid_airlines.index(classifier_input_data.get('Airline', valid_airlines[0])) if classifier_input_data.get('Airline') in valid_airlines else 0, key="cls_airline")
    classifier_input_data['Dep_Airport'] = st.selectbox("Departure Airport", top_categories['Dep_Airport'], key="cls_dep_airport")
    classifier_input_data['Arr_Airport'] = st.selectbox("Arrival Airport", top_categories['Arr_Airport'], key="cls_arr_airport")

    if classifier_input_data['Dep_Airport'] == classifier_input_data['Arr_Airport']:
        st.error("Departure and Arrival airports cannot be the same.")
        st.stop()

    classifier_input_data['DepTime_label'] = st.selectbox("Departure Time", ["Morning", "Afternoon", "Evening", "Night"], key="cls_deptime")

    for feature in set(classifier_features) - {'FlightDate', 'Day_Of_Week', 'Airline', 'Dep_Airport', 'Arr_Airport', 'DepTime_label'}:
        val = st.text_input(f"{feature}", value="")
        classifier_input_data[feature] = None if val.strip() == "" else val

    if st.button("Predict Delay Status", key="cls_predict"):
        classifier_input_df = pd.DataFrame([classifier_input_data], columns=classifier_features)
        for col in ['Dep_Airport', 'Arr_Airport', 'Airline']:
            classifier_input_df[col] = classifier_input_df[col].apply(lambda x: x if x in top_categories[col] else "Other")

        classifier_input_df = pd.get_dummies(classifier_input_df)
        expected_columns = classifier_model.feature_names_in_
        classifier_input_df = classifier_input_df.reindex(columns=expected_columns, fill_value=0)

        try:
            class_pred = classifier_model.predict(classifier_input_df)[0]
            st.subheader("Classification Prediction Results")
            result = "Delayed (15+ min)" if class_pred == 1 else "On Time"
            st.success(f"Delay Status: {result}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
