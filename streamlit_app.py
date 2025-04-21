import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Load saved model and encoder
def load_model(model_filename, encoder_filename):
    with open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(encoder_filename, 'rb') as encoders_file:
        encoder = pickle.load(encoders_file)
    return model, encoder

# Preprocess the input data before making predictions
def preprocess_input_data(df, encoder):
    # Preprocessing steps (same as in training)
    df['type_of_meal_plan'] = encoder.transform(df['type_of_meal_plan'])
    df['room_type_reserved'] = encoder.transform(df['room_type_reserved'])
    df['market_segment_type'] = encoder.transform(df['market_segment_type'])
    return df

# Prediction function
def make_prediction(model, encoder, input_data):
    # Preprocess input data
    input_data = preprocess_input_data(input_data, encoder)
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Return prediction result (0: Not Canceled, 1: Canceled)
    return 'Canceled' if prediction[0] == 1 else 'Not Canceled'

# Load the model and encoder
model, encoder = load_model('best_xgboost_model.pkl', 'label_encoders.pkl')

# Streamlit app interface
st.title('Hotel Booking Cancellation Prediction')

st.write("Please enter the following details:")

# Get user input
no_of_adults = st.number_input("Number of adults", min_value=1, max_value=10, value=1)
no_of_children = st.number_input("Number of children", min_value=0, max_value=10, value=0)
no_of_weekend_nights = st.number_input("Number of weekend nights", min_value=0, max_value=7, value=0)
no_of_week_nights = st.number_input("Number of week nights", min_value=0, max_value=7, value=0)
type_of_meal_plan = st.selectbox("Meal plan type", ["Room Only", "BB", "HB", "FB"])  # Example meal plans
required_car_parking_space = st.selectbox("Need car parking?", [0, 1])  # 0: No, 1: Yes
room_type_reserved = st.selectbox("Room type reserved", ["Room Type 1", "Room Type 2", "Room Type 3"])  # Example room types
lead_time = st.number_input("Lead time (days)", min_value=0, max_value=365, value=0)
arrival_year = st.number_input("Arrival year", [2017, 2018])
arrival_month = st.number_input("Arrival month", min_value=1, max_value=12, value=1)
arrival_date = st.number_input("Arrival date", min_value=1, max_value=31, value=1)
market_segment_type = st.selectbox("Market segment type", ["Direct", "Corporate", "Online TA", "Offline TA/TO"])  # Example
repeated_guest = st.selectbox("Is this a repeated guest?", [0, 1])  # 0: No, 1: Yes
no_of_previous_cancellations = st.number_input("Number of previous cancellations", min_value=0, max_value=10, value=0)
no_of_previous_bookings_not_canceled = st.number_input("Number of previous bookings not canceled", min_value=0, max_value=10, value=0)
avg_price_per_room = st.number_input("Average price per room (EUR)", min_value=1, max_value=1000, value=100)
no_of_special_requests = st.number_input("Number of special requests", min_value=0, max_value=10, value=0)

# Create a DataFrame for input
input_data = pd.DataFrame({
    'no_of_adults': [no_of_adults],
    'no_of_children': [no_of_children],
    'no_of_weekend_nights': [no_of_weekend_nights],
    'no_of_week_nights': [no_of_week_nights],
    'type_of_meal_plan': [type_of_meal_plan],
    'required_car_parking_space': [required_car_parking_space],
    'room_type_reserved': [room_type_reserved],
    'lead_time': [lead_time],
    'arrival_year': [arrival_year],
    'arrival_month': [arrival_month],
    'arrival_date': [arrival_date],
    'market_segment_type': [market_segment_type],
    'repeated_guest': [repeated_guest],
    'no_of_previous_cancellations': [no_of_previous_cancellations],
    'no_of_previous_bookings_not_canceled': [no_of_previous_bookings_not_canceled],
    'avg_price_per_room': [avg_price_per_room],
    'no_of_special_requests': [no_of_special_requests]
})

if st.button("Predict Cancellation"):
    # Make prediction
    prediction = make_prediction(model, encoder, input_data)
    st.write(f"The booking will be: {prediction}")
