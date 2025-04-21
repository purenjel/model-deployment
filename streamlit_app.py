import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Load the trained model and encoder
with open('best_xgboost_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('label_encoders.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

def preprocess_input_data(input_data):
    """Preprocess input data for the model."""
    # Apply the same preprocessing steps used during training (label encoding, etc.)
    input_data['type_of_meal_plan'] = encoder.transform([input_data['type_of_meal_plan']])[0]
    input_data['room_type_reserved'] = encoder.transform([input_data['room_type_reserved']])[0]
    input_data['market_segment_type'] = encoder.transform([input_data['market_segment_type']])[0]
    
    return input_data

def predict_booking_cancellation(input_data):
    """Predict the booking cancellation using the trained model."""
    input_data = preprocess_input_data(input_data)
    prediction = model.predict([input_data])
    return 'Cancelled' if prediction[0] == 1 else 'Not Cancelled'

# Streamlit UI
st.title("Hotel Booking Cancellation Prediction")
st.markdown("This app predicts if a hotel booking will be cancelled based on the provided data.")

# Input fields for user input
no_of_adults = st.number_input("Number of Adults", min_value=1, value=1)
no_of_children = st.number_input("Number of Children", min_value=0, value=0)
no_of_weekend_nights = st.number_input("Number of Weekend Nights", min_value=0, value=0)
no_of_week_nights = st.number_input("Number of Week Nights", min_value=0, value=0)
type_of_meal_plan = st.selectbox("Type of Meal Plan", ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Meal Plan 4'])
required_car_parking_space = st.selectbox("Required Car Parking Space", [0, 1])
room_type_reserved = st.selectbox("Room Type Reserved", ['Room Type 1', 'Room Type 2', 'Room Type 3'])
lead_time = st.number_input("Lead Time (Days)", min_value=1, value=1)
arrival_year = st.number_input("Arrival Year", min_value=2000, value=2025)
arrival_month = st.number_input("Arrival Month", min_value=1, max_value=12, value=5)
arrival_date = st.number_input("Arrival Date", min_value=1, max_value=31, value=1)
market_segment_type = st.selectbox("Market Segment Type", ['Segment 1', 'Segment 2', 'Segment 3'])
repeated_guest = st.selectbox("Repeated Guest", [0, 1])
no_of_previous_cancellations = st.number_input("Number of Previous Cancellations", min_value=0, value=0)
no_of_previous_bookings_not_canceled = st.number_input("Number of Previous Bookings Not Canceled", min_value=0, value=0)
avg_price_per_room = st.number_input("Average Price per Room (EUR)", min_value=1, value=100)
no_of_special_requests = st.number_input("Number of Special Requests", min_value=0, value=0)

# Input data dictionary
input_data = {
    'no_of_adults': no_of_adults,
    'no_of_children': no_of_children,
    'no_of_weekend_nights': no_of_weekend_nights,
    'no_of_week_nights': no_of_week_nights,
    'type_of_meal_plan': type_of_meal_plan,
    'required_car_parking_space': required_car_parking_space,
    'room_type_reserved': room_type_reserved,
    'lead_time': lead_time,
    'arrival_year': arrival_year,
    'arrival_month': arrival_month,
    'arrival_date': arrival_date,
    'market_segment_type': market_segment_type,
    'repeated_guest': repeated_guest,
    'no_of_previous_cancellations': no_of_previous_cancellations,
    'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
    'avg_price_per_room': avg_price_per_room,
    'no_of_special_requests': no_of_special_requests
}

# Prediction button
if st.button("Predict Booking Cancellation"):
    result = predict_booking_cancellation(input_data)
    st.write(f"The booking will be: {result}")
