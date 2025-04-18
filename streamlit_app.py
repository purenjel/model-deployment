import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Memuat model dan encoder yang sudah disimpan
def load_model_and_encoder():
    with open('best_xgboost_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    with open('label_encoders.pkl', 'rb') as encoders_file:
        encoder = pickle.load(encoders_file)
        
    return model, encoder

# Fungsi untuk melakukan prediksi
def predict_booking_status(model, encoder, input_data):
    # Menyandikan input data yang masuk menggunakan encoder
    input_data['type_of_meal_plan'] = encoder.transform([input_data['type_of_meal_plan']])[0]
    input_data['room_type_reserved'] = encoder.transform([input_data['room_type_reserved']])[0]
    input_data['market_segment_type'] = encoder.transform([input_data['market_segment_type']])[0]
    
    # Prediksi menggunakan model XGBoost
    prediction = model.predict(pd.DataFrame([input_data]))[0]
    
    return 'Canceled' if prediction == 1 else 'Not Canceled'

# Streamlit UI untuk menerima input
def main():
    st.title("Hotel Booking Status Prediction")

    # Input fields from the dataset
    no_of_adults = st.number_input("Number of Adults", min_value=1, value=1)
    no_of_children = st.number_input("Number of Children", min_value=0, value=0)
    no_of_weekend_nights = st.number_input("Number of Weekend Nights", min_value=0, value=0)
    no_of_week_nights = st.number_input("Number of Week Nights", min_value=0, value=0)
    
    type_of_meal_plan = st.selectbox("Meal Plan", options=["Full Board", "Half Board", "Bed and Breakfast", "No Meal"])
    required_car_parking_space = st.selectbox("Car Parking Required", options=[0, 1])
    room_type_reserved = st.selectbox("Room Type Reserved", options=["Standard", "Superior", "Deluxe", "Suite"])
    lead_time = st.number_input("Lead Time (Days)", min_value=1, value=1)
    arrival_year = st.number_input("Arrival Year", min_value=2000, value=2025)
    arrival_month = st.number_input("Arrival Month", min_value=1, max_value=12, value=1)
    market_segment_type = st.selectbox("Market Segment", options=["Corporate", "Online TA", "Offline TA/TO", "Direct"])
    repeated_guest = st.selectbox("Repeated Guest", options=[0, 1])
    no_of_previous_cancellations = st.number_input("Previous Cancellations", min_value=0, value=0)
    no_of_previous_bookings_not_canceled = st.number_input("Previous Bookings (Not Canceled)", min_value=0, value=0)
    avg_price_per_room = st.number_input("Average Price per Room", min_value=0, value=100)
    no_of_special_requests = st.number_input("Special Requests", min_value=0, value=0)

    input_data = {
        "no_of_adults": no_of_adults,
        "no_of_children": no_of_children,
        "no_of_weekend_nights": no_of_weekend_nights,
        "no_of_week_nights": no_of_week_nights,
        "type_of_meal_plan": type_of_meal_plan,
        "required_car_parking_space": required_car_parking_space,
        "room_type_reserved": room_type_reserved,
        "lead_time": lead_time,
        "arrival_year": arrival_year,
        "arrival_month": arrival_month,
        "market_segment_type": market_segment_type,
        "repeated_guest": repeated_guest,
        "no_of_previous_cancellations": no_of_previous_cancellations,
        "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
        "avg_price_per_room": avg_price_per_room,
        "no_of_special_requests": no_of_special_requests
    }

    if st.button("Predict Booking Status"):
        # Memuat model dan encoder
        model, encoder = load_model_and_encoder()
        
        # Melakukan prediksi
        result = predict_booking_status(model, encoder, input_data)
        
        # Menampilkan hasil
        st.success(f"The predicted booking status is: {result}")

if __name__ == "__main__":
    main()
