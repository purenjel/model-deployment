import pandas as pd
import streamlit as st
import numpy as np
import pickle

# Load the saved model and encoder separately
with open("best_xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    encoder = pickle.load(f)

# App layout and title
st.image("https://www.hoteldel.com/wp-content/uploads/2021/01/hotel-del-coronado-views-suite-K1TOS1-K1TOJ1-1600x900-1.jpg")  # Replace with an appropriate image for the hotel app
st.title("Hotel Booking Cancellation Prediction")

st.caption("This app predicts whether a hotel booking will be canceled based on guest booking details.")

# Sidebar description
st.sidebar.header("Required Input Fields")
st.sidebar.markdown("**Number of Adults**: The number of adults in the booking.")
st.sidebar.markdown("**Number of Children**: The number of children in the booking.")
st.sidebar.markdown("**Weekend Nights**: The number of nights booked for weekends.")
st.sidebar.markdown("**Week Nights**: The number of week nights booked.")
st.sidebar.markdown("**Meal Plan**: The meal plan chosen (e.g., Room Only, Half Board, etc.).")
st.sidebar.markdown("**Car Parking**: Whether the guest needs a car parking space (0: No, 1: Yes).")
st.sidebar.markdown("**Room Type**: The type of room booked.")
st.sidebar.markdown("**Lead Time**: The number of days between booking and check-in.")
st.sidebar.markdown("**Arrival Year**: The year of arrival.")
st.sidebar.markdown("**Arrival Month**: The month of arrival.")
st.sidebar.markdown("**Arrival Date**: The date of arrival.")
st.sidebar.markdown("**Market Segment**: The segment of the market the booking belongs to.")
st.sidebar.markdown("**Repeated Guest**: Whether the guest has made a previous booking (0: No, 1: Yes).")
st.sidebar.markdown("**Previous Cancellations**: The number of bookings previously canceled.")
st.sidebar.markdown("**Previous Bookings Not Canceled**: The number of bookings not canceled.")
st.sidebar.markdown("**Price per Room**: The average price per room (in Euro).")
st.sidebar.markdown("**Special Requests**: Number of special requests made by the guest.")

# Create the input fields for the user
input_data = {}
col1, col2, col3 = st.columns(3)

with col1:
    input_data['no_of_adults'] = st.slider("Number of Adults", 1, 10)
    input_data['no_of_children'] = st.slider("Number of Children", 0, 5)
    input_data['no_of_weekend_nights'] = st.slider("Number of Weekend Nights", 0, 7)
    input_data['no_of_week_nights'] = st.slider("Number of Week Nights", 0, 7)
    input_data['type_of_meal_plan'] = st.selectbox("Meal Plan", ['Not Selected', 'Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3'])  # Example meal plans
    input_data['required_car_parking_space'] = st.selectbox("Car Parking", [0, 1])  # 0: No, 1: Yes

with col2:
    input_data['room_type_reserved'] = st.selectbox("Room Type", ['Room Type 1', 'Room Type 2', 'Room Type 3'])  # Example room types
    input_data['lead_time'] = st.number_input("Lead Time (days)", 0, 365)
    input_data['arrival_year'] = st.selectbox("Arrival Year", [2017, 2018])
    input_data['arrival_month'] = st.selectbox("Arrival Month", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    input_data['arrival_date'] = st.slider("Arrival Date", 1, 31)
    input_data['market_segment_type'] = st.selectbox("Market Segment", ['Online', 'Offline', 'Corporate', 'Complementary', 'Aviation'])  # Example market segments

with col3:
    input_data['repeated_guest'] = st.selectbox("Repeated Guest", [0, 1])  # 0: No, 1: Yes
    input_data['no_of_previous_cancellations'] = st.slider("Previous Cancellations", 0, 10)
    input_data['no_of_previous_bookings_not_canceled'] = st.slider("Previous Bookings Not Canceled", 0, 10)
    input_data['avg_price_per_room'] = st.number_input("Average Price per Room (EUR)", 1, 1000)
    input_data['no_of_special_requests'] = st.slider("Special Requests", 0, 10)

# When the button is clicked, make a prediction
if st.button("Predict Cancellation"):
    # Convert the input data to a pandas DataFrame
    input_df = pd.DataFrame([input_data])

    # Apply the necessary preprocessing (like encoding categorical features)
    try:
        input_df['type_of_meal_plan'] = encoder.transform(input_df['type_of_meal_plan'])
        input_df['room_type_reserved'] = encoder.transform(input_df['room_type_reserved'])
        input_df['market_segment_type'] = encoder.transform(input_df['market_segment_type'])
    except Exception as e:
        st.error(f"Error during encoding: {e}")
        st.stop()

    # Make prediction using the trained XGBoost model
    prediction = model.predict(input_df)

    # Display the prediction result
    if prediction[0] == 1:
        st.write("The booking is likely to be Canceled.")
    else:
        st.write("The booking is likely to be Not Canceled.")
