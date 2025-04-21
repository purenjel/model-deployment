import pandas as pd
import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Load the saved model and encoders separately
with open("best_xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Preprocessing class
class HotelBookingPreprocessor:
    def __init__(self, df):
        self.df = df
        # Initialize separate encoders for each categorical feature
        self.meal_plan_encoder = LabelEncoder()
        self.room_type_encoder = LabelEncoder()
        self.market_segment_encoder = LabelEncoder()

    def fill_missing_values(self):
        """Fill missing values in the dataframe."""
        self.df['required_car_parking_space'].fillna(self.df['required_car_parking_space'].mode()[0], inplace=True)
        self.df['type_of_meal_plan'].fillna(self.df['type_of_meal_plan'].mode()[0], inplace=True)
        self.df['avg_price_per_room'].fillna(self.df['avg_price_per_room'].median(), inplace=True)
        self.df['no_of_adults'] = self.df['no_of_adults'].replace(0, 1)

    def encode_labels(self):
        """Encode categorical features using separate LabelEncoders."""
        self.df['type_of_meal_plan'] = self.meal_plan_encoder.fit_transform(self.df['type_of_meal_plan'])
        self.df['room_type_reserved'] = self.room_type_encoder.fit_transform(self.df['room_type_reserved'])
        self.df['market_segment_type'] = self.market_segment_encoder.fit_transform(self.df['market_segment_type'])

    def split_data(self):
        """Split the data into features (X) and target (y), but only return X for inference."""
        X = self.df.drop(columns=['booking_status'], errors='ignore')  # Ignore 'booking_status' if it doesn't exist
        return X

    def preprocess(self):
        """Execute the full preprocessing pipeline."""
        self.fill_missing_values()
        self.encode_labels()
        X = self.split_data()  # Only return features for inference, no target (y)
        return X


# Streamlit app layout and functionality
st.title("Hotel Booking Cancellation Prediction")
st.image("https://www.hoteldel.com/wp-content/uploads/2021/01/hotel-del-coronado-views-suite-K1TOS1-K1TOJ1-1600x900-1.jpg")
st.caption("This app predicts whether a hotel booking will be canceled based on guest booking details.")

# Sidebar for input fields
st.sidebar.header("Required Input Fields")
st.sidebar.markdown("**Number of Adults**: The number of adults in the booking.")
st.sidebar.markdown("**Number of Children**: The number of children in the booking.")
st.sidebar.markdown("**Weekend Nights**: The number of nights booked for weekends.")
st.sidebar.markdown("**Week Nights**: The number of week nights booked.")
st.sidebar.markdown("**Meal Plan**: The meal plan chosen (e.g., Room Only, Half Board, etc.).")
st.sidebar.markdown("**Car Parking**: Whether the guest needs a car parking space (Yes or No).")
st.sidebar.markdown("**Room Type**: The type of room booked.")
st.sidebar.markdown("**Lead Time**: The number of days between booking and check-in.")
st.sidebar.markdown("**Arrival Year**: The year of arrival.")
st.sidebar.markdown("**Arrival Month**: The month of arrival.")
st.sidebar.markdown("**Arrival Date**: The date of arrival.")
st.sidebar.markdown("**Market Segment**: The segment of the market the booking belongs to.")
st.sidebar.markdown("**Repeated Guest**: Whether the guest has made a previous booking (Yes or No).")
st.sidebar.markdown("**Previous Cancellations**: The number of bookings previously canceled.")
st.sidebar.markdown("**Previous Bookings Not Canceled**: The number of bookings not canceled.")
st.sidebar.markdown("**Price per Room**: The average price per room (in Euro).")
st.sidebar.markdown("**Special Requests**: Number of special requests made by the guest.")

# Input fields for user
input_data = {}
col1, col2, col3 = st.columns(3)

with col1:
    input_data['no_of_adults'] = st.selectbox("Number of Adults", [0, 1, 2, 3, 4])
    input_data['no_of_children'] = st.selectbox("Number of Children", [0, 1, 2, 3, 9, 10])
    input_data['no_of_weekend_nights'] = st.selectbox("Number of Weekend Nights", [0, 1, 2, 3, 4, 5, 6, 7])
    input_data['no_of_week_nights'] = st.selectbox("Number of Week Nights", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    input_data['type_of_meal_plan'] = st.selectbox("Meal Plan", ['Not Selected', 'Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3'])
    input_data['required_car_parking_space'] = st.selectbox("Car Parking", ['No', 'Yes'])  # Yes/No

with col2:
    input_data['room_type_reserved'] = st.selectbox("Room Type", ['Room_Type 1', 'Room_Type 4', 'Room_Type 6', 'Room_Type 2', 'Room_Type 5', 'Room_Type 7', 'Room_Type 3'])
    input_data['lead_time'] = st.number_input("Lead Time (days)", min_value=0, max_value=352)
    input_data['arrival_year'] = st.selectbox("Arrival Year", [2017, 2018])
    input_data['arrival_month'] = st.selectbox("Arrival Month", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    input_data['arrival_date'] = st.selectbox("Arrival Date", list(range(1, 32)))  # Values from 1 to 31
    input_data['market_segment_type'] = st.selectbox("Market Segment", ['Online', 'Offline', 'Corporate', 'Complementary', 'Aviation'])

with col3:
    input_data['repeated_guest'] = st.selectbox("Repeated Guest", ['No', 'Yes'])  # Yes/No
    input_data['no_of_previous_cancellations'] = st.selectbox("Previous Cancellations", [0, 1, 2, 3, 4, 5, 6, 11, 13])
    input_data['no_of_previous_bookings_not_canceled'] = st.selectbox("Previous Bookings Not Canceled", list(range(0, 58)))
    input_data['avg_price_per_room'] = st.number_input("Average Price per Room (EUR)", min_value=1.0, max_value=500.0)
    input_data['no_of_special_requests'] = st.selectbox("Special Requests", [0, 1, 2, 3, 4, 5])

# When the button is clicked, make a prediction
if st.button("Predict Cancellation"):
    # Convert the input data to a pandas DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess the data using HotelBookingPreprocessor
    preprocessor = HotelBookingPreprocessor(input_df)
    X_input, _ = preprocessor.preprocess()  # Get features after preprocessing

    # Make prediction using the trained XGBoost model
    prediction = model.predict(X_input)

    # Display the prediction result
    if prediction[0] == 1:
        st.write("The booking is likely to be Canceled.")
    else:
        st.write("The booking is likely to be Not Canceled.")
