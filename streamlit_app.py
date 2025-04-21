import streamlit as st
import pickle
import pandas as pd
import numpy as np

class HotelBookingApp:
    def __init__(self):
        self.model = self.load_model('xgboost_model.pkl')  # Change this to your model's filename
        self.encoders = self.load_model('label_encoders.pkl')  # Change this to your encoders' filename
        self.data = self.load_csv('Dataset_B_hotel.csv')  # Change this to your dataset's filename

    def load_model(self, path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.warning(f"⚠️ Failed to load model: {e}")
            return None

    def load_csv(self, path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.warning(f"⚠️ Failed to load CSV file: {e}")
            return None

    def encode_input(self, input_df):
        encoded_df = input_df.copy()
        for col in encoded_df.select_dtypes(include='object').columns:
            if col in self.encoders:
                le = self.encoders[col]
                encoded_df[col] = le.transform(encoded_df[col])
            else:
                encoded_df[col] = 0  # Default to 0 if the column is not found in encoders
        return encoded_df

    def predict(self, input_df):
        encoded_df = self.encode_input(input_df)
        prediction = self.model.predict(encoded_df)[0]
        probability = self.model.predict_proba(encoded_df)[0][1]
        return prediction, probability

    def run(self):
        st.title("🏨 Hotel Booking Cancellation Prediction")
        st.write("Predict whether a hotel booking will be **cancelled** or **not cancelled** based on the input data below")
        st.markdown("---")

        # Show the dataset
        if self.data is not None:
            st.subheader("📂 Dataset Preview")
            st.dataframe(self.data.head(50))  # Show a preview of the dataset
            st.markdown("---")

        st.subheader("✏️ Input Booking Information")

        # Collect user inputs
        user_input = pd.DataFrame([{
            'no_of_adults': st.number_input('Number of Adults', min_value=1, max_value=10, value=2),
            'no_of_children': st.number_input('Number of Children', min_value=0, max_value=10, value=0),
            'no_of_weekend_nights': st.number_input('Weekend Nights', min_value=0, max_value=10, value=1),
            'no_of_week_nights': st.number_input('Week Nights', min_value=0, max_value=10, value=2),
            'type_of_meal_plan': st.selectbox('Meal Plan Type', ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected']),
            'required_car_parking_space': st.selectbox('Car Parking Required?', [0, 1]),
            'room_type_reserved': st.selectbox('Room Type Reserved', ['Room_Type 1', 'Room_Type 2', 'Room_Type 3']),
            'lead_time': st.slider('Lead Time (days)', 0, 500, 45),
            'arrival_year': st.selectbox('Arrival Year', [2017, 2018]),
            'arrival_month': st.slider('Arrival Month', 1, 12, 7),
            'arrival_date': st.slider('Arrival Date', 1, 31, 15),
            'market_segment_type': st.selectbox('Market Segment Type', ['Online', 'Offline', 'Corporate']),
            'repeated_guest': st.selectbox('Repeated Guest?', [0, 1]),
            'no_of_previous_cancellations': st.slider('Previous Cancellations', 0, 10, 0),
            'no_of_previous_bookings_not_canceled': st.slider('Previous Non-Cancelled Bookings', 0, 10, 0),
            'avg_price_per_room': st.number_input('Average Price per Room', min_value=0.0, max_value=1000.0, value=100.0),
            'no_of_special_requests': st.slider('Special Requests', 0, 5, 1)
        }])

        # Predict the booking status when the button is clicked
        if st.button("🔮 Predict Booking Status"):
            pred, prob = self.predict(user_input)
            status = "✅ Not Cancelled" if pred == 0 else "❌ Cancelled"
            st.success(f"### Prediction: {status}")
            st.info(f"### Cancellation Probability: {prob:.2%}")
            st.markdown("#### 🔎 Data Used for Prediction")
            st.dataframe(user_input)

if __name__ == "__main__":
    app = HotelBookingApp()
    app.run()
