import streamlit as st
import pickle
import pandas as pd
import numpy as np

class HotelBookingApp:
    def __init__(self):
        # Load the pre-trained model and label encoders
        self.model, self.encoder = self.load_model('best_xgboost_model.pkl', 'label_encoders.pkl')

    def load_model(self, model_filename, encoder_filename):
        """Load the trained model and label encoders."""
        with open(model_filename, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(encoder_filename, 'rb') as encoders_file:
            encoder = pickle.load(encoders_file)
        return model, encoder

    def preprocess_input(self, input_df):
        """Preprocess user input: encode categorical features."""
        encoded_df = input_df.copy()
        
        # Encode categorical columns
        encoded_df['type_of_meal_plan'] = self.encoder.transform(encoded_df['type_of_meal_plan'])
        encoded_df['room_type_reserved'] = self.encoder.transform(encoded_df['room_type_reserved'])
        encoded_df['market_segment_type'] = self.encoder.transform(encoded_df['market_segment_type'])
        
        return encoded_df

    def predict(self, input_df):
        """Predict booking cancellation status."""
        input_df = self.preprocess_input(input_df)
        prediction = self.model.predict(input_df)
        probability = self.model.predict_proba(input_df)[0][1]
        return prediction[0], probability

    def run(self):
        st.title("🏨 Hotel Booking Cancellation Prediction")
        st.write("Predict whether a hotel booking will be **cancelled** or **not cancelled** based on the input data below")
        st.markdown("---")

        st.subheader("✏️ Input Booking Information")

        # Input fields for booking information
        user_input = pd.DataFrame([{
            'no_of_adults': st.number_input('Number of Adults', min_value=1, max_value=10, value=1),
            'no_of_children': st.number_input('Number of Children', min_value=0, max_value=10, value=0),
            'no_of_weekend_nights': st.number_input('Weekend Nights', min_value=0, max_value=7, value=0),
            'no_of_week_nights': st.number_input('Week Nights', min_value=0, max_value=7, value=0),
            'type_of_meal_plan': st.selectbox('Meal Plan Type', ['Room Only', 'BB', 'HB', 'FB']),
            'required_car_parking_space': st.selectbox('Car Parking Required?', [0, 1]),  # 0: No, 1: Yes
            'room_type_reserved': st.selectbox('Room Type Reserved', ['Room Type 1', 'Room Type 2', 'Room Type 3']),
            'lead_time': st.slider('Lead Time (days)', 0, 365, 30),
            'arrival_year': st.selectbox('Arrival Year', [2017, 2018]),
            'arrival_month': st.slider('Arrival Month', 1, 12, 1),
            'arrival_date': st.slider('Arrival Date', 1, 31, 1),
            'market_segment_type': st.selectbox('Market Segment Type', ['Direct', 'Corporate', 'Online TA', 'Offline TA/TO']),
            'repeated_guest': st.selectbox('Repeated Guest?', [0, 1]),
            'no_of_previous_cancellations': st.slider('Previous Cancellations', 0, 10, 0),
            'no_of_previous_bookings_not_canceled': st.slider('Previous Non-Cancelled Bookings', 0, 10, 0),
            'avg_price_per_room': st.number_input('Average Price per Room (EUR)', min_value=1.0, max_value=1000.0, value=100.0),
            'no_of_special_requests': st.slider('Special Requests', 0, 5, 0)
        }])

        if st.button("🔮 Predict Booking Status"):
            # Make prediction
            pred, prob = self.predict(user_input)
            status = "✅ Not Cancelled" if pred == 0 else "❌ Cancelled"
            st.success(f"### Prediction: {status}")
            st.info(f"### Cancellation Probability: {prob:.2%}")
            st.markdown("#### 🔎 Data Used for Prediction")
            st.dataframe(user_input)

if __name__ == "__main__":
    app = HotelBookingApp()
    app.run()
