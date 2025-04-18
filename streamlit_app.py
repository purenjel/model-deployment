import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Load the saved model and label encoder
def load_model():
    with open('best_xgboost_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('label_encoders.pkl', 'rb') as encoders_file:
        encoder_dict = pickle.load(encoders_file)
    return model, encoder_dict

# Prediction function
def predict_booking_status(input_data, model, encoder_dict):
    # Encoding categorical columns using the loaded label encoders
    try:
        input_data['type_of_meal_plan'] = encoder_dict['type_of_meal_plan'].transform([input_data['type_of_meal_plan']])
    except ValueError:
        st.error("Invalid value for type_of_meal_plan. Please choose from the available options.")
        return None
        
    try:
        input_data['room_type_reserved'] = encoder_dict['room_type_reserved'].transform([input_data['room_type_reserved']])
    except ValueError:
        st.error("Invalid value for room_type_reserved. Please choose from the available options.")
        return None
    
    try:
        input_data['market_segment_type'] = encoder_dict['market_segment_type'].transform([input_data['market_segment_type']])
    except ValueError:
        st.error("Invalid value for market_segment_type. Please choose from the available options.")
        return None

    # Predict using the model
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit interface
def main():
    st.title('Hotel Booking Cancellation Prediction')
    
    # Upload model and encoder
    model, encoder_dict = load_model()

    # Define valid options from label encoder
    valid_meal_plans = encoder_dict['type_of_meal_plan'].classes_
    valid_room_types = encoder_dict['room_type_reserved'].classes_
    valid_market_segments = encoder_dict['market_segment_type'].classes_

    # Input form for user to enter data
    st.header('Enter booking details')

    # Form inputs
    no_of_adults = st.number_input('Number of Adults', min_value=1, max_value=5, value=1)
    no_of_children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)
    no_of_weekend_nights = st.number_input('Number of Weekend Nights', min_value=0, max_value=7, value=0)
    no_of_week_nights = st.number_input('Number of Week Nights', min_value=0, max_value=7, value=0)
    
    type_of_meal_plan = st.selectbox('Meal Plan', valid_meal_plans)
    required_car_parking_space = st.selectbox('Required Car Parking Space', ['Yes', 'No'])
    room_type_reserved = st.selectbox('Room Type Reserved', valid_room_types)
    market_segment_type = st.selectbox('Market Segment Type', valid_market_segments)

    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        'no_of_adults': [no_of_adults],
        'no_of_children': [no_of_children],
        'no_of_weekend_nights': [no_of_weekend_nights],
        'no_of_week_nights': [no_of_week_nights],
        'type_of_meal_plan': [type_of_meal_plan],
        'required_car_parking_space': [1 if required_car_parking_space == 'Yes' else 0],
        'room_type_reserved': [room_type_reserved],
        'market_segment_type': [market_segment_type]
    })

    if st.button('Predict Booking Status'):
        prediction = predict_booking_status(input_data, model, encoder_dict)
        if prediction is not None:
            status = 'Canceled' if prediction == 1 else 'Not Canceled'
            st.write(f"The booking status is: {status}")

# Running the application
if __name__ == "__main__":
    main()
