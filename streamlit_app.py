import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Load the saved model and label encoder
def load_model():
    with open('best_xgboost_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('label_encoders.pkl', 'rb') as encoders_file:
        encoder = pickle.load(encoders_file)
    return model, encoder

# Preprocess input data (same as preprocessing done during training)
def preprocess_input(data, encoder):
    # Assuming the input data is a dict or pandas DataFrame
    df = pd.DataFrame([data])

    # Fill missing values (simulating the preprocessing)
    df['required_car_parking_space'].fillna(df['required_car_parking_space'].mode()[0], inplace=True)
    df['type_of_meal_plan'].fillna(df['type_of_meal_plan'].mode()[0], inplace=True)
    df['avg_price_per_room'].fillna(df['avg_price_per_room'].median(), inplace=True)
    df['no_of_adults'] = df['no_of_adults'].replace(0, 1)

    # Encode categorical columns
    df['type_of_meal_plan'] = encoder.transform(df['type_of_meal_plan'])
    df['room_type_reserved'] = encoder.transform(df['room_type_reserved'])
    df['market_segment_type'] = encoder.transform(df['market_segment_type'])

    return df

# Make prediction
def make_prediction(model, data, encoder):
    data_processed = preprocess_input(data, encoder)
    prediction = model.predict(data_processed)
    return prediction

# Streamlit app
st.title("Hotel Booking Cancellation Prediction")

# Inputs for the user to fill
st.header("Input Details:")
no_of_adults = st.number_input('Number of Adults', min_value=1, value=2)
no_of_children = st.number_input('Number of Children', min_value=0, value=0)
no_of_weekend_nights = st.number_input('Number of Weekend Nights', min_value=0, value=1)
no_of_week_nights = st.number_input('Number of Week Nights', min_value=0, value=1)
type_of_meal_plan = st.selectbox('Type of Meal Plan', options=['Breakfast', 'Half Board', 'Full Board'])
required_car_parking_space = st.selectbox('Required Car Parking Space (0 - No, 1 - Yes)', options=[0, 1])
room_type_reserved = st.selectbox('Room Type Reserved', options=['Single', 'Double', 'Suite'])
lead_time = st.number_input('Lead Time (Days)', min_value=1, value=10)
arrival_year = st.number_input('Arrival Year', min_value=2020, value=2025)
arrival_month = st.number_input('Arrival Month', min_value=1, max_value=12, value=5)
market_segment_type = st.selectbox('Market Segment Type', options=['Online', 'Offline', 'Corporate'])
repeated_guest = st.selectbox('Is this a repeated guest? (0 - No, 1 - Yes)', options=[0, 1])
no_of_previous_cancellations = st.number_input('Number of Previous Cancellations', min_value=0, value=0)
no_of_previous_bookings_not_canceled = st.number_input('Number of Previous Bookings Not Canceled', min_value=0, value=1)
avg_price_per_room = st.number_input('Average Price per Room (Euro)', min_value=0.0, value=100.0)
no_of_special_requests = st.number_input('Number of Special Requests', min_value=0, value=1)

# Prepare the data for prediction
user_data = {
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
    'market_segment_type': market_segment_type,
    'repeated_guest': repeated_guest,
    'no_of_previous_cancellations': no_of_previous_cancellations,
    'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
    'avg_price_per_room': avg_price_per_room,
    'no_of_special_requests': no_of_special_requests
}

# Load model and encoder
model, encoder = load_model()

# When the user clicks on the "Predict" button, make a prediction
if st.button('Predict Cancellation'):
    prediction = make_prediction(model, user_data, encoder)
    
    if prediction == 0:
        st.write("The booking is likely to **NOT** be canceled.")
    else:
        st.write("The booking is likely to be **CANCELED**.")

# Test Cases (Example)
# Test Case 1: Successful booking (not canceled)
test_case_1 = {
    'no_of_adults': 2,
    'no_of_children': 0,
    'no_of_weekend_nights': 1,
    'no_of_week_nights': 2,
    'type_of_meal_plan': 'Breakfast',
    'required_car_parking_space': 1,
    'room_type_reserved': 'Single',
    'lead_time': 5,
    'arrival_year': 2025,
    'arrival_month': 5,
    'market_segment_type': 'Online',
    'repeated_guest': 1,
    'no_of_previous_cancellations': 0,
    'no_of_previous_bookings_not_canceled': 3,
    'avg_price_per_room': 100.0,
    'no_of_special_requests': 1
}

# Test Case 2: Canceled booking
test_case_2 = {
    'no_of_adults': 1,
    'no_of_children': 2,
    'no_of_weekend_nights': 0,
    'no_of_week_nights': 3,
    'type_of_meal_plan': 'Half Board',
    'required_car_parking_space': 0,
    'room_type_reserved': 'Suite',
    'lead_time': 10,
    'arrival_year': 2025,
    'arrival_month': 6,
    'market_segment_type': 'Corporate',
    'repeated_guest': 0,
    'no_of_previous_cancellations': 2,
    'no_of_previous_bookings_not_canceled': 0,
    'avg_price_per_room': 150.0,
    'no_of_special_requests': 2
}

# Predict for Test Case 1 and Test Case 2
prediction_test_1 = make_prediction(model, test_case_1, encoder)
prediction_test_2 = make_prediction(model, test_case_2, encoder)

st.write(f"Test Case 1 Prediction: {'Canceled' if prediction_test_1 == 1 else 'Not Canceled'}")
st.write(f"Test Case 2 Prediction: {'Canceled' if prediction_test_2 == 1 else 'Not Canceled'}")
