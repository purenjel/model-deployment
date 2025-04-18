import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Fungsi untuk memuat model dan encoder yang sudah disimpan
def load_model():
    with open('best_xgboost_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('label_encoders.pkl', 'rb') as encoders_file:
        encoders = pickle.load(encoders_file)
    return model, encoders

# Fungsi untuk memproses input dari pengguna
def preprocess_input(user_input, encoders):
    # Mengonversi input ke format DataFrame
    input_data = pd.DataFrame([user_input])
    
    # Melakukan encoding pada kolom kategorikal
    input_data['type_of_meal_plan'] = encoders['type_of_meal_plan'].transform(input_data['type_of_meal_plan'])
    input_data['room_type_reserved'] = encoders['room_type_reserved'].transform(input_data['room_type_reserved'])
    input_data['market_segment_type'] = encoders['market_segment_type'].transform(input_data['market_segment_type'])
    
    return input_data

# Fungsi untuk memprediksi dengan model yang sudah dilatih
def predict(input_data, model):
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit UI
def run():
    st.title("Hotel Booking Prediction")
    
    # Input data dari pengguna
    st.subheader("Input Data")
    
    type_of_meal_plan = st.selectbox("Type of Meal Plan", ["Meal Plan 1", "Not Selected", "Meal Plan 2", "Meal Plan 3"])
    room_type_reserved = st.selectbox("Room Type Reserved", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])
    market_segment_type = st.selectbox("Market Segment Type", ["Online", "Offline", "Corporate", "Complementary", "Aviation"])
    no_of_adults = st.number_input("Number of Adults", min_value=1, max_value=4, value=1)
    no_of_children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    required_car_parking_space = st.selectbox("Required Car Parking Space", [0, 1])
    avg_price_per_room = st.number_input("Average Price per Room", min_value=0, max_value=500, value=100)
    
    # Menyusun input menjadi dictionary
    user_input = {
        "type_of_meal_plan": type_of_meal_plan,
        "room_type_reserved": room_type_reserved,
        "market_segment_type": market_segment_type,
        "no_of_adults": no_of_adults,
        "no_of_children": no_of_children,
        "required_car_parking_space": required_car_parking_space,
        "avg_price_per_room": avg_price_per_room
    }

    # Memuat model dan encoder
    model, encoders = load_model()
    
    # Proses input dan prediksi
    input_data = preprocess_input(user_input, encoders)
    prediction = predict(input_data, model)

    # Menampilkan hasil prediksi
    st.subheader("Prediction Result")
    if prediction == 0:
        st.write("Booking is **Not Canceled**")
    else:
        st.write("Booking is **Canceled**")

# Menjalankan aplikasi Streamlit
if __name__ == "__main__":
    run()
