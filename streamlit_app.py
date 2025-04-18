import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Fungsi untuk memuat model, encoder, dan scaler yang disimpan
def load_model(model_filename='best_xgb_model.pkl', encoder_filename='encoder.pkl', scaler_filename='scaler.pkl'):
    with open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)
        
    with open(encoder_filename, 'rb') as encoder_file:
        encoder = pickle.load(encoder_file)
        
    with open(scaler_filename, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
        
    return model, encoder, scaler

# Fungsi untuk melakukan prediksi berdasarkan input pengguna
def predict_booking_status(input_data, model, encoder, scaler):
    input_df = pd.DataFrame([input_data])
    input_df['type_of_meal_plan'] = encoder.transform(input_df['type_of_meal_plan'])
    input_df['room_type_reserved'] = encoder.transform(input_df['room_type_reserved'])
    input_df['market_segment_type'] = encoder.transform(input_df['market_segment_type'])
    numerical_features = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 'lead_time', 
                          'arrival_year', 'arrival_month', 'arrival_date', 'no_of_previous_cancellations', 
                          'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests']
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])
    prediction = model.predict(input_df)
    
    return prediction[0]

# Streamlit UI
def main():
    st.title('Hotel Booking Cancellation Prediction')

    model, encoder, scaler = load_model()

    with st.form(key='booking_form'):
        st.subheader('Input Data untuk Prediksi')

        type_of_meal_plan = st.selectbox('Pilih Tipe Paket Makanan', ['Meal Plan 1', 'Meal Plan 2', 'Not Selected'])
        room_type_reserved = st.selectbox('Pilih Tipe Kamar', ['Room_Type 1', 'Room_Type 2', 'Room_Type 3'])
        market_segment_type = st.selectbox('Pilih Segmen Pasar', ['Online', 'Offline', 'Corporate'])
        
        no_of_adults = st.number_input('Jumlah Dewasa', min_value=1, value=1)
        no_of_children = st.number_input('Jumlah Anak Kecil', min_value=0, value=0)
        no_of_weekend_nights = st.number_input('Jumlah Malam Akhir Pekan', min_value=0, value=0)
        no_of_week_nights = st.number_input('Jumlah Malam dalam Seminggu', min_value=1, value=1)
        lead_time = st.number_input('Lead Time (Hari)', min_value=1, value=10)
        arrival_year = st.number_input('Tahun Kedatangan', min_value=2020, value=2023)
        arrival_month = st.number_input('Bulan Kedatangan', min_value=1, max_value=12, value=1)
        arrival_date = st.number_input('Tanggal Kedatangan', min_value=1, max_value=31, value=1)
        no_of_previous_cancellations = st.number_input('Jumlah Pembatalan Sebelumnya', min_value=0, value=0)
        no_of_previous_bookings_not_canceled = st.number_input('Jumlah Pemesanan Sebelumnya yang Tidak Dibatalkan', min_value=0, value=0)
        avg_price_per_room = st.number_input('Harga Rata-rata Per Kamar', min_value=0.0, value=100.0)
        no_of_special_requests = st.number_input('Jumlah Permintaan Khusus', min_value=0, value=0)

        submit_button = st.form_submit_button(label='Prediksi Pembatalan')

        if submit_button:
            input_data = {
                'type_of_meal_plan': type_of_meal_plan,
                'room_type_reserved': room_type_reserved,
                'market_segment_type': market_segment_type,
                'no_of_adults': no_of_adults,
                'no_of_children': no_of_children,
                'no_of_weekend_nights': no_of_weekend_nights,
                'no_of_week_nights': no_of_week_nights,
                'lead_time': lead_time,
                'arrival_year': arrival_year,
                'arrival_month': arrival_month,
                'arrival_date': arrival_date,
                'no_of_previous_cancellations': no_of_previous_cancellations,
                'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
                'avg_price_per_room': avg_price_per_room,
                'no_of_special_requests': no_of_special_requests
            }

            result = predict_booking_status(input_data, model, encoder, scaler)

            if result == 1:
                st.success("Pembatalan Pemesanan Diprediksi (Canceled).")
            else:
                st.success("Pemesanannya Diprediksi Tidak Dibatalkan (Not Canceled).")

if __name__ == "__main__":
    main()
