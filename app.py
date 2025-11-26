import streamlit as st
import pandas as pd
import pickle

# ---- Load full pipeline model (preprocessor + PCA + model) ----
try:
    with open('full_model.pkl', 'rb') as file:
        full_model = pickle.load(file)
    st.success("Model berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Streamlit title
st.title('Prediksi Harga Rumah Jabodetabek')
st.write('Aplikasi untuk memprediksi harga rumah berdasarkan properti.')
st.write('**Catatan:** Hasil bersifat estimasi dan bukan acuan harga pasti.')

# Sidebar Input
st.sidebar.header('Input Properti Rumah')

def user_input_features():
    bedrooms = st.sidebar.slider('Jumlah Kamar Tidur', 1, 8, 3)
    bathrooms = st.sidebar.slider('Jumlah Kamar Mandi', 1, 4, 2)
    land_size_m2 = st.sidebar.slider('Luas Tanah (m2)', 10.0, 400.0, 100.0)
    building_size_m2 = st.sidebar.slider('Luas Bangunan (m2)', 10.0, 400.0, 90.0)
    carports = st.sidebar.slider('Jumlah Carport', 0, 3, 1)
    floors = st.sidebar.slider('Jumlah Lantai', 1, 3, 2)
    building_age = st.sidebar.slider('Usia Bangunan (tahun)', 0, 15, 0)
    garages = st.sidebar.slider('Jumlah Garasi', 0, 2, 0)

    city_options = ['Bekasi', 'Bogor', 'Depok', 'Jakarta Barat', 'Jakarta Pusat',
                    'Jakarta Selatan', 'Jakarta Timur', 'Jakarta Utara', 'Tangerang']
    furnishing_options = ['unfurnished', 'semi furnished', 'furnished']

    city = st.sidebar.selectbox('Kota', city_options)
    furnishing = st.sidebar.selectbox('Perabotan', furnishing_options)

    data = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'land_size_m2': land_size_m2,
        'building_size_m2': building_size_m2,
        'carports': carports,
        'floors': floors,
        'building_age': building_age,
        'garages': garages,
        'city': city,
        'furnishing': furnishing
    }
    return pd.DataFrame(data, index=[0])

# Reset button
if st.sidebar.button('Reset Input'):
    st.experimental_rerun()

df_input = user_input_features()

st.subheader('Parameter Input Pengguna:')
st.write(df_input)

# Validasi input
def validate_input(df):
    errors = []
    if df['building_size_m2'].iloc[0] > df['land_size_m2'].iloc[0]:
        errors.append("Luas bangunan tidak boleh lebih besar dari luas tanah.")
    return errors

# Predict button
if st.sidebar.button('Prediksi Harga'):
    validation_errors = validate_input(df_input)
    if validation_errors:
        for err in validation_errors:
            st.error(err)
        st.stop()

    try:
        # langsung prediksi dari full pipeline
        prediction = full_model.predict(df_input)[0]

        st.subheader("Hasil Prediksi Harga Rumah:")
        st.write(f"**Rp {prediction:,.2f}**")
        st.info("Prediksi ini berbasis machine learning dan hanya untuk referensi.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
