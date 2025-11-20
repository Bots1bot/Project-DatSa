
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# --- Load the trained model and preprocessors ---
# Muat model, preprocessor, dan PCA dari file pickle
try:
    with open('gradient_boosting_regressor_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('preprocessor.pkl', 'rb') as file:
        preprocessor = pickle.load(file)
    with open('pca_transformer.pkl', 'rb') as file:
        pca_transformer = pickle.load(file)
    st.success("Model dan preprocessor berhasil dimuat!")  # Feedback sukses
except Exception as e:
    st.error(f"Error loading model or preprocessors: {e}")
    st.stop()

# Streamlit app title
st.title('Prediksi Harga Rumah Jabodetabek')
st.write('Aplikasi untuk memprediksi harga rumah berdasarkan properti yang diberikan.')
st.write('**Disclaimer**: Prediksi ini adalah estimasi berdasarkan model machine learning dan bukan nilai pasti. Gunakan sebagai referensi saja.')

# Sidebar for user inputs
st.sidebar.header('Input Properti Rumah')

def user_input_features():
    # Sliders untuk fitur numerik
    bedrooms = st.sidebar.slider('Jumlah Kamar Tidur', 1, 8, 3)
    bathrooms = st.sidebar.slider('Jumlah Kamar Mandi', 1, 4, 2)
    land_size_m2 = st.sidebar.slider('Luas Tanah (m2)', 10.0, 400.0, 100.0)
    building_size_m2 = st.sidebar.slider('Luas Bangunan (m2)', 10.0, 400.0, 90.0)
    carports = st.sidebar.slider('Jumlah Carport', 0, 3, 1)
    floors = st.sidebar.slider('Jumlah Lantai', 1, 3, 2)
    building_age = st.sidebar.slider('Usia Bangunan (tahun)', 0, 15, 0)
    garages = st.sidebar.slider('Jumlah Garasi', 0, 2, 0)

    # Opsi kategorikal - hardcoded untuk deployment sederhana
    # Catatan: Jika dataset training berubah, update list ini secara manual atau muat dari file
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
    features = pd.DataFrame(data, index=[0])
    return features

# Tombol reset untuk mengatur ulang input
if st.sidebar.button('Reset Input'):
    st.experimental_rerun()  # Reload app untuk reset slider/selectbox

df_input_original = user_input_features()

st.subheader('Parameter Input Pengguna:')
st.write(df_input_original)

# Fungsi validasi input
def validate_input(df):
    errors = []
    if df['bedrooms'].iloc[0] < 1:
        errors.append("Jumlah kamar tidur minimal 1.")
    if df['building_size_m2'].iloc[0] > df['land_size_m2'].iloc[0]:
        errors.append("Luas bangunan tidak boleh lebih besar dari luas tanah.")
    if df['land_size_m2'].iloc[0] < 10 or df['building_size_m2'].iloc[0] < 10:
        errors.append("Luas tanah dan bangunan minimal 10 m².")
    return errors

# Make prediction
if st.sidebar.button('Prediksi Harga'):
    # Validasi input terlebih dahulu
    validation_errors = validate_input(df_input_original)
    if validation_errors:
        for error in validation_errors:
            st.error(error)
        st.stop()  # Cegah prediksi jika ada error
    
    try:
        # Pastikan urutan kolom sesuai dengan yang diharapkan oleh preprocessor
        expected_original_features_order = [
            'bedrooms', 'bathrooms', 'land_size_m2', 'building_size_m2',
            'carports', 'floors', 'building_age', 'garages', 'city', 'furnishing'
        ]
        
        # Reorder DataFrame untuk memastikan konsistensi
        input_data_for_preprocessing = df_input_original[expected_original_features_order]
        
        # Transformasi menggunakan preprocessor
        transformed_input = preprocessor.transform(input_data_for_preprocessing)
        
        # Terapkan PCA
        pca_transformed_input = pca_transformer.transform(transformed_input)
        
        # Prediksi harga
        prediction = model.predict(pca_transformed_input)
        
        # Tampilkan hasil
        st.subheader('Hasil Prediksi Harga Rumah:')
        st.write(f"Harga Diprediksi: Rp {prediction[0]:,.2f}")
        st.info("Prediksi ini berdasarkan data historis. Konsultasikan dengan ahli properti untuk nilai akurat.")
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        st.exception(e)  # Tampilkan traceback untuk debugging

