import joblib
import matplotlib.pyplot as plt
import numpy as np
from text_processing import process_text
import hydralit_components as hc
import streamlit as st

def home():
    st.title("Home")
    # Tambahkan konten untuk halaman home di sini

def predict_news():
    st.title("Prediksi Berita")
    st.write("Masukkan teks berita di bawah ini untuk mendeteksi apakah berita tersebut palsu atau tidak.")

    # Input text box
    input_text = st.text_area("Teks Berita", "")

    # Detect button
    if st.button("Deteksi"):
        if input_text:
            prediction, prob_fake, prob_true = predict_hoax(input_text)
            st.write("Prediksi:", prediction)
            st.write("Probabilitas Hoax:", prob_fake, "%")
            st.write("Probabilitas Valid:", prob_true, "%")
            
             # Create a pie chart for the probabilities
            labels = ['Hoax', 'Valid']
            probabilities = [prob_fake, prob_true]

            fig, ax = plt.subplots()
            ax.pie(probabilities, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

            st.pyplot(fig)  # Display the plot in Streamlit
        else:
            st.write("Mohon masukkan teks berita terlebih dahulu.")

def about():
    st.title("About")
    # Tambahkan konten untuk halaman about di sini

def data():
    st.title("Data")
    # Tambahkan konten untuk halaman data di sini

def main():
    st.sidebar.title("Menu")
    menu_options = ["Home", "Prediksi Berita", "About", "Data"]
    selected_menu = st.sidebar.selectbox("Pilih halaman", menu_options)

    if selected_menu == "Home":
        home()
    elif selected_menu == "Prediksi Berita":
        predict_news()
    elif selected_menu == "About":
        about()
    elif selected_menu == "Data":
        data()

if __name__ == '__main__':
    main()