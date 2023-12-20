import joblib
import matplotlib.pyplot as plt
import numpy as np
from text_processing import process_text  # Assuming this module contains your text processing functions
import hydralit_components as hc
import streamlit as st

# Load your machine learning models
model_svm = 'hoaxsvm.pkl'
model_rf = 'hoaxrf.pkl'

def predict_hoax_svm(text):
    prediction = model_svm.predict([text])[0]
    probabilities = model_svm.decision_function([text])[0]
    prob_fake = probabilities  # Assuming it's a decision function output
    prob_true = 1 - prob_fake

    return prediction, prob_fake, prob_true

def predict_hoax_rf(text):
    prediction = model_rf.predict([text])[0]
    probabilities = model_rf.predict_proba([text])[0]
    prob_fake = probabilities[0]
    prob_true = probabilities[1]

    return prediction, prob_fake, prob_true

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
            # Predictions for SVM model
            prediction_svm, prob_fake_svm, prob_true_svm = predict_hoax_svm(input_text)

            # Predictions for Random Forest model
            prediction_rf, prob_fake_rf, prob_true_rf = predict_hoax_rf(input_text)

            st.write("### SVM Model:")
            st.write("Prediksi:", prediction_svm)
            st.write("Nilai Keputusan (Decision Function):", prob_fake_svm)

            st.write("### Random Forest Model:")
            st.write("Prediksi:", prediction_rf)
            st.write("Probabilitas Hoax:", prob_fake_rf)
            st.write("Probabilitas Valid:", prob_true_rf)

            # Create pie charts for both models
            labels_svm = ['Hoax', 'Valid']
            probabilities_svm = [prob_fake_svm, prob_true_svm]

            labels_rf = ['Hoax', 'Valid']
            probabilities_rf = [prob_fake_rf, prob_true_rf]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

            ax1.pie(probabilities_svm, labels=labels_svm, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax1.set_title('SVM Model')

            ax2.pie(probabilities_rf, labels=labels_rf, autopct='%1.1f%%', startangle=90)
            ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            ax2.set_title('Random Forest Model')

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
