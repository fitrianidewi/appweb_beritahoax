import joblib
import matplotlib.pyplot as plt
import numpy as np
from text_processing import process_text  
import hydralit_components as hc
import streamlit as st
from wordcloud import WordCloud
import pandas as pd

# Load your machine learning models
model_svm = 'hoaxsvm.pkl'
model_rf = 'hoaxrf.pkl'

def predict_hoax_svm(text):
    prediction = model_svm.predict([text])[0]
    decision_function_values = model_svm.decision_function([text])[0]
    
    # Menggunakan fungsi sigmoid untuk mendapatkan nilai probabilitas antara 0 dan 1
    prob_fake = 1 / (1 + np.exp(-decision_function_values))
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
    st.write("Berita hoax adalah informasi palsu atau tidak benar yang disajikan sebagai berita nyata dengan tujuan untuk menyesatkan pembaca atau pendengar. Biasanya, berita hoax dibuat dan disebarkan secara sengaja dengan maksud tertentu, seperti memengaruhi opini publik, menciptakan ketegangan sosial, atau mendapatkan keuntungan pribadi.")

def predict_news():
    st.title("Prediksi Berita")
    st.write("Masukkan teks berita di bawah ini untuk mendeteksi apakah berita tersebut palsu atau tidak.")

    # Input text box
    input_text = st.text_area("Teks Berita", "")

    if st.button("Deteksi"):
        if input_text:
            # Predictions for SVM model
            prediction_svm, prob_fake_svm, prob_true_svm = predict_hoax_svm(input_text)

            # Predictions for Random Forest model
            prediction_rf, prob_fake_rf, prob_true_rf = predict_hoax_rf(input_text)

            # Menampilkan hasil dalam dua kolom
            col1, col2 = st.columns(2)

            with col1:
                st.write("### SVM Model:")
                #st.write("Prediksi:", prediction_svm)
                st.write("Nilai Keputusan (Decision Function):", prob_fake_svm)

            with col2:
                st.write("### Random Forest Model:")
                #st.write("Prediksi:", prediction_rf)
                st.write("Probabilitas Hoax:", prob_fake_rf)
                st.write("Probabilitas Valid:", prob_true_rf)

            # Menentukan hasil akhir berdasarkan probabilitas
            final_result_svm = "Fake" if prob_fake_svm > prob_true_svm else "Valid"
            final_result_rf = "Fake" if prob_fake_rf > prob_true_rf else "Valid"

            st.write("### Hasil Akhir:")
            st.write("SVM Model:", final_result_svm)
            st.write("Random Forest Model:", final_result_rf)
            
            # Visualisasi probabilitas
            labels = ['Hoax (SVM)', 'Valid (SVM)', 'Hoax (RF)', 'Valid (RF)']
            probabilities = [prob_fake_svm, prob_true_svm, prob_fake_rf, prob_true_rf]

            fig, ax = plt.subplots()
            bars = ax.bar(labels, probabilities, color=['red', 'green', 'blue', 'yellow'])

            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

            plt.ylabel('Probabilitas')
            plt.title('Probabilitas Prediksi')
            st.pyplot(fig)
            
            # Word cloud
            st.write("### Word Cloud:")
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(input_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
        else:
            st.write("Mohon masukkan teks berita terlebih dahulu.")

def about():
    st.title("About")
    st.write("Web aplikasi ini dibuat untuk menjadi Tugas Akhir untuk mata kuliah Aplikasi Web")

def data():
    st.title("Data")
    file_path = "Data_latih.csv"
    df = pd.read_csv(file_path)
    st.write(df)
    
    # Menyusun ulang dataset untuk membuatnya seimbang
    false_news = df[df['label'] == 1].sample(frac=1)
    true_fact = df[df['label'] == 0]
    balanced_df = pd.concat([true_fact, false_news[:len(true_fact) + 200]])
    
    # Menampilkan dataset yang telah disusun ulang
    st.title("Dataset Setelah Disusun Ulang")
    st.write(balanced_df)
    
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
