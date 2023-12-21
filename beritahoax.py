import nltk
import joblib
import matplotlib.pyplot as plt
import numpy as np
from text_processing import process_text  
import hydralit_components as hc
import streamlit as st
from wordcloud import WordCloud
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('punkt')
from streamlit_lottie import st_lottie


st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# Load your machine learning models
model_svm = 'cobahoaxsvm.pkl'
model_rf = 'cobahoaxrf.pkl'

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


def process_dataframe(df):
    # Proses teks di dalam DataFrame
    df['processed_text'] = df['judul'].apply(process_text)
    return df

def most_common_words(text_series, title):
    # Menggabungkan semua teks dalam satu string
    all_text = ' '.join(text_series.astype(str))

    # Tokenisasi dan hitung frekuensi kata
    tokens = process_text(all_text)
    frequency = nltk.FreqDist(tokens)

    # Membuat DataFrame untuk kata-kata yang paling sering muncul
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()), "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns="Frequency", n=20)

    # Menampilkan visualisasi kata-kata yang sering muncul
    st.subheader(f"Kata-kata yang Sering Muncul ({title})")
    fig, ax = plt.subplots()
    ax.bar(df_frequency["Word"], df_frequency["Frequency"], color='blue')
    plt.xticks(rotation='vertical')
    plt.xlabel('Kata')
    plt.ylabel('Frekuensi')
    plt.title(f'Kata-kata yang Sering Muncul ({title})')
    st.pyplot(fig)

def generate_wordcloud(text_series, title):
    # Menggabungkan semua teks dalam satu string
    all_text = ' '.join(text_series.astype(str))

    # Membuat dan menampilkan WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'WordCloud ({title})')
    st.pyplot(plt)

    

def home():
    col1, col2 = st.columns([5,4])
    with col1:
        title = '''<br>
        <span style='color: black; font-weight: 700'>Hoax News Detector '''
        st.markdown(f"<h1 style='text-align: left; font-size: 60px; font-weight: 900;'>{title}</h1>", unsafe_allow_html=True)
        sub_title = '''Accelerate your insight into misinformation with our cutting-edge hoax detection web tool. Uncover the power of effortless data analysis through Streamlit. Your go-to solution for transforming news data into vibrant visualizations, effortlessly accessible and user-friendly. Elevate your understanding and stay one step ahead in the fight against fake news.'''
        st.markdown(f"<p style='text-align: justify; font-size: 18px; color: #555555;margin-bottom:310px'>{sub_title}</p>", unsafe_allow_html=True)


    with col2:
        st_lottie("https://lottie.host/f6c4d923-c06f-4a68-92b9-ad63c1b09683/msJtvE0eeF.json")
    
    st.write("Berita hoax adalah informasi palsu atau tidak benar yang disajikan sebagai berita nyata dengan tujuan untuk menyesatkan pembaca atau pendengar. Biasanya, berita hoax dibuat dan disebarkan secara sengaja dengan maksud tertentu, seperti memengaruhi opini publik, menciptakan ketegangan sosial, atau mendapatkan keuntungan pribadi.")
    
    st.write("Di dunia informasi yang semakin kompleks dan terhubung, penyebaran berita hoax menjadi ancaman serius. Berita hoax adalah informasi palsu atau tidak benar yang disajikan sebagai berita yang sebenarnya, dengan tujuan untuk menyesatkan pembaca atau pendengar. Praktik ini biasanya dilakukan dengan sengaja, dengan motivasi seperti mempengaruhi opini publik, menciptakan ketegangan sosial, atau bahkan untuk keuntungan pribadi.")
    st.write("Berita hoax sangat merugikan masyarakat, karena dapat menyebabkan kebingungan, mempengaruhi pengambilan keputusan yang salah, dan memicu konflik yang tidak perlu. Oleh karena itu, penting bagi kita semua untuk menjadi konsumen berita yang cerdas dan kritis. Kita perlu memverifikasi informasi sebelum mempercayainya, menggunakan sumber berita yang terpercaya, dan berbagi informasi yang akurat kepada orang lain.")
    st.write("Dalam era informasi digital, kecerdasan kita dalam memilah berita menjadi kunci untuk menjaga integritas informasi dan membangun masyarakat yang berdasarkan pada pengetahuan yang benar.")
    
    
    st.error('WARNING : Streamlit ini masih belum sempurna, termasuk dalam pendeteksian berita')

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
            
        else:
            st.write("Mohon masukkan teks berita terlebih dahulu.")

def about():
    col1, col2 = st.columns([5,4])
    with col1:
        st.title("About")
        st.write("Web aplikasi ini dibuat untuk menjadi Tugas Akhir untuk mata kuliah Aplikasi Web")
    
        # Add your name, NIM, and department
        name = "Fitriani Dewi"
        nim = "21537141020"
        department = "Teknologi Informasi"
    
        # Display personal information
        st.subheader(":woman: Personal Information:")
        st.write(f"Nama: {name}")
        st.write(f"NIM: {nim}")
        st.write(f"Jurusan: {department}")
    with col2:
        st_lottie("https://lottie.host/ad4a4d21-39c6-4114-86fd-37f6b77f25d7/ePlOVVkNyA.json")
    

def data():
    st.title("Data")
    file_path = "Data_latih.csv"
    df = pd.read_csv(file_path)
    st.write(df)

    # Proses DataFrame
    df = process_dataframe(df)

    # Menyusun ulang dataset untuk membuatnya seimbang
    false_news = df[df['label'] == 1].sample(frac=1)
    true_fact = df[df['label'] == 0]
    balanced_df = pd.concat([true_fact, false_news[:len(true_fact) + 200]])

    show_balanced_df = st.checkbox("Tampilkan Dataset Setelah Disusun Ulang")
    if show_balanced_df:
        st.title("Dataset Setelah Disusun Ulang")
        st.write(balanced_df)

    # Checkbox untuk menampilkan WordCloud
    show_wordcloud = st.checkbox("Tampilkan WordCloud")
    if show_wordcloud:
        st.title("WordCloud:")
        st.subheader("WordCloud Data Fake")
        generate_wordcloud(false_news['judul'], "WordCloud Data Fake")

        st.subheader("WordCloud Data Truth")
        generate_wordcloud(true_fact['judul'], "WordCloud Data Valid")

    # Checkbox untuk menampilkan kata-kata yang sering muncul
    show_most_common_words = st.checkbox("Tampilkan Kata-kata yang Sering Muncul")
    if show_most_common_words:
        most_common_words(false_news['judul'], "Data Fake")
        most_common_words(true_fact['judul'], "Data Truth")

    
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
