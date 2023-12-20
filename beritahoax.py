import joblib
import matplotlib.pyplot as plt
import numpy as np
from text_processing import process_text  
import hydralit_components as hc
import streamlit as st
from wordcloud import WordCloud
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

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

def generate_wordcloud(data, title):
    all_words = ' '.join(data)
    wordcloud = WordCloud(width=800, height=500, max_font_size=110, collocations=False).generate(all_words)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(title)
    st.pyplot(fig)
    
def most_common_words(data, title, num_words=10):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data)
    feature_names = vectorizer.get_feature_names_out()

    word_counts = X.sum(axis=0).A1
    word_counts_df = pd.DataFrame({'Word': feature_names, 'Count': word_counts})
    sorted_word_counts = word_counts_df.sort_values(by='Count', ascending=False).head(num_words)

    st.title(f"Most Common Words in {title}:")
    st.write(sorted_word_counts)
    
    # Visualisasi kata-kata yang sering muncul
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(sorted_word_counts['Word'], sorted_word_counts['Count'], color='skyblue')
    ax.set_xticklabels(sorted_word_counts['Word'], rotation=45, ha='right')
    ax.set_xlabel('Word')
    ax.set_ylabel('Count')
    ax.set_title(f"Top {num_words} Most Common Words in {title}")
    st.pyplot(fig)

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
    
    show_balanced_df = st.checkbox("Tampilkan Dataset Setelah Disusun Ulang")
    if show_balanced_df:
        st.title("Dataset Setelah Disusun Ulang")
        st.write(balanced_df)
        
    show_wordcloud = st.checkbox("Tampilkan WordCloud")
    if show_wordcloud:
        st.title("WordCloud:")
        st.subheader("WordCloud Data Fake")
        generate_wordcloud(false_news['judul'], "WordCloud Data Fake")
        
        st.subheader("WordCloud Data Truth")
        generate_wordcloud(true_fact['judul'], "WordCloud Data Valid")
        
        
    # Checkbox untuk menampilkan WordCloud
    show_wordcloud = st.checkbox("Tampilkan Kata-kata yang Sering Muncul")
    if show_wordcloud:
        # Analisis kata-kata umum
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
