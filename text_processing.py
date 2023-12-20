import nltk
import re
import json
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary

# Membaca kamus slangword dari file
with open("slangwords_dict.txt", "r") as file:
    kamus_slangword = json.load(file)

# Ambil Stopword bawaan
stop_factory = StopWordRemoverFactory().get_stop_words()
more_stopword = ['dengan', 'bahwa', 'ia', 'oleh']

# Merge stopword
new_stopword = stop_factory + more_stopword

dictionary = ArrayDictionary(new_stopword)
stopword = StopWordRemover(dictionary)

def process_text(review):
    review = review.lower()
    review = re.sub(r'http\S+', '', review)
    # Remove @username
    review = re.sub('@[^\s]+', '', review)
    # Remove #tagger
    review = re.sub(r'#([^\s]+)', '', review)
    # Remove angka termasuk angka yang berada dalam string
    # Remove non ASCII chars
    review = re.sub(r'[^\x00-\x7f]', r'', review)
    review = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', review)
    review = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", review)
    review = re.sub(r'\\u\w\w\w\w', '', review)
    # Remove simbol, angka dan karakter aneh
    review = re.sub(r"[.,:;+!\-_<^/=?\"'\(\)\d\*]", " ", review)
    # Pattern to look for three or more repetitions of any character, including newlines (contoh goool -> gool).
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    review = pattern.sub(r"\1\1", review)

    while '  ' in review:
        review = review.replace('  ', ' ')

    review = review.split(' ')
    content = []
    for kata in review:
        if kata in kamus_slangword:
            kata = kamus_slangword[kata]
        content.append(kata)
    review = ' '.join(content)

    # stopwords = open(stopwords_Reduced.txt', 'r').read().split()
    # content = []
    # filteredtext = [word for word in review.split() if word not in stopwords]
    # content.append(" ".join(filteredtext))
    # review = content
    review = stopword.remove(review)
    token = nltk.word_tokenize(review)
    return token
