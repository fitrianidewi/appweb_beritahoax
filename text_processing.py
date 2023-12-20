import nltk
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary

# Kamus slangword
kamus_slangword = {
    'kpn': 'kapan',
    'gmn': 'bagaimana',
    # Tambahkan kamus slangword lainnya sesuai kebutuhan
}

# Daftar stopwords tambahan
more_stopword = ['dengan', 'bahwa', 'ia', 'oleh']

# Ambil Stopword bawaan
stop_factory = StopWordRemoverFactory().get_stop_words()

# Gabungkan stopword
new_stopword = stop_factory + more_stopword

dictionary = ArrayDictionary(new_stopword)
stopword = StopWordRemover(dictionary)

def process_text(review):
    review = review.lower()
    review = re.sub(r'http\S+', '', review)
    review = re.sub('@[^\s]+', '', review)
    review = re.sub(r'#([^\s]+)', '', review)
    review = re.sub(r'[^\x00-\x7f]', r'', review)
    review = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', review)
    review = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", review)
    review = re.sub(r'\\u\w\w\w\w', '', review)
    review = re.sub(r"[.,:;+!\-_<^/=?\"'\(\)\d\*]", " ", review)
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

    review = stopword.remove(review)
    token = nltk.word_tokenize(review)
    return token
