import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_stop_words_from_text(word_list, stopwords):
    out = [word for word in word_list if word not in stopwords]
    return ' '.join(out)


def calc_tf_idf(corpus, stopwords_l=None, top_n=20, ngram_range=(1, 2)):
    corpus_cleaned = [remove_punctuation(doc.lower()) for doc in corpus]
    if stopwords_l:
        corpus_cleaned = [remove_stop_words_from_text(doc.split(), stopwords_l) for doc in corpus_cleaned]
        vectorizer = TfidfVectorizer(stop_words=stopwords_l, lowercase=True, ngram_range=ngram_range)
    else:
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=ngram_range)
    tfidf_m = vectorizer.fit_transform(corpus_cleaned)
    feature_array = vectorizer.get_feature_names_out()
    tfidf_sorting = np.argsort(tfidf_m.toarray()).flatten()[::-1]
    top = feature_array[tfidf_sorting].tolist()[:top_n]
    print(top)
    return top
