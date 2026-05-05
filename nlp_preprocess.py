import streamlit as st
import nltk
import re
import snowballstemmer
from nltk.corpus import stopwords

@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)

download_nltk_data()

turkish_stopwords = set(stopwords.words('turkish'))
stemmer = snowballstemmer.stemmer('turkish')

def turkish_tokenizer(text):
    text = text.lower()
    text = re.sub(r'[^a-zçğıöşü\s]', '', text)
    tokens = text.split()
    # Stemming
    stems = [stemmer.stemWord(word) for word in tokens if word not in turkish_stopwords]
    return stems

def sentence_tokenize(text):
    return [s for s in nltk.sent_tokenize(text) if len(s.split()) > 5]