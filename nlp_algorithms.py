import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge import Rouge
from nlp_preprocess import turkish_tokenizer, turkish_stopwords

def get_keywords(sentences, top_n=5):
    vectorizer = TfidfVectorizer(stop_words=list(turkish_stopwords))
    tfidf = vectorizer.fit_transform(sentences)

    scores = np.array(tfidf.sum(axis=0)).flatten()
    words = vectorizer.get_feature_names_out()

    ranked = sorted(zip(scores, words), reverse=True)
    return [w for _, w in ranked[:top_n]]

def generate_headline(summary):
    return summary.split(".")[0]

def tfidf_summary(sentences, top_n):
    vec = TfidfVectorizer(tokenizer=turkish_tokenizer, token_pattern=None)
    tfidf = vec.fit_transform(sentences)

    scores = np.array(tfidf.sum(axis=1)).flatten()
    ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    return " ".join([s for _, s in ranked[:top_n]])

def tfidf_summary_with_index(sentences, top_n):
    vec = TfidfVectorizer(tokenizer=turkish_tokenizer, token_pattern=None)
    tfidf = vec.fit_transform(sentences)

    scores = np.array(tfidf.sum(axis=1)).flatten()
    ranked = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)

    selected = [i for _, i, _ in ranked[:top_n]]
    summary = " ".join([s for _, _, s in ranked[:top_n]])

    return summary, selected

def textrank_summary(sentences, top_n):
    vec = TfidfVectorizer(tokenizer=turkish_tokenizer, token_pattern=None)
    tfidf = vec.fit_transform(sentences)

    matrix = (tfidf * tfidf.T).toarray()
    graph = nx.from_numpy_array(matrix)
    scores = nx.pagerank(graph)

    ranked = sorted(((scores[i], s, i) for i, s in enumerate(sentences)), reverse=True)

    summary = " ".join([s for _, s, _ in ranked[:top_n]])
    selected = [i for _, _, i in ranked[:top_n]]

    return summary, graph, scores, selected

def calculate_rouge(reference, summary):
    if not summary.strip() or not reference.strip():
        return None
    
    rouge = Rouge()
    scores = rouge.get_scores(summary, reference)
    return scores[0]