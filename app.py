import streamlit as st
import nltk
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import feedparser
import requests
from bs4 import BeautifulSoup
from readability import Document
import re
import snowballstemmer # <-- YENİ KÜTÜPHANE BURADA
from nltk.corpus import stopwords
from pyvis.network import Network
import streamlit.components.v1 as components

# DOWNLOADS

@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')

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



def highlight_text(sentences, selected):
    return " ".join(
        [f"<mark>{s}</mark>" if i in selected else s for i, s in enumerate(sentences)]
    )


def highlight_compare(sentences, tr_selected, tf_selected):
    result = ""
    for i, s in enumerate(sentences):
        if i in tr_selected and i in tf_selected:
            result += f"<mark style='background-color: violet'>{s}</mark> "
        elif i in tr_selected:
            result += f"<mark style='background-color: lightcoral'>{s}</mark> "
        elif i in tf_selected:
            result += f"<mark style='background-color: lightblue'>{s}</mark> "
        else:
            result += s + " "
    return result


def get_keywords(sentences, top_n=5):
    vectorizer = TfidfVectorizer(stop_words=list(turkish_stopwords))
    tfidf = vectorizer.fit_transform(sentences)

    scores = np.array(tfidf.sum(axis=0)).flatten()
    words = vectorizer.get_feature_names_out()

    ranked = sorted(zip(scores, words), reverse=True)
    return [w for _, w in ranked[:top_n]]


def generate_headline(summary):
    return summary.split(".")[0]


# NEWS

def get_full_article(url):
    try:
        r = requests.get(url, timeout=5)
        doc = Document(r.text)
        html = doc.summary()

        soup = BeautifulSoup(html, "html.parser")
        return " ".join([p.get_text() for p in soup.find_all("p")])
    except:
        return ""


def get_news(source="BBC"):
    feeds = {
        "BBC": "https://feeds.bbci.co.uk/turkce/rss.xml",
        "CNN": "http://rss.cnn.com/rss/edition.rss",
        "TRT": "https://www.trthaber.com/sondakika.rss"
    }

    feed = feedparser.parse(feeds.get(source))
    articles = []

    for e in feed.entries[:5]:
        text = get_full_article(e.link)

        if len(text) < 200:
            text = e.title + " " + e.get("description", "")

        articles.append({"title": e.title, "text": text})

    return articles


# NLP CORE

def sentence_tokenize(text):
    return [s for s in nltk.sent_tokenize(text) if len(s.split()) > 5]


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


# GRAPH HELPERS

def draw_interactive_graph(nx_graph, scores, sentences, selected=None, font_size=40, base_size=30, multiplier=300):
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black", cdn_resources="remote")
    net.barnes_hut() 
    
    net.set_options(f"""
    var options = {{
      "configure": {{
        "filter": ["physics"]
      }},
      "nodes": {{
        "font": {{
          "size": {font_size},
          "face": "Tahoma",
          "color": "#000000",
          "vadjust": 15
        }}
      }}
    }}
    """)
    
    for i in range(len(scores)):
        color = "#ff4b4b" if selected and i in selected else "#1f77b4"
        
        size = base_size + (float(scores[i]) * multiplier) 
        hover_text = f"Skor: {scores[i]:.3f}<br><br>{sentences[i]}"
        
        net.add_node(i, label=str(i), title=hover_text, color=color, size=size)
        
    for u, v, data in nx_graph.edges(data=True):
        net.add_edge(u, v, value=data.get('weight', 0.5))
        
    net.save_graph("graph.html")
    with open("graph.html", "r", encoding="utf-8") as f:
        return f.read()


def draw_interactive_overlay_graph(nx_graph, sentences, tr_selected, tf_selected, font_size=40, base_size=40):
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black", cdn_resources="remote")
    net.barnes_hut() 
    
    net.set_options(f"""
    var options = {{
      "configure": {{
        "filter": ["physics"]
      }},
      "nodes": {{
        "font": {{
          "size": {font_size},
          "face": "Tahoma",
          "color": "#000000",
          "vadjust": 15
        }}
      }}
    }}
    """)
    
    for i in range(len(sentences)):
        if i in tr_selected and i in tf_selected:
            color = "#8A2BE2"
            label_text = "Ortak Seçim"
        elif i in tr_selected:
            color = "#ff4b4b"
            label_text = "Sadece TextRank"
        elif i in tf_selected:
            color = "#1f77b4"
            label_text = "Sadece TF-IDF"
        else:
            color = "#cccccc"
            label_text = "Seçilmedi"
            
        size = base_size 
        hover_text = f"<b>{label_text}</b><br><br>{sentences[i]}"
        
        net.add_node(i, label=str(i), title=hover_text, color=color, size=size)
        
    for u, v, data in nx_graph.edges(data=True):
        net.add_edge(u, v, value=data.get('weight', 0.5))
        
    net.save_graph("overlay_graph.html")
    with open("overlay_graph.html", "r", encoding="utf-8") as f:
        return f.read()
    
def draw_graph(graph, scores, selected=None):
    plt.figure(figsize=(7, 5))
    pos = nx.spring_layout(graph, seed=42)

    if selected is None:
        node_colors = ["lightblue"] * len(scores)
    else:
        node_colors = ["red" if i in selected else "lightblue" for i in range(len(scores))]

    node_sizes = [scores[i] * 3000 for i in range(len(scores))]

    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_edges(graph, pos, alpha=0.3)
    nx.draw_networkx_labels(graph, pos)

    plt.title("Graph")
    plt.axis("off")
    return plt


def build_tfidf_graph(sentences, threshold=0.2):
    vec = TfidfVectorizer(tokenizer=turkish_tokenizer, token_pattern=None)
    tfidf = vec.fit_transform(sentences)

    sim = (tfidf * tfidf.T).toarray()

    # 🔥 weak connections remove
    sim[sim < threshold] = 0

    graph = nx.from_numpy_array(sim)
    scores = np.sum(sim, axis=1)

    return graph, scores


def draw_overlay_graph(sentences, tr_selected, tf_selected):
    vec = TfidfVectorizer(tokenizer=turkish_tokenizer, token_pattern=None)
    tfidf = vec.fit_transform(sentences)

    sim = (tfidf * tfidf.T).toarray()
    graph = nx.from_numpy_array(sim)

    pos = nx.spring_layout(graph, seed=42)

    plt.figure(figsize=(7, 5))

    colors = []
    for i in range(len(sentences)):
        if i in tr_selected and i in tf_selected:
            colors.append("purple")
        elif i in tr_selected:
            colors.append("red")
        elif i in tf_selected:
            colors.append("blue")
        else:
            colors.append("lightgray")

    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=1000)
    nx.draw_networkx_edges(graph, pos, alpha=0.2)
    nx.draw_networkx_labels(graph, pos)

    plt.title("Overlay Graph (TextRank vs TF-IDF)")
    plt.axis("off")
    return plt


# UI

st.title("🧠 NLP News Summarization Dashboard")

st.sidebar.title("Ayarlar")

app_mode = st.sidebar.selectbox("Mod", ["Manuel Metin", "Haber Dashboard"])
source = st.sidebar.selectbox("Kaynak", ["BBC", "CNN", "TRT"])
top_n = st.sidebar.slider("Özet Cümle Sayısı", 1, 5, 2)

# DİNAMİK GRAFİK AYARLARI 
st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Grafik Ayarları")

ui_font_size = st.sidebar.slider("Yazı Boyutu", 10, 80, 60)
ui_base_size = st.sidebar.slider("Taban Düğüm Boyutu", 10, 80, 40)
ui_multiplier = st.sidebar.slider("Skor Çarpanı (Düğüm Büyümesi)", 100, 1000, 400)
# ------------------------------------------

if app_mode == "Haber Dashboard":
    articles = get_news(source)
    titles = [a["title"] for a in articles]


    selected_title = st.selectbox("Haber seç", titles)
    text = next(a["text"] for a in articles if a["title"] == selected_title)
else:
    text = st.text_area("Metni gir:", height=200)

algo_mode = st.selectbox("Algoritma", ["TextRank", "TF-IDF", "Both (Comparison)"])


# SESSION STATE 

if "is_summarized" not in st.session_state:
    st.session_state.is_summarized = False

if st.button("🚀 Özetle"):
    st.session_state.is_summarized = True

# MAIN

if st.session_state.is_summarized:

    if not text.strip():
        st.warning("Lütfen özetlenecek bir metin girin.")
    else:
        sentences = sentence_tokenize(text)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📌 Summary", "🕸️ Graphs", "📊 Analysis", "📄 Text"]
        )

        with tab1:
            if algo_mode == "TextRank":
                summary, graph, scores, selected = textrank_summary(sentences, top_n)
                st.success(summary)

            elif algo_mode == "TF-IDF":
                summary = tfidf_summary(sentences, top_n)
                st.success(summary)
                selected = []

            else:
                tr_summary, graph, scores, tr_selected = textrank_summary(sentences, top_n)
                tf_summary, tf_selected = tfidf_summary_with_index(sentences, top_n)

                st.subheader("TextRank")
                st.success(tr_summary)

                st.subheader("TF-IDF")
                st.info(tf_summary)

                summary = tr_summary

        with tab2:
            if algo_mode == "TextRank":
                st.subheader("🕸️ İnteraktif TextRank Ağ Grafiği")
                st.info("Düğümleri (noktaları) fareyle sürükleyebilir, üzerine gelip cümleleri okuyabilirsiniz!")
                html_code = draw_interactive_graph(graph, scores, sentences, selected, ui_font_size, ui_base_size, ui_multiplier)
                components.html(html_code, height=1250)

            elif algo_mode == "TF-IDF":
                st.subheader("🕸️ İnteraktif TF-IDF Grafiği")
                st.info("Düğümleri sürükleyerek cümleler arası ilişkileri keşfedin.")
                g, s = build_tfidf_graph(sentences)
                html_code = draw_interactive_graph(g, s, sentences, None, ui_font_size, ui_base_size, ui_multiplier)
                components.html(html_code, height=1250)

            else:
                st.subheader("🕸️ İnteraktif Karşılaştırma (Overlay) Grafiği")
                st.markdown("""
                **Renk Kodları:**  
                🔴 Sadece TextRank | 🔵 Sadece TF-IDF | 🟣 İki Algoritmanın Ortak Seçimi
                """)
                g, _ = build_tfidf_graph(sentences) 
                html_code = draw_interactive_overlay_graph(g, sentences, tr_selected, tf_selected, ui_font_size, ui_base_size)
                components.html(html_code, height=1250)

        with tab3:
            st.metric("Cümle", len(sentences))
            st.metric("Özet", top_n)
            st.metric("Sıkıştırma", f"%{round(len(summary)/len(text)*100,2)}")

            st.subheader("Keywords")
            st.write(", ".join(get_keywords(sentences)))

            st.subheader("Headline")
            st.info(generate_headline(summary))

        with tab4:
            if algo_mode == "Both (Comparison)":
                st.markdown(highlight_compare(sentences, tr_selected, tf_selected),
                            unsafe_allow_html=True)
                st.markdown("""
                🔴 TextRank  
                🔵 TF-IDF  
                🟣 Ortak
                """)
            elif algo_mode == "TextRank":
                st.markdown(highlight_text(sentences, selected),
                            unsafe_allow_html=True)
            else:
                st.write(text)

st.markdown("---")
st.caption("🚀 NLP Summarization Project - Final Version")