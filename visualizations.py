import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from pyvis.network import Network
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nlp_preprocess import turkish_tokenizer, turkish_stopwords

from nlp_preprocess import turkish_tokenizer

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

def build_tfidf_graph(sentences, threshold=0.2):
    vec = TfidfVectorizer(tokenizer=turkish_tokenizer, token_pattern=None)
    tfidf = vec.fit_transform(sentences)

    sim = (tfidf * tfidf.T).toarray()
    sim[sim < threshold] = 0

    graph = nx.from_numpy_array(sim)
    scores = np.sum(sim, axis=1)

    return graph, scores

def draw_wordcloud(text):
    """
    Metindeki kelime frekanslarına göre görsel bir kelime bulutu oluşturur.
    """
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        colormap='viridis',
        max_words=100,
        stopwords=turkish_stopwords
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig