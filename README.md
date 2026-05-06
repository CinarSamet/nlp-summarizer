# 🧠 NLP Text Summarizer

Bu proje, metinleri otomatik olarak özetleyen **extractive (çıkarımsal) doğal dil işleme (NLP)** uygulamasıdır.  
TextRank ve TF-IDF algoritmalarını kullanarak metindeki en önemli cümleleri seçer ve anlamlı bir özet oluşturur.

---

## 🚀 Özellikler

- ✂️ Extractive Text Summarization (çıkarımsal özetleme)
- 🧠 TextRank algoritması
- 📊 TF-IDF tabanlı özetleme
- 🇹🇷 Türkçe dil desteği
- 🌐 Haber çekme (RSS feed üzerinden)
- 📈 Grafik ve analiz ekranı
- 🎯 Anahtar kelime çıkarımı
- 🖍️ Özet karşılaştırma (TextRank vs TF-IDF)
- 💡 Tamamen klasik NLP (API / LLM bağımsız)

---

## 🧩 Kullanılan Teknolojiler

- Python  
- Streamlit  
- NLTK  
- Scikit-learn  
- NetworkX  
- Matplotlib  
- BeautifulSoup  
- Feedparser  

---

## 🏗️ Nasıl Çalışır?

Bu proje **extractive summarization** yaklaşımını kullanır.  
Yani yeni cümle üretmez, metin içerisinden en önemli cümleleri seçer.

- **TF-IDF**: Kelime önemine göre cümleleri skorlar  
- **TextRank**: Cümleler arası benzerlik grafı oluşturur ve PageRank uygular  

---

## ⚙️ Kurulum

```bash
git clone https://github.com/CinarSamet/nlp-summarizer.git
cd nlp-summarizer

python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

---

## ▶️ Çalıştırma

```bash
streamlit run app.py
```

Tarayıcıda aç:
```
http://localhost:8501
```

---

## 📊 Uygulama Bölümleri

### 📝 Metin Özeti
- Kullanıcıdan alınan metni özetler  
- TextRank ve TF-IDF sonuçlarını karşılaştırır  

### 📰 News Dashboard
- RSS kaynaklarından haber çeker  
- Haberleri otomatik özetler  

### 📈 Analiz
- Kelime frekans grafikleri  
- Cümle skorları  
- Önemli kelimeler  

---

## 🔍 Örnek Kullanım

```python
from summarizer import summarize_text

text = "Uzun bir metin buraya yazılır..."
summary = summarize_text(text, method="textrank")

print(summary)
```

---

## 📂 Proje Yapısı

```
nlp-summarizer/
│
├── app.py
├── data_scraper.py
├── file_utils.py
├── nlp_algorithms.py
├── nlp_preprocess.py
├── requirements.txt
└── visualizations.py
```

---

## 🧪 Geliştirme Fikirleri

- 🔥 Chrome Extension (seçili metni anlık özetleme)
- 🤖 LLM destekli hybrid summarization
- 🌍 Çoklu dil desteği
- 📄 PDF / DOCX desteği
- ⚡ Real-time summarization API

---

## 📌 Notlar

- Bu proje tamamen **klasik NLP yöntemleri** kullanır  
- Harici API veya ücretli servis bağımlılığı yoktur  
- Eğitim amaçlı ve geliştirilebilir yapıdadır  

---

## 👨‍💻 Geliştirici

**Samet Çınar**  
Computer Engineering Student  

---

## 📜 Lisans

MIT License
