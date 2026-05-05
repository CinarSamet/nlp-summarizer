import streamlit as st
import streamlit.components.v1 as components

# Diğer modüllerden importlar
from data_scraper import get_news
from nlp_preprocess import sentence_tokenize
from nlp_algorithms import textrank_summary, tfidf_summary, tfidf_summary_with_index, get_keywords, generate_headline, calculate_rouge
from visualizations import draw_interactive_graph, draw_interactive_overlay_graph, highlight_text, highlight_compare, build_tfidf_graph, draw_wordcloud
from file_utils import process_uploaded_file, read_dataset

st.set_page_config(page_title="NLP Summarizer", layout="wide") # Sayfayı genişletir

st.title("🧠 NLP News Summarization Dashboard")

st.sidebar.title("⚙️ Kontrol Paneli")

# MOD SEÇİMİ
st.sidebar.markdown("Çalışma Modu")
app_mode = st.sidebar.selectbox("Veri Kaynağı:", ["Manuel Metin", "Tekli Dosya Yükleme", "Veri Seti Yükleme", "Haber Dashboard"])

st.sidebar.markdown("---")

with st.sidebar.expander("⚙️ Özetleme Ayarları", expanded=False):
    algo_mode = st.selectbox("Algoritma Seçimi:", ["TextRank", "TF-IDF", "Both (Comparison)"])
    top_n = st.slider("Özet Uzunluğu (Cümle):", 1, 5, 2)



text = "" 

if app_mode == "Haber Dashboard":
    st.info("💡 Lütfen önce bir haber kaynağı, ardından özetlemek istediğiniz haberi seçin.")
    
    source = st.selectbox("🌐 Haber Kaynağı", ["BBC", "CNN", "TRT"])
    
    articles = get_news(source)
    if articles:
        titles = [a["title"] for a in articles]
        selected_title = st.selectbox("📰 Haber Seç", titles)
        text = next((a["text"] for a in articles if a["title"] == selected_title), "")

elif app_mode == "Tekli Dosya Yükleme":
    st.info("💡 Desteklenen formatlar: PDF, DOCX (Word), TXT")
    uploaded_file = st.file_uploader("Bir dosya yükleyin", type=["pdf", "docx", "txt"])
    
    if uploaded_file is not None:
        text = process_uploaded_file(uploaded_file)
        if text.strip():
            st.success(f"✅ {uploaded_file.name} başarıyla okundu! Aşağıdan özetleyebilirsiniz.")
            with st.expander("Yüklenen Metni İncele (İsteğe Bağlı)"):
                st.write(text)
        else:
            st.error("Dosya okunamadı veya içi boş.")

elif app_mode == "Veri Seti Yükleme":
    st.info("💡 Desteklenen formatlar: CSV, Excel (.xlsx, .xls)")
    uploaded_dataset = st.file_uploader("Bir Haber Veri Seti Yükleyin", type=["csv", "xlsx", "xls"])
    
    if uploaded_dataset is not None:
        df = read_dataset(uploaded_dataset)
        
        if df is not None and not df.empty:
            st.success(f"✅ Veri seti başarıyla yüklendi! (Toplam {len(df)} satır bulundu)")
            
            # Dinamik Sütun Eşleştirme 
            st.markdown("### ⚙️ Veri Seti Ayarları")
            col1, col2 = st.columns(2)
            
            # Sütunları listele
            columns = list(df.columns)
            
            title_col = col1.selectbox("Başlık Sütunu (Opsiyonel)", ["Seçmek İstemiyorum"] + columns)
            text_col = col2.selectbox("Haber Metni Sütunu (Zorunlu)", columns, index=len(columns)-1) 
            
            st.markdown("### 📰 İşlem Yapılacak Haberi Seçin")
            
            if title_col != "Seçmek İstemiyorum":
                # Başlığa göre haber seçimi
                titles = df[title_col].astype(str).tolist()
                selected_title = st.selectbox("Listeden Bir Haber Seçin:", titles)
                
                row_index = df[df[title_col] == selected_title].index[0]
                text = str(df.loc[row_index, text_col])
            else:
                # Sadece indeks numarasına göre satır seçimi
                selected_index = st.number_input("Özetlenecek Satır Numarası (Index):", min_value=0, max_value=len(df)-1, value=0)
                text = str(df.loc[selected_index, text_col])
                
            with st.expander("Seçilen Metni İncele"):
                st.write(text)
                
            # TOPLU ÖZETLEME VE İNDİRME KISMI 
            st.markdown("---")
            st.markdown("### 🚀 Toplu Özetleme ve Dışa Aktarma")
            st.info(f"Seçili olan '{algo_mode}' algoritması ile tüm satırları otomatik özetleyip bilgisayarınıza indirebilirsiniz.")
            
            if st.button("Tüm Veri Setini Özetle"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                summaries = []
                total_rows = len(df)
                
                for i in range(total_rows):
                    row_text = str(df.loc[i, text_col])
                    
                    if not row_text.strip() or len(row_text.split()) < 5:
                        summaries.append("Metin çok kısa veya boş.")
                    else:
                        try:
                            row_sentences = sentence_tokenize(row_text)
                            
                            if len(row_sentences) <= top_n:
                                summaries.append(row_text)
                            else:
                                if algo_mode == "TextRank":
                                    summary, _, _, _ = textrank_summary(row_sentences, top_n)
                                elif algo_mode == "TF-IDF":
                                    summary = tfidf_summary(row_sentences, top_n)
                                else:
                                    summary, _, _, _ = textrank_summary(row_sentences, top_n)
                                
                                summaries.append(summary)
                        except Exception:
                            summaries.append("Özetleme Sırasında Hata Oluştu")
                    
                    progress = (i + 1) / total_rows
                    progress_bar.progress(progress)
                    status_text.text(f"İşleniyor: {i + 1} / {total_rows} satır tamamlandı. Lütfen bekleyin...")
                
                df["Cikarilan_Ozet"] = summaries
                st.success("🎉 Tüm veri seti başarıyla özetlendi!")
                
                csv_data = df.to_csv(index=False).encode('utf-8-sig')
                
                st.download_button(
                    label="📥 Özetli Veri Setini İndir (CSV)",
                    data=csv_data,
                    file_name="ozetlenmis_haber_veri_seti.csv",
                    mime="text/csv",
                    help="Bu dosyayı doğrudan Excel ile açabilirsiniz."
                )
        else:
            st.error("Dosya okunamadı veya içi boş.")

else:
    text = st.text_area("Metni gir:", height=200)


# SESSION STATE 
if "is_summarized" not in st.session_state:
    st.session_state.is_summarized = False

if st.button("🚀 Seçili Metni Özetle"):
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
            with st.expander("🎨 Grafik Görünüm Ayarları (Tıklayıp Açabilirsiniz)"):
                g_col1, g_col2, g_col3 = st.columns(3)
                ui_font_size = g_col1.slider("Yazı Boyutu", 10, 80, 60)
                ui_base_size = g_col2.slider("Taban Düğüm Boyutu", 10, 80, 40)
                ui_multiplier = g_col3.slider("Skor Çarpanı", 100, 1000, 400)
            
            st.markdown("---")

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
            col1, col2, col3 = st.columns(3)
            col1.metric("Cümle Sayısı", len(sentences))
            col2.metric("Özet Cümle", top_n)
            col3.metric("Sıkıştırma Oranı", f"%{round(len(summary)/len(text)*100,2)}")

            st.markdown("---")
            st.subheader("🔑 Anahtar Kelimeler")
            st.write(", ".join(get_keywords(sentences)))

            st.subheader("📰 Başlık Önerisi")
            st.info(generate_headline(summary))

            st.markdown("---")
            st.subheader("☁️ Kelime Bulutu (Orijinal Metin)")
            # Kelime bulutu grafiğini çizdiriyoruz
            wc_fig = draw_wordcloud(text)
            st.pyplot(wc_fig)

            st.markdown("---")
            st.subheader("📈 ROUGE Başarı Metrikleri")
            st.caption("ROUGE skorları, çıkarılan özetin orijinal metin içerisindeki bilgi yoğunluğunu (örtüşmeyi) ölçer. 1.0'a ne kadar yakınsa o kadar çok anahtar bilgi tutulmuş demektir.")
            
            scores = calculate_rouge(text, summary)
            if scores:
                r1_col, r2_col, rl_col = st.columns(3)
                # F-Score (f) değerlerini gösteriyoruz
                r1_col.metric("ROUGE-1 (Tekli Kelime)", f"{scores['rouge-1']['f']:.2f}")
                r2_col.metric("ROUGE-2 (İkili Kelime)", f"{scores['rouge-2']['f']:.2f}")
                rl_col.metric("ROUGE-L (Uzun Cümle)", f"{scores['rouge-l']['f']:.2f}")

        with tab4:
            if algo_mode == "Both (Comparison)":
                st.markdown(highlight_compare(sentences, tr_selected, tf_selected), unsafe_allow_html=True)
                st.markdown("""
                🔴 TextRank  
                🔵 TF-IDF  
                🟣 Ortak
                """)
            elif algo_mode == "TextRank":
                st.markdown(highlight_text(sentences, selected), unsafe_allow_html=True)
            else:
                st.write(text)

st.markdown("---")
st.caption("🚀 NLP Summarization Project - Modular Version")