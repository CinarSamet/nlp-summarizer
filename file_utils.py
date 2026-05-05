import io
from pypdf import PdfReader
import docx
import pandas as pd
import streamlit as st

def read_pdf(file):
    text = ""
    try:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        pass
    return text

def read_docx(file):
    text = ""
    try:
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        pass
    return text

def read_txt(file):
    try:
        return file.getvalue().decode("utf-8")
    except Exception as e:
        return ""

def process_uploaded_file(uploaded_file):
    """Yüklenen dosyanın uzantısına göre doğru okuyucu fonksiyonu tetikler."""
    if uploaded_file.name.lower().endswith('.pdf'):
        return read_pdf(uploaded_file)
    elif uploaded_file.name.lower().endswith('.docx'):
        return read_docx(uploaded_file)
    elif uploaded_file.name.lower().endswith('.txt'):
        return read_txt(uploaded_file)
    else:
        return ""
    
def read_dataset(file):
    """CSV veya Excel formatındaki veri setlerini Pandas DataFrame'e çevirir."""
    try:
        if file.name.lower().endswith('.csv'):
            # En yaygın veri seti kodlamaları
            encodings_to_try = ['utf-8', 'utf-8-sig', 'windows-1254', 'iso-8859-9', 'latin1']
            
            for enc in encodings_to_try:
                try:
                    file.seek(0)
                    # Önce noktalı virgül (;) ile deniyoruz (Türkçe setlerde %90 böyledir)
                    df = pd.read_csv(file, encoding=enc, sep=';', on_bad_lines='skip')
                    
                    # Eğer tüm veriyi tek bir sütuna yığdıysa, ayırıcı aslında virgül (,) demektir
                    if len(df.columns) == 1:
                        file.seek(0)
                        df = pd.read_csv(file, encoding=enc, sep=',', on_bad_lines='skip')
                    return df
                except UnicodeDecodeError:
                    # Bu kodlama uymadı, çökme ve diğerine geç
                    continue 
            
            # 🚨 EĞER HİÇBİRİ İŞE YARAMAZSA (SON ÇARE):
            file.seek(0)
            # encoding_errors='replace' komutu, o okuyamadığı 0x9e karakterini çöktürmek yerine "?" yapar.
            df = pd.read_csv(file, encoding='utf-8', sep=';', on_bad_lines='skip', encoding_errors='replace')
            if len(df.columns) == 1:
                file.seek(0)
                df = pd.read_csv(file, encoding='utf-8', sep=',', on_bad_lines='skip', encoding_errors='replace')
            return df
            
        elif file.name.lower().endswith(('.xls', '.xlsx')):
            file.seek(0)
            return pd.read_excel(file)
            
    except Exception as e:
        st.error(f"🚨 Kritik Okuma Hatası: {e}")
        return None
        
    return None