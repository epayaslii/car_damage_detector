import streamlit as st
import os
import requests
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from fpdf import FPDF
from PIL import Image

# --- 1. MODEL AYARLARI VE İNDİRME ---
# Hugging Face linkini ve yerel dosya adını tanımlıyoruz
MODEL_URL = "https://huggingface.co/elizpayasli/car_dd/resolve/main/best%20(3).pt"
MODEL_PATH = "best.pt"

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Yapay zeka modeli Hugging Face'ten indiriliyor, lütfen bekleyin..."):
            try:
                response = requests.get(MODEL_URL, stream=True)
                if response.status_code == 200:
                    with open(MODEL_PATH, "wb") as f:
                        for chunk in response.iter_content(chunk_size=1024*1024):
                            if chunk:
                                f.write(chunk)
                    st.success("Model başarıyla yüklendi!")
                else:
                    st.error(f"İndirme başarısız. Hata kodu: {response.status_code}")
                    return None
            except Exception as e:
                st.error(f"İndirme hatası: {e}")
                return None
    return YOLO(MODEL_PATH)

# --- 2. PDF RAPOR OLUŞTURMA FONKSİYONU ---
def create_pdf(df, total_cost):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    
    # Rapor Başlığı
    pdf.cell(200, 10, txt="Araba Hasar Ekspertiz Raporu", ln=True, align='C')
    pdf.ln(10)
    
    # Tablo Başlıkları
    pdf.set_font("Arial", "B", 12)
    pdf.cell(60, 10, "Hasar Tipi", 1)
    pdf.cell(60, 10, "Guven Skoru", 1)
    pdf.cell(60, 10, "Maliyet ($)", 1)
    pdf.ln()
    
    # Tablo İçeriği
    pdf.set_font("Arial", "", 12)
    for index, row in df.iterrows():
        pdf.cell(60, 10, str(row["Hasar Tipi"]), 1)
        pdf.cell(60, 10, str(row["Güven Skoru"]), 1)
        pdf.cell(60, 10, str(row["Tahmini Maliyet ($)"]), 1)
        pdf.ln()
        
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt=f"Toplam Onarim Tutari: ${total_cost:,.2f}", ln=True)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(200, 10, txt="Bu rapor yapay zeka tarafindan otomatik olarak olusturulmustur.", ln=True)
    
    # PDF'i byte formatına çevir (Streamlit indirmesi için)
    return pdf.output(dest='S').encode('latin-1', 'ignore')

# --- 3. SAYFA AYARLARI VE MODEL YÜKLEME ---
st.set_page_config(page_title="Araba Hasar Analizi", layout="wide")

# Sol panel (Sidebar) - DHL ve Boğaziçi vizyonunu buraya ekledik
with st.sidebar:
    st.image("https://www.boun.edu.tr/assets/images/logo/logo_en.png", width=100)
    st.title("Proje Hakkında")
    st.info("Bu uygulama, Boğaziçi Üniversitesi Endüstri ve Bilgisayar Mühendisliği öğrencisi tarafından geliştirilmiş bir AI tabanlı hasar tespit sistemidir.")
    st.markdown("---")
    st.write("🔧 **Model:** YOLOv11")
    st.write("📊 **Analiz:** Maliyet Tahminleme")

st.title("🚗 Profesyonel Araba Hasar Tespit ve Maliyet Analizi")
st.markdown("Aracınızın fotoğrafını yükleyin, yapay zeka hasarları tespit etsin ve onarım maliyetini hesaplasın.")

model = download_and_load_model()

# --- 4. FİYAT LİSTESİ ---
price_dictionary = {
    'doorouter-dent': 150.0,
    'fender-dent': 120.0,
    'front-bumper-dent': 200.0,
    'Headlight-Damage': 350.0,
    'Front-Windscreen-Damage': 500.0,
    'doorouter-scratch': 50.0,
    'bonnet-dent': 250.0,
    'rear-bumper-dent': 180.0,
    'quaterpanel-dent': 100.0, # Resmindeki sınıflara göre ekledim
    'default': 100.0
}

# --- 5. ANA UYGULAMA MANTIĞI ---
if model:
    uploaded_file = st.file_uploader("Hasarlı araç fotoğrafı yükleyin...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Resmi Oku
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Analiz Başlat
        with st.spinner('Yapay zeka analiz ediyor...'):
            results = model.predict(source=image, conf=0.15)
            result = results[0]
        
        ann_image = image_rgb.copy()
        report_data = []

        # Sonuçları İşle
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            prob = float(box.conf[0])
            
            cost = price_dictionary.get(label, price_dictionary['default'])
            report_data.append({
                "Hasar Tipi": label, 
                "Güven Skoru": f"%{prob*100:.1f}", 
                "Tahmini Maliyet ($)": cost
            })

            # Görsel üzerine çizim (Kırmızı kutu ve beyaz yazı)
            cv2.rectangle(ann_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(ann_image, f"{label} ${cost}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Arayüzü İkiye Böl
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Tespit Edilen Hasarlar")
            st.image(ann_image, use_container_width=True)

        with col2:
            st.subheader("Ekspertiz Raporu")
            if report_data:
                df = pd.DataFrame(report_data)
                st.dataframe(df, use_container_width=True)
                
                total_cost = df["Tahmini Maliyet ($)"].sum()
                st.metric("Toplam Tahmini Onarım Tutarı", f"${total_cost:,.2f}")
                
                # PDF İndirme Butonu
                pdf_bytes = create_pdf(df, total_cost)
                st.download_button(
                    label="📄 Ekspertiz Raporunu PDF İndir",
                    data=pdf_bytes,
                    file_name="araba_hasar_raporu.pdf",
                    mime="application/pdf",
                )
                
                st.write("---")
                st.bar_chart(df.set_index("Hasar Tipi")["Tahmini Maliyet ($)"])
            else:
                st.success("Harika! Herhangi bir hasar tespit edilemedi.")

else:
    st.warning("Model yüklenemedi. Lütfen Hugging Face bağlantısını ve internetinizi kontrol edin.")
