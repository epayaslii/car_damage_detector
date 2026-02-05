import streamlit as st
import cv2
import os
import pandas as pd
import numpy as np
from ultralytics import YOLO
from PIL import Image

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Araba Hasar Analizi", layout="wide")
st.title("🚗 Profesyonel Araba Hasar Tespit ve Maliyet Analizi")
st.markdown("Bu uygulama, **YOLO11** modelini kullanarak araç üzerindeki hasarları tespit eder ve onarım maliyeti çıkarır.")

# --- MODEL YÜKLEME ---
# Burayı kendi best.pt dosyanın yoluna göre güncelle!
MODEL_PATH = 'best (3).pt'

@st.cache_resource # Modelin her seferinde tekrar yüklenip uygulamayı yavaşlatmasını engeller
def load_model():
    if os.path.exists(MODEL_PATH):
        return YOLO(MODEL_PATH)
    return None

model = load_model()

# --- FİYAT LİSTESİ ---
price_dictionary = {
    'doorouter-dent': 150.0,
    'fender-dent': 120.0,
    'front-bumper-dent': 200.0,
    'Headlight-Damage': 350.0,
    'Front-Windscreen-Damage': 500.0,
    'doorouter-scratch': 50.0,
    'bonnet-dent': 250.0,
    'rear-bumper-dent': 180.0,
    'default': 100.0
}

# --- ARAYÜZ ---
if model is None:
    st.error(f"❌ '{MODEL_PATH}' dosyası bulunamadı! Lütfen model dosyasını proje klasörüne koyun.")
else:
    uploaded_file = st.file_uploader("Hasarlı araç fotoğrafı yükleyin...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Resmi Oku
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Tahmin Yap (Senin modelinle)
        with st.spinner('Analiz ediliyor...'):
            results = model.predict(source=image, conf=0.15) # Conf değerini ihtiyacına göre ayarla
            result = results[0]
        
        # Görselleştirme ve Veri Toplama
        ann_image = image_rgb.copy()
        report_data = []

        for box in result.boxes:
            # Koordinatlar ve Sınıf Bilgisi
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            prob = float(box.conf[0])
            
            cost = price_dictionary.get(label, price_dictionary['default'])
            report_data.append({"Hasar Tipi": label, "Güven Skoru": f"%{prob*100:.1f}", "Tahmini Maliyet ($)": cost})

            # Kutu Çizimi
            cv2.rectangle(ann_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(ann_image, f"{label} ${cost}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Sonuçları Göster
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Tespit Edilen Hasarlar")
            st.image(ann_image, use_container_width=True)

        with col2:
            st.subheader("Ekspertiz Raporu")
            if report_data:
                df = pd.DataFrame(report_data)
                st.dataframe(df)
                
                total_cost = df["Tahmini Maliyet ($)"].sum()
                st.metric("Toplam Onarım Tutarı", f"${total_cost:,.2f}")
                
                # Plotting (Senin kodundaki matplotlib'in Streamlit hali)
                st.bar_chart(df.set_index("Hasar Tipi")["Tahmini Maliyet ($)"])
            else:
                st.info("Herhangi bir hasar tespit edilemedi.")

# --- EĞİTİM (TRAINING) HAKKINDA NOT ---
with st.expander("Eğitim (Training) Bilgileri"):
    st.write("""
    Streamlit bir sunum arayüzüdür. Model eğitimi (`model.train`) genellikle terminalden veya Notebook üzerinden yapılır. 
    Eğer eğitimi buradan başlatmak isterseniz, bir buton eklenebilir ancak tarayıcı sekmesi eğitim bitene kadar yanıt vermeyebilir.
    """)
