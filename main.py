import streamlit as st
import os
import requests
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO


# --- 1. HUGGING FACE MODEL İNDİRME SİSTEMİ ---
# Senin paylaştığın linki buraya tanımladık
MODEL_URL = "https://huggingface.co/elizpayasli/car_dd/resolve/main/best%20(3).pt"
MODEL_PATH = "best.pt" # İçeride bu isimle kullanacağız, kafa karışıklığına son!

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Model Hugging Face üzerinden indiriliyor..."):
            try:
                # Hugging Face direkt indirmeye izin verdiği için Session veya Token gerekmez
                response = requests.get(MODEL_URL, stream=True)
                if response.status_code == 200:
                    with open(MODEL_PATH, "wb") as f:
                        for chunk in response.iter_content(chunk_size=1024*1024): # 1MB'lık parçalar
                            if chunk:
                                f.write(chunk)
                    st.success("YOLOv11 modeli başarıyla yüklendi!")
                else:
                    st.error(f"Dosya indirilemedi. Hata kodu: {response.status_code}")
                    return None
            except Exception as e:
                st.error(f"İndirme sırasında bir hata oluştu: {e}")
                return None
    
    return YOLO(MODEL_PATH)

# --- 2. SAYFA AYARLARI ---
st.set_page_config(page_title="Araba Hasar Analizi", layout="wide")
st.title("🚗 Profesyonel Araba Hasar Tespit ve Maliyet Analizi")
st.markdown("Bu uygulama, **YOLO11** modelini kullanarak araç üzerindeki hasarları tespit eder ve onarım maliyeti çıkarır.")

model = download_and_load_model()

# Uygulamanın geri kalan kısmını (Fiyat Listesi ve Arayüz) altına ekleyebilirsin...

# --- 3. FİYAT LİSTESİ ---
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

# --- 4. ARAYÜZ VE ANALİZ ---
if model is None:
    st.error("❌ Model yüklenemedi. Lütfen Google Drive bağlantısını kontrol edin.")
else:
    uploaded_file = st.file_uploader("Hasarlı araç fotoğrafı yükleyin...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with st.spinner('Analiz ediliyor...'):
            results = model.predict(source=image, conf=0.15)
            result = results[0]
        
        ann_image = image_rgb.copy()
        report_data = []

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

            cv2.rectangle(ann_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(ann_image, f"{label} ${cost}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

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
                st.bar_chart(df.set_index("Hasar Tipi")["Tahmini Maliyet ($)"])
            else:
                st.info("Herhangi bir hasar tespit edilemedi.")

with st.expander("Eğitim (Training) Bilgileri"):
    st.write("Model eğitimi terminal üzerinden tamamlanmış ve ağırlıklar (best.pt) Google Drive üzerinden sisteme entegre edilmiştir.")
