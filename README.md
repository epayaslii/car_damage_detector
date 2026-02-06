# car_damage_detector
Streamlit website for the Car DD: https://cardamagedetector-wscmllni8pkugxamvrcvtj.streamlit.app/

# 🚗 AI Car Damage Analysis & Cost Estimation

An advanced computer vision application built with **Streamlit** and **YOLOv11** that detects vehicle damage from images and provides real-time repair cost estimations.

## 🌟 Features
* **Automated Detection:** Identifies dents, scratches, and glass damage using a custom-trained YOLOv11 model.
* **Cost Forecasting:** Calculates estimated repair costs based on a predefined price dictionary.
* **Professional Reporting:** Generates a downloadable PDF expertise report for the analysis.
* **Interactive UI:** Clean, English-language interface with visual bounding boxes and data charts.

## 🛠️ Tech Stack
* **Frontend:** [Streamlit](https://streamlit.io/)
* **AI Model:** [Ultralytics YOLOv11](https://docs.ultralytics.com/)
* **Language:** Python 3.x
* **Libraries:** OpenCV, Pandas, FPDF, PIL

## 🧠 How It Works: The Pipeline

The application follows a structured Computer Vision pipeline to move from a raw image to a financial estimate:

1. **Image Preprocessing:** The uploaded image is decoded via OpenCV and resized to the model's expected input dimensions.
2. **Inference (YOLOv11):** The YOLO (You Only Look Once) model performs a single pass over the image to predict bounding boxes and class probabilities simultaneously.
3. **Post-Processing:** * **Non-Maximum Suppression (NMS):** Filters out overlapping boxes to ensure each damage is counted only once.
    * **Annotation:** The system draws bounding boxes and labels the detected damage directly on the image.
4. **Cost Mapping:** Each detected class (e.g., `fender-dent`) is matched against a regional price dictionary to calculate the total repair estimation.


## 🛠️ Technical Deep Dive

### The Model: YOLOv8
This project utilizes the latest **YOLOv8** architecture, which offers significant improvements in:
* **Precision:** Better localization of small scratches that previous models might miss.
* **Speed:** Near-instant inference even on CPU-based Streamlit Cloud instances.
* **Efficiency:** High mAP (mean Average Precision) with a lower parameter count.

### Automated Workflow
The app is designed to be "zero-config" for the user. 
* **Dynamic Loading:** The model is fetched from Hugging Face on the first run using `requests` and cached locally via `st.cache_resource` to ensure fast subsequent loads.
* **State Management:** Streamlit's reactive framework handles the transition from "Upload" to "Analysis" without page refreshes.

## 📈 Price Estimation Logic
The current estimation is based on a **fixed-price dictionary** model:
| Damage Type | Estimated Base Cost |
| :--- | :--- |
| Front Windscreen | $500.00 |
| Headlight Damage | $350.00 |
| Major Dents | $150.00 - $250.00 |
| Minor Scratches | $50.00 |

*Note: These prices are constants defined in `main.py` and can be adjusted based on local market rates.*

## 📋 Future Roadmap
- [ ] **GPS Integration:** Adjusting labor costs based on user location.
- [ ] **Multiple Angles:** Combining results from 4-side photos of a vehicle.
- [ ] **Severity Scoring:** Differentiating between "Deep Scratch" and "Surface Scratch" for better accuracy.
