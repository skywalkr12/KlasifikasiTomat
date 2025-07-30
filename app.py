import streamlit as st
import pandas as pd
from datetime import datetime
from PIL import Image
import os

from helper import load_model, predict_image

st.set_page_config(page_title="Deteksi Penyakit Tomat", page_icon="🍅", layout="centered")

# Warna custom
st.markdown("""
    <style>
    .stApp {
        background-color: #f7fff5;
    }
    .main > div {
        background: #ffffff;
        border-radius: 12px;
        padding: 15px;
    }
    .css-1d391kg {background: #e1f7e1 !important;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    .stDownloadButton>button {
        background-color: #c62828;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Session state untuk histori
if "history" not in st.session_state:
    st.session_state["history"] = []

# Logo
logo_path = "images/logo.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=120)

st.title("🍅 Deteksi Penyakit Tomat")
st.write("Upload gambar daun tomat untuk mendeteksi jenis penyakitnya. Model dilatih pada 10 kelas penyakit.")

# Model
model = load_model()

# Upload file
uploaded_file = st.file_uploader("Pilih gambar daun tomat", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", width=300)
    
    if st.button("Prediksi"):
        label, probs = predict_image(model, image)
        conf = round(probs.max() * 100, 2)
        st.success(f"**Hasil Prediksi: {label}** (confidence: {conf}%)")

        # Simpan ke histori
        st.session_state["history"].append({
            "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "File": uploaded_file.name,
            "Prediksi": label,
            "Confidence": f"{conf}%"
        })

# Histori
st.subheader("📜 Histori Prediksi (Session)")
if st.session_state["history"]:
    df = pd.DataFrame(st.session_state["history"])
    st.dataframe(df, use_container_width=True)

    # Export CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Histori CSV", csv, "histori_prediksi.csv", "text/csv", key="download-csv")
else:
    st.info("Belum ada histori prediksi di sesi ini.")

st.markdown("---")
st.markdown("""
**Ikuti saya:** 
[LinkedIn](https://www.linkedin.com/in/namamu/) | 
[Instagram](https://www.instagram.com/username/) | 
[Facebook](https://www.facebook.com/username/)
""")