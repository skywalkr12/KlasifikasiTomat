
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from helper import load_model, predict_image, CLASS_NAMES

st.title("🔍 Prediksi Penyakit Tomat")

if "history" not in st.session_state:
    st.session_state["history"] = []

uploaded_file = st.file_uploader("Upload gambar daun tomat", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", width=250)

    model = load_model()
    label, probs = predict_image(model, image)

    st.success(f"Hasil Prediksi: **{label}**")

    # Visualisasi probabilitas
    fig, ax = plt.subplots()
    ax.barh(CLASS_NAMES, probs, color="green")
    ax.set_xlabel("Probabilitas")
    st.pyplot(fig)

    # Simpan ke histori
    st.session_state["history"].append({
        "Tanggal": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Nama File": uploaded_file.name,
        "Prediksi": label,
        "Probabilitas (%)": f"{max(probs)*100:.2f}"
    })

# Tampilkan histori
if st.session_state["history"]:
    st.subheader("📜 Histori Prediksi")
    df = pd.DataFrame(st.session_state["history"])
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "histori_prediksi.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size:14px;'>
Dibuat oleh <b>Farhan</b><br>
🔗 <a href="https://linkedin.com/" target="_blank">LinkedIn</a> | 
<a href="https://instagram.com/" target="_blank">Instagram</a> | 
<a href="https://facebook.com/" target="_blank">Facebook</a>
</div>
""", unsafe_allow_html=True)
