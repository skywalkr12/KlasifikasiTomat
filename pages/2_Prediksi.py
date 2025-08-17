# prediksi.py
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

# Ambil API dari helper.py (pastikan helper.py yang terbaru)
from helper import (
    load_model,
    show_prediction_and_cam,  # menampilkan input, prediksi, top-k, dan overlay Grad-CAM
    gradcam_on_pil,           # untuk Grad-CAM kelas terpilih manual
    CLASS_NAMES
)

# ====== Setup Halaman ======
st.set_page_config(page_title="Prediksi Penyakit Tomat + Grad-CAM", layout="wide")
st.title("🔍 Prediksi Penyakit Tomat + Grad-CAM")

if "history" not in st.session_state:
    st.session_state["history"] = []

# ====== Sidebar Controls ======
with st.sidebar:
    st.header("Pengaturan")
    alpha = st.slider("Transparansi Heatmap (α)", 0.0, 1.0, 0.45, 0.05)
    topk  = st.slider("Jumlah alternatif (Top-k)", 1, min(5, len(CLASS_NAMES)), 3, 1)
    show_full_chart = st.checkbox("Tampilkan chart probabilitas lengkap", True)
    sort_desc       = st.checkbox("Urutkan chart dari probabilitas tertinggi", True)
    st.markdown("---")
    st.caption("Grad-CAM: menyorot area citra yang berkontribusi pada prediksi model.")

# ====== Model (cached oleh Streamlit) ======
model = load_model()

# ====== Uploader ======
uploaded_file = st.file_uploader("Upload gambar daun tomat", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Tampilkan prediksi + Grad-CAM + Top-k langsung (helper akan render UI dalam 2 kolom)
    overlay, cam, used_idx, probs_all = show_prediction_and_cam(
        model, image, alpha=alpha, topk=topk
    )

    # ====== Chart Probabilitas Lengkap ======
    if show_full_chart:
        st.subheader("📊 Probabilitas per Kelas (Lengkap)")
        probs = np.array(probs_all)
        class_names = CLASS_NAMES

        # Indeks kelas untuk di-plot
        idxs = np.arange(len(class_names))
        if sort_desc:
            idxs = np.argsort(-probs)

        fig, ax = plt.subplots()
        ax.barh([class_names[i] for i in idxs], probs[idxs], height=0.6)
        ax.invert_yaxis()  # kelas dengan probabilitas terbesar di atas
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probabilitas")
        ax.set_ylabel("Kelas")
        st.pyplot(fig)

    # ====== Lihat Grad-CAM untuk Kelas Lain (opsional) ======
    with st.expander("🎯 Lihat Grad-CAM untuk kelas tertentu (opsional)"):
        target_label = st.selectbox(
            "Pilih kelas untuk divisualisasikan",
            options=CLASS_NAMES,
            index=used_idx
        )
        target_idx = CLASS_NAMES.index(target_label)
        if target_idx != used_idx:
            overlay2, _, _, _, _ = gradcam_on_pil(
                model, image, class_idx=target_idx, alpha=alpha
            )
            st.image(
                overlay2,
                caption=f"Grad-CAM → {target_label}",
                use_container_width=True
            )

    # ====== Simpan Riwayat ======
    st.session_state["history"].append({
        "Tanggal": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Nama File": uploaded_file.name,
        "Prediksi": CLASS_NAMES[used_idx],
        "Probabilitas (%)": f"{float(probs_all[used_idx]) * 100:.2f}"
    })

# ====== Tabel Riwayat + Unduh ======
if st.session_state["history"]:
    st.subheader("📜 Histori Prediksi")
    df = pd.DataFrame(st.session_state["history"])
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "histori_prediksi.csv", "text/csv")

# ====== Footer ======
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size:14px;'>
Dibuat oleh <b>Muhammad Sahrul Farhan</b><br>
🔗 <a href="https://linkedin.com/" target="_blank">LinkedIn</a> | 
<a href="https://instagram.com/" target="_blank">Instagram</a> | 
<a href="https://facebook.com/" target="_blank">Facebook</a>
</div>
""", unsafe_allow_html=True)
