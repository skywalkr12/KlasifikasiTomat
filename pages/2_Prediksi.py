# prediksi.py
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

from helper import (
    load_model,
    show_prediction_and_cam,
    gradcam_on_pil,
    CLASS_NAMES
)

st.set_page_config(page_title="Prediksi Penyakit Tomat + Grad-CAM", layout="wide")
st.title("🔍 Prediksi Penyakit Tomat + Grad-CAM (aman)")

if "history" not in st.session_state:
    st.session_state["history"] = []

# ----- Sidebar -----
with st.sidebar:
    st.header("Pengaturan Visualisasi")
    target_layer_name = st.selectbox(
        "Layer target Grad-CAM",
        options=["conv4_prepool", "conv3_prepool", "conv2_prepool", "res2"],
        index=0
    )
    alpha = st.slider("Transparansi Heatmap (α)", 0.0, 1.0, 0.45, 0.05)
    topk  = st.slider("Jumlah alternatif (Top-k)", 1, min(5, len(CLASS_NAMES)), 3, 1)
    mask_bg = st.checkbox("Mask background (fokus ke daun)", True)
    blend_with_res2 = st.checkbox("Blend dengan res2 (stabilkan semantik)", True)
    st.markdown("---")
    show_full_chart = st.checkbox("Tampilkan chart probabilitas lengkap", True)
    sort_desc = st.checkbox("Urutkan chart menurun", True)

# ----- Model (pakai cache-bust agar benar-benar reload) -----
model = load_model(cache_bust="noinplace-v3")

# ----- Uploader -----
uploaded_file = st.file_uploader("Upload gambar daun tomat", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Prediksi + Grad-CAM
    overlay, cam, used_idx, probs_all = show_prediction_and_cam(
        model, image,
        alpha=alpha,
        topk=topk,
        target_layer_name=target_layer_name,
        include_brown=True,
        lesion_boost=True, lesion_weight=0.5,
        mask_bg=mask_bg,
        blend_with_res2=blend_with_res2
    )

    # Chart probabilitas lengkap
    if show_full_chart:
        st.subheader("📊 Probabilitas per Kelas")
        probs = np.array(probs_all)
        idxs = np.argsort(-probs) if sort_desc else np.arange(len(CLASS_NAMES))
        fig, ax = plt.subplots()
        ax.barh([CLASS_NAMES[i] for i in idxs], probs[idxs], height=0.6)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probabilitas")
        ax.set_ylabel("Kelas")
        st.pyplot(fig)

    # Grad-CAM untuk kelas lain (opsional)
    with st.expander("🎯 Lihat Grad-CAM untuk kelas tertentu"):
        target_label = st.selectbox("Pilih kelas", CLASS_NAMES, index=used_idx)
        target_idx = CLASS_NAMES.index(target_label)
        overlay2, _, _, _, _ = gradcam_on_pil(
            model, image,
            target_layer_name=target_layer_name,
            include_brown=True,
            lesion_boost=True, lesion_weight=0.5,
            class_idx=target_idx,
            alpha=alpha,
            mask_bg=mask_bg,
            blend_with_res2=blend_with_res2
        )
        st.image(overlay2, caption=f"Grad-CAM ({target_layer_name}) → {target_label}", use_container_width=True)

    # Simpan riwayat
    st.session_state["history"].append({
        "Tanggal": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Nama File": uploaded_file.name,
        "Prediksi": CLASS_NAMES[used_idx],
        "Probabilitas (%)": f"{float(probs_all[used_idx]) * 100:.2f}",
        "Layer": target_layer_name,
        "MaskBG": mask_bg,
        "BlendRes2": blend_with_res2
    })

# Riwayat + unduh
if st.session_state["history"]:
    st.subheader("📜 Histori Prediksi")
    df = pd.DataFrame(st.session_state["history"])
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "histori_prediksi.csv", "text/csv")

st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size:14px;'>
Dibuat oleh <b>Muhammad Sahrul Farhan | 51421076</b><br>
🔗 <a href="https://linkedin.com/" target="https://id.linkedin.com/in/muhammad-sahrul-farhan">LinkedIn</a> | 
<a href="https://instagram.com/" target="https://www.instagram.com/eitcheien">Instagram</a> | 
<a href="https://facebook.com/" target="https://www.facebook.com/skywalkr12">Facebook</a>
</div>
""", unsafe_allow_html=True)
