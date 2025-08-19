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
    CLASS_NAMES,
    format_probs_for_display   # <<— util baru
)

st.set_page_config(page_title="Prediksi Penyakit Tomat + Grad-CAM", layout="wide")
st.title("🔍 Prediksi Penyakit Tomat + Fitur Grad-CAM")

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
    # 🔧 Toggle yang kamu minta:
    erode_border = st.checkbox("Erosi tepi mask 1px (redam pinggiran daun)", True)
    lesion_boost = st.checkbox("Deteksi bintik (aktifkan lesion prior)", True)
    lesion_weight = st.slider("Bobot deteksi bintik (lesion prior)", 0.0, 1.0, 0.5, 0.05)

    st.markdown("---")
    st.subheader("Mode Tampilan Probabilitas")
    display_mode = st.selectbox(
        "Pilih mode tampilan",
        options=["raw (asli, max bisa 100%)", "temperature (sebar T>1)", "topk_renorm (visual)"],
        index=0
    )
    temperature = st.slider("Temperature (T)", 1.0, 5.0, 1.8, 0.1)
    eps = st.slider("Probabilitas minimum per kelas (ε)", 0.0, 0.01, 0.0, 0.001,
                    help="Opsional untuk menghindari 0 murni. Di-normalisasi ulang otomatis.")

    show_full_chart = st.checkbox("Tampilkan chart probabilitas lengkap", True)
    sort_desc = st.checkbox("Urutkan chart menurun", True)

# ----- Model (pakai cache-bust agar benar-benar reload) -----
model = load_model(cache_bust="noinplace-v3")

# ----- Uploader -----
uploaded_file = st.file_uploader("Upload gambar daun tomat", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Prediksi + Grad-CAM (kini probs_raw dikembalikan tanpa pemotongan)
    overlay, cam, used_idx, probs_raw = show_prediction_and_cam(
        model, image,
        alpha=alpha,
        topk=topk,
        target_layer_name=target_layer_name,
        include_brown=True,                          # bantu deteksi cokelat
        lesion_boost=lesion_boost, lesion_weight=lesion_weight,
        mask_bg=mask_bg,
        blend_with_res2=blend_with_res2,
        erode_border=erode_border
    )

    # Tentukan mode display
    mode_key = display_mode.split()[0]  # "raw" | "temperature" | "topk_renorm"
    probs_disp = format_probs_for_display(
        probs_raw,
        mode="raw" if mode_key == "raw" else ("temperature" if mode_key == "temperature" else "topk_renorm"),
        temperature=temperature,
        eps=eps,
        topk=topk
    )

    # Panel kiri-kanan (tambahkan info confidence & alternatif)
    col1, col2 = st.columns([1,1])
    with col1:
        st.image(image, caption="Input", use_container_width=True)
        st.write(f"**Prediksi**: {CLASS_NAMES[used_idx]}  \n**Confidence (display)**: {float(probs_disp[used_idx]):.2%}")
        # Urutan alternatif selalu dari probabilitas mentah (keputusan model)
        topk_ = min(topk, len(CLASS_NAMES))
        order = np.argsort(-probs_raw)[:topk_]
        st.markdown("**Alternatif (Top-k)**")
        st.markdown("\n".join([
            f"{'★' if i==used_idx else '•'} {CLASS_NAMES[i]}: {probs_disp[i]:.2%} (raw: {probs_raw[i]:.2%})"
            for i in order
        ]))
    with col2:
        st.image(overlay, caption=f"Grad-CAM ({target_layer_name}) → {CLASS_NAMES[used_idx]}", use_container_width=True)

    # Chart probabilitas lengkap
    if show_full_chart:
        st.subheader("📊 Probabilitas per Kelas")
        probs = np.array(probs_disp)
        idxs = np.argsort(-probs) if sort_desc else np.arange(len(CLASS_NAMES))
        fig, ax = plt.subplots()
        ax.barh([CLASS_NAMES[i] for i in idxs], probs[idxs], height=0.6)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probabilitas")
        ax.set_ylabel("Kelas")
        st.pyplot(fig)

    # Grad-CAM untuk kelas lain (tetap Grad-CAM standar)
    with st.expander("🎯 Lihat Grad-CAM untuk kelas tertentu"):
        target_label = st.selectbox("Pilih kelas", CLASS_NAMES, index=used_idx)
        target_idx = CLASS_NAMES.index(target_label)
        overlay2, _, _, _, _ = gradcam_on_pil(
            model, image,
            target_layer_name=target_layer_name,
            include_brown=True,
            lesion_boost=lesion_boost, lesion_weight=lesion_weight,
            class_idx=target_idx,
            alpha=alpha,
            mask_bg=mask_bg,
            blend_with_res2=blend_with_res2,
            erode_border=erode_border
        )
        st.image(overlay2, caption=f"Grad-CAM ({target_layer_name}) → {target_label}", use_container_width=True)

    # Simpan riwayat (pakai display prob agar konsisten dengan UI)
    st.session_state["history"].append({
        "Tanggal": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Nama File": uploaded_file.name,
        "Prediksi": CLASS_NAMES[used_idx],
        "Probabilitas (%)": f"{float(probs_disp[used_idx]) * 100:.2f}",
        "Layer": target_layer_name,
        "MaskBG": mask_bg,
        "BlendRes2": blend_with_res2,
        "ErodeBorder": erode_border,
        "LesionBoost": lesion_boost,
        "LesionWeight": lesion_weight,
        "ModeDisplay": mode_key,
        "Temperature": temperature,
        "Eps": eps
    })

# Riwayat + unduh
if st.session_state["history"]:
    st.subheader("📜 Histori Prediksi")
    df = pd.DataFrame(st.session_state["history"])
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "histori_prediksi.csv", "text/csv")

st.write("""
Sebagai Catatan: Ini adalah alat diagnosis dengan bantuan Kecerdasan Buatan dan sebaiknya digunakan hanya sebagai panduan.
Untuk diagnosis konklusif, konsultasikan dengan ahli patologi tanaman profesional.
""")

st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size:14px;'>
<b>© - 2025 | Muhammad Sahrul Farhan | 51421076</b><br>
🔗 <a href="https://www.linkedin.com/in/muhammad-sahrul-farhan/" target="blank_">LinkedIn</a> |
<a href="https://www.instagram.com/eitcheien/" target="blank_">Instagram</a> |
<a href="https://www.facebook.com/skywalkr12" target="blank_">Facebook</a>
</div>
""", unsafe_allow_html=True)
