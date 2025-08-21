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

# === Gate "tomato-only" versi LAB (ringkas) ===
# requires: opencv-python-headless
import cv2

def _leaf_mask_lab(img_rgb,
                   L_min=25, L_max=245,
                   a_green_max=-5, a_brown_min=12, b_yellow_min=10):
    """Mask daun tomat: hijau (a* negatif), kuning (b* positif), cokelat (a* & b* positif)."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    a = A.astype(np.int16) - 128
    b = B.astype(np.int16) - 128
    green  = (a <= a_green_max) & (L >= L_min) & (L <= L_max)
    yellow = (a >  a_green_max) & (a < a_brown_min) & (b >= b_yellow_min) & (L >= L_min) & (L <= L_max)
    brown  = (a >= a_brown_min) & (b >= b_yellow_min) & (L >= L_min)
    m = (green | yellow | brown).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((5,5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    m = cv2.dilate(m, np.ones((3,3), np.uint8), 1)
    return m

def _largest_component_stats(mask01):
    """Ambil komponen terbesar sebagai daun utama; kembalikan fraksi area & solidity."""
    num, labels = cv2.connectedComponents(mask01)
    if num <= 1:
        return 0.0, 0.0
    best, area = 0, 0
    for lb in range(1, num):
        a = int((labels == lb).sum())
        if a > area:
            best, area = lb, a
    comp = (labels == best).astype(np.uint8)
    frac = comp.mean()
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return float(frac), 0.0
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    sol = float(cv2.contourArea(cnt) / (cv2.contourArea(hull) + 1e-6))
    return float(frac), sol

def tomato_gate(pil_image, min_mask_frac=0.08, max_mask_frac=0.95, min_solidity=0.25):
    """Return: (accept: bool, info: dict). Tolak jika bukan daun tomat/kurang layak."""
    rgb = np.array(pil_image.convert("RGB"))
    m = _leaf_mask_lab(rgb)
    frac, sol = _largest_component_stats(m)
    reasons = []
    if frac < min_mask_frac: reasons.append(f"mask terlalu kecil ({frac:.2f})")
    if frac > max_mask_frac: reasons.append(f"mask terlalu besar ({frac:.2f})")
    if sol  < min_solidity:  reasons.append(f"solidity rendah ({sol:.2f})")
    return (len(reasons) == 0), {"mask_frac": frac, "solidity": sol, "reasons": reasons}
# === End gate ===

st.set_page_config(page_title="Prediksi Penyakit Tomat + Grad-CAM", layout="wide")
st.title("🔍 Prediksi Penyakit Tomat + Fitur Grad-CAM")

if "history" not in st.session_state:
    st.session_state["history"] = []

# ------ Util display: batasi hanya tampilan agar tak pernah 100% ------
DISPLAY_CAP = 0.9999  # 99.99% maksimum di UI

def cap_for_display(p: float, cap: float = DISPLAY_CAP) -> float:
    return p if p < cap else cap

def fmt_pct(p: float, cap: float = DISPLAY_CAP, decimals: int = 2) -> str:
    q = cap_for_display(float(p), cap)
    return f"{q*100:.{decimals}f}%"

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
    erode_border = st.checkbox("Erosi tepi mask 1px (redam pinggiran daun)", True)
    lesion_boost = st.checkbox("Deteksi bintik (aktifkan lesion prior)", True)
    lesion_weight = st.slider("Bobot deteksi bintik (lesion prior)", 0.0, 1.0, 0.5, 0.05)

    st.markdown("---")
    show_full_chart = st.checkbox("Tampilkan chart probabilitas lengkap", True)
    sort_desc = st.checkbox("Urutkan chart menurun", True)

# ----- Model -----
model = load_model(cache_bust="noinplace-v3")

# ----- Uploader -----
uploaded_file = st.file_uploader("Upload gambar daun tomat", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # === Gate "tomato-only": tolak jika bukan daun tomat/kurang layak ===
    accept, info_gate = tomato_gate(image)  # ambang default sudah aman
    if not accept:
        st.error("❌ Ditolak: bukan daun tomat / kualitas kurang memadai → " + ", ".join(info_gate["reasons"]))
        st.stop()

    # Prediksi + Grad-CAM (helper TIDAK merender apa pun)
    overlay, cam, used_idx, probs_raw = show_prediction_and_cam(
        model, image,
        alpha=alpha,
        topk=topk,
        target_layer_name=target_layer_name,
        include_brown=True,
        lesion_boost=lesion_boost, lesion_weight=lesion_weight,
        mask_bg=mask_bg,
        blend_with_res2=blend_with_res2,
        erode_border=erode_border
    )

    # === DUA PANEL: KIRI INPUT, KANAN GRAD-CAM ===
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Input", use_container_width=True)
        st.caption(f"Gate: mask_frac={info_gate['mask_frac']:.2f}, solidity={info_gate['solidity']:.2f}")
    with col2:
        st.image(
            overlay,
            caption=f"Grad-CAM ({target_layer_name}) → {CLASS_NAMES[used_idx]} • Confidence: {fmt_pct(probs_raw[used_idx])}",
            use_container_width=True
        )
    st.caption("---")

    # Alternatif (Top-k) — teks saja (tidak menambah gambar utama)
    topk_ = min(topk, len(CLASS_NAMES))
    order = np.argsort(-probs_raw)[:topk_]
    st.markdown("**Alternatif (Top-k)**")
    st.markdown("\n".join([
        f"{'★' if i==used_idx else '•'} {CLASS_NAMES[i]}: {fmt_pct(probs_raw[i])}"
        for i in order
    ]))

    # Chart probabilitas lengkap (opsional)
    if show_full_chart:
        st.subheader("📊 Probabilitas per Kelas")
        probs_plot = np.minimum(np.array(probs_raw, dtype=float), DISPLAY_CAP)
        idxs = np.argsort(-probs_plot) if sort_desc else np.arange(len(CLASS_NAMES))
        fig, ax = plt.subplots()
        ax.barh([CLASS_NAMES[i] for i in idxs], probs_plot[idxs], height=0.6)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probabilitas (dibatasi < 100%)")
        ax.set_ylabel("Kelas")
        st.pyplot(fig)

    # Grad-CAM untuk kelas lain (opsional, di expander)
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

    # Histori (simpan angka tampilan)
    st.session_state["history"].append({
        "Tanggal": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Nama File": uploaded_file.name,
        "Prediksi": CLASS_NAMES[used_idx],
        "Probabilitas (display)": fmt_pct(probs_raw[used_idx]),
        "Layer": target_layer_name,
        "MaskBG": mask_bg,
        "BlendRes2": blend_with_res2,
        "ErodeBorder": erode_border,
        "LesionBoost": lesion_boost,
        "LesionWeight": lesion_weight
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
