# prediksi.py
# -- Gate "tomato-only" sudah terintegrasi (LAB + anti-skin YCrCb) --
# -- Tambahkan ke requirements.txt: opencv-python-headless>=4.9.0 --

import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import cv2  # <— untuk gate

from helper import (
    load_model,
    show_prediction_and_cam,
    CLASS_NAMES
)

# ========== Gate TOMATO-ONLY (LAB + anti-skin, ringkas) ==========
def _leaf_mask_lab(img_rgb,
                   L_min=25, L_max=245,
                   a_green_max=-5, a_brown_min=12, b_yellow_min=10):
    """
    Mask daun tomat di ruang CIELAB (OpenCV skala 0..255):
    - hijau: a* relatif negatif
    - kuning: b* positif (a* netral/positif kecil)
    - cokelat/nekrosis: a* & b* positif
    """
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
    """
    Ambil komponen terbesar sebagai kandidat daun.
    Return: (mask_frac, solidity)
    """
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
    return float(frac), float(sol)

def _green_ratio_hsv(img_rgb, mask01):
    """Proporsi piksel hijau (HSV) di dalam mask."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    green = ((H>=35) & (H<=85) & (S>=28) & (V>=40)).astype(np.uint8)
    g_in = int((green & mask01).sum())
    area = int(mask01.sum()) + 1
    return float(g_in) / float(area)

def _skin_in_mask_ratio_ycrcb(img_rgb, mask01):
    """Proporsi piksel berkarakter kulit (YCrCb) di dalam mask (untuk menolak wajah)."""
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    Y, Cr, Cb = ycrcb[...,0], ycrcb[...,1], ycrcb[...,2]
    # rentang kulit yang umum (longgar)
    skin = ((Cr>=135) & (Cr<=180) & (Cb>=85) & (Cb<=135)).astype(np.uint8)
    s_in = int((skin & mask01).sum())
    area = int(mask01.sum()) + 1
    return float(s_in) / float(area)

def tomato_gate(pil_image,
                min_mask_frac=0.08, max_mask_frac=0.95, min_solidity=0.25,
                min_green_ratio=0.12,    # bukti hijau minimal
                max_skin_in_mask=0.35):  # jika skin>35% di dalam mask → tolak
    """
    Return:
      accept(bool), info(dict: mask_frac, solidity, green_ratio, skin_ratio, reasons[list])
    """
    rgb = np.array(pil_image.convert("RGB"))
    mask = _leaf_mask_lab(rgb)
    frac, sol = _largest_component_stats(mask)
    green_r   = _green_ratio_hsv(rgb, mask)
    skin_r    = _skin_in_mask_ratio_ycrcb(rgb, mask)

    reasons = []
    if frac < min_mask_frac: reasons.append(f"mask kecil ({frac:.2f})")
    if frac > max_mask_frac: reasons.append(f"mask terlalu besar ({frac:.2f})")
    if sol  < min_solidity:  reasons.append(f"solidity rendah ({sol:.2f})")
    if green_r < min_green_ratio: reasons.append(f"hijau rendah ({green_r:.2f})")
    if skin_r  > max_skin_in_mask: reasons.append(f"pola kulit terdeteksi ({skin_r:.2f})")

    return (len(reasons) == 0), {
        "mask_frac": frac, "solidity": sol,
        "green_ratio": green_r, "skin_ratio": skin_r,
        "reasons": reasons
    }
# ========== END Gate ==========

st.set_page_config(page_title="Prediksi Penyakit Tomat + Grad-CAM", layout="wide")
st.title("🔍 Prediksi Penyakit Tomat + Fitur Grad-CAM")

if "history" not in st.session_state:
    st.session_state["history"] = []

# ------ Util display: batasi tampilan agar tidak 100% ------
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

    # === Gate: hanya ijinkan daun tomat ===
    accept, info_gate = tomato_gate(image)  # ambang default aman
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
        st.caption(
            f"Gate: mask_frac={info_gate['mask_frac']:.2f} • "
            f"solidity={info_gate['solidity']:.2f} • "
            f"green={info_gate['green_ratio']:.2f} • "
            f"skin={info_gate['skin_ratio']:.2f}"
        )
    with col2:
        st.image(
            overlay,
            caption=f"Grad-CAM ({target_layer_name}) → {CLASS_NAMES[used_idx]} • Confidence: {fmt_pct(probs_raw[used_idx])}",
            use_container_width=True
        )
    st.caption("---")

    # Alternatif (Top-k) — teks
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

    # (Dihapus) — Grad-CAM untuk kelas tertentu

    # Histori
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
        "LesionWeight": lesion_weight,
        "Gate_mask_frac": f"{info_gate['mask_frac']:.2f}",
        "Gate_solidity": f"{info_gate['solidity']:.2f}",
        "Gate_green": f"{info_gate['green_ratio']:.2f}",
        "Gate_skin": f"{info_gate['skin_ratio']:.2f}"
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
