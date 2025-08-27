# prediksi.py
# -- Gate "tomato-only" (LAB + anti-skin, sederhana) + Prediksi Kelas
# -- + Deteksi Kekuningan (chlorosis) & Indikator Kelayuan (wilt)
# Tambahkan ke requirements.txt: opencv-python-headless>=4.9.0

import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import cv2

from helper import (
    load_model,
    predict_image,   # ← gunakan prediksi saja (tanpa Grad-CAM)
    CLASS_NAMES
)

# ========== Gate TOMATO-ONLY (LAB + anti-skin, ringkas) ==========
def _leaf_mask_lab(img_rgb,
                   L_min=25, L_max=245,
                   a_green_max=-5, a_brown_min=12, b_yellow_min=10):
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
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    green = ((H>=35) & (H<=85) & (S>=28) & (V>=40)).astype(np.uint8)  # cv2: H∈[0,179]
    g_in = int((green & mask01).sum())
    area = int(mask01.sum()) + 1
    return float(g_in) / float(area)

def _skin_in_mask_ratio_ycrcb(img_rgb, mask01):
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    Y, Cr, Cb = ycrcb[...,0], ycrcb[...,1], ycrcb[...,2]
    skin = ((Cr>=135) & (Cr<=180) & (Cb>=85) & (Cb<=135)).astype(np.uint8)
    s_in = int((skin & mask01).sum())
    area = int(mask01.sum()) + 1
    return float(s_in) / float(area)

def tomato_gate(pil_image,
                min_mask_frac=0.08, max_mask_frac=0.95, min_solidity=0.25,
                min_green_ratio=0.12, max_skin_in_mask=0.35):
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
        "reasons": reasons, "mask": mask
    }
# ========== END Gate ==========

# ========== Analisis Kekuningan & Kelayuan ==========
def _color_masks_hsv(img_rgb, leaf_mask01):
    """Segmentasi dalam ruang HSV cv2 (H∈[0,179], S,V∈[0,255])."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]

    # Rentang empiris yang “aman” untuk daun tomat
    green  = ((H>=35) & (H<=85)  & (S>=28) & (V>=40)).astype(np.uint8)
    yellow = ((H>=20) & (H<=35)  & (S>=60) & (V>=60)).astype(np.uint8)  # chlorosis
    brown1 = ((H>=5)  & (H<=20)  & (S>=50) & (V>=25) & (V<=210)).astype(np.uint8)
    brown2 = ((H<5)               & (S>=60) & (V>=15) & (V<=180)).astype(np.uint8)
    brown  = (brown1 | brown2).astype(np.uint8)

    # Batasi ke area daun
    green  = green  & leaf_mask01
    yellow = yellow & leaf_mask01
    brown  = brown  & leaf_mask01
    total  = np.clip(green | yellow | brown, 0, 1).astype(np.uint8)
    area = int(total.sum()) + 1

    stats = {
        "green_ratio":  float(green.sum())  / area,
        "yellow_ratio": float(yellow.sum()) / area,
        "brown_ratio":  float(brown.sum())  / area,
        "area_leaf_px": int(area - 1)
    }
    return {"green":green, "yellow":yellow, "brown":brown, "total":total}, stats

def _shape_metrics_for_wilt(leaf_mask01):
    """Ekstrak metrik bentuk untuk indikasi kelayuan (solidity & roughness)."""
    num, labels = cv2.connectedComponents(leaf_mask01)
    if num <= 1:
        return {"solidity": 0.0, "roughness": 0.0}
    # ambil komponen terbesar
    best, area = 0, 0
    for lb in range(1, num):
        a = int((labels == lb).sum())
        if a > area:
            best, area = lb, a
    comp = (labels == best).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {"solidity": 0.0, "roughness": 0.0}
    cnt  = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    area_cnt  = max(cv2.contourArea(cnt), 1.0)
    area_hull = max(cv2.contourArea(hull), 1.0)
    perim = cv2.arcLength(cnt, closed=True)

    solidity = float(area_cnt / area_hull)
    # Shape factor: 1 untuk lingkaran; makin besar → tepi makin “kasar / berlekuk”
    shape_factor = float((perim**2) / (4.0 * np.pi * area_cnt))
    # Normalisasi empiris → 0..1 (≈1 “kasar sekali”)
    roughness = float(np.clip((shape_factor - 1.0) / 1.2, 0.0, 1.0))
    return {"solidity": solidity, "roughness": roughness}

def _wilt_and_chlorosis_scores(color_stats, shape_stats):
    """
    Skor 0..1 (semakin besar semakin parah). Heuristik terkontrol:
    - Chlorosis: bergantung pada rasio kuning dan penurunan hijau.
    - Wilt: bergantung pada (1 - solidity) dan roughness tepi.
    """
    y = color_stats["yellow_ratio"]
    g = color_stats["green_ratio"]
    chl = np.clip(0.7*(y/0.25) + 0.3*((1.0-g)/0.5), 0.0, 1.0)  # ~25% kuning → 0.7
    wilt = np.clip(0.6*((1.0 - shape_stats["solidity"])/0.75) + 0.4*(shape_stats["roughness"]), 0.0, 1.0)
    return float(chl), float(wilt)

def _make_color_overlay(pil_img, masks, alpha=0.45):
    """Overlay warna: green→(0,255,0), yellow→(255,255,0), brown→(255,80,0)."""
    base = np.asarray(pil_img.convert("RGB")).astype(np.float32)
    H, W, _ = base.shape
    overlay = base.copy()
    color_map = {
        "yellow": np.array([255, 255,   0], dtype=np.float32),
        "brown":  np.array([255,  80,   0], dtype=np.float32),
        "green":  np.array([  0, 255,   0], dtype=np.float32),
    }
    for key in ["yellow", "brown", "green"]:
        m = masks[key].astype(bool)
        if m.any():
            layer = np.zeros_like(base); layer[m] = color_map[key]
            overlay = (1-alpha)*overlay + alpha*layer
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)

# ========== Streamlit UI ==========
st.set_page_config(page_title="Prediksi Penyakit Tomat + Deteksi Kekuningan/Kelayuan", layout="wide")
st.title("🟡 Deteksi Kekuningan & Kelayuan + 🔍 Prediksi Penyakit Tomat")

if "history" not in st.session_state:
    st.session_state["history"] = []

DISPLAY_CAP = 0.9900
def cap_for_display(p: float, cap: float = DISPLAY_CAP) -> float:
    return p if p < cap else cap
def fmt_pct(p: float, cap: float = DISPLAY_CAP, decimals: int = 2) -> str:
    q = cap_for_display(float(p), cap)
    return f"{q*100:.{decimals}f}%"

# ----- CUSTOM CSS for colored metrics -----
st.markdown("""
<style>
    /* Umum untuk nilai metric */
    .st-emotion-cache-1f06a9k.ef3psqc12 > div[data-testid="stMetricValue"] {
        font-weight: normal !important; /* Memastikan tidak bold */
    }

    /* Rasio Kuning */
    div[data-testid="stMetricLabel"]:has(div[title="Rasio Kuning"]) + div > div[data-testid="stMetricValue"] {
        color: #FFD700; /* Kuning emas */
    }

    /* Rasio Cokelat */
    div[data-testid="stMetricLabel"]:has(div[title="Rasio Cokelat"]) + div > div[data-testid="stMetricValue"] {
        color: #A0522D; /* Sienn (Cokelat) */
    }

    /* Solidity (kompaksi) */
    div[data-testid="stMetricLabel"]:has(div[title="Solidity (kompaksi)"]) + div > div[data-testid="stMetricValue"] {
        color: #404040; /* Abu-abu gelap/kehitaman */
    }

    /* Roughness (tepi) */
    div[data-testid="stMetricLabel"]:has(div[title="Roughness (tepi)"]) + div > div[data-testid="stMetricValue"] {
        color: #3CB371; /* Medium Sea Green (Hijau) */
    }
</style>
""", unsafe_allow_html=True)


# ----- Sidebar -----
with st.sidebar:
    st.header("💡 Panduan Penggunaan")
    st.markdown("""
    - **Unggah Gambar:** Siapkan foto daun tomat Anda (format JPG, JPEG, PNG).
    - **Lihat Hasil:** Model akan otomatis menganalisis dan menampilkan potensi penyakit setelah gambar diunggah.
    """)
    st.info("Pastikan gambar hanya menampilkan satu daun tomat dalam kondisi pencahayaan yang baik untuk hasil terbaik.", icon="⚠️")
    st.markdown("---")
    
    st.header("Pengaturan Tampilan")
    topk  = st.slider("Jumlah alternatif (Top-k)", 1, min(5, len(CLASS_NAMES)), 3, 1)
    st.markdown("---")
    show_full_chart = st.checkbox("Tampilkan chart probabilitas lengkap", True)
    sort_desc = st.checkbox("Urutkan chart menurun", True)

# ----- Model -----
model = load_model()

# ----- Uploader -----
uploaded_file = st.file_uploader("Upload gambar daun tomat", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    accept, info_gate = tomato_gate(image)
    if not accept:
        st.error("❌ Ditolak: bukan daun tomat / kualitas kurang memadai → " + ", ".join(info_gate["reasons"]))
        st.stop()
    leaf_mask01 = info_gate["mask"].astype(np.uint8)

    pred_name, probs_raw, _ = predict_image(model, image)
    used_idx = int(np.argmax(probs_raw))

    rgb = np.array(image.convert("RGB"))
    color_masks, color_stats = _color_masks_hsv(rgb, leaf_mask01)
    shape_stats = _shape_metrics_for_wilt(leaf_mask01)
    chlorosis_score, wilt_score = _wilt_and_chlorosis_scores(color_stats, shape_stats)
    color_overlay = _make_color_overlay(image, color_masks, alpha=0.45)

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
            color_overlay,
            caption=f"Segmentasi Warna — Prediksi: {CLASS_NAMES[used_idx]} ({fmt_pct(probs_raw[used_idx])})",
            use_container_width=True
        )

    with st.container(border=True):
        st.subheader("📊 Deteksi Kekuningan & Kelayuan")
        
        with st.container(border=True):
            st.write("""
            **Analisis Gejala Visual:** Informasi di bawah ini memetakan gejala visual seperti kekuningan (klorosis) dan kelayuan daun, **ini bukan diagnosis final**. Diperlukan pemeriksaan lapang lebih lanjut untuk konfirmasi.
            """)
        
        st.markdown(" ")

        with st.container(border=True):
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.metric("Rasio Kuning", fmt_pct(color_stats["yellow_ratio"]))
            with mcol2:
                st.metric("Rasio Cokelat", fmt_pct(color_stats["brown_ratio"]))
            with mcol3:
                st.metric("Solidity (kompaksi)", f"{shape_stats['solidity']:.2f}")
            with mcol4:
                st.metric("Roughness (tepi)", f"{shape_stats['roughness']:.2f}")

        st.markdown(" ")

        pcol1, pcol2 = st.columns(2)
        with pcol1:
            st.markdown(f"**Skor Kekuningan (0–1):** `{chlorosis_score:.2f}`")
            st.progress(min(max(chlorosis_score,0.0),1.0))
        with pcol2:
            st.markdown(f"**Skor Kelayuan (0–1):** `{wilt_score:.2f}`")
            st.progress(min(max(wilt_score,0.0),1.0))
    
    st.markdown(" ")
    topk_ = min(topk, len(CLASS_NAMES))
    order = np.argsort(-probs_raw)[:topk_]
    st.markdown("**Alternatif (Top-k)**")
    st.markdown("\n".join([
        f"{'★' if i==used_idx else '•'} {CLASS_NAMES[i]}: {fmt_pct(probs_raw[i])}"
        for i in order
    ]))

    if show_full_chart:
        st.subheader("📊 Probabilitas per Kelas")
        probs_plot = np.minimum(np.array(probs_raw, dtype=float), DISPLAY_CAP)
        idxs = np.argsort(-probs_plot) if sort_desc else np.arange(len(CLASS_NAMES))
        fig, ax = plt.subplots()
        ax.barh([CLASS_NAMES[i] for i in idxs], probs_plot[idxs], height=0.6)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probabilitas")
        ax.set_ylabel("Kelas")
        st.pyplot(fig)

    st.session_state["history"].append({
        "Tanggal": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Nama File": uploaded_file.name,
        "Prediksi": CLASS_NAMES[used_idx],
        "Probabilitas (display)": fmt_pct(probs_raw[used_idx]),
        "Rasio_Kuning": fmt_pct(color_stats["yellow_ratio"]),
        "Rasio_Cokelat": fmt_pct(color_stats["brown_ratio"]),
        "Solidity": f"{shape_stats['solidity']:.2f}",
        "Roughness": f"{shape_stats['roughness']:.2f}",
        "Skor_Chlorosis": f"{chlorosis_score:.2f}",
        "Skor_Wilt": f"{wilt_score:.2f}"
    })

if st.session_state["history"]:
    st.subheader("📜 Histori Prediksi")
    df = pd.DataFrame(st.session_state["history"])
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "histori_prediksi.csv", "text/csv")

st.info( "Perlu diingat: Ini adalah alat diagnosis dengan bantuan Kecerdasan Buatan dan sebaiknya digunakan hanya sebagai panduan. Untuk diagnosis konklusif, konsultasikan dengan ahli patologi tanaman profesional."
)
st.markdown(
    """
<div style='text-align:center; font-size:14px;'>
<b>© 2025 | Muhammad Sahrul Farhan | 51421076</b><br>
🔗 <a href="https://www.linkedin.com/in/muhammad-sahrul-farhan/" target="_blank">LinkedIn</a> |
<a href="https://www.instagram.com/eitcheien/" target="_blank">Instagram</a> |
<a href="https://www.facebook.com/skywalkr12" target="_blank">Facebook</a>
</div>
""",
    unsafe_allow_html=True,
)
