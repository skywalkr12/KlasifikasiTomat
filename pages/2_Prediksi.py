# prediksi.py
# -- Gate "tomato-only" sederhana + Prediksi + Grad-CAM
# -- + Deteksi Kekuningan & Kelayuan
# -- + Mode CAM: Standar / Brown-aware / Wilt-aware / Brown+Wilt
# requires: opencv-python-headless>=4.9.0

import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import cv2

from helper import (
    load_model,
    show_prediction_and_cam,
    CLASS_NAMES
)

# ========= Util kecil =========
def _normalize01(arr):
    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    return (arr - mn) / (mx - mn + 1e-8)

def _overlay_jet(pil_img, cam01, alpha=0.45):
    base = np.asarray(pil_img.convert("RGB")).astype(np.float32) / 255.0
    hm = (np.clip(cam01, 0.0, 1.0) * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    out = (1 - alpha) * base + alpha * heat
    out = np.clip(out, 0, 1)
    return Image.fromarray((out*255).astype(np.uint8))

# ========== Gate TOMATO-ONLY (LAB + anti-skin, ringkas) ==========
def _leaf_mask_lab_union(img_rgb,
                         L_min=25, L_max=245,
                         a_green_max=-5, a_brown_min=12, b_yellow_min=10):
    """Mask awal berbasis LAB untuk green/yellow/brown."""
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

def _largest_component(mask01):
    """Ambil komponen terbesar sebagai mask daun; return (comp, frac, solidity)."""
    num, labels = cv2.connectedComponents(mask01)
    if num <= 1:
        return np.zeros_like(mask01, dtype=np.uint8), 0.0, 0.0
    best, area = 0, 0
    for lb in range(1, num):
        a = int((labels == lb).sum())
        if a > area:
            best, area = lb, a
    comp = (labels == best).astype(np.uint8)
    frac = comp.mean()
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return comp, float(frac), 0.0
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    sol = float(cv2.contourArea(cnt) / (cv2.contourArea(hull) + 1e-6))
    return comp, float(frac), float(sol)

def _green_ratio_hsv(img_rgb, mask01):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    green = ((H>=35) & (H<=85) & (S>=28) & (V>=40)).astype(np.uint8)
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
                min_mask_frac=0.06, max_mask_frac=0.90, min_solidity=0.22,
                min_green_ratio=0.10, max_skin_in_mask=0.35):
    """
    Return:
      accept(bool), info(dict: metrics..., 'leaf_mask': uint8 mask komponen terbesar)
    """
    rgb = np.array(pil_image.convert("RGB"))
    union = _leaf_mask_lab_union(rgb)
    comp, frac, sol = _largest_component(union)                # ← hanya komponen terbesar
    green_r   = _green_ratio_hsv(rgb, comp)
    skin_r    = _skin_in_mask_ratio_ycrcb(rgb, comp)

    reasons = []
    if frac < min_mask_frac: reasons.append(f"mask kecil ({frac:.2f})")
    if frac > max_mask_frac: reasons.append(f"mask terlalu besar ({frac:.2f})")
    if sol  < min_solidity:  reasons.append(f"solidity rendah ({sol:.2f})")
    if green_r < min_green_ratio: reasons.append(f"hijau rendah ({green_r:.2f})")
    if skin_r  > max_skin_in_mask: reasons.append(f"pola kulit terdeteksi ({skin_r:.2f})")

    return (len(reasons) == 0), {
        "mask_frac": frac, "solidity": sol,
        "green_ratio": green_r, "skin_ratio": skin_r,
        "reasons": reasons,
        "leaf_mask": comp
    }
# ========== END Gate ==========

# ========== Analisis warna & prior ==========
def _color_masks_hsv(img_rgb, leaf_mask01):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    green  = ((H>=35) & (H<=85)  & (S>=28) & (V>=40)).astype(np.uint8)
    yellow = ((H>=20) & (H<=35)  & (S>=60) & (V>=60)).astype(np.uint8)
    brown1 = ((H>=5)  & (H<=20)  & (S>=50) & (V>=25) & (V<=210)).astype(np.uint8)
    brown2 = ((H<5)               & (S>=60) & (V>=15) & (V<=180)).astype(np.uint8)
    brown  = (brown1 | brown2).astype(np.uint8)

    green  = green  & leaf_mask01
    yellow = yellow & leaf_mask01
    brown  = brown  & leaf_mask01
    total  = np.clip(green | yellow | brown, 0, 1).astype(np.uint8)
    area = int(total.sum()) + 1

    stats = {
        "green_ratio":  float(green.sum())  / area,
        "yellow_ratio": float(yellow.sum()) / area,
        "brown_ratio":  float(brown.sum())  / area,
        "area_leaf_px": int(area-1)
    }
    return {"green":green, "yellow":yellow, "brown":brown, "total":total}, stats

def _build_brown_prior(mask_brown, leaf_mask01, k_blur=9):
    m = (mask_brown & leaf_mask01).astype(np.uint8)
    prior = cv2.GaussianBlur(m*255, (k_blur, k_blur), 0).astype(np.float32) / 255.0
    if prior.max() > 0:
        prior = prior / prior.max()
    return prior

def _build_wilt_prior_inside(leaf_mask01):
    """
    Wilt prior DI-DALAM daun:
      - inner edge band = leaf - erode(leaf)
      - concavity touch = dilate(hull-leaf) ∩ inner_edge_band
      - prior = 0.7*concavity_touch + 0.3*inner_edge_strength (LoG)
    """
    leaf = leaf_mask01.astype(np.uint8)
    if leaf.sum() == 0:
        return np.zeros_like(leaf, dtype=np.float32)

    # inner edge band (pita 1-2 px di dalam daun)
    k = np.ones((3,3), np.uint8)
    inner_edge = cv2.subtract(leaf, cv2.erode(leaf, k, 1))

    # convex hull deficit lalu proyeksikan ke tepi dalam
    cnts, _ = cv2.findContours(leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros_like(leaf, dtype=np.float32)
    cnt  = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    hull_mask = np.zeros_like(leaf, dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, hull, 1)
    deficit_outside = cv2.subtract(hull_mask, leaf)           # di LUAR daun
    concavity_touch = cv2.dilate(deficit_outside, k, 1) & inner_edge  # hanya tepi-dalam yg bersentuhan

    # edge strength di dalam daun (LoG ringan pada grayscale)
    # siapkan grayscale dibatasi ke leaf (pakai distance untuk keamanan bila tak ada gambar)
    edge_strength = inner_edge.copy().astype(np.float32)
    edge_strength = cv2.GaussianBlur(edge_strength, (5,5), 0)

    prior = 0.7*concavity_touch.astype(np.float32) + 0.3*edge_strength
    prior = cv2.GaussianBlur(prior, (7,7), 0)
    prior = prior * (leaf>0)  # clamp ke dalam daun
    if prior.max() > 1e-6:
        prior = prior / prior.max()
    return prior

def _make_color_overlay(pil_img, masks, alpha=0.45):
    base = np.asarray(pil_img.convert("RGB")).astype(np.float32)
    overlay = base.copy()
    color_map = {"yellow": [255,255,0], "brown": [255,80,0], "green": [0,255,0]}
    for key in ["yellow", "brown", "green"]:
        m = masks[key].astype(bool)
        if m.any():
            layer = np.zeros_like(base); layer[m] = np.array(color_map[key], np.float32)
            overlay = (1-alpha)*overlay + alpha*layer
    return Image.fromarray(np.clip(overlay,0,255).astype(np.uint8))

# ========== Streamlit ==========
st.set_page_config(page_title="Prediksi Tomat + Grad-CAM (Brown/Wilt-aware)", layout="wide")
st.title("🔍 Prediksi Penyakit Tomat + Grad-CAM (Mode Brown/Wilt) + Deteksi Kekuningan/Kelayuan")

if "history" not in st.session_state:
    st.session_state["history"] = []

DISPLAY_CAP = 0.9999
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
    cam_mode = st.selectbox(
        "Mode CAM",
        options=["Standar", "Brown-aware", "Wilt-aware", "Brown+Wilt"],
        index=0
    )
    alpha = st.slider("Transparansi Heatmap (α)", 0.0, 1.0, 0.45, 0.05)
    topk  = st.slider("Jumlah alternatif (Top-k)", 1, min(5, len(CLASS_NAMES)), 3, 1)
    st.markdown("---")
    mask_bg = st.checkbox("Mask background (fokus ke daun)", True)
    blend_with_res2 = st.checkbox("Blend dengan res2 (stabilkan semantik)", True)
    erode_border = st.checkbox("Erosi tepi mask 1px (redam pinggiran daun)", True)
    lesion_boost = st.checkbox("Deteksi bintik (aktifkan lesion prior)", True)
    lesion_weight = st.slider("Bobot deteksi bintik (lesion prior)", 0.0, 1.0, 0.5, 0.05)
    st.markdown("---")
    show_full_chart = st.checkbox("Tampilkan chart probabilitas lengkap", True)
    sort_desc = st.checkbox("Urutkan chart menurun", True)

# ----- Model -----
model = load_model(weights_path="model/resnet9(99,16).pt")

# ----- Uploader -----
uploaded_file = st.file_uploader("Upload gambar daun tomat", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    rgb = np.array(image)

    # 1) Gate (dengan mask komponen terbesar)
    accept, info_gate = tomato_gate(image)
    if not accept:
        st.error("❌ Ditolak: bukan daun tomat / kualitas kurang memadai → " + ", ".join(info_gate["reasons"]))
        st.stop()
    leaf_mask01 = info_gate["leaf_mask"].astype(np.uint8)

    # 2) Prediksi + CAM dasar (sudah termask di helper bila mask_bg=True)
    overlay_std, cam_base, used_idx, probs_raw = show_prediction_and_cam(
        model, image,
        alpha=alpha, topk=topk, target_layer_name=target_layer_name,
        include_brown=True, lesion_boost=lesion_boost, lesion_weight=lesion_weight,
        mask_bg=mask_bg, blend_with_res2=blend_with_res2, erode_border=erode_border
    )

    # 3) Warna & prior (dibatasi leaf_mask)
    color_masks, color_stats = _color_masks_hsv(rgb, leaf_mask01)
    brown_prior = _build_brown_prior(color_masks["brown"], leaf_mask01, k_blur=9)
    wilt_prior  = _build_wilt_prior_inside(leaf_mask01)

    # 4) Re-weight CAM + clamp ke daun lagi (anti bocor)
    cam_mod = cam_base.copy()
    if cam_mode == "Brown-aware":
        cam_mod = _normalize01(cam_base * (1.0 + 1.0 * brown_prior))
    elif cam_mode == "Wilt-aware":
        cam_mod = _normalize01(cam_base * (1.0 + 1.0 * wilt_prior))
    elif cam_mode == "Brown+Wilt":
        cam_mod = _normalize01(cam_base * (1.0 + 0.9 * brown_prior + 0.9 * wilt_prior))
    # clamp to leaf area AGAIN
    cam_mod = cam_mod * (leaf_mask01.astype(np.float32))

    overlay = _overlay_jet(image, cam_mod, alpha=alpha)
    color_overlay = _make_color_overlay(image, color_masks, alpha=0.45)

    # 5) Shape metrics untuk ringkasan kelayuan
    # (kita hitung cepat lagi di sini)
    def _shape_metrics_for_wilt(leaf_mask01):
        num, labels = cv2.connectedComponents(leaf_mask01)
        if num <= 1: return {"solidity": 0.0, "roughness": 0.0}
        best, area = 0, 0
        for lb in range(1, num):
            a = int((labels == lb).sum())
            if a > area:
                best, area = lb, a
        comp = (labels == best).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return {"solidity": 0.0, "roughness": 0.0}
        cnt  = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        area_cnt  = max(cv2.contourArea(cnt), 1.0)
        area_hull = max(cv2.contourArea(hull), 1.0)
        perim = cv2.arcLength(cnt, closed=True)
        solidity = float(area_cnt / area_hull)
        shape_factor = float((perim**2) / (4.0 * np.pi * area_cnt))
        roughness = float(np.clip((shape_factor - 1.0) / 1.2, 0.0, 1.0))
        return {"solidity": solidity, "roughness": roughness}

    shape_stats = _shape_metrics_for_wilt(leaf_mask01)
    chlorosis_score = float(np.clip(0.7*(color_stats["yellow_ratio"]/0.25) + 0.3*((1.0-color_stats["green_ratio"])/0.5), 0.0, 1.0))
    wilt_score      = float(np.clip(0.6*((1.0 - shape_stats["solidity"])/0.75) + 0.4*(shape_stats["roughness"]), 0.0, 1.0))

    # === Panel tampilan ===
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.image(image, caption="Input", width="stretch")
        st.caption(
            f"Gate: mask_frac={info_gate['mask_frac']:.2f} • "
            f"solidity={info_gate['solidity']:.2f} • "
            f"green={info_gate['green_ratio']:.2f} • "
            f"skin={info_gate['skin_ratio']:.2f}"
        )
    with col2:
        st.image(
            overlay,
            caption=f"Grad-CAM ({cam_mode}, {target_layer_name}) → {CLASS_NAMES[used_idx]} • Confidence: {fmt_pct(probs_raw[used_idx])}",
            width="stretch"
        )
    with col3:
        prior_combo = _normalize01(0.5*brown_prior + 0.5*wilt_prior) * leaf_mask01
        st.image(_overlay_jet(image, prior_combo, alpha=0.45),
                 caption="Peta Prior (dibatasi daun)", width="stretch")

    # Ringkasan metrik
    st.subheader("📊 Deteksi Kekuningan & Kelayuan")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1: st.metric("Rasio Kuning", f"{color_stats['yellow_ratio']*100:.2f}%")
    with mcol2: st.metric("Rasio Cokelat", f"{color_stats['brown_ratio']*100:.2f}%")
    with mcol3: st.metric("Solidity", f"{shape_stats['solidity']:.2f}")
    with mcol4: st.metric("Roughness", f"{shape_stats['roughness']:.2f}")

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.markdown(f"**Skor Kekuningan (0–1):** `{chlorosis_score:.2f}`")
        st.progress(min(max(chlorosis_score,0.0),1.0))
    with pcol2:
        st.markdown(f"**Skor Kelayuan (0–1):** `{wilt_score:.2f}`")
        st.progress(min(max(wilt_score,0.0),1.0))

    # Alternatif (Top-k)
    topk_ = min(topk, len(CLASS_NAMES))
    order = np.argsort(-probs_raw)[:topk_]
    st.markdown("**Alternatif (Top-k)**")
    st.markdown("\n".join([
        f"{'★' if i==used_idx else '•'} {CLASS_NAMES[i]}: {fmt_pct(probs_raw[i])}"
        for i in order
    ]))

    # Chart probabilitas (opsional)
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

    # Histori
    st.session_state["history"].append({
        "Tanggal": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Nama File": uploaded_file.name,
        "Prediksi": CLASS_NAMES[used_idx],
        "Probabilitas (display)": fmt_pct(probs_raw[used_idx]),
        "Layer": target_layer_name,
        "CAM_Mode": cam_mode,
        "MaskBG": mask_bg,
        "BlendRes2": blend_with_res2,
        "ErodeBorder": erode_border,
        "LesionBoost": lesion_boost,
        "LesionWeight": lesion_weight,
        "Rasio_Kuning": f"{color_stats['yellow_ratio']*100:.2f}%",
        "Rasio_Cokelat": f"{color_stats['brown_ratio']*100:.2f}%",
        "Solidity": f"{shape_stats['solidity']:.2f}",
        "Roughness": f"{shape_stats['roughness']:.2f}",
        "Skor_Chlorosis": f"{chlorosis_score:.2f}",
        "Skor_Wilt": f"{wilt_score:.2f}"
    })

# Riwayat + unduh
if st.session_state["history"]:
    st.subheader("📜 Histori Prediksi")
    df = pd.DataFrame(st.session_state["history"])
    st.dataframe(df, width="stretch")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "histori_prediksi.csv", "text/csv")

st.write("""
Catatan: CAM 'Brown/Wilt-aware' hanya membimbing heatmap ke area yang heuristiknya cocok (warna cokelat / tepi berkonkav).
Ini bukan bukti patologi final. Gunakan bersama interpretasi ahli.
""")
