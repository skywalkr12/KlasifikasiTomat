# prediksi.py
# — Prediksi + Grad-CAM (robust res2) + Deteksi Kekuningan/Kelayuan —
# requires: opencv-python-headless>=4.9.0, matplotlib

import streamlit as st
from PIL import Image
import numpy as np, cv2, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime

from helper import load_model, show_prediction_and_cam, CLASS_NAMES

# ===== util =====
def _normalize01(x):
    x = x.astype(np.float32); mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn + 1e-8)

def _overlay_jet(pil_img, cam01, alpha=0.45):
    base = np.asarray(pil_img.convert("RGB")).astype(np.float32) / 255.0
    hm = (np.clip(cam01, 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    out = np.clip((1 - alpha) * base + alpha * heat, 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8))

# ===== Gate sederhana (LAB + anti-skin) =====
def _leaf_mask_lab_union(img_rgb, L_min=25, L_max=245, a_green_max=-5, a_brown_min=12, b_yellow_min=10):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, A, B = lab[...,0], lab[...,1], lab[...,2]
    a = A.astype(np.int16) - 128; b = B.astype(np.int16) - 128
    green  = (a <= a_green_max) & (L >= L_min) & (L <= L_max)
    yellow = (a >  a_green_max) & (a < a_brown_min) & (b >= b_yellow_min) & (L >= L_min) & (L <= L_max)
    brown  = (a >= a_brown_min) & (b >= b_yellow_min) & (L >= L_min)
    m = (green | yellow | brown).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((5,5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    m = cv2.dilate(m, np.ones((3,3), np.uint8), 1)
    return m

def _largest_component(mask01):
    num, labels = cv2.connectedComponents(mask01)
    if num <= 1: return np.zeros_like(mask01, np.uint8), 0.0, 0.0
    best, area = 0, 0
    for lb in range(1, num):
        a = int((labels == lb).sum())
        if a > area: best, area = lb, a
    comp = (labels == best).astype(np.uint8)
    frac = comp.mean()
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return comp, float(frac), 0.0
    cnt = max(cnts, key=cv2.contourArea); hull = cv2.convexHull(cnt)
    sol = float(cv2.contourArea(cnt) / (cv2.contourArea(hull) + 1e-6))
    return comp, float(frac), float(sol)

def _green_ratio_hsv(img_rgb, mask01):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    green = ((H>=35) & (H<=85) & (S>=28) & (V>=40)).astype(np.uint8)
    g_in = int((green & mask01).sum()); area = int(mask01.sum()) + 1
    return float(g_in) / float(area)

def _skin_in_mask_ratio_ycrcb(img_rgb, mask01):
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    Y, Cr, Cb = ycrcb[...,0], ycrcb[...,1], ycrcb[...,2]
    skin = ((Cr>=135) & (Cr<=180) & (Cb>=85) & (Cb<=135)).astype(np.uint8)
    s_in = int((skin & mask01).sum()); area = int(mask01.sum()) + 1
    return float(s_in) / float(area)

def tomato_gate(pil_image, min_mask_frac=0.06, max_mask_frac=0.90, min_solidity=0.22, min_green_ratio=0.10, max_skin_in_mask=0.35):
    rgb = np.array(pil_image.convert("RGB"))
    union = _leaf_mask_lab_union(rgb)
    comp, frac, sol = _largest_component(union)
    green_r = _green_ratio_hsv(rgb, comp); skin_r = _skin_in_mask_ratio_ycrcb(rgb, comp)
    reasons = []
    if frac < min_mask_frac: reasons.append(f"mask kecil ({frac:.2f})")
    if frac > max_mask_frac: reasons.append(f"mask terlalu besar ({frac:.2f})")
    if sol  < min_solidity:  reasons.append(f"solidity rendah ({sol:.2f})")
    if green_r < min_green_ratio: reasons.append(f"hijau rendah ({green_r:.2f})")
    if skin_r  > max_skin_in_mask: reasons.append(f"pola kulit terdeteksi ({skin_r:.2f})")
    return (len(reasons) == 0), {"mask_frac": frac, "solidity": sol, "green_ratio": green_r, "skin_ratio": skin_r, "reasons": reasons, "leaf_mask": comp}

# ===== Warna & prior =====
def _color_masks_hsv(img_rgb, leaf_mask01):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    green  = ((H>=35) & (H<=85)  & (S>=28) & (V>=40)).astype(np.uint8)
    yellow = ((H>=20) & (H<=35)  & (S>=60) & (V>=60)).astype(np.uint8)
    brown1 = ((H>=5)  & (H<=20)  & (S>=50) & (V>=25) & (V<=210)).astype(np.uint8)
    brown2 = ((H<5)               & (S>=60) & (V>=15) & (V<=180)).astype(np.uint8)
    brown  = (brown1 | brown2).astype(np.uint8)
    green, yellow, brown = green & leaf_mask01, yellow & leaf_mask01, brown & leaf_mask01
    total = np.clip(green | yellow | brown, 0, 1).astype(np.uint8)
    area = int(total.sum()) + 1
    stats = {"green_ratio": float(green.sum())/area, "yellow_ratio": float(yellow.sum())/area, "brown_ratio": float(brown.sum())/area, "area_leaf_px": int(area-1)}
    return {"green":green,"yellow":yellow,"brown":brown,"total":total}, stats

def _build_brown_prior(mask_brown, leaf_mask01, k_blur=9):
    m = (mask_brown & leaf_mask01).astype(np.uint8)
    prior = cv2.GaussianBlur(m*255, (k_blur,k_blur), 0).astype(np.float32)/255.0
    prior = prior * (leaf_mask01.astype(np.float32))
    if prior.max() <= 1e-6:  # fallback anti-hijau supaya tetap ada sinyal
        prior = (leaf_mask01.astype(np.float32)) - cv2.GaussianBlur((leaf_mask01 & (1-mask_brown)).astype(np.float32), (7,7), 0)
        prior = np.clip(prior, 0, 1)
    return prior / (prior.max() + 1e-8)

def _build_wilt_prior_inside(leaf_mask01):
    leaf = leaf_mask01.astype(np.uint8)
    if leaf.sum() == 0: return np.zeros_like(leaf, np.float32)
    k = np.ones((3,3), np.uint8)
    inner = cv2.subtract(leaf, cv2.erode(leaf, k, 1))  # pita tepi di dalam daun
    prior = cv2.GaussianBlur(inner.astype(np.float32), (7,7), 0)
    prior = prior * (leaf>0)
    return prior / (prior.max() + 1e-8)

def _make_color_overlay(pil_img, masks, alpha=0.45):
    base = np.asarray(pil_img.convert("RGB")).astype(np.float32)
    overlay = base.copy()
    cmap = {"yellow":[255,255,0], "brown":[255,80,0], "green":[0,255,0]}
    for key in ["yellow","brown","green"]:
        m = masks[key].astype(bool)
        if m.any():
            layer = np.zeros_like(base); layer[m] = np.array(cmap[key], np.float32)
            overlay = (1-alpha)*overlay + alpha*layer
    return Image.fromarray(np.clip(overlay,0,255).astype(np.uint8))

# ===== UI =====
st.set_page_config(page_title="Prediksi Tomat + Grad-CAM (res2 fix)", layout="wide")
st.title("🔍 Prediksi Penyakit Tomat + Grad-CAM (res2 diperbaiki) + Kekuningan/Kelayuan")

if "history" not in st.session_state: st.session_state["history"] = []
DISPLAY_CAP = 0.9999
def fmt_pct(p, cap=DISPLAY_CAP, decimals=2):
    q = p if p < cap else cap
    return f"{q*100:.{decimals}f}%"

with st.sidebar:
    st.header("Pengaturan Visualisasi")
    target_layer_name = st.selectbox("Layer target Grad-CAM", ["conv4_prepool","conv3_prepool","conv2_prepool","res2"], index=0)
    cam_mode = st.selectbox("Mode CAM", ["Standar","Brown-aware","Wilt-aware","Brown+Wilt"], index=0)
    alpha = st.slider("Transparansi Heatmap (α)", 0.0, 1.0, 0.45, 0.05)
    topk  = st.slider("Jumlah alternatif (Top-k)", 1, min(5, len(CLASS_NAMES)), 3, 1)
    st.markdown("---")
    mask_bg = st.checkbox("Mask background (fokus ke daun)", True)
    blend_with_res2 = st.checkbox("Blend dengan res2 (stabilkan semantik)", True)
    erode_border = st.checkbox("Erosi tepi mask 1px", True)
    lesion_boost = st.checkbox("Lesion prior (brown/dark) di CAM", True)
    lesion_weight = st.slider("Bobot lesion prior", 0.0, 1.0, 0.5, 0.05)
    st.markdown("---")
    show_full_chart = st.checkbox("Tampilkan chart probabilitas lengkap", True)
    sort_desc = st.checkbox("Urutkan chart menurun", True)

model = load_model(weights_path="model/resnet9(99,16).pt")

uploaded_file = st.file_uploader("Upload gambar daun tomat", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    rgb = np.array(image)

    # 1) Gate + mask komponen terbesar
    accept, info_gate = tomato_gate(image)
    if not accept:
        st.error("❌ Ditolak: bukan daun tomat / kualitas kurang → " + ", ".join(info_gate["reasons"]))
        st.stop()
    leaf_mask01 = info_gate["leaf_mask"].astype(np.uint8)
    if leaf_mask01.mean() < 1e-4:
        st.warning("Mask daun sangat kecil; CAM tidak akan dipotong (auto-bypass).")

    # 2) Prediksi + CAM dasar (pakai external mask & res2 fallback)
    overlay_std, cam_base, used_idx, probs_raw = show_prediction_and_cam(
        model, image,
        alpha=alpha, topk=topk, target_layer_name=target_layer_name,
        include_brown=True, lesion_boost=lesion_boost, lesion_weight=lesion_weight,
        mask_bg=mask_bg, blend_with_res2=blend_with_res2, erode_border=erode_border,
        external_leaf_mask=leaf_mask01, auto_res2_fallback=True
    )

    # 3) Priors di dalam daun (selalu non-zero setelah normalisasi)
    color_masks, color_stats = _color_masks_hsv(rgb, leaf_mask01)
    brown_prior = _build_brown_prior(color_masks["brown"], leaf_mask01, k_blur=9)
    wilt_prior  = _build_wilt_prior_inside(leaf_mask01)

    # 4) Re-weight CAM sesuai mode
    cam_mod = cam_base.copy()
    if cam_mode == "Brown-aware":
        cam_mod = _normalize01(cam_base * (1.0 + brown_prior))
    elif cam_mode == "Wilt-aware":
        cam_mod = _normalize01(cam_base * (1.0 + wilt_prior))
    elif cam_mode == "Brown+Wilt":
        cam_mod = _normalize01(cam_base * (1.0 + 0.9*brown_prior + 0.9*wilt_prior))

    # 5) Overlay & prior debug
    overlay = _overlay_jet(image, cam_mod * (leaf_mask01.astype(np.float32) if mask_bg else 1.0), alpha=alpha)
    prior_combo = _normalize01((0.5*brown_prior + 0.5*wilt_prior) * (leaf_mask01.astype(np.float32)))

    # === Tampilan ===
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.image(image, caption="Input", width='stretch')
        st.caption(f"Gate: mask_frac={info_gate['mask_frac']:.2f} • solidity={info_gate['solidity']:.2f} • green={info_gate['green_ratio']:.2f} • skin={info_gate['skin_ratio']:.2f}")
    with col2:
        st.image(overlay, caption=f"Grad-CAM ({cam_mode}, {target_layer_name}) → {CLASS_NAMES[used_idx]} • Confidence: {probs_raw[used_idx]*100:.2f}%", width='stretch')
    with col3:
        st.image(_overlay_jet(image, prior_combo, alpha=0.45), caption="Peta Prior (dibatasi daun)", width='stretch')

    # Ringkasan & histori
    st.subheader("📊 Kekuningan & Kelayuan")
    m1,m2,m3,m4 = st.columns(4)
    with m1: st.metric("Rasio Kuning", f"{color_stats['yellow_ratio']*100:.2f}%")
    with m2: st.metric("Rasio Cokelat", f"{color_stats['brown_ratio']*100:.2f}%")
    # metrik bentuk kecil (kasar)
    def _shape_metrics(leaf_mask01):
        num, labels = cv2.connectedComponents(leaf_mask01)
        if num<=1: return 0.0, 0.0
        best, area = 0, 0
        for lb in range(1,num):
            a=int((labels==lb).sum()); 
            if a>area: best, area = lb, a
        comp=(labels==best).astype(np.uint8)
        cnts,_=cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return 0.0, 0.0
        cnt=max(cnts, key=cv2.contourArea)
        hull=cv2.convexHull(cnt)
        area_cnt=max(cv2.contourArea(cnt),1.0)
        area_hull=max(cv2.contourArea(hull),1.0)
        perim=cv2.arcLength(cnt,True)
        solidity=float(area_cnt/area_hull)
        shape_factor=float((perim**2)/(4.0*np.pi*area_cnt))
        rough=float(np.clip((shape_factor-1.0)/1.2,0.0,1.0))
        return solidity, rough
    sol, rough = _shape_metrics(leaf_mask01)
    with m3: st.metric("Solidity", f"{sol:.2f}")
    with m4: st.metric("Roughness", f"{rough:.2f}")

    topk_ = min(topk, len(CLASS_NAMES))
    order = np.argsort(-probs_raw)[:topk_]
    st.markdown("**Alternatif (Top-k)**")
    st.markdown("\n".join([f"{'★' if i==used_idx else '•'} {CLASS_NAMES[i]}: {fmt_pct(float(probs_raw[i]))}" for i in order]))

    if st.checkbox("Tampilkan chart probabilitas lengkap", value=show_full_chart):
        probs_plot = np.minimum(np.array(probs_raw, dtype=float), DISPLAY_CAP)
        idxs = np.argsort(-probs_plot) if sort_desc else np.arange(len(CLASS_NAMES))
        fig, ax = plt.subplots()
        ax.barh([CLASS_NAMES[i] for i in idxs], probs_plot[idxs], height=0.6)
        ax.invert_yaxis(); ax.set_xlim(0,1)
        ax.set_xlabel("Probabilitas (dibatasi < 100%)"); ax.set_ylabel("Kelas")
        st.pyplot(fig)

    st.session_state["history"].append({
        "Tanggal": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Nama File": uploaded_file.name,
        "Prediksi": CLASS_NAMES[used_idx],
        "Probabilitas": f"{probs_raw[used_idx]*100:.2f}%",
        "Layer": target_layer_name, "CAM_Mode": cam_mode
    })

if st.session_state["history"]:
    st.subheader("📜 Histori Prediksi")
    df = pd.DataFrame(st.session_state["history"])
    st.dataframe(df, width='stretch')
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "histori_prediksi.csv", "text/csv")
