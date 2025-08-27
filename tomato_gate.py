# tomato_gate.py
# Gate sederhana untuk daun tomat berbasis LAB (+opsi HSV), dengan komponen-terbesar,
# metrik green_ratio & skin_ratio. Kompatibel dengan prediksi.py terbaru.

import numpy as np
import cv2
from PIL import Image

# --- Mask LAB: stabil ke bayangan; tangkap hijau–kuning–cokelat daun ---
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
    # morfologi ringan untuk merapikan area
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((5,5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    m = cv2.dilate(m, np.ones((3,3), np.uint8), 1)
    return m

# --- (opsional) Mask HSV: lebih longgar; bisa digabung ke union jika perlu ---
def _leaf_mask_hsv(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    green  = (H>=35)&(H<=85)&(S>=28)&(V>=40)
    yellow = (H>=20)&(H<=34)&(S>=35)&(V>=50)
    m = np.where(green | yellow, 1, 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((5,5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    m = cv2.dilate(m, np.ones((3,3), np.uint8), 1)
    return m

# --- Ambil komponen terbesar + metrik bentuk ---
def _largest_component(mask01):
    num, labels = cv2.connectedComponents(mask01)
    if num <= 1:
        return np.zeros_like(mask01, np.uint8), 0.0, 0, 0.0
    best, area = 0, 0
    for lb in range(1, num):
        a = int((labels == lb).sum())
        if a > area:
            best, area = lb, a
    comp = (labels == best).astype(np.uint8)
    frac = comp.mean()
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return comp, float(frac), int(area), 0.0
    cnt  = max(cnts, key=cv2.contourArea)
    a    = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    ha   = cv2.contourArea(hull) if hull is not None else 1.0
    solidity = float(a / (ha + 1e-6))
    return comp, float(frac), int(a), solidity

# --- Metrik warna & anti-skin ---
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
    # rentang kulit yang umum (longgar)
    skin = ((Cr>=135) & (Cr<=180) & (Cb>=85) & (Cb<=135)).astype(np.uint8)
    s_in = int((skin & mask01).sum())
    area = int(mask01.sum()) + 1
    return float(s_in) / float(area)

# --- API utama gate ---
def tomato_gate(pil_image, mode="union",
                min_mask_frac=0.06, max_mask_frac=0.90, min_solidity=0.22,
                min_green_ratio=0.10,    # bukti hijau minimal (hindari background)
                max_skin_in_mask=0.35):  # jika skin > 35% di dalam mask → tolak
    """
    mode: 'lab' | 'hsv' | 'union' (LAB ∪ HSV)
    return:
      (accept: bool,
       info: dict{
           mask_frac, mask_area, solidity, green_ratio, skin_ratio, reasons,
           leaf_mask  # << komponen terbesar (H,W) {0,1}
       })
    """
    rgb = np.array(pil_image.convert("RGB"))

    m_lab = _leaf_mask_lab(rgb) if mode in ("lab","union") else None
    m_hsv = _leaf_mask_hsv(rgb) if mode in ("hsv","union") else None
    if mode == "lab":   m_union = m_lab
    elif mode == "hsv": m_union = m_hsv
    else:               m_union = ((m_lab | m_hsv) > 0).astype(np.uint8)

    # Pakai komponen terbesar sebagai leaf mask final
    comp, frac, area, sol = _largest_component(m_union)
    green_r = _green_ratio_hsv(rgb, comp)
    skin_r  = _skin_in_mask_ratio_ycrcb(rgb, comp)

    reasons = []
    if frac < min_mask_frac: reasons.append(f"mask kecil ({frac:.2f})")
    if frac > max_mask_frac: reasons.append(f"mask terlalu besar ({frac:.2f})")
    if sol  < min_solidity:  reasons.append(f"solidity rendah ({sol:.2f})")
    if green_r < min_green_ratio: reasons.append(f"hijau rendah ({green_r:.2f})")
    if skin_r  > max_skin_in_mask: reasons.append(f"pola kulit terdeteksi ({skin_r:.2f})")

    return (len(reasons) == 0), {
        "mask_frac": frac, "mask_area": area, "solidity": sol,
        "green_ratio": green_r, "skin_ratio": skin_r,
        "reasons": reasons,
        "leaf_mask": comp  # <- yang dipakai CAM/prior untuk clamp
    }

# --- Visual helper (opsional) ---
def draw_mask_overlay(pil_image, mask01, alpha=0.45, color=(0,255,0)):
    """Overlay mask biner (0/1) di atas gambar."""
    rgb = np.array(pil_image.convert("RGB")).astype(np.float32)
    color_img = np.zeros_like(rgb); color_img[...,0]=color[0]; color_img[...,1]=color[1]; color_img[...,2]=color[2]
    mask3 = mask01[...,None].astype(np.float32)
    out = (1-alpha)*rgb + alpha*(mask3*color_img + (1-mask3)*rgb)
    return Image.fromarray(np.clip(out,0,255).astype(np.uint8))
