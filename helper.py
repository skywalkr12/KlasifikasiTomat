# helper.py — Grad-CAM untuk ResNet9-variant
# Fokus: cegah bayangan ikut terdeteksi tanpa mematikan nekrosis cokelat.

import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import cm

# ===== Streamlit (opsional) =====
try:
    import streamlit as st
except ImportError:
    class _Dummy:
        def cache_resource(self, **kw):
            def deco(f): return f
            return deco
    st = _Dummy()

# ===== Utils model =====
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class SimpleResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 3, 1, 1)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(3, 3, 3, 1, 1)
        self.relu2 = nn.ReLU(inplace=False)
    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.conv2(out)
        return self.relu2(out) + x

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        return F.cross_entropy(out, labels)
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}
    def validation_epoch_end(self, outputs):
        losses = torch.stack([x['val_loss'] for x in outputs]).mean()
        accs   = torch.stack([x['val_acc'] for x in outputs]).mean()
        return {'val_loss': losses.item(), 'val_acc': accs.item()}

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False)
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

# ===== Model (ResNet9-variant; nama dibiarkan ResNet18 agar kompatibel) =====
class ResNet18(ImageClassificationBase):
    def __init__(self, num_diseases=10, in_channels=3):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)               # 256
        self.conv2 = ConvBlock(64, 128, pool=True)            # 64
        self.res1  = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)           # 16
        self.conv4 = ConvBlock(256, 512, pool=True)           # 4
        self.res2  = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten(), nn.Dropout(0.3), nn.Linear(512, num_diseases))
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out); out = self.res1(out) + out
        out = self.conv3(out); out = self.conv4(out)
        out = self.res2(out) + out
        return self.classifier(out)

# ===== Kelas =====
CLASS_NAMES = [
 'Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight',
 'Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato__Tomato_mosaic_virus','Tomato_healthy'
]

# ===== Transform (samakan dengan training!) =====
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
    # Jika training pakai Normalize, aktifkan lagi di sini.
    # transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def _disable_inplace_relu(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.ReLU): m.inplace = False
    return model

@st.cache_resource
def load_model(cache_bust: str = "noinplace-v7"):
    model = ResNet18(num_diseases=len(CLASS_NAMES), in_channels=3)
    sd = torch.load("model/resnet_97_56.pt", map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd: sd = sd["model_state_dict"]
    sd = {(k.replace("module.","") if k.startswith("module.") else k): v for k,v in sd.items()}
    model.load_state_dict(sd, strict=True)
    _disable_inplace_relu(model)
    model.eval()
    return model

@torch.no_grad()
def predict_image(model, image):
    x = transform(image).unsqueeze(0)
    out = model(x)
    probs = torch.softmax(out[0], dim=0)
    idx = torch.argmax(probs).item()
    return CLASS_NAMES[idx], probs.numpy()

# ====== Morfologi kecil (tanpa OpenCV) ======
def _dilate(mask: np.ndarray, k: int = 5) -> np.ndarray:
    pad = k // 2
    padded = np.pad(mask, ((pad,pad),(pad,pad)), mode="edge")
    out = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            out[i,j] = padded[i:i+k, j:j+k].max()
    return out

def _erode(mask: np.ndarray, k: int = 5) -> np.ndarray:
    pad = k // 2
    padded = np.pad(mask, ((pad,pad),(pad,pad)), mode="edge")
    out = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            out[i,j] = padded[i:i+k, j:j+k].min()
    return out

def _close(mask: np.ndarray, k: int = 5) -> np.ndarray:
    return _erode(_dilate(mask, k), k)

def _largest_cc(mask: np.ndarray) -> np.ndarray:
    # ambil komponen terkoneksi terbesar (4-connected)
    H, W = mask.shape
    visited = np.zeros((H,W), dtype=bool)
    best_cnt, best = 0, None
    for i in range(H):
        for j in range(W):
            if mask[i,j] and not visited[i,j]:
                stack = [(i,j)]; visited[i,j] = True; cnt = 0; comp = []
                while stack:
                    y,x = stack.pop(); comp.append((y,x)); cnt += 1
                    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = y+dy, x+dx
                        if 0<=ny<H and 0<=nx<W and mask[ny,nx] and not visited[ny,nx]:
                            visited[ny,nx]=True; stack.append((ny,nx))
                if cnt > best_cnt: best_cnt, best = cnt, comp
    out = np.zeros_like(mask)
    if best is not None:
        for (y,x) in best: out[y,x] = 1
    return out

# ====== Mask daun dgn penjaga bayangan ======
def _leaf_mask_shadow_guard(pil_img: Image.Image, k_close:int=5, k_halo:int=5) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Return (mask_leaf 0/1, S_norm, brownness_norm)."""
    hsv = np.asarray(pil_img.convert("HSV")).astype(np.int32)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    rgb = np.asarray(pil_img.convert("RGB")).astype(np.float32)/255.0
    r,g,b = rgb[...,0], rgb[...,1], rgb[...,2]

    green  = (H>=60) & (H<=130) & (S>=50) & (V>=35)
    yellow = (H>=35) & (H<=59)  & (S>=50) & (V>=45)
    brown  = (H>=8)  & (H<=35)  & (S>=45) & (V>=20) & (V<=210)
    base = (green | yellow | brown).astype(np.uint8)
    base = _close(base, k_close)
    base = _largest_cc(base)                 # ambil daun utama

    # piksel gelap yang “cokelat-ish” (bukan abu-abu)
    very_dark_brownish = ((V<=60) & (S>=35) & (((H>=8)&(H<=35)) | ((H<8)&(S>=60))))  # oranye/brown gelap
    halo = _dilate(base, k_halo)
    mask = np.clip(base + (very_dark_brownish & (halo>0)).astype(np.uint8), 0, 1)

    # normalisasi pembantu
    S_norm = S.astype(np.float32)/255.0
    brownness = np.clip((r-g),0,1) + np.clip((r-b),0,1)
    if brownness.max()>0: brownness /= brownness.max()
    return mask.astype(np.float32), S_norm, brownness

# ====== Lesion prior (brown + dark, downweight shadow) ======
def _lesion_prior(pil_img: Image.Image, shadow_guard: bool, S_norm: np.ndarray | None) -> np.ndarray:
    rgb = np.asarray(pil_img.convert("RGB")).astype(np.float32)/255.0
    r,g,b = rgb[...,0], rgb[...,1], rgb[...,2]
    brownness = np.clip((r-g),0,1) + np.clip((r-b),0,1)
    if brownness.max()>0: brownness /= brownness.max()
    V = np.asarray(pil_img.convert("HSV"))[...,2].astype(np.float32)/255.0
    darkness = np.clip((0.8 - V)/0.8, 0, 1)
    prior = 0.6*brownness + 0.4*darkness
    # haluskan 3x3
    k=3; pad=k//2
    padded = np.pad(prior, ((pad,pad),(pad,pad)), mode="edge")
    sm = np.zeros_like(prior)
    for i in range(prior.shape[0]):
        for j in range(prior.shape[1]):
            sm[i,j] = padded[i:i+k, j:j+k].mean()
    prior = np.clip(sm, 0, 1)
    if shadow_guard and S_norm is not None:
        # turunkan prior pada area saturasi rendah (bayangan)
        prior = prior * (S_norm**0.6)
    return prior

# ====== Grad-CAM core ======
def _normalize_pos(cam: torch.Tensor):
    cam = torch.relu(cam)
    mn, mx = cam.min(), cam.max()
    return (cam - mn) / (mx - mn + 1e-8) if (mx - mn) > 1e-8 else torch.zeros_like(cam)

def _normalize_signed(cam_raw: torch.Tensor):
    pos = torch.relu(cam_raw); neg = torch.relu(-cam_raw)
    if pos.max()>0: pos/=pos.max()
    if neg.max()>0: neg/=neg.max()
    return pos - neg  # [-1..1]

def _upsample(cam: torch.Tensor, size_hw: tuple[int,int]) -> torch.Tensor:
    cam = cam[None,None,...]
    cam = F.interpolate(cam, size=size_hw, mode="bilinear", align_corners=False)
    return cam[0,0]

def _overlay_rgb(pil_img: Image.Image, heat: np.ndarray, alpha: float=0.45) -> Image.Image:
    base = np.asarray(pil_img.convert("RGB")).astype(np.float32)/255.0
    out = (1-alpha)*base + alpha*heat
    return Image.fromarray((np.clip(out,0,1)*255).astype(np.uint8))

def _overlay_pos(pil_img: Image.Image, cam01: np.ndarray, alpha: float=0.45) -> Image.Image:
    heat = cm.get_cmap("jet")(np.clip(cam01,0,1))[..., :3]
    return _overlay_rgb(pil_img, heat, alpha)

def _overlay_signed(pil_img: Image.Image, cam_signed: np.ndarray, alpha: float=0.45) -> Image.Image:
    norm = (np.clip(cam_signed,-1,1)+1.0)/2.0
    heat = cm.get_cmap("coolwarm")(norm)[..., :3]
    return _overlay_rgb(pil_img, heat, alpha)

def get_target_layer(model: nn.Module, name: str):
    if name == "res2":           return model.res2
    if name == "conv4_prepool":  return model.conv4[1]  # BN
    if name == "conv3_prepool":  return model.conv3[1]  # BN
    if name == "conv2_prepool":  return model.conv2[1]  # BN
    raise ValueError(f"target_layer tidak dikenal: {name}")

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.tl = target_layer
        self.A = None; self.G = None
        self.h1 = self.tl.register_forward_hook(self._fh)
        try:    self.h2 = self.tl.register_full_backward_hook(self._bh)
        except AttributeError: self.h2 = self.tl.register_backward_hook(self._bh)
    def _fh(self, m, inp, out):  self.A = out.detach().clone()
    def _bh(self, m, gin, gout): self.G = gout[0].detach().clone()
    def remove(self): self.h1.remove(); self.h2.remove()
    def compute_raw(self, x: torch.Tensor, class_idx: int | None = None):
        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)
            out = self.model(x)
            probs = torch.softmax(out, dim=1)[0]
            if class_idx is None: class_idx = int(torch.argmax(probs))
            out[0, class_idx].backward(retain_graph=False)
            A = self.A[0]; G = self.G[0]
            w = G.view(G.size(0), -1).mean(dim=1)
            cam_raw = torch.sum(w[:,None,None]*A, dim=0)  # bisa ±
        return cam_raw, class_idx, probs.detach().cpu().numpy()

# ====== API utama ======
def gradcam_on_pil(
    model: nn.Module,
    pil_img: Image.Image,
    target_layer_name: str = "conv3_prepool",
    class_idx: int | None = None,
    alpha: float = 0.45,
    cam_mode: str = "pos",            # "pos" | "abs" | "signed"
    mask_bg: bool = True,
    soft_mask_base: float = 0.2,      # non-leaf tetap diberi bobot base
    include_brown_black: bool = True,
    lesion_boost: bool = True,
    lesion_weight: float = 0.5,
    blend_with_res2: bool = False,
    shadow_guard: bool = True,        # ↓ turunkan bobot shadow
    shadow_gamma: float = 0.6,
):
    x = transform(pil_img).unsqueeze(0)

    # CAM layer target
    tl = get_target_layer(model, target_layer_name)
    eg = GradCAM(model, tl)
    try:
        cam_raw, used_idx, probs = eg.compute_raw(x, class_idx=class_idx)
    finally:
        eg.remove()

    # Blend dgn res2 (upsample terlebih dahulu)
    if blend_with_res2 and target_layer_name != "res2":
        eg2 = GradCAM(model, get_target_layer(model,"res2"))
        try:
            cam2_raw, _, _ = eg2.compute_raw(x, class_idx=used_idx)
        finally:
            eg2.remove()
        cam2_raw = _upsample(cam2_raw, (cam_raw.shape[0], cam_raw.shape[1]))
        cam_raw = 0.6*cam_raw + 0.4*cam2_raw

    # Mode CAM
    if cam_mode == "pos":
        cam = _normalize_pos(cam_raw); signed = None
    elif cam_mode == "abs":
        cam = _normalize_pos(torch.abs(cam_raw)); signed = None
    elif cam_mode == "signed":
        signed = _normalize_signed(cam_raw).cpu(); cam = None
    else:
        raise ValueError("cam_mode harus 'pos', 'abs', atau 'signed'.")

    # Upsample ke ukuran gambar
    H, W = pil_img.size[1], pil_img.size[0]
    cam_up = _upsample(cam if signed is None else signed, (H, W))

    # Mask daun + shadow guard
    S_norm = None; brownness = None
    if mask_bg:
        if include_brown_black:
            m_leaf, S_norm, brownness = _leaf_mask_shadow_guard(pil_img, k_close=5, k_halo=5)  # 0/1
        else:
            hsv = np.asarray(pil_img.convert("HSV")).astype(np.int32)
            Hh,Ss,Vv = hsv[...,0], hsv[...,1], hsv[...,2]
            m_leaf = (((Hh>=60)&(Hh<=130)&(Ss>=50)&(Vv>=35)) | ((Hh>=35)&(Hh<=59)&(Ss>=50)&(Vv>=45))).astype(np.float32)

        w = soft_mask_base + (1.0 - soft_mask_base) * m_leaf  # soft-mask

        if shadow_guard:
            # skor bayangan: gelap & tidak brownish → turunkan bobot
            if S_norm is None:
                hsv = np.asarray(pil_img.convert("HSV")).astype(np.int32)
                S_norm = (hsv[...,1].astype(np.float32)/255.0)
            if brownness is None:
                rgb = np.asarray(pil_img.convert("RGB")).astype(np.float32)/255.0
                r,g,b = rgb[...,0], rgb[...,1], rgb[...,2]
                brownness = np.clip((r-g),0,1) + np.clip((r-b),0,1)
                if brownness.max()>0: brownness /= brownness.max()
            shadow_score = (1.0 - S_norm) * (1.0 - brownness)
            w = w * (1.0 - float(shadow_gamma) * shadow_score)
            w = np.clip(w, 0.0, 1.0)

        cam_up = cam_up * torch.tensor(w, dtype=cam_up.dtype, device=cam_up.device)

    # Lesion prior (downweight shadow via S_norm)
    if lesion_boost and cam_mode != "signed":
        prior = _lesion_prior(pil_img, shadow_guard, S_norm)
        cam_up = cam_up * (1.0 + float(lesion_weight) * torch.tensor(prior, dtype=cam_up.dtype, device=cam_up.device))

    # Overlay
    if cam_mode == "signed":
        cam_np = torch.clamp(cam_up, -1, 1).cpu().numpy()
        overlay = _overlay_signed(pil_img, cam_np, alpha=alpha)
    else:
        cam_np = _normalize_pos(cam_up).cpu().numpy()
        overlay = _overlay_pos(pil_img, cam_np, alpha=alpha)

    pred_label = CLASS_NAMES[used_idx] if 0 <= used_idx < len(CLASS_NAMES) else str(used_idx)
    return overlay, cam_np, pred_label, used_idx, probs

def show_prediction_and_cam(
    model: nn.Module,
    pil_img: Image.Image,
    alpha: float = 0.45,
    topk: int = 3,
    target_layer_name: str = "conv3_prepool",
    cam_mode: str = "pos",
    mask_bg: bool = True,
    soft_mask_base: float = 0.2,
    include_brown_black: bool = True,
    lesion_boost: bool = True,
    lesion_weight: float = 0.5,
    blend_with_res2: bool = False,
    shadow_guard: bool = True,
    shadow_gamma: float = 0.6,
):
    with torch.no_grad():
        x = transform(pil_img).unsqueeze(0)
        out = model(x)
        probs_all = torch.softmax(out[0], dim=0).cpu().numpy()
        used_idx = int(np.argmax(probs_all))
    pred_label = CLASS_NAMES[used_idx]

    overlay, cam, _, _, _ = gradcam_on_pil(
        model, pil_img,
        target_layer_name=target_layer_name,
        class_idx=used_idx,
        alpha=alpha,
        cam_mode=cam_mode,
        mask_bg=mask_bg,
        soft_mask_base=soft_mask_base,
        include_brown_black=include_brown_black,
        lesion_boost=lesion_boost,
        lesion_weight=lesion_weight,
        blend_with_res2=blend_with_res2,
        shadow_guard=shadow_guard,
        shadow_gamma=shadow_gamma
    )

    try:
        col1, col2 = st.columns([1,1])
        with col1:
            st.image(pil_img, caption="Input", use_container_width=True)
            st.write(f"**Prediksi**: {pred_label}  \n**Confidence**: {float(probs_all.max()):.2%}")
            topk_ = min(topk, len(CLASS_NAMES))
            order = np.argsort(-probs_all)[:topk_]
            st.markdown("**Alternatif (Top-k)**")
            st.markdown("\n".join([f"{'★' if i==used_idx else '•'} {CLASS_NAMES[i]}: {probs_all[i]:.2%}" for i in order]))
        with col2:
            mode_text = "Grad-CAM" if cam_mode=="pos" else ("|CAM|" if cam_mode=="abs" else "Signed CAM")
            st.image(overlay, caption=f"{mode_text} ({target_layer_name}) → {pred_label}", use_container_width=True)
    except Exception:
        pass

    return overlay, cam, used_idx, probs_all
