# helper.py (No-inplace ReLU + safe BN hooks + cache-bust)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from matplotlib import cm

# Streamlit opsional
try:
    import streamlit as st
except ImportError:
    class _Dummy:
        def cache_resource(self, **kw):
            def deco(f): return f
            return deco
    st = _Dummy()

# ========= Utils =========
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
        accs   = torch.stack([x['val_acc']  for x in outputs]).mean()
        return {'val_loss': losses.item(), 'val_acc': accs.item()}

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False)
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

# ========= Model (ResNet9-variant) =========
class ResNet18(ImageClassificationBase):
    def __init__(self, num_diseases=10, in_channels=3):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)               # 256 -> 256
        self.conv2 = ConvBlock(64, 128, pool=True)            # 256 -> 64
        self.res1  = nn.Sequential(ConvBlock(128, 128),
                                   ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)           # 64 -> 16
        self.conv4 = ConvBlock(256, 512, pool=True)           # 16 -> 4
        self.res2  = nn.Sequential(ConvBlock(512, 512),
                                   ConvBlock(512, 512))
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),  # 4x4 -> 1x1
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, num_diseases)
        )
    def forward(self, xb):
        out = self.conv1(xb)          # 256
        out = self.conv2(out)         # 64
        out = self.res1(out) + out
        out = self.conv3(out)         # 16
        out = self.conv4(out)         # 4
        out = self.res2(out) + out
        return self.classifier(out)

# ========= Kelas =========
CLASS_NAMES = [
 'Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy'
]

# Penting: samakan dengan training!
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
    # Jika training pakai Normalize, aktifkan juga di sini.
    # transforms.Normalize(mean=[...], std=[...]),
])

# ========= Patch helper =========
def _disable_inplace_relu(model: nn.Module):
    """Setiap nn.ReLU di model dipaksa inplace=False (tambahan safety post-load)."""
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    return model

# ========= Loader (dengan cache-bust arg) =========
@st.cache_resource
def load_model(cache_bust: str = "noinplace-v3"):
    model = ResNet18(num_diseases=len(CLASS_NAMES), in_channels=3)
    sd = torch.load("model/resnet_97_56.pt", map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    sd = { (k.replace("module.","") if k.startswith("module.") else k): v for k,v in sd.items() }
    model.load_state_dict(sd, strict=True)
    _disable_inplace_relu(model)  # <<— patch setelah load
    model.eval()
    return model

# ========= Prediksi =========
@torch.no_grad()
def predict_image(model, image):
    img = transform(image).unsqueeze(0)
    outputs = model(img)
    probs = torch.softmax(outputs[0], dim=0)
    pred_idx = torch.argmax(probs).item()
    return CLASS_NAMES[pred_idx], probs.numpy()

# ========= Grad-CAM =========
def _normalize_cam(cam: torch.Tensor):
    cam = torch.relu(cam)
    cam_min, cam_max = cam.min(), cam.max()
    if (cam_max - cam_min) > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)
    return cam

def _upsample_cam(cam: torch.Tensor, size_hw: tuple[int,int]) -> torch.Tensor:
    cam = cam[None, None, ...]
    cam = F.interpolate(cam, size=size_hw, mode="bilinear", align_corners=False)
    return cam[0,0]

def _overlay(pil_img: Image.Image, cam_hw01: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = np.asarray(pil_img.convert("RGB")).astype(np.float32) / 255.0
    H, W = base.shape[:2]
    cam_hw01 = np.clip(cam_hw01, 0.0, 1.0)
    heat = cm.get_cmap("jet")(cam_hw01)[..., :3]
    out = (1 - alpha) * base + alpha * heat
    out = np.clip(out, 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8))

def _simple_leaf_mask(pil_img: Image.Image) -> np.ndarray:
    hsv = np.asarray(pil_img.convert("HSV")).astype(np.int32)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    green  = (H >= 60) & (H <= 130) & (S >= 60) & (V >= 40)
    yellow = (H >= 35) & (H <= 59)  & (S >= 60) & (V >= 50)
    mask = (green | yellow).astype(np.float32)
    # box blur 3x3
    if mask.sum() > 0:
        k = 3
        pad = k // 2
        padded = np.pad(mask, ((pad,pad),(pad,pad)), mode="edge")
        smoothed = np.zeros_like(mask)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                smoothed[i,j] = padded[i:i+k, j:j+k].mean()
        mask = np.clip(smoothed, 0, 1)
    return mask

def get_target_layer(model: nn.Module, name: str):
    """
    Hook ke BN (bukan ReLU) untuk aman dari in-place:
      - "conv4_prepool" -> model.conv4[1] (BN), resolusi ~16x16
      - "conv3_prepool" -> model.conv3[1] (BN), resolusi ~16x16
      - "conv2_prepool" -> model.conv2[1] (BN), resolusi ~64x64
      - "res2"          -> output block (4x4, sangat semantik)
    """
    if name == "res2":
        return model.res2
    if name == "conv4_prepool":
        return model.conv4[1]
    if name == "conv3_prepool":
        return model.conv3[1]
    if name == "conv2_prepool":
        return model.conv2[1]
    raise ValueError(f"target_layer tidak dikenal: {name}")

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer
        self._A = None
        self._G = None
        self.h1 = target_layer.register_forward_hook(self._fhook)
        try:
            self.h2 = target_layer.register_full_backward_hook(self._bhook)
        except AttributeError:
            self.h2 = target_layer.register_backward_hook(self._bhook)

    def _fhook(self, module, inp, out):
        self._A = out.detach().clone()  # aman dari view+inplace

    def _bhook(self, module, gin, gout):
        self._G = gout[0].detach().clone()

    def remove(self):
        self.h1.remove()
        self.h2.remove()

    def compute(self, x: torch.Tensor, class_idx: int | None = None):
        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)
            out = self.model(x)                # 1 x C
            probs = torch.softmax(out, dim=1)[0]
            if class_idx is None:
                class_idx = int(torch.argmax(probs).item())
            score = out[0, class_idx]
            score.backward(retain_graph=False)

            A = self._A[0]      # (C,H,W)
            G = self._G[0]      # (C,H,W)
            weights = G.view(G.size(0), -1).mean(dim=1)          # (C,)
            cam = torch.sum(weights[:, None, None] * A, dim=0)   # (H,W)
            cam = _normalize_cam(cam)
        return cam, class_idx, probs.detach().cpu().numpy()

def gradcam_on_pil(
    model: nn.Module,
    pil_img: Image.Image,
    target_layer_name: str = "conv4_prepool",
    class_idx: int | None = None,
    alpha: float = 0.45,
    mask_bg: bool = False,
    blend_with_res2: bool = False,
):
    x = transform(pil_img).unsqueeze(0)

    # CAM utama
    tl = get_target_layer(model, target_layer_name)
    engine = GradCAM(model, tl)
    try:
        cam, used_idx, probs = engine.compute(x, class_idx=class_idx)
    finally:
        engine.remove()

    # Opsional: blend dengan res2
    if blend_with_res2 and target_layer_name != "res2":
        tl2 = get_target_layer(model, "res2")
        engine2 = GradCAM(model, tl2)
        try:
            cam2, _, _ = engine2.compute(x, class_idx=used_idx)
            if cam2.shape != cam.shape:
                cam2 = _upsample_cam(cam2, (cam.shape[0], cam.shape[1]))  # naikkan 4x4 → 16x16 (atau 64x64)
            cam = 0.6 * cam + 0.4 * cam2
            cam = _normalize_cam(cam)  # re-normalisasi setelah gabung
            
    # Upsample ke ukuran input
    H, W = pil_img.size[1], pil_img.size[0]
    cam_up = _upsample_cam(cam, (H, W))

    # Mask background
    if mask_bg:
        m = _simple_leaf_mask(pil_img)
        cam_up = cam_up * torch.tensor(m, dtype=cam_up.dtype)

    cam_up = _normalize_cam(cam_up).cpu().numpy()
    overlay = _overlay(pil_img, cam_up, alpha=alpha)
    pred_label = CLASS_NAMES[used_idx] if 0 <= used_idx < len(CLASS_NAMES) else str(used_idx)
    return overlay, cam_up, pred_label, used_idx, probs

def show_prediction_and_cam(
    model: nn.Module,
    pil_img: Image.Image,
    alpha: float = 0.45,
    topk: int = 3,
    target_layer_name: str = "conv4_prepool",
    mask_bg: bool = False,
    blend_with_res2: bool = False,
):
    # Prediksi
    with torch.no_grad():
        x = transform(pil_img).unsqueeze(0)
        out = model(x)
        probs_all = torch.softmax(out[0], dim=0).cpu().numpy()
        used_idx = int(np.argmax(probs_all))
    pred_label = CLASS_NAMES[used_idx]

    # Grad-CAM
    overlay, cam, _, _, _ = gradcam_on_pil(
        model, pil_img,
        target_layer_name=target_layer_name,
        class_idx=used_idx,
        alpha=alpha,
        mask_bg=mask_bg,
        blend_with_res2=blend_with_res2
    )

    # Tampilkan (jika di Streamlit)
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
            st.image(overlay, caption=f"Grad-CAM ({target_layer_name}) → {pred_label}", use_container_width=True)
    except Exception:
        pass

    return overlay, cam, used_idx, probs_all
