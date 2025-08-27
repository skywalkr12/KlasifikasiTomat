# helper.py
# — ResNet9-variant + Grad-CAM + Leaf&Brown Mask + Lesion Prior —
# Tidak ada rendering Streamlit di file ini. Semua tampilan diatur dari prediksi.py.

import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import cm

# ========= Util umum =========
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
class ResNet9(ImageClassificationBase):
    def __init__(self, num_diseases=10, in_channels=3):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)               # 256
        self.conv2 = ConvBlock(64, 128, pool=True)            # 64
        self.res1  = nn.Sequential(ConvBlock(128, 128),
                                   ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)           # 16
        self.conv4 = ConvBlock(256, 512, pool=True)           # 4
        self.res2  = nn.Sequential(ConvBlock(512, 512),
                                   ConvBlock(512, 512))
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),  # 4x4 -> 1x1
            nn.Flatten(),
            nn.Dropout(0.2),
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

# ========= Daftar kelas =========
CLASS_NAMES = [
 'Bacterial_spot',
 'Early_blight',
 'Late_blight',
 'Leaf_Mold',
 'Septoria_leaf_spot',
 'Spider_mites_Two_spotted_spider_mite',
 'Target_Spot',
 'Tomato_Tomato_YellowLeaf__Curl_Virus',
 'Tomato_Tomato_mosaic_virus',
 'healthy'
]

# ========= Transform (samakan dengan training!) =========
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ========= Loader =========
def load_model(cache_bust: str = "noinplace-v5"):
    model = ResNet9(num_diseases=len(CLASS_NAMES), in_channels=3)
    sd = torch.load("model/resnet9(99,16).pt", map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    sd = { (k.replace("module.","") if k.startswith("module.") else k): v for k,v in sd.items() }
    model.load_state_dict(sd, strict=True)
    # matikan inplace ReLU (aman untuk Grad-CAM)
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    model.eval()
    return model

# ========= Prediksi (RAW, tanpa clipping) =========
@torch.no_grad()
def predict_image(model, image):
    x = transform(image).unsqueeze(0)
    out = model(x)
    probs_raw = torch.softmax(out[0], dim=0).cpu().numpy()  # sum=1
    idx = int(np.argmax(probs_raw))
    return CLASS_NAMES[idx], probs_raw, out[0].detach().cpu().numpy()

# ========= CAM util =========
def _normalize_cam(cam: torch.Tensor):
    cam = torch.relu(cam)
    mn, mx = cam.min(), cam.max()
    if (mx - mn) > 1e-8: cam = (cam - mn) / (mx - mn)
    else:                cam = torch.zeros_like(cam)
    return cam

def _upsample_cam(cam: torch.Tensor, size_hw: tuple[int,int]) -> torch.Tensor:
    cam = cam[None, None, ...]
    cam = F.interpolate(cam, size=size_hw, mode="bilinear", align_corners=False)
    return cam[0,0]

def _overlay(pil_img: Image.Image, cam_hw01: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = np.asarray(pil_img.convert("RGB")).astype(np.float32) / 255.0
    cam_hw01 = np.clip(cam_hw01, 0.0, 1.0)
    heat = cm.get_cmap("jet")(cam_hw01)[..., :3]
    out = (1 - alpha) * base + alpha * heat
    out = np.clip(out, 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8))

# --- Erosi sederhana (min-filter) untuk mask float 0..1 ---
def _erode_min(mask: np.ndarray, k: int = 3, iters: int = 1) -> np.ndarray:
    if k <= 1 or iters <= 0: return mask
    m = mask.copy()
    for _ in range(iters):
        pad = k // 2
        padded = np.pad(m, ((pad,pad),(pad,pad)), mode="edge")
        out = np.zeros_like(m)
        H, W = m.shape
        for i in range(H):
            for j in range(W):
                out[i,j] = np.min(padded[i:i+k, j:j+k])
        m = out
    return m

# ========= Mask daun (Green+Yellow+Brown) =========
def _leaf_mask_hsv_with_brown(pil_img: Image.Image) -> np.ndarray:
    """
    HSV 0..255:
      hijau:   H∈[60,130], S≥60, V≥40
      kuning:  H∈[35,59],  S≥60, V≥50
      cokelat: H∈[8,35],   S≥50, V∈[20,200]
      cokelat gelap: H≤8,  S≥60, V∈[10,180]
    """
    hsv = np.asarray(pil_img.convert("HSV")).astype(np.int32)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    green  = (H>=60) & (H<=130) & (S>=60) & (V>=40)
    yellow = (H>=35) & (H<=59)  & (S>=60) & (V>=50)
    brown1 = (H>=8)  & (H<=35)  & (S>=50) & (V>=20) & (V<=200)
    brown2 = (H<=8)  & (S>=60)  & (V>=10) & (V<=180)
    mask = (green | yellow | brown1 | brown2).astype(np.float32)
    # halus tipis (box 3x3)
    if mask.sum() > 0:
        k = 3; pad = k // 2
        padded = np.pad(mask, ((pad,pad),(pad,pad)), mode="edge")
        smoothed = np.zeros_like(mask)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                smoothed[i,j] = padded[i:i+k, j:j+k].mean()
        mask = np.clip(smoothed, 0, 1)
    return mask

# ========= Lesion prior (brownness + darkness) =========
def _lesion_prior_brown(pil_img: Image.Image) -> np.ndarray:
    rgb = np.asarray(pil_img.convert("RGB")).astype(np.float32) / 255.0
    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    brownness = np.clip((r - g), 0, 1) + np.clip((r - b), 0, 1)
    mx = brownness.max()
    brownness = brownness / (mx + 1e-6) if mx > 0 else brownness
    V = np.asarray(pil_img.convert("HSV"))[...,2].astype(np.float32) / 255.0
    darkness = np.clip((0.8 - V) / 0.8, 0, 1)
    prior = 0.6 * brownness + 0.4 * darkness
    # haluskan box 3x3
    k = 3; pad = k // 2
    padded = np.pad(prior, ((pad,pad),(pad,pad)), mode="edge")
    sm = np.zeros_like(prior)
    for i in range(prior.shape[0]):
        for j in range(prior.shape[1]):
            sm[i,j] = padded[i:i+k, j:j+k].mean()
    return np.clip(sm, 0, 1)

# ========= Chlorosis suppressor (turunkan CAM pada kuning cerah non-brown) =========
def _chlorosis_mask(pil_img: Image.Image) -> np.ndarray:
    hsv = np.asarray(pil_img.convert("HSV")).astype(np.int32)
    H,S,V = hsv[...,0], hsv[...,1], hsv[...,2]
    yellow = (H>=35) & (H<=60) & (S>=60) & (V>=110)  # kuning kuat & terang
    rgb = np.asarray(pil_img.convert("RGB")).astype(np.float32) / 255.0
    r,g,b = rgb[...,0], rgb[...,1], rgb[...,2]
    non_brown = ( (r - g) < 0.02 ) & ( (r - b) < 0.02 )  # tidak kemerahan
    m = (yellow & non_brown).astype(np.float32)
    # haluskan ringan
    if m.sum() > 0:
        k = 3; pad = k // 2
        padded = np.pad(m, ((pad,pad),(pad,pad)), mode="edge")
        sm = np.zeros_like(m)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                sm[i,j] = padded[i:i+k, j:j+k].mean()
        m = np.clip(sm, 0, 1)
    return m

def _resize_like(cam_src: torch.Tensor, cam_ref: torch.Tensor) -> torch.Tensor:
    if cam_src.shape == cam_ref.shape: return cam_src
    return _upsample_cam(cam_src, (cam_ref.shape[0], cam_ref.shape[1]))

# ========= Target layer Grad-CAM =========
def get_target_layer(model: nn.Module, name: str):
    if name == "res2":            return model.res2
    if name == "conv4_prepool":   return model.conv4[1]  # BN sebelum MaxPool
    if name == "conv3_prepool":   return model.conv3[1]
    if name == "conv2_prepool":   return model.conv2[1]
    raise ValueError(f"target_layer tidak dikenal: {name}")

# ========= Grad-CAM standar =========
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self._A = None
        self._G = None
        self.h1 = target_layer.register_forward_hook(self._fhook)
        try:
            self.h2 = target_layer.register_full_backward_hook(self._bhook)
        except AttributeError:
            self.h2 = target_layer.register_backward_hook(self._bhook)
    def _fhook(self, module, inp, out):       self._A = out.detach().clone()
    def _bhook(self, module, gin, gout):      self._G = gout[0].detach().clone()
    def remove(self):                         self.h1.remove(); self.h2.remove()

    def compute(self, x: torch.Tensor, class_idx: int | None = None):
        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)
            out = self.model(x)                # 1 x C
            probs = torch.softmax(out, dim=1)[0]
            if class_idx is None:
                class_idx = int(torch.argmax(probs).item())
            score = out[0, class_idx]
            score.backward(retain_graph=False)
            A = self._A[0]                                # (C,H,W)
            G = self._G[0]                                # (C,H,W)
            weights = G.view(G.size(0), -1).mean(dim=1)   # (C,)
            cam = torch.sum(weights[:, None, None] * A, dim=0)   # (H,W)
            cam = _normalize_cam(cam)
        return cam, class_idx, probs.detach().cpu().numpy()

# ========= API utama (tidak merender) =========
def gradcam_on_pil(
    model: nn.Module,
    pil_img: Image.Image,
    target_layer_name: str = "res2",
    class_idx: int | None = None,
    alpha: float = 0.45,
    mask_bg: bool = True,
    include_brown: bool = True,
    lesion_boost: bool = True,
    lesion_weight: float = 0.5,
    blend_with_res2: bool = False,
    erode_border: bool = True,
    suppress_chlorosis: bool = True,
    chlorosis_weight: float = 0.35
):
    x = transform(pil_img).unsqueeze(0)

    # CAM utama
    tl = get_target_layer(model, target_layer_name)
    engine = GradCAM(model, tl)
    try:
        cam, used_idx, probs = engine.compute(x, class_idx=class_idx)
    finally:
        engine.remove()

    # Blend dengan res2 (4x4) jika diminta
    if blend_with_res2 and target_layer_name != "res2":
        tl2 = get_target_layer(model, "res2")
        engine2 = GradCAM(model, tl2)
        try:
            cam2, _, _ = engine2.compute(x, class_idx=used_idx)
        finally:
            engine2.remove()
        cam2 = _resize_like(cam2, cam)
        cam  = _normalize_cam(0.6 * cam + 0.4 * cam2)

    # Upsample ke ukuran gambar asli
    H, W = pil_img.size[1], pil_img.size[0]
    cam_up = _upsample_cam(cam, (H, W))

    # Mask daun + erosi tepi
    if mask_bg:
        if include_brown:
            m = _leaf_mask_hsv_with_brown(pil_img)
        else:
            hsv = np.asarray(pil_img.convert("HSV")).astype(np.int32)
            Hh, Ss, Vv = hsv[...,0], hsv[...,1], hsv[...,2]
            green  = (Hh>=60) & (Hh<=130) & (Ss>=60) & (Vv>=40)
            yellow = (Hh>=35) & (Hh<=59)  & (Ss>=60) & (Vv>=50)
            m = (green | yellow).astype(np.float32)
        if erode_border:
            m = _erode_min(m, k=3, iters=1)
        cam_up = cam_up * torch.tensor(m, dtype=cam_up.dtype, device=cam_up.device)

    # Lesion prior
    if lesion_boost:
        prior = _lesion_prior_brown(pil_img)
        cam_up = cam_up * (1.0 + float(lesion_weight) *
                           torch.tensor(prior, dtype=cam_up.dtype, device=cam_up.device))

    # Turunkan respon pada klorosis (kuning terang non-brown) agar tidak salah fokus
    if suppress_chlorosis:
        chl = _chlorosis_mask(pil_img)
        cam_up = cam_up * (1.0 - float(chlorosis_weight) *
                           torch.tensor(chl, dtype=cam_up.dtype, device=cam_up.device))

    cam_up = _normalize_cam(cam_up).cpu().numpy()
    overlay = _overlay(pil_img, cam_up, alpha=alpha)
    pred_label = CLASS_NAMES[used_idx] if 0 <= used_idx < len(CLASS_NAMES) else str(used_idx)
    return overlay, cam_up, pred_label, used_idx, probs

def show_prediction_and_cam(
    model: nn.Module,
    pil_img: Image.Image,
    alpha: float = 0.45,
    topk: int = 3,
    target_layer_name: str = "res2",
    mask_bg: bool = True,
    include_brown: bool = True,
    lesion_boost: bool = True,
    lesion_weight: float = 0.5,
    blend_with_res2: bool = False,
    erode_border: bool = True,
    suppress_chlorosis: bool = True,
    chlorosis_weight: float = 0.35
):
    # Prediksi RAW (tanpa clipping)
    with torch.no_grad():
        x = transform(pil_img).unsqueeze(0)
        out = model(x)
        probs_raw = torch.softmax(out[0], dim=0).cpu().numpy()
        used_idx = int(np.argmax(probs_raw))

    # Grad-CAM untuk kelas teratas
    overlay, cam, _, _, _ = gradcam_on_pil(
        model, pil_img,
        target_layer_name=target_layer_name,
        class_idx=used_idx,
        alpha=alpha,
        mask_bg=mask_bg,
        include_brown=include_brown,
        lesion_boost=lesion_boost,
        lesion_weight=lesion_weight,
        blend_with_res2=blend_with_res2,
        erode_border=erode_border,
        suppress_chlorosis=suppress_chlorosis,
        chlorosis_weight=chlorosis_weight
    )

    return overlay, cam, used_idx, probs_raw




