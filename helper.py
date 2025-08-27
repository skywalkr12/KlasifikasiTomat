# helper.py
# — ResNet9-variant + Grad-CAM (robust res2) —
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

def ConvBlock(in_c, out_c, pool=False):
    layers = [nn.Conv2d(in_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=False)]
    if pool: layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, num_diseases=10, in_channels=3):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)               # 256
        self.conv2 = ConvBlock(64, 128, pool=True)            # 64
        self.res1  = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)           # 16
        self.conv4 = ConvBlock(256, 512, pool=True)           # 4
        self.res2  = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten(), nn.Dropout(0.2), nn.Linear(512, num_diseases))
    def forward(self, xb):
        out = self.conv1(xb)          
        out = self.conv2(out)         
        out = self.res1(out) + out
        out = self.conv3(out)         
        out = self.conv4(out)         
        out = self.res2(out) + out
        return self.classifier(out)

CLASS_NAMES = [
 'Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite','Tomato_Target_Spot',
 'Tomato_Tomato_YellowLeaf__Curl_Virus','Tomato_Tomato_mosaic_virus','Tomato_healthy'
]

transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

def load_model(weights_path="model/resnet9(99,16).pt", cache_bust="v1"):
    model = ResNet9(num_diseases=len(CLASS_NAMES), in_channels=3)
    sd = torch.load(weights_path, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd: sd = sd["model_state_dict"]
    sd = { (k.replace("module.","") if k.startswith("module.") else k): v for k,v in sd.items() }
    model.load_state_dict(sd, strict=True)
    for m in model.modules():
        if isinstance(m, nn.ReLU): m.inplace = False
    model.eval()
    return model

@torch.no_grad()
def predict_image(model, image: Image.Image):
    x = transform(image).unsqueeze(0)
    out = model(x)
    probs_raw = torch.softmax(out[0], dim=0).cpu().numpy()
    idx = int(np.argmax(probs_raw))
    return CLASS_NAMES[idx], probs_raw, out[0].detach().cpu().numpy()

# ========= CAM util =========
def _normalize_cam(cam: torch.Tensor):
    cam = torch.relu(cam)
    mn, mx = cam.min(), cam.max()
    return (cam - mn) / (mx - mn + 1e-8) if (mx - mn) > 0 else torch.zeros_like(cam)

def _upsample_cam(cam: torch.Tensor, size_hw):
    cam = cam[None, None, ...]
    cam = F.interpolate(cam, size=size_hw, mode="bilinear", align_corners=False)
    return cam[0,0]

def _overlay(pil_img: Image.Image, cam01: np.ndarray, alpha=0.45):
    base = np.asarray(pil_img.convert("RGB")).astype(np.float32) / 255.0
    cam01 = np.clip(cam01, 0, 1)
    heat = cm.get_cmap("jet")(cam01)[..., :3]
    out = np.clip((1 - alpha) * base + alpha * heat, 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8))

def _erode_min(mask: np.ndarray, k=3, iters=1):
    if k <= 1 or iters <= 0: return mask
    m = mask.copy(); pad = k // 2
    for _ in range(iters):
        padded = np.pad(m, ((pad,pad),(pad,pad)), mode="edge")
        out = np.zeros_like(m)
        H, W = m.shape
        for i in range(H):
            for j in range(W):
                out[i,j] = padded[i:i+k, j:j+k].min()
        m = out
    return m

def _leaf_mask_hsv_with_brown(pil_img: Image.Image) -> np.ndarray:
    hsv = np.asarray(pil_img.convert("HSV")).astype(np.int32)
    H, S, V = hsv[...,0], hsv[...,1], hsv[...,2]
    green  = (H>=60) & (H<=130) & (S>=60) & (V>=40)
    yellow = (H>=35) & (H<=59)  & (S>=60) & (V>=50)
    brown1 = (H>=8)  & (H<=35)  & (S>=50) & (V>=20) & (V<=200)
    brown2 = (H<=8)  & (S>=60)  & (V>=10) & (V<=180)
    mask = (green | yellow | brown1 | brown2).astype(np.float32)
    if mask.sum() > 0:
        k = 3; pad = k // 2
        padded = np.pad(mask, ((pad,pad),(pad,pad)), mode="edge")
        sm = np.zeros_like(mask)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                sm[i,j] = padded[i:i+k, j:j+k].mean()
        mask = np.clip(sm, 0, 1)
    return mask

def _lesion_prior_brown(pil_img: Image.Image) -> np.ndarray:
    rgb = np.asarray(pil_img.convert("RGB")).astype(np.float32) / 255.0
    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    brownness = np.clip((r - g), 0, 1) + np.clip((r - b), 0, 1)
    brownness = brownness / (brownness.max() + 1e-6)
    V = np.asarray(pil_img.convert("HSV"))[...,2].astype(np.float32) / 255.0
    darkness = np.clip((0.8 - V) / 0.8, 0, 1)
    prior = 0.6 * brownness + 0.4 * darkness
    k = 3; pad = k // 2
    padded = np.pad(prior, ((pad,pad),(pad,pad)), mode="edge")
    sm = np.zeros_like(prior)
    for i in range(prior.shape[0]):
        for j in range(prior.shape[1]):
            sm[i,j] = padded[i:i+k, j:j+k].mean()
    return np.clip(sm, 0, 1)

def _resize_like(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a if a.shape == b.shape else _upsample_cam(a, (b.shape[0], b.shape[1]))

# ========= Target layer (pakai CONV terakhir agar grad kaya) =========
def get_target_layer(model: nn.Module, name: str):
    if name == "res2":            return model.res2[1][0]       # Conv2d terakhir di res2
    if name == "conv4_prepool":   return model.conv4[0]         # Conv2d
    if name == "conv3_prepool":   return model.conv3[0]         # Conv2d
    if name == "conv2_prepool":   return model.conv2[0]         # Conv2d
    raise ValueError(f"target_layer tidak dikenal: {name}")

# ========= Grad-CAM =========
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self._A = None; self._G = None
        self.h1 = target_layer.register_forward_hook(self._fhook)
        try:
            self.h2 = target_layer.register_full_backward_hook(self._bhook)
        except AttributeError:
            self.h2 = target_layer.register_backward_hook(self._bhook)
    def _fhook(self, m, i, o): self._A = o.detach().clone()
    def _bhook(self, m, gi, go): self._G = go[0].detach().clone()
    def remove(self): self.h1.remove(); self.h2.remove()
    def compute(self, x: torch.Tensor, class_idx: int | None = None):
        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)
            out = self.model(x)
            probs = torch.softmax(out, dim=1)[0]
            if class_idx is None: class_idx = int(torch.argmax(probs).item())
            out[0, class_idx].backward()
            A = self._A[0]; G = self._G[0]
            w = G.view(G.size(0), -1).mean(dim=1)
            cam = torch.sum(w[:, None, None] * A, dim=0)
            cam = _normalize_cam(cam)
        return cam, class_idx, probs.detach().cpu().numpy()

# ========= API utama =========
def gradcam_on_pil(
    model: nn.Module, pil_img: Image.Image,
    target_layer_name: str = "conv3_prepool", class_idx: int | None = None,
    alpha: float = 0.45, mask_bg: bool = True, include_brown: bool = True,
    lesion_boost: bool = True, lesion_weight: float = 0.5,
    blend_with_res2: bool = False, erode_border: bool = True,
    external_leaf_mask: np.ndarray | None = None,
    auto_res2_fallback: bool = True
):
    x = transform(pil_img).unsqueeze(0)

    # CAM utama
    tl = get_target_layer(model, target_layer_name)
    engine = GradCAM(model, tl)
    try:
        cam, used_idx, probs = engine.compute(x, class_idx=class_idx)
    finally:
        engine.remove()

    # Fallback untuk 'res2' bila datar
    if target_layer_name == "res2" and auto_res2_fallback:
        flat = float((cam.max() - cam.min()).item()) < 1e-6 or float(cam.mean().item()) < 1e-5
        if flat:
            try:
                tl2 = model.res2[0][0]   # conv pada blok pertama res2
                engine2 = GradCAM(model, tl2); cam2, used_idx, probs = engine2.compute(x, class_idx=used_idx); engine2.remove()
                cam = cam2
            except Exception:
                tl3 = get_target_layer(model, "conv3_prepool")
                engine3 = GradCAM(model, tl3); cam3, used_idx, probs = engine3.compute(x, class_idx=used_idx); engine3.remove()
                cam = cam3

    # Blend dengan res2 (opsional)
    if blend_with_res2 and target_layer_name != "res2":
        tl2 = get_target_layer(model, "res2")
        engine2 = GradCAM(model, tl2)
        try:
            cam2, _, _ = engine2.compute(x, class_idx=used_idx)
        finally:
            engine2.remove()
        cam = _normalize_cam(0.6 * _resize_like(cam, cam2) + 0.4 * _resize_like(cam2, cam))

    # Upsample ke ukuran gambar asli
    H, W = pil_img.size[1], pil_img.size[0]
    cam_up = _upsample_cam(cam, (H, W))

    # Mask daun (pakai external bila ada)
    if mask_bg:
        if external_leaf_mask is not None:
            m = np.asarray(external_leaf_mask).astype(np.float32)
            if m.max() > 1.0: m = (m > 0).astype(np.float32)
            if m.mean() < 1e-4:  # mask rusak → jangan zero-kan CAM
                m = np.ones_like(m, dtype=np.float32)
        else:
            m = _leaf_mask_hsv_with_brown(pil_img).astype(np.float32)
        if erode_border: m = _erode_min(m, k=3, iters=1)
        cam_up = cam_up * torch.tensor(m, dtype=cam_up.dtype, device=cam_up.device)

    if lesion_boost:
        prior = _lesion_prior_brown(pil_img)
        cam_up = cam_up * (1.0 + float(lesion_weight) * torch.tensor(prior, dtype=cam_up.dtype, device=cam_up.device))

    cam_up = _normalize_cam(cam_up).cpu().numpy()
    overlay = _overlay(pil_img, cam_up, alpha=alpha)
    pred_label = CLASS_NAMES[used_idx] if 0 <= used_idx < len(CLASS_NAMES) else str(used_idx)
    return overlay, cam_up, pred_label, used_idx, probs

def show_prediction_and_cam(
    model: nn.Module, pil_img: Image.Image, alpha: float = 0.45, topk: int = 3,
    target_layer_name: str = "conv3_prepool", mask_bg: bool = True,
    include_brown: bool = True, lesion_boost: bool = True, lesion_weight: float = 0.5,
    blend_with_res2: bool = False, erode_border: bool = True,
    external_leaf_mask: np.ndarray | None = None, auto_res2_fallback: bool = True
):
    with torch.no_grad():
        x = transform(pil_img).unsqueeze(0)
        out = model(x)
        probs_raw = torch.softmax(out[0], dim=0).cpu().numpy()
        used_idx = int(np.argmax(probs_raw))

    overlay, cam, _, _, _ = gradcam_on_pil(
        model, pil_img, target_layer_name=target_layer_name, class_idx=used_idx,
        alpha=alpha, mask_bg=mask_bg, include_brown=include_brown,
        lesion_boost=lesion_boost, lesion_weight=lesion_weight,
        blend_with_res2=blend_with_res2, erode_border=erode_border,
        external_leaf_mask=external_leaf_mask, auto_res2_fallback=auto_res2_fallback
    )
    return overlay, cam, used_idx, probs_raw
