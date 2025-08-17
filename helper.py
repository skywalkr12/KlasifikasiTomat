# helper.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import streamlit as st

import numpy as np
from PIL import Image
from matplotlib import cm  # untuk colormap 'jet'

# --------- Utils ---------
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# (opsional, tidak dipakai di model ini)
class SimpleResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
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

# --------- Blocks ---------
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        # sesuai arsitektur training-mu: turunkan /4 per stage
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

# --------- Model (ini sebenarnya ResNet9-variant) ---------
class ResNet18(ImageClassificationBase):
    """
    Catatan: Ini BUKAN ResNet18 resmi; ini arsitektur ResNet9-variant yang kamu pakai.
    """
    def __init__(self, num_diseases=10, in_channels=3):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)   # 256 -> 64
        self.res1  = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)  # 64 -> 16
        self.conv4 = ConvBlock(256, 512, pool=True)  # 16 -> 4
        self.res2  = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),  # 4x4 -> 1x1
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, num_diseases)
        )
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        return self.classifier(out)

# --------- Kelas ---------
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

# Pastikan sama dengan training (256 & tanpa normalize)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# --------- Loader ---------
@st.cache_resource
def load_model():
    model = ResNet18(num_diseases=len(CLASS_NAMES), in_channels=3)
    sd = torch.load("model/resnet_97_56.pt", map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    # hapus prefix 'module.' jika perlu
    sd = { (k.replace("module.","") if k.startswith("module.") else k): v for k,v in sd.items() }
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model

# --------- Predict (tanpa Grad) ---------
def predict_image(model, image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs[0], dim=0)
        pred_idx = torch.argmax(probs).item()
    return CLASS_NAMES[pred_idx], probs.numpy()

# --------- Grad-CAM Core ---------
class GradCAM:
    """
    Implementasi Grad-CAM generik untuk model PyTorch.
    - target_layer: module yang akan di-hook (contoh: model.res2)
    - bekerja tanpa retrain: hanya pakai forward + backward dari model terlatih
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        # forward hook
        self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        # backward hook (full jika tersedia, fallback jika tidak)
        try:
            self.bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)
        except AttributeError:
            self.bwd_handle = target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        # simpan aktivasi sebelum detach supaya tetap ada graph untuk backward;
        # tapi untuk perhitungan CAM nanti kita .detach()
        self._activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output adalah tuple; ambil grad wrt output module
        self._gradients = grad_output[0]

    def remove_hooks(self):
        if hasattr(self, "fwd_handle") and self.fwd_handle is not None:
            self.fwd_handle.remove()
        if hasattr(self, "bwd_handle") and self.bwd_handle is not None:
            self.bwd_handle.remove()

    @torch.no_grad()
    def _normalize_cam(self, cam: torch.Tensor):
        cam = torch.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def __call__(self, img_tensor: torch.Tensor, class_idx: int | None = None):
        """
        img_tensor: tensor shape (1, C, H, W) hasil transform (JANGAN pakai no_grad di luar).
        Return: (cam_np [Hc,Wc in feature space], class_idx, logits_softmax[classes])
        """
        # pastikan mode eval
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        # forward
        outputs = self.model(img_tensor)             # (1, num_classes)
        probs = torch.softmax(outputs, dim=1)[0]     # (num_classes,)
        if class_idx is None:
            class_idx = int(torch.argmax(probs).item())

        # backward terhadap skor target
        score = outputs[0, class_idx]
        score.backward(retain_graph=False)

        # ambil activations & gradients dari target layer
        A = self._activations.detach()[0]  # (C, H, W)
        G = self._gradients.detach()[0]    # (C, H, W)

        # bobot channel = average pooling grad per channel
        weights = G.view(G.size(0), -1).mean(dim=1)  # (C,)
        cam = torch.sum(weights[:, None, None] * A, dim=0)  # (H, W)
        cam = self._normalize_cam(cam).cpu().numpy()

        # bersihkan grad untuk panggilan berikutnya
        self.model.zero_grad(set_to_none=True)
        return cam, class_idx, probs.detach().cpu().numpy()

# --------- Visual Utilities ---------
def _resize_to(img_like: np.ndarray, size_hw: tuple[int,int]) -> np.ndarray:
    """
    Resize array [H,W] -> [H0,W0] dengan PIL bilinear.
    """
    H0, W0 = size_hw
    pil = Image.fromarray((img_like * 255).astype(np.uint8))
    pil = pil.resize((W0, H0), resample=Image.BILINEAR)
    out = np.asarray(pil).astype(np.float32) / 255.0
    return out

def overlay_cam_on_image(pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    Blend citra RGB asli dengan heatmap 'jet' dari CAM (range 0..1).
    """
    pil_img = pil_img.convert("RGB")
    img = np.asarray(pil_img).astype(np.float32) / 255.0  # H,W,3
    H, W = img.shape[:2]
    cam_rs = _resize_to(cam, (H, W))                      # H,W
    heat = cm.get_cmap('jet')(cam_rs)[..., :3]            # H,W,3 in 0..1
    blended = (1 - alpha) * img + alpha * heat
    blended = np.clip(blended, 0, 1)
    return Image.fromarray((blended * 255).astype(np.uint8))

# --------- High-level API untuk Streamlit ---------
def gradcam_on_pil(model: nn.Module, pil_img: Image.Image, target_layer: nn.Module | None = None,
                   class_idx: int | None = None, alpha: float = 0.45):
    """
    Hitung Grad-CAM pada satu gambar PIL.
    Return:
      overlay_pil, raw_cam(np.ndarray Hc x Wc), pred_label(str), pred_idx(int), probs(np.ndarray)
    """
    # siapkan input tensor (tanpa no_grad!)
    x = transform(pil_img).unsqueeze(0)
    # pilih target layer (default: residu terakhir, yaitu model.res2)
    target_layer = target_layer or getattr(model, "res2", None)
    if target_layer is None:
        raise ValueError("Target layer untuk Grad-CAM tidak ditemukan. Pastikan model punya atribut 'res2'.")

    cam_engine = GradCAM(model, target_layer)
    try:
        cam, used_idx, probs = cam_engine(x, class_idx=class_idx)
    finally:
        cam_engine.remove_hooks()

    overlay = overlay_cam_on_image(pil_img, cam, alpha=alpha)
    pred_label = CLASS_NAMES[used_idx] if 0 <= used_idx < len(CLASS_NAMES) else str(used_idx)
    return overlay, cam, pred_label, used_idx, probs

def show_prediction_and_cam(model: nn.Module, pil_img: Image.Image, alpha: float = 0.45, topk: int = 3):
    """
    Utility siap-pakai untuk Streamlit:
    - tampilkan prediksi top-1 + Grad-CAM overlay
    - tampilkan alternatif top-k
    """
    # Prediksi biasa (no_grad OK karena ini hanya buat tampilan)
    pred_label, probs_np = predict_image(model, pil_img)

    # Grad-CAM (tanpa no_grad)
    overlay, cam, _, used_idx, probs_all = gradcam_on_pil(model, pil_img, alpha=alpha)

    # Tampilkan di Streamlit
    col1, col2 = st.columns([1,1])
    with col1:
        st.image(pil_img, caption="Input", use_container_width=True)
        st.write(f"**Prediksi**: {pred_label}  \n**Confidence**: {float(probs_np.max()):.2%}")
        # top-k alternatif
        topk = min(topk, len(CLASS_NAMES))
        top_idx = np.argsort(-probs_all)[:topk]
        alt_lines = []
        for i in range(topk):
            cls = CLASS_NAMES[top_idx[i]]
            p = float(probs_all[top_idx[i]])
            prefix = "★ " if top_idx[i] == used_idx else "• "
            alt_lines.append(f"{prefix}{cls}: {p:.2%}")
        st.markdown("**Alternatif (Top-k)**  \n" + "\n".join(alt_lines))
    with col2:
        st.image(overlay, caption=f"Grad-CAM → {CLASS_NAMES[used_idx]}", use_container_width=True)
    return overlay, cam, used_idx, probs_all
