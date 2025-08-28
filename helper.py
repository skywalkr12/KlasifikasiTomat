# helper.py
# — Robust .pt loader (ResNet9/ResNet18 autodetect) + Temperature Scaling —
# Tidak ada rendering Streamlit di file ini. Semua tampilan diatur dari klasifikasi.py.

import os
import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# ================= Util umum =================
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
        out = self.conv1(x)
        out = self.relu1(out)
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
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

# ================= Model (ResNet9-variant) =================
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

# ================= Global info untuk ditampilkan di UI =================
LOAD_NOTES: list[str] = []

def get_load_notes() -> list[str]:
    return LOAD_NOTES

# ================= Daftar kelas (fallback; dioverride jika ada di checkpoint) =================
CLASS_NAMES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_YellowLeaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy",
]

# ================= Transform (samakan dengan training; tanpa augmentasi acak di inferensi) =================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ================= Temperature Scaling =================
DEFAULT_T = 0.9137
TEMPERATURE = DEFAULT_T

# ================= Helper loading =================
def _strip_module(sd: dict) -> dict:
    return { (k.replace("module.", "") if k.startswith("module.") else k): v for k, v in sd.items() }

def _infer_num_classes_from_sd(sd: dict, fallback: int) -> int:
    # Cari bobot linear akhir (num_classes, in_features)
    cand_keys = [k for k in sd.keys() if k.endswith(".weight")]
    for k in ["classifier.3.weight", "classifier.2.weight", "fc.weight", "head.weight", "linear.weight"]:
        if k in sd and sd[k].ndim == 2:
            return int(sd[k].shape[0])
    for k in cand_keys:
        if sd[k].ndim == 2:  # heuristic
            return int(sd[k].shape[0])
    return int(fallback)

def _looks_like_resnet18(sd_keys: list[str]) -> bool:
    # Ciri khas torchvision ResNet18: layer1/2/3/4, bn1, conv1, fc
    return any(k.startswith("layer1.") for k in sd_keys) and any(k.startswith("layer2.") for k in sd_keys)

def _build_model_for_sd(sd: dict, num_classes: int) -> nn.Module:
    if _looks_like_resnet18(list(sd.keys())):
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        LOAD_NOTES.append("Autodetect arsitektur: ResNet18.")
        return m
    LOAD_NOTES.append("Autodetect arsitektur: ResNet9.")
    return ResNet9(num_diseases=num_classes, in_channels=3)

def _load_checkpoint_generic(path: str):
    """
    Return (model, class_names, temperature)
    Mendukung:
      1) nn.Module / TorchScript (model utuh)
      2) dict checkpoint berisi:
           - model_state_dict (preferred) ATAU langsung state_dict
           - temperature (opsional)
           - class_names (opsional)
    """
    obj = torch.load(path, map_location="cpu")

    # (1) model utuh
    if isinstance(obj, torch.jit.ScriptModule) or isinstance(obj, nn.Module):
        model = obj
        class_names = CLASS_NAMES
        T = DEFAULT_T
        LOAD_NOTES.append("Checkpoint berisi model utuh.")
        return model, class_names, T

    # (2) dict
    if not isinstance(obj, dict):
        raise RuntimeError("Format .pt tidak dikenal. Harus nn.Module/TorchScript atau dict checkpoint/state_dict.")

    class_names = obj.get("class_names", CLASS_NAMES)
    T = float(obj.get("temperature", DEFAULT_T))

    if "model_state_dict" in obj:
        sd = _strip_module(obj["model_state_dict"])
    else:
        sd = _strip_module(obj)

    num_classes = _infer_num_classes_from_sd(sd, fallback=len(class_names))
    model = _build_model_for_sd(sd, num_classes)

    # Coba strict=True; jika gagal, jatuhkan ke strict=False dan catat missing/unexpected
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        LOAD_NOTES.append("strict=True gagal saat load_state_dict; mencoba strict=False.")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            LOAD_NOTES.append(f"Missing keys: {', '.join(missing[:10])}" + (" ..." if len(missing) > 10 else ""))
        if unexpected:
            LOAD_NOTES.append(f"Unexpected keys: {', '.join(unexpected[:10])}" + (" ..." if len(unexpected) > 10 else ""))
    return model, class_names, T

def load_model(model_path: str | None = None):
    """
    Memuat model dari path tetap: model/resnet9_finetuned.pt
    """
    global CLASS_NAMES, TEMPERATURE
    LOAD_NOTES.clear()

    fixed_path = "model/resnet9_finetuned.pt"
    if not os.path.exists(fixed_path):
        raise FileNotFoundError(f"File model tidak ditemukan di: {fixed_path}")

    model, class_names, T = _load_checkpoint_generic(fixed_path)
    CLASS_NAMES = list(class_names) if isinstance(class_names, (list, tuple)) else CLASS_NAMES
    TEMPERATURE = float(T) if T is not None else DEFAULT_T

    # pastikan ReLU tidak inplace
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    model.eval()
    return model

# ================= Prediksi (Softmax dengan Temperature Scaling) =================
@torch.no_grad()
def predict_image(model, image: Image.Image):
    """
    Return:
      - label prediksi (string)
      - probs_T: probabilitas setelah temperature scaling (numpy, sum=1)
      - logits: vektor logit mentah (numpy)
    """
    x = transform(image).unsqueeze(0)
    logits = model(x)[0]  # (C,)
    probs_T = torch.softmax(logits / float(TEMPERATURE), dim=0).cpu().numpy()
    idx = int(np.argmax(probs_T))
    return CLASS_NAMES[idx], probs_T, logits.detach().cpu().numpy()
