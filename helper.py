# helper.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import streamlit as st

# =========================
#  Konfigurasi ringan
# =========================
# Ganti di sini bila perlu, atau override lewat ENV vars:
#   ARCH_NAME : "resnet18" | "resnet9plus"
#   MODEL_PATH: path checkpoint .pt/.pth
ARCH_NAME_DEFAULT = "resnet18"
MODEL_PATH_DEFAULT = "model/resnet_97_56.pt"
IMG_SIZE = 256

# Kalau training pakai Normalize, aktifkan & sesuaikan mean-std
USE_NORMALIZE = False
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# =========================
#  Kelas (10 label)
# =========================
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

# =========================
#  Util & Base
# =========================
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

# =========================
#  Arsitektur LAMA (ResNet9-variant kamu)
#  - Tetap pakai MaxPool tahap tengah (/4)
#  - Head AdaptiveMaxPool(1) agar size-agnostic, tapi tetap "max"
# =========================
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))  # konsisten dgn training kamu
    return nn.Sequential(*layers)

class ResNet18(ImageClassificationBase):
    """Ini bukan ResNet18 resmi; ini varian ResNet9 sesuai proyekmu."""
    def __init__(self, num_diseases=10, in_channels=3):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)   # 256 -> 64
        self.res1  = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)  # 64 -> 16
        self.conv4 = ConvBlock(256, 512, pool=True)  # 16 -> 4
        self.res2  = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),   # global MAX adaptif → [B,512,1,1]
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

# =========================
#  ResNet9Plus (opsional)
#  - SE, DropPath kecil, GAP/GeM di akhir
#  - pool_ks=4 untuk input 256; kalau 224 → pool_ks=2
# =========================
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class GeMPool2d(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps
    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p)
        return x

class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, pool=False, pool_ks=4, dropout2d=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=pool_ks))
        if dropout2d > 0:
            layers.append(nn.Dropout2d(p=dropout2d))
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_se=True, drop_path_prob=0.0, dropout2d=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)
        self.se    = SEBlock(channels) if use_se else nn.Identity()
        self.dp    = DropPath(drop_path_prob)
        self.do2d  = nn.Dropout2d(dropout2d) if dropout2d > 0 else nn.Identity()
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.dp(out)
        out = out + identity
        out = self.relu(out)
        return self.do2d(out)

class ResNet9Plus(nn.Module):
    def __init__(
        self, in_channels: int, num_classes: int,
        *, pool_ks=4, use_se=True, drop_path_prob=0.05,
        dropout2d=0.05, dropout_fc=0.5, global_pool="avg", hidden_dim=256
    ):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, 64,  pool=False, pool_ks=pool_ks, dropout2d=dropout2d)
        self.conv2 = ConvBNReLU(64,         128, pool=True,  pool_ks=pool_ks, dropout2d=dropout2d)
        self.res1  = ResidualBlock(128, use_se=use_se, drop_path_prob=drop_path_prob, dropout2d=dropout2d)
        self.conv3 = ConvBNReLU(128,        256, pool=True,  pool_ks=pool_ks, dropout2d=dropout2d)
        self.conv4 = ConvBNReLU(256,        512, pool=True,  pool_ks=pool_ks, dropout2d=dropout2d)
        self.res2  = ResidualBlock(512, use_se=use_se, drop_path_prob=drop_path_prob, dropout2d=dropout2d)
        self.global_pool = GeMPool2d(p=3.0) if global_pool.lower()=="gem" else nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(512, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc), nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x); x = self.res1(x)
        x = self.conv3(x)
        x = self.conv4(x); x = self.res2(x)
        x = self.global_pool(x)          # [B,512,1,1]
        return self.head(x)

# =========================
#  Transform (samakan dgn training)
# =========================
_tfms = [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
if USE_NORMALIZE:
    _tfms.append(transforms.Normalize(mean=MEAN, std=STD))
transform = transforms.Compose(_tfms)

# =========================
#  Loader
# =========================
@st.cache_resource
def load_model():
    arch = os.getenv("ARCH_NAME", ARCH_NAME_DEFAULT).lower().strip()
    model_path = os.getenv("MODEL_PATH", MODEL_PATH_DEFAULT)

    num_classes = len(CLASS_NAMES)
    # Instansiasi model sesuai pilihan
    if arch == "resnet9plus":
        pool_ks = 4 if IMG_SIZE==256 else 2
        model = ResNet9Plus(in_channels=3, num_classes=num_classes,
                            pool_ks=pool_ks, use_se=True,
                            drop_path_prob=0.02, dropout2d=0.0,
                            dropout_fc=0.3, global_pool="avg", hidden_dim=256)
    else:
        arch = "resnet18"  # fallback
        model = ResNet18(num_diseases=num_classes, in_channels=3)

    # Muat checkpoint
    try:
        sd = torch.load(model_path, map_location="cpu")
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        # hapus prefix 'module.' bila perlu
        sd = { (k.replace("module.","") if k.startswith("module.") else k): v for k,v in sd.items() }
        model.load_state_dict(sd, strict=True)
        st.info(f"✅ Model '{arch}' dimuat dari: {model_path}")
    except FileNotFoundError:
        st.warning(f"⚠️ Checkpoint tidak ditemukan: {model_path}. Model '{arch}' akan berisi bobot random.")
    except RuntimeError as e:
        # Mismatch state_dict (mis. beda arsitektur). Coba non-strict sebagai darurat.
        st.warning(f"⚠️ Gagal strict load: {e}\nMencoba non-strict (beberapa layer mungkin tidak terisi).")
        try:
            model.load_state_dict(sd, strict=False)
        except Exception as e2:
            st.error(f"❌ Gagal memuat checkpoint ke arsitektur '{arch}': {e2}")
            # jika benar-benar tidak cocok, jatuhkan tanpa bobot
    model.eval()
    return model

# =========================
#  Predict
# =========================
def predict_image(model, image):
    """
    Mengembalikan (label_str, probs_numpy)
    """
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits[0], dim=0)
        pred_idx = int(torch.argmax(probs).item())
    return CLASS_NAMES[pred_idx], probs.cpu().numpy()
