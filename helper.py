# helper.py
# — ResNet9 (inference only) + prediksi top-k —
# Tidak ada rendering Streamlit di file ini.

import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# ========= Util umum (dipakai saat training; aman dibiarkan) =========
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
 'Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato_Target_Spot',
 'Tomato_Tomato_YellowLeaf__Curl_Virus',
 'Tomato_Tomato_mosaic_virus',
 'Tomato_healthy'
]

# ========= Transform (samakan dengan training!) =========
train_tfms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor()
])

# ========= Loader =========
def load_model(weights_path: str = "model/resnet9(99,16).pt", cache_bust: str = "v1"):
    model = ResNet9(num_diseases=len(CLASS_NAMES), in_channels=3)
    sd = torch.load(weights_path, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    sd = { (k.replace("module.","") if k.startswith("module.") else k): v for k,v in sd.items() }
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model

# ========= Prediksi =========
@torch.no_grad()
def predict_image(model, image: Image.Image):
    """
    Return:
      label_top1: str
      probs: np.ndarray shape (num_classes,) — softmax
      logits: np.ndarray shape (num_classes,) — raw
    """
    x = transform(image).unsqueeze(0)
    out = model(x)
    probs = torch.softmax(out[0], dim=0).cpu().numpy()
    label_top1 = CLASS_NAMES[int(np.argmax(probs))]
    return label_top1, probs, out[0].detach().cpu().numpy()

@torch.no_grad()
def predict_topk(model, image: Image.Image, k: int = 3):
    _, probs, _ = predict_image(model, image)
    k = max(1, min(k, len(CLASS_NAMES)))
    idxs = np.argsort(-probs)[:k]
    return [(CLASS_NAMES[i], float(probs[i])) for i in idxs]

