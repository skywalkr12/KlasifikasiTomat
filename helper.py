import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import streamlit as st

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

# --------- Model ---------
class ResNet18(ImageClassificationBase):
    """
    Ini BUKAN ResNet18 resmi; ini arsitektur ResNet9-variant yang kamu pakai.
    Kami beri default in_channels=3 agar instansiasi di Streamlit sederhana.
    """
    def __init__(self, num_diseases=10, in_channels=3):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)   # 256 -> 64
        self.res1  = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)  # 64 -> 16
        self.conv4 = ConvBlock(256, 512, pool=True)  # 16 -> 4
        self.res2  = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        # Head size-agnostic, tetap "max" agar konsisten:
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),  # ganti dari MaxPool2d(4)
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

# Pastikan sama dengan training (kamu memang pakai 256 & tanpa normalize)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
    # kalau training pakai normalize, tambahkan di sini juga
    # transforms.Normalize(mean=[...], std=[...]),
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

# --------- Predict ---------
def predict_image(model, image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs[0], dim=0)
        pred_idx = torch.argmax(probs).item()
    return CLASS_NAMES[pred_idx], probs.numpy()
