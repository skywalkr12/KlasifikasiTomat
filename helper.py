import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import streamlit as st

# --- Class Model ---
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  
        loss = F.cross_entropy(out, labels) 
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    
        loss = F.cross_entropy(out, labels)   
        acc = accuracy(out, labels)           
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['val_loss'], result['val_acc']))

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class SimpleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, xb):
        residual = xb
        out = F.relu(self.bn1(self.conv1(xb)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ResNet18(ImageClassificationBase):
    def __init__(self, num_classes=9):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.res1 = nn.Sequential(
            SimpleResidualBlock(64, 64),
            SimpleResidualBlock(64, 64)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.res2 = nn.Sequential(
            SimpleResidualBlock(128, 128),
            SimpleResidualBlock(128, 128)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out)
        out = self.conv3(out)
        out = self.res2(out)
        out = self.conv4(out)
        return self.classifier(out)


# --- Kelas Nama ---
CLASS_NAMES = [
    "Early Blight", "Late Blight", "Leaf Mold", "Septoria Leaf Spot", 
    "Spider Mites", "Target Spot", "Yellow Leaf Curl Virus", 
    "Mosaic Virus", "Healthy"
]

# --- Transformasi ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# --- Load model dengan state_dict ---
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None, num_classes=9)   # atau sesuai arsitektur yang dipakai
    state_dict = torch.load("model/resnet_97_56.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_image(model, image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        pred_idx = torch.argmax(probs).item()
    return CLASS_NAMES[pred_idx], probs.numpy()
