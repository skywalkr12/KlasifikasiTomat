
import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

CLASS_NAMES = [
    "Early Blight", "Late Blight", "Leaf Mold", "Septoria Leaf Spot", 
    "Spider Mites", "Target Spot", "Yellow Leaf Curl Virus", 
    "Mosaic Virus", "Healthy"
]

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()   # langsung ke tensor [0-1]
])

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
