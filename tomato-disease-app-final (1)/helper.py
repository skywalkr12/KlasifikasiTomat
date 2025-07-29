
import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms

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
    model = torch.load("model/plant-disease-model-complete.pth", map_location="cpu")
    model.eval()
    return model

def predict_image(model, image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        pred_idx = torch.argmax(probs).item()
    return CLASS_NAMES[pred_idx], probs.numpy()
