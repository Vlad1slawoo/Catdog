import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('resnet18_full_model.pth', map_location=device)
model.eval()

def predict(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        predicted = (torch.sigmoid(output) > 0.5).float().item()
    return "Псина обнаружена" if predicted else "Это котик!"

st.title('Котик я милый или псина сутулая')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = predict(image)
    st.write(f'The uploaded image most likely belongs to a {label}.')

