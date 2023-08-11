import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загружаем модель resnet18
model = torch.load('resnet18_full_model.pth', map_location=device)
model.eval()

# Загружаем предварительно обученную модель Inception
inception = models.inception_v3(pretrained=True).to(device)
inception.eval()


def predict_cat_dog(image):
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
    return "обнаружена псина" if predicted else "котик!"

def predict_inception(image):
    transform = transforms.Compose([
        transforms.Resize(299), # Inception требует изображения размером 299x299
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = inception(image)
        _, preds = outputs.max(1)
        return preds.item() # возвращаем индекс класса

st.title('Классификатор изображений')

# Добавляем выбор страницы
page = st.sidebar.selectbox("Выберите классификатор", ["Inception", "Котик или Псина"])

if page == "Inception":
    st.header("Inception ImageNet Классификатор")

    uploaded_file = st.file_uploader("Выберите фотографию", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Загруженная фотография', use_column_width=True)
        st.write("Определяем...")
        class_idx = predict_inception(image)
        # Чтение списка классов ImageNet из файла
        with open('imagenet1000_clsidx_to_labels.txt', 'r') as f:
            classes = [line.strip() for line in f.readlines()]


        def get_class_name(class_idx):
            return classes[class_idx]


        # В функции predict_inception:

        def predict_inception(image):
            ...
            class_name = get_class_name(preds.item())
            return class_name


        class_name = get_class_name(class_idx)
        st.write(f"Изображение принадлежит классу {class_name}")
else:
    st.header("Классификатор Котик или Псина")

    uploaded_file = st.file_uploader("Выберите фотографию", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Загруженная фотография', use_column_width=True)
        st.write("Определяем...")
        label = predict_cat_dog(image)
        st.write(f'Скорее всего на фотографии {label}.')
