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

res = torch.load('path_to_saved_model.pth', map_location=device)
res.eval()

page = st.sidebar.selectbox("Выберите классификатор", ["Inception", "Котик или Псина","Кровушка"])

if page == "Inception":
    st.header("Inception ImageNet Классификатор")

    uploaded_file = st.file_uploader("Выберите фотографию", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        def predict_inception(image):
            transform = transforms.Compose([
                transforms.Resize(299),  # Inception требует изображения размером 299x299
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            image = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = inception(image)
                _, preds = outputs.max(1)
                return preds.item()
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
elif page == ("Котик или Псина"):
    st.header("Котик я милый или псина дрожащая")

    uploaded_file = st.file_uploader("Выберите фотографию", type=["jpg", "png", "jpeg"])


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
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Загруженная фотография', use_column_width=True)
        st.write("Определяем...")
        label = predict_cat_dog(image)
        st.write(f'Скорее всего на фотографии {label}.')
elif page == "Кровушка":
    st.header("Классификатор на основе новой ResNet модели")

    uploaded_file = st.file_uploader("Выберите фотографию", type=["jpg", "png", "jpeg"])

    def predict_resnet_model(image):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = res(image)
            predicted_class_idx = torch.argmax(output, 1).item()

        # Используем словарь для получения имени класса по индексу
        class_dict = {'EOSINOPHIL': 0, 'LYMPHOCYTE': 1, 'MONOCYTE': 2, 'NEUTROPHIL': 3}
        index_to_class = {v: k for k, v in class_dict.items()}
        return index_to_class[predicted_class_idx]
        with torch.no_grad():
            output = res(image)
            predicted_class_idx = torch.argmax(output, 1).item()

        # Используем словарь для получения имени класса по индексу
        class_dict = {'EOSINOPHIL': 0, 'LYMPHOCYTE': 1, 'MONOCYTE': 2, 'NEUTROPHIL': 3}
        index_to_class = {v: k for k, v in class_dict.items()}
        return index_to_class[predicted_class_idx]

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Загруженная фотография', use_column_width=True)
        st.write("Определяем...")
        label = predict_resnet_model(image)
        st.write(f'дальше будет что-то на латыни {label}.')




