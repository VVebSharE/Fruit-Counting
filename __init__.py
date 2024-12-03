import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from core.data.dataset import CustomCocoDataset

# from ..core.Network.direct import

# Load dataset and model
dataset = CustomCocoDataset(
    root="D:/DATA/APPLE/Roboflow/valid",
    annFile="D:/DATA/APPLE/Roboflow/valid/_annotations.coco.json",
    transform=ToTensor(),
)


# model = YourModel()
# model.load_state_dict(torch.load("path/to/model.pth"))
# model.eval()
def model(img):
    return {"bboxes": [[0, 0, 100, 100]], "count": 1}


# Function to visualize dataset images
def visualize_dataset(index):
    img, label = dataset[index]

    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(np.array(img).transpose(1, 2, 0))  # Convert tensor to image

    for bbox in label["bboxes"]:
        x, y, w, h = bbox
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor="blue", facecolor="none"
        )
        ax.add_patch(rect)

    ax.text(
        10,
        10,
        f"Count: {label['count']}",
        fontsize=12,
        color="white",
        bbox=dict(facecolor="black", alpha=0.5),
    )

    st.pyplot(fig)


# Function to perform inference and visualize predictions
def model_inference(image):
    # Preprocess the image
    transform = ToTensor()
    img_tensor = transform(image).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        prediction = model(img_tensor)

    # Extract predictions (example: replace with your model's output structure)
    predicted_bboxes = prediction["bboxes"]
    predicted_count = prediction["count"]

    # Visualize predictions
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(image)

    for bbox in predicted_bboxes:
        x, y, w, h = bbox
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)

    ax.text(
        10,
        10,
        f"Predicted Count: {predicted_count}",
        fontsize=12,
        color="white",
        bbox=dict(facecolor="black", alpha=0.5),
    )

    st.pyplot(fig)


# Streamlit App Layout
st.title("Fruit Counting Model Viewer")

# Dataset Viewer
st.sidebar.header("Dataset Viewer")
dataset_index = st.sidebar.slider("Dataset Index", 0, len(dataset) - 1, 0)
visualize_dataset(
    dataset_index
)  # Automatically update the dataset image when slider changes

# Model Inference
st.sidebar.header("Model Inference")
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    if st.sidebar.button("Run Model"):
        model_inference(img)
