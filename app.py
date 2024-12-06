import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import streamlit as st

# Streamlit App Title
st.title("Image Classification with DenseNet and MobileNet")

# Configure device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: *{device}*")

# Load model function
@st.cache_resource
def load_model(model_type: str, model_path: str):
    """Load a model based on its type and checkpoint."""
    if "small" in model_type.lower():
        model = models.mobilenet_v3_small(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    elif "large" in model_type.lower():
        model = models.mobilenet_v3_large(pretrained=True)
        model.classifier[3] = nn.Linear(1280, 2)
    elif "dense" in model_type.lower():
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

# Load selected model
model_types = ["MobileNetV3 Small", "MobileNetV3 Large", "DenseNet 121"]
model_choice = st.selectbox("Select Model Type:", model_types)
model_file=""
if model_choice=="DenseNet 121":
    model_file= "densenet_balanced_best_model.pth"
if model_choice=="MobileNetV3 Small":
    model_file= "mobilenet_small_Adv_consol.pth"
if model_choice=="MobileNetV3 Large":
    model_file= "mobilenet_large_consol_Adv.pth"        
if st.button("Load Model"):
    if not os.path.exists(model_file):
        st.error("Model file does not exist. Please check the path.")
    else:
        model = load_model(model_choice.lower(), model_file)
        st.success(f"{model_choice} model loaded successfully!")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Upload and process image
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Apply transformations to the image
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Load the model (for example, DenseNet)
    model_type = "dense"  # or any other model type you need to load
    model = load_model(model_type)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)
        prob = outputs.softmax(dim=1)[0]
        pred = torch.argmax(prob).item()
        label = "Real" if pred == 1 else "Fake"
    
    # Display prediction and confidence
    st.write(f"Prediction: *{label}*")
    st.write(f"Confidence: *{prob[pred].item():.2f}*")
