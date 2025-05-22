import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import pickle
from torchvision import transforms

# Load model and classes
with open("resnet50_leaf_model.pkl", "rb") as f:
    model_bundle = pickle.load(f)

model = model_bundle["model"]
class_names = model_bundle["class_names"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Streamlit UI
st.title("🍃 Leaf Disease Classifier")
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose page", ["Home", "Predict Disease"])

if app_mode == "Home":
    st.markdown("### Welcome to the Leaf Disease Prediction App!")
    st.markdown("Upload a leaf image in the **Predict Disease** section to identify the disease.")

elif app_mode == "Predict Disease":
    uploaded_file = st.file_uploader("Upload a Leaf Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            top_prob, top_class = torch.max(probs, 1)

        st.success(f"### Predicted Disease: **{class_names[top_class.item()]}**")
        st.info(f"Confidence: **{top_prob.item()*100:.2f}%**")
