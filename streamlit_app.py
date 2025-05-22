import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import pickle
from torchvision import transforms

# ---------------------- LEAF ANIMATION CSS ---------------------- #
leaf_animation_css = """
<style>
body {
    background: #f4f6f8;
    overflow-x: hidden;
}

.leaf {
    position: fixed;
    width: 40px;
    height: 40px;
    background-image: url('https://cdn-icons-png.flaticon.com/512/427/427735.png'); /* Leaf icon */
    background-size: contain;
    background-repeat: no-repeat;
    opacity: 0.5;
    animation: floatLeaf 20s linear infinite;
}

@keyframes floatLeaf {
    0% {
        transform: translateY(100vh) rotate(0deg);
        left: calc(100% * var(--i));
        opacity: 0;
    }
    30% {
        opacity: 0.5;
    }
    100% {
        transform: translateY(-10vh) rotate(360deg);
        opacity: 0;
    }
}

.leaf-container {
    z-index: 0;
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    pointer-events: none;
}
</style>

<div class="leaf-container">
    <div class="leaf" style="--i: 0.1; animation-delay: 0s;"></div>
    <div class="leaf" style="--i: 0.2; animation-delay: 2s;"></div>
    <div class="leaf" style="--i: 0.3; animation-delay: 4s;"></div>
    <div class="leaf" style="--i: 0.4; animation-delay: 1s;"></div>
    <div class="leaf" style="--i: 0.5; animation-delay: 3s;"></div>
    <div class="leaf" style="--i: 0.6; animation-delay: 5s;"></div>
    <div class="leaf" style="--i: 0.7; animation-delay: 6s;"></div>
    <div class="leaf" style="--i: 0.8; animation-delay: 7s;"></div>
</div>
"""

st.markdown(leaf_animation_css, unsafe_allow_html=True)

# ---------------------- APP TITLE ---------------------- #
st.title("🍃 Leaf Disease Classifier")
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose page", ["Home", "Predict Disease"])

# ---------------------- PAGE: HOME ---------------------- #
if app_mode == "Home":
    st.markdown("### Welcome to the futuristic Leaf Disease Prediction App! 🍃")
    st.markdown("Use the **Predict Disease** section to upload a leaf image and identify possible diseases.")
    st.info("This app uses a deep learning model fine-tuned on various plant leaf diseases.")

# ---------------------- PAGE: PREDICTION ---------------------- #
elif app_mode == "Predict Disease":
    uploaded_file = st.file_uploader("Upload a Leaf Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Apply transform and predict
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        image_tensor = transform(image).unsqueeze(0)

        # Load model
        with open("resnet50_leaf_model.pkl", "rb") as f:
            model_bundle = pickle.load(f)

        model = model_bundle["model"]
        class_names = model_bundle["class_names"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        image_tensor = image_tensor.to(device)

        with st.spinner("🔍 Classifying... please wait a moment"):
            with torch.no_grad():
                outputs = model(image_tensor)
                probs = F.softmax(outputs, dim=1)
                top_prob, top_class = torch.max(probs, 1)

        st.success(f"### Predicted Disease: **{class_names[top_class.item()]}**")
        st.info(f"Confidence: **{top_prob.item()*100:.2f}%**")
