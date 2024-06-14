import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import requests
import os

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the pre-trained model
class AlexNet(nn.Module):
    def __init__(self, num_classes=1):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Download model from GitHub
@st.cache_resource
def download_model():
    try:
        url = 'https://github.com/nandeeshhu/AI_FAKE_IMAGE_CLASSIFIER/blob/my-new-branch/ai_imageclassifier_1.pth'
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes

        # Check if the content type is appropriate for a model file
        content_type = response.headers.get('Content-Type')
        if 'application/octet-stream' not in content_type:
            raise ValueError("Downloaded file is not a valid model checkpoint.")

        with open('ai_imageclassifier_1.pth', 'wb') as f:
            f.write(response.content)

        # Validate the file
        if os.path.getsize('ai_imageclassifier_1.pth') == 0:
            raise ValueError("Downloaded file is empty.")
        
        return 'ai_imageclassifier_1.pth'
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while downloading the model: {e}")
        return None
    except ValueError as ve:
        st.error(f"File validation error: {ve}")
        return None

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AlexNet()
model_path = download_model()
if model_path:
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        st.error(f"Error loading the model: {e}")
else:
    st.error("Model could not be loaded. Please check the download path or internet connection.")


# Streamlit app
st.title("Image Classification with AlexNet")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg", key="nand18")

if uploaded_file is not None:
    # Convert the file to an image
    image = Image.open(uploaded_file).convert("RGB")

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Perform prediction
    with torch.no_grad():
        output = model(image_tensor)
        prediction = output.item()

    # Convert prediction to label
    predicted_label = 'Real' if prediction >= 0.5 else 'Fake'
    if(predicted_label == 'Real'):
        prediction = 100*(prediction)
    else:
        prediction = 100*(1-prediction)

    # Display the image with the predicted label
    st.image(image, caption=f'Predicted Label: {predicted_label} ({prediction:.2f})', use_column_width=True)

    # Highlighted and centered text
    st.markdown(f"""
        <div style="text-align: center; font-size: 24px; font-weight: bold; color: #007BFF;">
            Predicted Label: {predicted_label}
        </div>
        <div style="text-align: center; font-size: 20px;">
            Prediction Score (Confidence): {prediction:.2f}%
        </div>
    """, unsafe_allow_html=True)
    st.write("**Note:** This model is not 100% accurate and may make mistakes on some unseen instances.")


# Sidebar for optional details
st.sidebar.header("Options")

# Toggle to show/hide model architecture
if st.sidebar.checkbox("Show Model Architecture"):
    # Function to create a summary of the model
    def model_summary(model):
        summary_str = ""
        summary_str += "Model Architecture:\n"
        summary_str += "-" * 80 + "\n"
        for name, layer in model.named_children():
            summary_str += f"{name}: {layer}\n"
        summary_str += "-" * 80 + "\n"
        return summary_str

    st.sidebar.subheader("Model Architecture")
    st.sidebar.text(model_summary(model))

# Display model evaluation metrics
st.sidebar.subheader("Model Evaluation Metrics")
st.sidebar.write("Accuracy: 99.5%")
st.sidebar.write("Precision: 99.0%")
st.sidebar.write("Recall: 100%")

# Add dataset information
st.sidebar.subheader("Dataset Information")
st.sidebar.write("This model is trained on datasets collected from various domains of living things(including human) images. The datasets were collected through web scraping from Google and include a variety of categories.")

st.sidebar.markdown(f"""
        <div style="font-size: 15px; font-weight: bold; color: #007BFF;">
            Developed By:
        </div>
    """, unsafe_allow_html=True)

st.sidebar.write("Nandeesh H U")
st.sidebar.write("10nandeeshhu@gmail.com")
