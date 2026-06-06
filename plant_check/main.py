# main.py

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
import os
from models.PlantDiseaseModel import PlantDiseaseModel # Your CNN class
from transformers import pipeline # For the text model
import io
from PIL import Image
from torchvision import transforms
import joblib  
import json    

# --- 1. DEFINE YOUR MODELS AND CLASSES ---

# Pydantic model for text input
class TextPredictionInputModel(BaseModel):
    input: str

# Load the class names from the file
with open('image_class_names.json', 'r') as f:
    CLASS_NAMES = json.load(f)

# --- 2. LOAD MODELS ON STARTUP ---

# Get the absolute path to the directory where main.py is
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Load Text Model (Absolute Path) ---
TEXT_MODEL_PATH = os.path.join(BASE_DIR, "disease_detection_model")
text_classifier = pipeline("text-classification", model=TEXT_MODEL_PATH)

# --- FIX #1: Load Text Encoder using an Absolute Path ---
TEXT_ENCODER_PATH = os.path.join(BASE_DIR, "text_label_encoder.joblib")
text_encoder = joblib.load(TEXT_ENCODER_PATH)
# -----------------------------------------------------

# --- Load Image Model ---
# 1. Create an instance of your model's architecture
image_model = PlantDiseaseModel()

# 2. Build the full, absolute path to the image model weights
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_disease_cnn.pth")

# --- FIX #2: Load weights using 'with open' (more robust to spaces in path) ---
try:
    with open(IMAGE_MODEL_PATH, 'rb') as f:
        weights = torch.load(f, map_location=torch.device('cpu'), weights_only=False)
    
    image_model.load_state_dict(weights)

except FileNotFoundError:
    print(f"FATAL ERROR: The file was not found at the path:")
    print(f"{IMAGE_MODEL_PATH}")
    print("Please ensure the file 'plant_disease_cnn.pth' exists in the 'models' folder.")
    exit()
# -------------------------------------------------------------------------

# 3. Set the model to evaluation mode
image_model.eval()


# --- 3. CREATE FASTAPI APP ---
app = FastAPI(title="Plant Doc API")


# --- 4. DEFINE API ENDPOINTS ---
@app.get("/health-check")
def health_check():
    return {"status": "Ok"}

@app.post("/image-prediction")
async def image_predict(file: UploadFile = File(...)):
    # Read the image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB") # Use .convert("RGB") to ensure 3 channels

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4759, 0.5003, 0.4266],
                             std=[0.2102, 0.1888, 0.2262])
    ])
    
    # Preprocess the image and add a batch dimension
    img_tensor = transform(image).unsqueeze(0)

    # Make a prediction
    with torch.no_grad():
        outputs = image_model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_score = confidence.item()
        
    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "confidence": f"{confidence_score:.4f}"
    }


@app.post("/text-prediction")
def text_predict(input_data: TextPredictionInputModel):
    # 1. Use the text classification pipeline
    raw_prediction = text_classifier(input_data.input)[0] # Get the first result
    
    # 2. Extract the predicted label ID
    label_id = int(raw_prediction['label'].split('_')[-1])
    
    # 3. Use the loaded encoder to get the human-readable name
    predicted_label_name = text_encoder.inverse_transform([label_id])[0]
    
    # 4. Return a clean, user-friendly JSON response
    return {
        "input_text": input_data.input,
        "predicted_disease": predicted_label_name,
        "confidence": f"{raw_prediction['score']:.4f}"
    }