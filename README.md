🌿 Plant Doc API
Dual-Mode Plant Disease Detection — Image CNN + Text Classification
Python FastAPI PyTorch HuggingFace License

Detect plant diseases from a leaf photo or a text description — served via a production-ready REST API.

📌 Overview
Plant Doc API is a FastAPI-based backend that offers two independent disease detection modes:

Mode	Input	Model
🖼️ Image	Leaf photo (JPG/PNG)	Custom CNN (PlantDiseaseModel) trained on PlantVillage
📝 Text	Symptom description	Fine-tuned BERT-based text classifier
Both endpoints return the predicted disease name and a confidence score. The system covers 14 plant species across 38 disease/healthy classes.

🌱 Supported Plants & Classes
14 plant species — Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

26 disease classes (e.g., Apple Scab, Tomato Late Blight, Grape Black Rot)
12 healthy classes (one per applicable species)
38 total output classes
🏗️ Architecture
Client Request
      │
      ▼
┌─────────────────────────────────┐
│         FastAPI Backend          │
│                                  │
│  POST /image-prediction          │
│  ┌──────────────────────────┐   │
│  │  PIL Image → Transform   │   │
│  │  → PlantDiseaseModel CNN │   │
│  │  → Softmax → Class Name  │   │
│  └──────────────────────────┘   │
│                                  │
│  POST /text-prediction           │
│  ┌──────────────────────────┐   │
│  │  Text → BERT Classifier  │   │
│  │  → LabelEncoder.inverse  │   │
│  │  → Disease Name          │   │
│  └──────────────────────────┘   │
│                                  │
│  GET  /health-check              │
└─────────────────────────────────┘
CNN Architecture (PlantDiseaseModel)
Input (3 × 224 × 224)
  → Conv2d(3→16) + ReLU + MaxPool
  → Conv2d(16→32) + ReLU + MaxPool
  → Conv2d(32→64) + ReLU + MaxPool
  → AdaptiveAvgPool2d(7×7)
  → Flatten → Linear(3136→256) + ReLU + Dropout(0.5)
  → Linear(256→38)
Output: 38-class logits
🛠️ Tech Stack
Component	Technology
API Framework	FastAPI + Uvicorn
Image Model	Custom CNN (PyTorch) — plant_disease_cnn.pth
Text Model	Fine-tuned BERT (HuggingFace Transformers)
Label Encoding	scikit-learn LabelEncoder (joblib)
Image Processing	torchvision transforms + Pillow
Training Notebooks	Jupyter (separate image + text pipelines)
🚀 Getting Started
1. Clone the Repository
git clone https://github.com/YOUR_USERNAME/plant-doc-api.git
cd plant-doc-api
2. Install Dependencies
pip install -r requirements.txt
⚠️ PyTorch 2.8 is listed in requirements. Install the correct CUDA version for your GPU from pytorch.org if needed.

3. Model Weights
The text model weights (disease_detection_model/model.safetensors, ~256MB) are tracked via Git LFS.

# Install Git LFS if not already installed
git lfs install
git lfs pull
Or download manually from the Releases page and place in disease_detection_model/.

4. Run the API
uvicorn main:app --reload
API will be live at http://127.0.0.1:8000

Interactive docs at http://127.0.0.1:8000/docs (Swagger UI)

📡 API Reference
GET /health-check
{ "status": "Ok" }
POST /image-prediction
Upload a leaf image file.

Request:

curl -X POST "http://127.0.0.1:8000/image-prediction" \
  -F "file=@tomato_leaf.jpg"
Response:

{
  "filename": "tomato_leaf.jpg",
  "predicted_class": "Tomato___Late_blight",
  "confidence": "0.9341"
}
POST /text-prediction
Describe symptoms as plain text.

Request:

curl -X POST "http://127.0.0.1:8000/text-prediction" \
  -H "Content-Type: application/json" \
  -d '{"input": "brown spots on tomato leaves with yellow halo"}'
Response:

{
  "input_text": "brown spots on tomato leaves with yellow halo",
  "predicted_disease": "Tomato___Early_blight",
  "confidence": "0.8812"
}
📁 Project Structure
plant-doc-api/
│
├── main.py                          # FastAPI app — all endpoints
├── requirements.txt                 # All dependencies (pinned)
├── image_class_names.json           # 38 class label list
├── text_label_encoder.joblib        # Fitted LabelEncoder for text model
│
├── models/
│   ├── PlantDiseaseModel.py         # CNN architecture definition
│   ├── plant_disease_cnn.pth        # Trained CNN weights (~3.2MB)
│   └── __init__.py
│
├── disease_detection_model/         # Fine-tuned BERT text classifier
│   ├── config.json
│   ├── model.safetensors            # ⚠️ 256MB — tracked via Git LFS
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── vocab.txt
│
├── MAINtrain_image_model.ipynb      # CNN training pipeline
└── MAINtrain_text_model.ipynb       # BERT fine-tuning pipeline
🏋️ Training
Two separate Jupyter notebooks handle model training:

Notebook	Model	Dataset
MAINtrain_image_model.ipynb	CNN (PlantDiseaseModel)	PlantVillage (image)
MAINtrain_text_model.ipynb	BERT text classifier	Symptom-disease text data
Image preprocessing normalization used:

Mean: [0.4759, 0.5003, 0.4266]
Std: [0.2102, 0.1888, 0.2262]
⚙️ Configuration
Key values to tune in main.py:

Parameter	Value	Description
num_classes	38	Total disease/healthy classes
Resize	(224, 224)	Input image size for CNN
map_location	cpu	Change to cuda for GPU inference
weights_only	False	PyTorch model load flag
🔮 Roadmap
 Dockerize the API for one-command deployment
 Add confidence threshold filtering (reject low-confidence predictions)
 Expand to more plant species
 Add a Gradio/Streamlit frontend demo
 Replace custom CNN with fine-tuned EfficientNet/ResNet backbone
📄 License
MIT License — free to use, modify, and distribute.

Built with FastAPI · PyTorch · HuggingFace Transformers
