# TruthLens — DeepFake Detection Web App

## Quick Start

### 1. Train the model (Google Colab)
Open `DeepFake_Training_Ensemble.ipynb` in Colab with a T4 GPU.
After training, download `deepfake_export.zip` and extract `deepfake_ensemble.pkl`.

### 2. Set up the Flask app
```bash
cd deepfake_app
pip install -r requirements.txt

# Place your trained model here:
cp /path/to/deepfake_ensemble.pkl .

python app.py
# → Open http://localhost:5000
```

### 3. Optional: Enable LLaVA explanations
```bash
# Install Ollama from https://ollama.com
ollama pull llava
ollama serve   # runs on localhost:11434
# Then toggle "LLaVA explanation" ON in the UI
```

## File structure
```
deepfake_app/
├── app.py                    # Flask backend
├── deepfake_ensemble.pkl     # ← place trained model here
├── requirements.txt
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   ├── uploads/              # temp upload storage
│   └── results/              # saved result images
└── templates/
    └── index.html
```

## What the app does
1. User uploads image or video
2. Flask extracts frames, runs through XceptionNet + EfficientNet-B4+DCT ensemble
3. Grad-CAM++ heatmaps generated for both models
4. Real-time streaming results via SSE (Server-Sent Events)
5. Optional: LLaVA forensic explanation via Ollama

## Running without the PKL
The app starts in demo mode with random weights if `deepfake_ensemble.pkl`
is not found. Train the model first for real predictions.
