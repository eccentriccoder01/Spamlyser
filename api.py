from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ensemble_classifier_method import EnsembleSpamClassifier, ModelPerformanceTracker
from app import MODEL_OPTIONS, ENSEMBLE_METHODS, get_ensemble_predictions

app = FastAPI(title="SpamLyser Real-Time API", description="API for real-time spam detection using SpamLyser ensemble models.")

class PredictRequest(BaseModel):
    message: str
    method: Optional[str] = "majority_voting"

class PredictResponse(BaseModel):
    label: str
    confidence: float
    spam_probability: float
    method: str
    details: Optional[str] = None

# Load models and ensemble classifier
performance_tracker = ModelPerformanceTracker()
ensemble_classifier = EnsembleSpamClassifier(performance_tracker=performance_tracker)

# Load all models once
_loaded_models = {}
for model_name in MODEL_OPTIONS:
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        model_id = MODEL_OPTIONS[model_name]["id"]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        _loaded_models[model_name] = pipe
    except Exception as e:
        _loaded_models[model_name] = None

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    message = request.message
    method = request.method if request.method in ENSEMBLE_METHODS else "majority_voting"
    predictions = get_ensemble_predictions(message, _loaded_models)
    if not predictions:
        return PredictResponse(label="UNKNOWN", confidence=0.0, spam_probability=0.0, method=method, details="No model predictions available.")
    result = ensemble_classifier.get_ensemble_prediction(predictions, method)
    return PredictResponse(label=result["label"], confidence=result["confidence"], spam_probability=result["spam_probability"], method=method, details=result.get("details", ""))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)