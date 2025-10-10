"""
Model initialization and verification module
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def verify_model_availability():
    """
    Verify that required ML frameworks and models are available
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU for model inference.")
    
    try:
        # Initialize a simple test model to verify transformers setup
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
        return True
    except Exception as e:
        print(f"Error loading transformer models: {str(e)}")
        return False

# Pre-load models
MODEL_STATUS = verify_model_availability()