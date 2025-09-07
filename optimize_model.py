import torch
from transformers import AutoModelForSequenceClassification

# Example: DistilBERT model id
MODEL_ID = "distilbert-base-uncased"

# Load PyTorch model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

# Quantization (PyTorch dynamic quantization)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
print("Model quantized with torch.quantization.quantize_dynamic.")

# Save quantized model
torch.save(quantized_model.state_dict(), "distilbert_quantized.pth")
print("Quantized model saved as distilbert_quantized.pth")
