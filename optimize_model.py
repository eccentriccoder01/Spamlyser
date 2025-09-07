import torch
from transformers import AutoModelForSequenceClassification
import onnx
import onnxruntime as ort

# Example: DistilBERT model id
MODEL_ID = "distilbert-base-uncased"
ONNX_PATH = "distilbert.onnx"

# Load PyTorch model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

# Dummy input for export (batch_size=1, sequence_length=8)
dummy_input = torch.randint(0, 100, (1, 8))

# Export to ONNX
with torch.no_grad():
    torch.onnx.export(
        model,
        dummy_input,
        "distilbert.onnx",
        input_names=["input_ids"],
        output_names=["output"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}, "output": {0: "batch_size"}},
        opset_version=14
    )
print(f"Model exported to {ONNX_PATH}")

# Quantization (PyTorch static quantization example)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
print("Model quantized with torch.quantization.quantize_dynamic.")

# Save quantized model
torch.save(quantized_model.state_dict(), "distilbert_quantized.pth")
print("Quantized model saved as distilbert_quantized.pth")

# ONNX inference test
ort_session = ort.InferenceSession(ONNX_PATH)
outputs = ort_session.run(None, {"input_ids": dummy_input.numpy()})
print("ONNX inference output:", outputs)
