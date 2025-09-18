import torch
from transformers import AutoModelForSequenceClassification

MODEL_ID = "distilbert-base-uncased"
ONNX_PATH = "distilbert.onnx"

model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

dummy_input = torch.randint(0, 100, (1, 8))

with torch.no_grad():
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input_ids"],
        output_names=["output"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}, "output": {0: "batch_size"}},
        opset_version=14
    )
print(f"Model exported to {ONNX_PATH}")