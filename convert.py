import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
import subprocess
import sys

class TerrainClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    
    def forward(self, x):
        return self.net(x)

model = TerrainClassifier()
model.load_state_dict(torch.load("terrain_model.pth"))
model.eval()

dummy_input = torch.randn(1, 6)
torch.onnx.export(
    model, dummy_input, "terrain_model.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=18
)
print("Exported to ONNX")

# use sys.executable to guarantee we call THIS venv's python, not system python
result = subprocess.run([
    sys.executable, "-m", "tf2onnx.convert",
    "--onnx", "terrain_model.onnx",
    "--output", "terrain_model.tflite",
    "--target", "tflite"
], capture_output=True, text=True)

print(result.stdout)
print(result.stderr)
print("Converted to TFLite")

interpreter = tf.lite.Interpreter(model_path="terrain_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

test_input = np.random.randn(1, 6).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
print(f"Output shape: {output.shape}")
print(f"Raw output: {output}")
print("TFLite model working correctly!")