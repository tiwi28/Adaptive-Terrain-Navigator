import torch
import torch.nn as nn
import numpy as np
import pickle

class TerrainClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )
    
    def forward(self, x):
        return self.net(x)

model = TerrainClassifier()
model.load_state_dict(torch.load("terrain_model.pth", weights_only=True))
model.eval()

# extract weights and biases from each Linear layer
w1 = model.net[0].weight.detach().numpy()  # Linear(6,16)
b1 = model.net[0].bias.detach().numpy()
w2 = model.net[2].weight.detach().numpy()  # Linear(16,8)
b2 = model.net[2].bias.detach().numpy()
w3 = model.net[4].weight.detach().numpy()  # Linear(8,3)
b3 = model.net[4].bias.detach().numpy()


# load scaler for normalization values
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# load label encoder to know class order
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)


# Generating C++ header file
def format_array_1d(name, arr):
    values = ", ".join(f"{v:.6f}f" for v in arr)
    return f"const float {name}[{len(arr)}] = {{{values}}};\n"

def format_array_2d(name, arr):
    rows, cols = arr.shape
    lines = [f"const float {name}[{rows}][{cols}] = {{"]
    for row in arr:
        values = ", ".join(f"{v:.6f}f" for v in row)
        lines.append(f"  {{{values}}},")
    lines.append("};\n")
    return "\n".join(lines)

with open("model_weights.h", "w") as f:
    f.write("#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n")
    
    f.write(format_array_2d("W1", w1))
    f.write(format_array_1d("BIAS1", b1))
    f.write(format_array_2d("W2", w2))
    f.write(format_array_1d("BIAS2", b2))   
    f.write(format_array_2d("W3", w3))
    f.write(format_array_1d("BIAS3", b3))

    
    
    
    
    f.write(format_array_1d("SCALER_MEAN", scaler.mean_))
    f.write(format_array_1d("SCALER_SCALE", scaler.scale_))
    
    f.write(f'const char* CLASSES[3] = {{"{le.classes_[0]}", "{le.classes_[1]}", "{le.classes_[2]}"}};\n')
    
    f.write("\n#endif\n")

