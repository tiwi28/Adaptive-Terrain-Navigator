# Terrain-Adaptive Robot Navigator

An autonomous 2WD robot that classifies the surface it's driving on — hardwood, carpet, or incline — using a neural network running entirely on an Arduino Uno, then adapts its motor behavior in real time. No cloud, no laptop, no external compute once deployed.

## Overview

This project builds a complete embedded ML pipeline from scratch: collecting real IMU sensor data, training a classifier, and deploying it to a memory-constrained microcontroller (2KB RAM) using a hand-written C++ inference engine.

```
MPU-6050 (IMU) → Arduino → trained neural net (on-device) → terrain prediction → motor adjustment
```

## Hardware

- Arduino Uno R3 (Elegoo)
- MPU-6050 accelerometer/gyroscope (soldered header pins)
- 2WD chassis kit — DC gear motors, wheels, battery box
- L298N dual H-bridge motor driver
- Breadboard + jumper wires

## Pipeline

**1. Data Collection (`collect.py`)**
A custom Python + `pyserial` pipeline reads live IMU readings (accelerometer + gyroscope, 6 values) from the Arduino over serial and saves labeled CSV files. Roughly 3,400 labeled readings were collected across three terrain classes by manually simulating each surface with the sensor.

**2. Model Training (`train.py`)**
Two classifiers were trained and compared:
- **Random Forest** (scikit-learn) — baseline, 100% test accuracy
- **Neural network** (PyTorch, 6→16→8→3 fully connected, ReLU) — 97% test accuracy after tuning

The neural net was chosen for deployment despite marginally lower accuracy, since it generalizes better and is what the project set out to explore.

**3. Weight Export (`export_weights.py`)**
Rather than deploying via TensorFlow Lite Micro, the trained PyTorch weights are extracted directly as NumPy arrays and written into C++ source (`model_weights.h` / `model_weights.cpp`) as `extern`-declared float arrays.

**4. On-Device Inference (`model_inference.cpp`)**
A hand-written forward pass (matrix multiply → ReLU → matrix multiply → ReLU → matrix multiply → argmax) runs natively on the Arduino, no external ML library required. Input features are normalized on-device using the exported `StandardScaler` mean/scale values before inference.

**5. Motor Control** *(in progress)*
An L298N motor driver will translate the live terrain prediction into adjusted motor behavior — e.g., different power/speed profiles for carpet vs. hardwood vs. incline.

## Engineering Decisions

**Why not TensorFlow Lite Micro?**
The standard PyTorch → ONNX → TFLite conversion pipeline was attempted first but hit repeated, unresolvable dependency conflicts on Windows (`onnx-tf`, `ai-edge-torch`, and `tf2onnx` each failed or turned out to solve the wrong direction of conversion). Given the network's small size, hand-rolling the forward pass in C++ was a more reliable and arguably more appropriate engineering choice than fighting a heavyweight conversion toolchain — it also removes any runtime dependency on TFLite Micro entirely.

**Why a smaller network (6→16→8→3) instead of the original (6→128→64→3)?**
The first trained model didn't fit in the Arduino Uno's 2KB of RAM — a single weight matrix alone required 32KB. The network was resized to fit the hardware constraint, then retrained with more training epochs to compensate for the smaller model's slower convergence, recovering accuracy back to ~97%.

## Repo Structure

```
collect.py            # serial data collection → labeled CSV
train.py               # trains Random Forest + PyTorch classifier
export_weights.py      # extracts trained weights → C++ source
model_weights.h/.cpp    # exported model weights (extern-declared)
model_inference.cpp    # hand-written C++ forward pass + prediction
hardwood.csv / carpet.csv / incline.csv   # collected training data
terrain_model.pth       # trained PyTorch model
scaler.pkl / label_encoder.pkl  # preprocessing artifacts
```

## Status

- [x] Hardware assembled and wired
- [x] IMU data collection pipeline
- [x] Model training (Random Forest + neural net)
- [x] On-device C++ inference deployed and verified — live predictions printing from real sensor data
- [ ] Motor driver wiring (L298N)
- [ ] Adaptive motor control logic
- [ ] Real-world retraining on robot-collected (vs. hand-simulated) data

## Future Work

Once motors are wired, the plan is to collect a second dataset from the robot actually driving on each surface (rather than hand-simulated motion) and retrain the model on that — real-world sensor signatures from wheel/motor vibration should differ meaningfully from hand simulation and should improve real-world robustness.