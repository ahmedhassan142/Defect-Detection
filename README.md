---
title: Steel Surface Defect Detection
emoji: 🏭
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.12.0
app_file: app.py
pinned: false
license: mit
---

# 🏭 Steel Surface Defect Detection Model

A deep learning model that classifies 6 types of steel surface defects with 92%+ accuracy.

![Demo GIF](https://example.com/path/to/demo.gif) <!-- Replace with actual demo GIF -->

## 🔍 Model Details
- **Architecture**: Custom CNN with EfficientNet backbone
- **Input Size**: 256x256 RGB images
- **Output Classes**: 6 defect types
  - Crazing
  - Inclusion
  - Patches
  - Pitted Surface
  - Rolled-in Scale
  - Scratches

## 🚀 Live Demo
Try the model in your browser:  
👉 [Live Gradio Demo](https://huggingface.co/spaces/Ahmedhassan54/Defect-Detection-Model)

## 🛠️ Usage

### Python Inference
```python
from huggingface_hub import hf_hub_download
import tensorflow as tf
import cv2
import numpy as np

# Download model
model_path = hf_hub_download(
    repo_id="Ahmedhassan54/Defect_Detection_Model",
    filename="defect_model.h5"
)

# Load model
model = tf.keras.models.load_model(model_path)

# Preprocess image
img = cv2.imread("your_image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))
img = np.expand_dims(img/255.0, axis=0)

# Predict
pred = model.predict(img)
class_names = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']
print(f"Predicted: {class_names[np.argmax(pred[0])]} ({np.max(pred[0]):.2%})")