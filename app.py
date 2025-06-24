import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from huggingface_hub import hf_hub_download
import os

# Suppress all unnecessary logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.get_logger().setLevel('ERROR')

# Constants
CLASS_NAMES = ['Patches', 'Pitted', 'Scratches', 'Rolled', 'Crazing', 'Inclusion']
MODEL_REPO = "Ahmedhassan54/Defect_Detection_Model"
MODEL_FILE = "best_defect_model.h5"

# Load model
def load_model():
    tf.keras.backend.clear_session()
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    model = tf.keras.models.load_model(model_path, compile=False)
    model.trainable = False
    return model

model = load_model()

# Prediction function
def predict_defect(image):
    if image is None:
        return "No image", 0, {"x": CLASS_NAMES, "y": [0]*len(CLASS_NAMES)}
    
    # Preprocess
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=-1)
    image = cv2.resize(image, (256, 256))
    image = np.expand_dims(image.astype('float32') / 255.0, axis=0)
    
    # Predict
    predictions = model.predict(image, verbose=0)
    scores = tf.nn.softmax(predictions[0]).numpy()
    
    return (
        CLASS_NAMES[np.argmax(scores)],
        float(np.max(scores) * 100),
        {"x": CLASS_NAMES, "y": [float(s) for s in scores]}
    )

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🏭 Steel Surface Defect Detection")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="numpy")
            btn = gr.Button("Detect Defect")
        
        with gr.Column():
            label = gr.Label()
            confidence = gr.Number(label="Confidence (%)")
            plot = gr.BarPlot(x=CLASS_NAMES, y=[0]*6, vertical=False)

    btn.click(
        predict_defect,
        inputs=img_input,
        outputs=[label, confidence, plot]
    )

if __name__ == "__main__":
    demo.launch(debug=True,share=False)