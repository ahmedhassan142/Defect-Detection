import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress ALL TensorFlow messages
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Completely disable GPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

# Import TensorFlow after setting environment variables
import tensorflow as tf
tf.autograph.set_verbosity(0)  # Disable autograph warnings
tf.get_logger().setLevel('ERROR')  # Only show errors

# Now import other dependencies
import gradio as gr
import numpy as np
import cv2
from huggingface_hub import hf_hub_download
import logging
logging.basicConfig(level=logging.ERROR)  # Only log errors

# Configuration
CLASS_NAMES = ['Patches', 'Pitted', 'Scratches', 'Rolled', 'Crazing', 'Inclusion']
MODEL_REPO = "Ahmedhassan54/Defect_Detection_Model"
MODEL_FILE = "best_defect_model.h5"

def load_model():
    """Load model with complete CPU isolation"""
    # Force CPU-only operation
    tf.config.set_visible_devices([], 'GPU')
    
    # Download model
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        cache_dir="model_cache"
    )
    
    # Load with CPU optimizations
    model = tf.keras.models.load_model(model_path, compile=False)
    model.trainable = False
    
    # Warm up model
    dummy_input = np.zeros((1, 256, 256, 3), dtype=np.float32)
    model.predict(dummy_input, verbose=0)
    
    return model

model = load_model()

def preprocess_image(image):
    """Optimized CPU image processing"""
    if image is None:
        return None
        
    if len(image.shape) == 2:  # Grayscale
        image = np.stack((image,)*3, axis=-1)
    elif image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]
    
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    return np.expand_dims(image.astype('float32') / 255.0, axis=0)

def predict_defect(image):
    """Silent prediction function"""
    try:
        processed_image = preprocess_image(image)
        if processed_image is None:
            return "Invalid Image", 0, {"x": CLASS_NAMES, "y": [0]*len(CLASS_NAMES)}
        
        predictions = model.predict(processed_image, verbose=0)
        scores = tf.nn.softmax(predictions[0]).numpy()
        
        return (
            CLASS_NAMES[np.argmax(scores)],
            float(np.max(scores) * 100),
            {"x": CLASS_NAMES, "y": [float(s) for s in scores]}
        )
    except Exception:
        return "Error", 0, {"x": CLASS_NAMES, "y": [0]*len(CLASS_NAMES)}

# Create interface
with gr.Blocks() as demo:
    gr.Markdown("# 🏭 Steel Surface Defect Detection")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="numpy")
            btn = gr.Button("Detect Defect")
        
        with gr.Column():
            label = gr.Label()
            confidence = gr.Number(label="Confidence (%)")
            plot = gr.BarPlot(x=CLASS_NAMES, y=[0]*len(CLASS_NAMES), vertical=False)

    btn.click(predict_defect, inputs=img_input, outputs=[label, confidence, plot])

if __name__ == "__main__":
    # Launch with minimal logging
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    
    demo.launch(
       
        server_port=7860,
        show_error=False,
        debug=False
    )