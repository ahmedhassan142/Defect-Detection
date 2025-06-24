import os
import sys
import warnings

# 1. COMPLETE SILENCE - Suppress ALL possible warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress ALL TensorFlow output
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Completely disable GPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
warnings.filterwarnings('ignore')  # Suppress Python warnings

# 2. Redirect stderr to suppress remaining TensorFlow messages
class SuppressStderr:
    def __enter__(self):
        self.original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self.original_stderr

with SuppressStderr():
    import tensorflow as tf
    tf.autograph.set_verbosity(0)
    tf.get_logger().setLevel('ERROR')
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)

# 3. Now import other dependencies quietly
import numpy as np
import cv2
from huggingface_hub import hf_hub_download
import gradio as gr

# 4. Configuration
CLASS_NAMES = ['Patches', 'Pitted', 'Scratches', 'Rolled', 'Crazing', 'Inclusion']
MODEL_REPO = "Ahmedhassan54/Defect_Detection_Model"
MODEL_FILE = "best_defect_model.h5"

# 5. Model Loading with Complete CPU Isolation
def load_model():
    """Silent model loading with CPU optimizations"""
    tf.config.set_visible_devices([], 'GPU')  # Explicitly disable GPU
    
    # Download model silently
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        cache_dir="model_cache",
        quiet=True  # Suppress download progress
    )
    
    # Load model with CPU optimizations
    model = tf.keras.models.load_model(model_path, compile=False)
    model.trainable = False
    
    # Warm up model silently
    dummy_input = np.zeros((1, 256, 256, 3), dtype=np.float32)
    model.predict(dummy_input, verbose=0)
    
    return model

model = load_model()

# 6. Image Processing
def preprocess_image(image):
    """Silent image preprocessing"""
    if image is None:
        return None
        
    if len(image.shape) == 2:  # Grayscale
        image = np.stack((image,)*3, axis=-1)
    elif image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]
    
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    return np.expand_dims(image.astype('float32') / 255.0, axis=0)

# 7. Prediction Function
def predict_defect(image):
    """Completely silent prediction"""
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

# 8. Create Interface
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

# 9. Silent Launch
if __name__ == "__main__":
    # Final suppression of any remaining messages
    with SuppressStderr():
        demo.launch(
          
            server_port=7860,
            show_error=False,
            debug=False,
            quiet=True  # Suppress Gradio startup messages
        )