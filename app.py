import os
import sys
import warnings
import time
import threading

# 1. COMPLETE SILENCE CONFIGURATION
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# 2. SILENT TENSORFLOW IMPORT
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

# 3. IMPORTS
import numpy as np
import cv2
from huggingface_hub import hf_hub_download
import gradio as gr

# 4. CONFIGURATION
CLASS_NAMES = ['Patches', 'Pitted', 'Scratches', 'Rolled', 'Crazing', 'Inclusion']
MODEL_REPO = "Ahmedhassan54/Defect_Detection_Model"
MODEL_FILE = "best_defect_model.h5"

# 5. MODEL LOADING
def load_model():
    """Load model with error handling"""
    try:
        tf.config.set_visible_devices([], 'GPU')
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            cache_dir="model_cache"
        )
        model = tf.keras.models.load_model(model_path, compile=False)
        model.trainable = False
        # Warm up
        dummy_input = np.zeros((1, 256, 256, 3), dtype=np.float32)
        model.predict(dummy_input, verbose=0)
        return model
    except Exception as e:
        print(f"Model loading failed: {str(e)}")
        return None

# 6. PRELOAD MODEL IN BACKGROUND
model = None
model_loaded = False

def load_model_background():
    global model, model_loaded
    model = load_model()
    model_loaded = True

threading.Thread(target=load_model_background, daemon=True).start()

# 7. IMAGE PROCESSING
def preprocess_image(image):
    """Fast image preprocessing"""
    if image is None:
        return None
        
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    return np.expand_dims(image.astype('float32') / 255.0, axis=0)

# 8. PREDICTION FUNCTION
def predict_defect(image, progress=gr.Progress()):
    """Prediction with progress tracking"""
    if not model_loaded:
        progress(0, desc="Model still loading, please wait...")
        return "Model loading...", 0, {"x": CLASS_NAMES, "y": [0]*len(CLASS_NAMES)}
    
    progress(0.2, desc="Processing image...")
    processed_image = preprocess_image(image)
    if processed_image is None:
        return "Invalid Image", 0, {"x": CLASS_NAMES, "y": [0]*len(CLASS_NAMES)}
    
    progress(0.5, desc="Analyzing defects...")
    predictions = model.predict(processed_image, verbose=0)
    scores = tf.nn.softmax(predictions[0]).numpy()
    
    progress(0.9, desc="Finalizing results...")
    return (
        CLASS_NAMES[np.argmax(scores)],
        float(np.max(scores) * 100),
        {"x": CLASS_NAMES, "y": [float(s) for s in scores]}
    )

# 9. GRADIO INTERFACE
with gr.Blocks() as demo:
    gr.Markdown("# 🏭 Steel Surface Defect Detection")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="numpy", label="Upload Image")
            btn = gr.Button("Detect Defect", variant="primary")
            status = gr.Textbox(label="Status", value="Loading model...", interactive=False)
        
        with gr.Column():
            label = gr.Label(label="Predicted Defect")
            confidence = gr.Number(label="Confidence (%)")
            plot = gr.BarPlot(
                x=CLASS_NAMES, 
                y=[0]*len(CLASS_NAMES),
                label="Class Probabilities",
                vertical=False
            )

    # Prediction function
    btn.click(
        fn=predict_defect,
        inputs=img_input,
        outputs=[label, confidence, plot],
        show_progress="minimal"
    )
    
    # Model status update
    def update_status():
        return "Model ready!" if model_loaded else "Loading model..."
    
    # Use a separate thread to update status
    def status_updater():
        while not model_loaded:
            time.sleep(1)
            demo.queue().update(update_status, outputs=status)
    
    threading.Thread(target=status_updater, daemon=True).start()

# 10. LAUNCH APPLICATION
if __name__ == "__main__":
    with SuppressStderr():
        demo.launch(
          
            server_port=7860,
            show_error=False
        )