import os
import sys
import warnings
import time
import threading
import numpy as np
import cv2
import gradio as gr
from huggingface_hub import hf_hub_download

# 1. SILENCE ALL WARNINGS
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

# 3. CONFIGURATION
CLASS_NAMES = ['Patches', 'Pitted', 'Scratches', 'Rolled', 'Crazing', 'Inclusion']
MODEL_REPO = "Ahmedhassan54/Defect_Detection_Model"
MODEL_FILE = "best_defect_model.h5"

# 4. MODEL LOADING
model = None
model_ready = False

def load_model():
    global model, model_ready
    try:
        tf.config.set_visible_devices([], 'GPU')
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            cache_dir="model_cache"
        )
        model = tf.keras.models.load_model(model_path, compile=False)
        model.trainable = False
        # Warm up model
        dummy_input = np.zeros((1, 256, 256, 3), dtype=np.float32)
        model.predict(dummy_input, verbose=0)
        model_ready = True
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model loading failed: {str(e)}")

# Load model in background
threading.Thread(target=load_model, daemon=True).start()

# 5. IMAGE PROCESSING
def preprocess_image(image):
    if image is None:
        return None
        
    if len(image.shape) == 2:  # Grayscale
        image = np.stack((image,)*3, axis=-1)
    elif image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]
    
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    return np.expand_dims(image.astype('float32') / 255.0, axis=0)

# 6. PREDICTION FUNCTION WITH IMMEDIATE FEEDBACK
def predict_defect(image):
    if not model_ready:
        return "Model still loading...", 0, {
            "x": CLASS_NAMES,
            "y": [0]*len(CLASS_NAMES),
            "title": "Loading model..."
        }
    
    try:
        processed_image = preprocess_image(image)
        if processed_image is None:
            return "Invalid image", 0, {
                "x": CLASS_NAMES,
                "y": [0]*len(CLASS_NAMES),
                "title": "Error"
            }
        
        predictions = model.predict(processed_image, verbose=0)
        scores = tf.nn.softmax(predictions[0]).numpy()
        
        return (
            CLASS_NAMES[np.argmax(scores)],
            float(np.max(scores) * 100),
            {
                "x": CLASS_NAMES,
                "y": [float(score) for score in scores],
                "title": "Defect Probabilities"
            }
        )
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return "Error during prediction", 0, {
            "x": CLASS_NAMES,
            "y": [0]*len(CLASS_NAMES),
            "title": "Error"
        }

# 7. GRADIO INTERFACE WITH IMMEDIATE FEEDBACK
with gr.Blocks(title="Steel Defect Detection") as demo:
    gr.Markdown("# 🏭 Steel Surface Defect Detection")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload Steel Surface Image", 
                type="numpy",
                height=300
            )
            detect_btn = gr.Button(
                "Detect Defect", 
                variant="primary",
                interactive=True
            )
            status_text = gr.Textbox(
                label="Status",
                value="Loading model..." if not model_ready else "Ready",
                interactive=False
            )
        
        with gr.Column():
            result_label = gr.Label(
                label="Predicted Defect",
                value="Waiting for input..."
            )
            confidence_output = gr.Number(
                label="Confidence Score (%)",
                value=0,
                minimum=0,
                maximum=100
            )
            plot_output = gr.BarPlot(
                x=CLASS_NAMES,
                y=[0]*len(CLASS_NAMES),
                label="Defect Probabilities",
                vertical=False,
                height=300
            )
    
    # Prediction function
    detect_btn.click(
        fn=predict_defect,
        inputs=image_input,
        outputs=[result_label, confidence_output, plot_output],
        api_name="predict"
    )
    
    # Update status when model loads
    def update_status():
        return "Ready" if model_ready else "Loading model..."
    
    demo.load(
        fn=update_status,
        outputs=status_text,
        every=1
    )

# 8. LAUNCH APPLICATION
if __name__ == "__main__":
    with SuppressStderr():
        demo.launch(
           
            server_port=7860,
            show_error=True,
            debug=False
        )