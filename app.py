import os
import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from huggingface_hub import hf_hub_download

# 1. Force CPU-only mode and suppress all warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU completely
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
tf.get_logger().setLevel('ERROR')  # Set TensorFlow logger to only show errors

# 2. Configuration
CLASS_NAMES = ['Patches', 'Pitted', 'Scratches', 'Rolled', 'Crazing', 'Inclusion']
MODEL_REPO = "Ahmedhassan54/Defect_Detection_Model"
MODEL_FILE = "best_defect_model.h5"

# 3. Model Loading with CPU Optimization
def load_model():
    """Load the model with CPU-specific optimizations"""
    # Clear any existing sessions
    tf.keras.backend.clear_session()
    
    # Configure TensorFlow to use CPU only
    tf.config.set_visible_devices([], 'GPU')  # Hide GPU
    tf.config.threading.set_intra_op_parallelism_threads(2)  # Limit CPU threads
    tf.config.threading.set_inter_op_parallelism_threads(2)
    
    # Download and load model
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        cache_dir="model_cache"
    )
    model = tf.keras.models.load_model(model_path, compile=False)
    model.trainable = False  # Freeze model weights
    
    # Warm up the model with dummy input
    dummy_input = np.zeros((1, 256, 256, 3), dtype=np.float32)
    model.predict(dummy_input, verbose=0)
    
    return model

# 4. Load the model
model = load_model()

# 5. Optimized Image Processing
def preprocess_image(image):
    """CPU-optimized image preprocessing"""
    if image is None:
        return None
        
    # Handle different image formats
    if len(image.shape) == 2:  # Grayscale
        image = np.stack((image,)*3, axis=-1)
    elif image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]
    
    # Fast resize and normalization
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    return np.expand_dims(image.astype('float32') / 255.0, axis=0)

# 6. Prediction Function
def predict_defect(image):
    """Run prediction with CPU optimizations"""
    try:
        processed_image = preprocess_image(image)
        if processed_image is None:
            return "Invalid Image", 0, {"x": CLASS_NAMES, "y": [0]*len(CLASS_NAMES)}
        
        predictions = model.predict(processed_image, verbose=0)
        scores = tf.nn.softmax(predictions[0]).numpy()
        
        return (
            CLASS_NAMES[np.argmax(scores)],  # Predicted class
            float(np.max(scores) * 100),     # Confidence
            {"x": CLASS_NAMES, "y": [float(s) for s in scores]}  # Plot data
        )
    except Exception:
        return "Error", 0, {"x": CLASS_NAMES, "y": [0]*len(CLASS_NAMES)}

# 7. Gradio Interface
with gr.Blocks(title="Steel Defect Detector (CPU Mode)") as demo:
    gr.Markdown("# 🏭 Steel Surface Defect Detection (CPU Version)")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="Upload Image", type="numpy")
            btn = gr.Button("Detect Defect", variant="primary")
        
        with gr.Column():
            label = gr.Label(label="Predicted Defect")
            confidence = gr.Number(label="Confidence (%)", precision=2)
            plot = gr.BarPlot(
                label="Class Probabilities",
                x=CLASS_NAMES,
                y=[0]*len(CLASS_NAMES),
                vertical=False,
                height=300
            )
    
    btn.click(
        fn=predict_defect,
        inputs=img_input,
        outputs=[label, confidence, plot]
    )

# 8. Launch Application
if __name__ == "__main__":
    demo.launch(
       
        server_port=7860,
        share=False
    )