import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from huggingface_hub import hf_hub_download
import logging
import os

# Configure environment and logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define class names
CLASS_NAMES = sorted(['Patches', 'Pitted', 'Scratches', 'Rolled', 'Crazing', 'Inclusion'])

def load_model():
    """Load the model with optimizations for CPU"""
    try:
        logger.info("Starting model download...")
        
        # Explicitly disable GPU
        tf.config.set_visible_devices([], 'GPU')
        
        # Download model (with corrected repo_id spelling)
        model_path = hf_hub_download(
            repo_id="Ahmedhassan54/Defect_Detection_Model",
            filename="best_defect_model.h5",
            cache_dir="model_cache"
        )
        logger.info(f"Model downloaded to: {model_path}")
        
        # Load model with optimizations
        model = tf.keras.models.load_model(model_path, compile=False)
        model.trainable = False  # Freeze model weights
        tf.keras.backend.clear_session()  # Clean up
        
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        return None

# Load model at startup
model = load_model()

def preprocess_image(image):
    """Optimized image preprocessing pipeline"""
    try:
        if image is None:
            raise ValueError("No image provided")
            
        # Efficient color conversion
        if len(image.shape) == 2:  # Grayscale
            image = np.stack((image,)*3, axis=-1)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
            
        # Optimized resize and normalization
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        return np.expand_dims(image.astype('float32') / 255.0, axis=0)
        
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        return None

def predict_defect(image):
    """Run prediction with proper error handling"""
    try:
        if model is None:
            raise ValueError("Model not loaded")
            
        processed_image = preprocess_image(image)
        if processed_image is None:
            raise ValueError("Image processing failed")
        
        # Run prediction with minimal overhead
        predictions = model.predict(processed_image, verbose=0)
        scores = tf.nn.softmax(predictions[0]).numpy()
        
        # Prepare outputs
        predicted_class = CLASS_NAMES[np.argmax(scores)]
        confidence = float(np.max(scores) * 100)
        plot_data = {"x": CLASS_NAMES, "y": [float(s) for s in scores]}
        
        return predicted_class, confidence, plot_data
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return "Error", 0.0, {"x": CLASS_NAMES, "y": [0.0]*len(CLASS_NAMES)}

# Create Gradio interface
with gr.Blocks(title="Steel Defect Detector") as demo:
    gr.Markdown("# 🏭 Steel Surface Defect Detection")
    gr.Markdown("Upload an image to detect surface defects")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Steel Surface Image", type="numpy")
            submit_btn = gr.Button("Detect Defect", variant="primary")
        
        with gr.Column():
            label_output = gr.Label(label="Predicted Defect")
            confidence_output = gr.Number(
                label="Confidence Score (%)",
                minimum=0,
                maximum=100,
                precision=2
            )
            plot_output = gr.BarPlot(
                label="Class Probabilities",
                x=CLASS_NAMES,
                y=[0]*len(CLASS_NAMES),
                vertical=False,
                height=300
            )
    
    # Prediction workflow with loading state
    loading = gr.Textbox(visible=False, label="Status")
    
    submit_btn.click(
        fn=lambda: gr.Textbox("Processing...", visible=True),
        outputs=loading,
        queue=False
    ).then(
        fn=predict_defect,
        inputs=image_input,
        outputs=[label_output, confidence_output, plot_output],
        queue=False
    ).then(
        fn=lambda: gr.Textbox(visible=False),
        outputs=loading,
        queue=False
    )
    
    # Clear functionality
    clear_btn = gr.Button("Clear")
    clear_btn.click(
        fn=lambda: [None, "", 0, {"x": CLASS_NAMES, "y": [0]*len(CLASS_NAMES)}],
        outputs=[image_input, label_output, confidence_output, plot_output],
        queue=False
    )

# Launch the application
if __name__ == "__main__":
    demo.launch(
     
        share=False
    )