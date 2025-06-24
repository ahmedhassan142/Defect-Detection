import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from huggingface_hub import hf_hub_download
import logging
import os

# Disable unnecessary TensorFlow logging and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASS_NAMES = sorted(['Patches', 'Pitted', 'Scratches', 'Rolled', 'Crazing', 'Inclusion'])

def load_model():
    try:
        logger.info("Starting model download...")
        
        # Disable GPU as we're seeing CUDA errors
        tf.config.set_visible_devices([], 'GPU')
        
        model_path = hf_hub_download(
            repo_id="Ahmedhassan54/Defect_Detection_Model",
            filename="best_defect_model.h5",
            cache_dir="model_cache",
            resume_download=True  # Enable resuming interrupted downloads
        )
        logger.info(f"Model downloaded to: {model_path}")
        
        # Load model with custom_objects if needed
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Disable training specific operations to optimize for inference
        model.trainable = False
        tf.keras.backend.clear_session()
        
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

# Load model at startup
model = load_model()

def preprocess_image(image):
    """Optimized image preprocessing"""
    try:
        if image is None:
            raise ValueError("No image provided")
            
        # Convert to RGB if needed (more efficient handling)
        if image.ndim == 2:
            image = np.stack((image,)*3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
            
        # Use optimized resize and normalization
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        return np.expand_dims(image.astype('float32') / 255.0, axis=0)
        
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        return None

def predict_defect(image):
    """Optimized prediction function"""
    try:
        if model is None:
            raise ValueError("Model not loaded")
            
        if image is None:
            raise ValueError("No image provided")
            
        processed_image = preprocess_image(image)
        if processed_image is None:
            raise ValueError("Image processing failed")
        
        # Run prediction with optimized settings
        predictions = model.predict(
            processed_image,
            verbose=0,  # Disable prediction logging
            batch_size=1  # Explicit batch size
        )
        
        scores = tf.nn.softmax(predictions[0]).numpy()
        predicted_class = CLASS_NAMES[np.argmax(scores)]
        confidence = float(np.max(scores) * 100)
        
        # Prepare plot data
        plot_data = {
            "x": CLASS_NAMES,
            "y": [float(score) for score in scores],
        }
        
        return predicted_class, confidence, plot_data
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return "Error", 0.0, {"x": CLASS_NAMES, "y": [0.0]*len(CLASS_NAMES)}

def create_interface():
    with gr.Blocks(title="Steel Defect Detection") as demo:
        gr.Markdown("# 🏭 Steel Surface Defect Detection")
        gr.Markdown("Upload an image of steel surface to detect defects")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Upload Image", type="numpy")
                submit_btn = gr.Button("Detect Defect", variant="primary")
            
            with gr.Column():
                label_output = gr.Label(label="Predicted Defect")
                confidence_output = gr.Number(
                    label="Confidence Score (%)",
                    minimum=0,
                    maximum=100
                )
                plot_output = gr.BarPlot(
                    label="Class Probabilities",
                    x=CLASS_NAMES,
                    y=[0]*len(CLASS_NAMES),
                    vertical=False,
                    height=300
                )
        
        # Add loading state
        loading = gr.Textbox(visible=False)
        
        submit_btn.click(
            fn=lambda: gr.Textbox("Processing...", visible=True),
            outputs=loading
        ).then(
            fn=predict_defect,
            inputs=image_input,
            outputs=[label_output, confidence_output, plot_output]
        ).then(
            fn=lambda: gr.Textbox(visible=False),
            outputs=loading
        )
        
        # Clear button
        clear_btn = gr.Button("Clear")
        clear_btn.click(
            fn=lambda: [None, "", 0, {"x": CLASS_NAMES, "y": [0]*len(CLASS_NAMES)}],
            outputs=[image_input, label_output, confidence_output, plot_output]
        )
    
    return demo

if __name__ == "__main__":
    # Configure for production
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        enable_queue=True,  # Enable queue for better handling
        max_threads=2,     # Limit threads to prevent memory issues
        share=False        # Disable public sharing
    )