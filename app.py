import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from huggingface_hub import hf_hub_download
import logging
import os

# 1. Environment Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Explicitly disable GPU

# 2. Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 3. Constants
CLASS_NAMES = sorted(['Patches', 'Pitted', 'Scratches', 'Rolled', 'Crazing', 'Inclusion'])
MODEL_REPO = "Ahmedhassan54/Defect_Detection_Model"
MODEL_FILE = "best_defect_model.h5"

# 4. Model Loading with Error Handling
def load_model():
    """Load the model with optimizations for CPU"""
    try:
        logger.info("🚀 Starting model download...")
        
        # Clear any existing TensorFlow sessions
        tf.keras.backend.clear_session()
        
        # Download model
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            cache_dir="model_cache",
            force_download=False,
            resume_download=True
        )
        logger.info(f"✅ Model downloaded to: {model_path}")
        
        # Load with optimizations
        model = tf.keras.models.load_model(model_path, compile=False)
        model.trainable = False  # Freeze model weights
        
        # Warm up the model
        dummy_input = np.zeros((1, 256, 256, 3), dtype=np.float32)
        model.predict(dummy_input, verbose=0)
        
        logger.info("🎉 Model loaded and warmed up successfully")
        return model
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {str(e)}")
        return None

# 5. Image Processing
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
        logger.error(f"🖼️ Image processing error: {str(e)}")
        return None

# 6. Prediction Function
def predict_defect(image):
    """Run prediction with proper error handling"""
    try:
        if model is None:
            raise ValueError("Model not loaded - please check logs")
            
        processed_image = preprocess_image(image)
        if processed_image is None:
            raise ValueError("Image processing failed")
        
        # Run prediction
        predictions = model.predict(processed_image, verbose=0)
        scores = tf.nn.softmax(predictions[0]).numpy()
        
        # Prepare outputs
        predicted_class = CLASS_NAMES[np.argmax(scores)]
        confidence = float(np.max(scores) * 100)
        plot_data = {
            "x": CLASS_NAMES,
            "y": [float(score) for score in scores],
            "label": "Probability",
            "color": "steelblue"
        }
        
        return predicted_class, confidence, plot_data
        
    except Exception as e:
        logger.error(f"🔴 Prediction failed: {str(e)}")
        return "Error", 0.0, {
            "x": CLASS_NAMES,
            "y": [0.0]*len(CLASS_NAMES),
            "label": "Probability",
            "color": "lightgray"
        }

# 7. Load Model at Startup
model = load_model()

# 8. Gradio Interface
with gr.Blocks(title="Steel Defect Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏭 Steel Surface Defect Detection")
    gr.Markdown("Upload an image to detect surface defects (Patches, Pitted, Scratches, etc.)")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Steel Surface Image",
                type="numpy",
                height=300
            )
            with gr.Row():
                submit_btn = gr.Button("Detect Defect", variant="primary")
                clear_btn = gr.Button("Clear")
        
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
                height=300,
                width=500
            )
    
    # Status indicator
    status = gr.Textbox(label="Status", visible=False)
    
    # Prediction workflow
    submit_btn.click(
        fn=lambda: gr.Textbox("🔄 Processing...", visible=True),
        outputs=status,
        queue=False
    ).then(
        fn=predict_defect,
        inputs=image_input,
        outputs=[label_output, confidence_output, plot_output],
        queue=False
    ).then(
        fn=lambda: gr.Textbox(visible=False),
        outputs=status,
        queue=False
    )
    
    # Clear functionality
    clear_btn.click(
        fn=lambda: [None, "", 0, {"x": CLASS_NAMES, "y": [0]*len(CLASS_NAMES)}],
        outputs=[image_input, label_output, confidence_output, plot_output],
        queue=False
    )

# 9. Launch Application
if __name__ == "__main__":
    demo.launch(
      
        share=False,
        show_error=True,
        debug=False
    )