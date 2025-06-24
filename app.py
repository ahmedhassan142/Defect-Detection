import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from huggingface_hub import hf_hub_download
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASS_NAMES = sorted(['Patches', 'Pitted', 'Scratches', 'Rolled', 'Crazing', 'Inclusion'])

def load_model():
    try:
        logger.info("Starting model download...")
        model_path = hf_hub_download(
            repo_id="Ahmedhassan54/Defect_Detection_Model", 
            filename="best_defect_model.h5",
            cache_dir="model_cache"
        )
        logger.info(f"Model downloaded to: {model_path}")
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

def preprocess_image(image):
    """Preprocess the image for model prediction"""
    try:
        logger.info("Starting image preprocessing...")
        if image is None:
            raise ValueError("No image provided")
            
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        image = cv2.resize(image, (256, 256))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_defect(image):
    """Make prediction and return results"""
    try:
        if model is None:
            raise ValueError("Model not loaded")
        
        if image is None:
            raise ValueError("No image provided")
        
        processed_image = preprocess_image(image)
        if processed_image is None:
            raise ValueError("Image processing failed")
        
        predictions = model.predict(processed_image)
        scores = tf.nn.softmax(predictions[0]).numpy()
        
        predicted_class = CLASS_NAMES[np.argmax(scores)]
        confidence = float(np.max(scores) * 100)
        
        # Ensure class_probs has all classes in correct order
        class_probs = {class_name: float(scores[i]) for i, class_name in enumerate(CLASS_NAMES)}
        
        # Return format must match output components:
        # 1. Label (string)
        # 2. Confidence (number)
        # 3. BarPlot data (dict with 'x' and 'y' keys)
        plot_data = {
            "x": CLASS_NAMES,
            "y": [float(score) for score in scores],
        }
        
        return predicted_class, confidence, plot_data
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        # Return properly formatted fallback values
        plot_data = {
            "x": CLASS_NAMES,
            "y": [0.0] * len(CLASS_NAMES),
        }
        return "Error in prediction", 0.0, plot_data

with gr.Blocks() as demo:
    gr.Markdown("# 🏭 Steel Surface Defect Detection")
    gr.Markdown("Upload an image of steel surface to detect defects like Patches, Pitted, Scratches, etc.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Steel Surface Image", type="numpy")
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
    
    submit_btn.click(
        fn=predict_defect,
        inputs=image_input,
        outputs=[label_output, confidence_output, plot_output]
    )

if __name__ == "__main__":
    demo.launch(debug=True)