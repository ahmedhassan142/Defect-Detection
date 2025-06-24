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
        
        logger.info("Loading model...")
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
        
        # Convert to RGB if not already
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3:  # Already RGB
            pass
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        logger.info(f"Image shape after color conversion: {image.shape}")
        
        image = cv2.resize(image, (256, 256))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        
        logger.info(f"Final preprocessed image shape: {image.shape}")
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_defect(image):
    """Make prediction and return results"""
    try:
        logger.info("Starting prediction...")
        
        if model is None:
            error_msg = "Model not loaded - please check logs"
            logger.error(error_msg)
            return error_msg, 0.0, {c: 0.0 for c in CLASS_NAMES}
        
        processed_image = preprocess_image(image)
        if processed_image is None:
            error_msg = "Image processing failed - please check logs"
            logger.error(error_msg)
            return error_msg, 0.0, {c: 0.0 for c in CLASS_NAMES}
        
        logger.info("Running model prediction...")
        predictions = model.predict(processed_image)
        logger.info("Prediction completed")
        
        scores = tf.nn.softmax(predictions[0]).numpy()
        predicted_class = CLASS_NAMES[np.argmax(scores)]
        confidence = float(np.max(scores) * 100)
        
        class_probs = {CLASS_NAMES[i]: float(scores[i]) for i in range(len(CLASS_NAMES))}
        
        logger.info(f"Prediction results - Class: {predicted_class}, Confidence: {confidence}%")
        return predicted_class, confidence, class_probs
        
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        logger.error(error_msg)
        return error_msg, 0.0, {c: 0.0 for c in CLASS_NAMES}

def create_interface():
    with gr.Blocks(title="Steel Defect Detection") as demo:
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
                    precision=2,
                    minimum=0,
                    maximum=100
                )
                plot_output = gr.BarPlot(
                    x=CLASS_NAMES,
                    y=[0]*len(CLASS_NAMES),
                    label="Class Probabilities",
                    vertical=False,
                    height=300
                )
        
        submit_btn.click(
            fn=predict_defect,
            inputs=image_input,
            outputs=[label_output, confidence_output, plot_output],
            api_name="predict"
        )
        
        # Add a clear button
        clear_btn = gr.Button("Clear")
        clear_btn.click(
            fn=lambda: [None, "", 0, {c: 0 for c in CLASS_NAMES}],
            outputs=[image_input, label_output, confidence_output, plot_output]
        )
        
        # Progress indicator
        progress = gr.Textbox(visible=False)
        
        # Error display
        error_display = gr.Textbox(label="Error Messages", visible=False)
    
    return demo

if __name__ == "__main__":
    logger.info("Launching application...")
    demo = create_interface()
    demo.launch(debug=True)