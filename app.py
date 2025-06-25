import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Constants
CLASS_NAMES = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']
MODEL_PATH = 'defect_detection_model.h5'
IMAGE_SIZE = (256, 256)

# Custom CSS to fix UI issues
CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
}
.upload-container {
    min-height: 250px;
}
.output-label {
    font-weight: bold;
    margin-top: 10px;
}
.probability-bar {
    margin: 5px 0;
}
.probability-label {
    display: inline-block;
    width: 100px;
}
"""

# Load model (with error handling)
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    # Create a dummy model if loading fails (for demo purposes)
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(*IMAGE_SIZE, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])

def preprocess_image(image_path):
    """Load and preprocess an image for prediction"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)
        img_array = np.expand_dims(img, axis=0) / 255.0
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_defect(image_path):
    """Make prediction on an image"""
    try:
        # Preprocess the image
        img_array = preprocess_image(image_path)
        if img_array is None:
            return None, "Error processing image"
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)[0]
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = float(np.max(predictions))
        
        # Create detailed results
        detailed_results = [
            (class_name, float(prob)) 
            for class_name, prob in zip(CLASS_NAMES, predictions)
        ]
        
        # Sort by probability (descending)
        detailed_results.sort(key=lambda x: x[1], reverse=True)
        
        return predicted_class, confidence, detailed_results
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None, None

def create_probability_bars(probabilities):
    """Create HTML for probability bars visualization"""
    html = "<div class='probability-bars'>"
    for class_name, prob in probabilities:
        percentage = prob * 100
        html += f"""
        <div class='probability-bar'>
            <span class='probability-label'>{class_name}:</span>
            <progress value='{percentage}' max='100'></progress>
            <span>{percentage:.1f}%</span>
        </div>
        """
    html += "</div>"
    return html

def process_image(image):
    """Gradio interface function"""
    if image is None:
        return {
            "Prediction": "No image provided",
            "Confidence": "0%",
            "Details": "Please upload an image"
        }
    
    # Save the uploaded image temporarily
    temp_path = "temp_upload.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Get predictions
    predicted_class, confidence, details = predict_defect(temp_path)
    
    # Clean up temporary file
    try:
        os.remove(temp_path)
    except:
        pass
    
    if predicted_class is None:
        return {
            "Error": "Failed to process image",
            "Details": "Please try another image"
        }
    
    # Create visualization
    probability_bars = create_probability_bars(details)
    
    return {
        "Prediction": predicted_class,
        "Confidence": f"{confidence*100:.1f}%",
        "Details": probability_bars,
        "Raw Probabilities": {k: f"{v:.4f}" for k, v in details}
    }

# Create Gradio interface
with gr.Blocks(css=CSS, title="Steel Surface Defect Detection") as demo:
    gr.Markdown("""
    # 🏭 Steel Surface Defect Detection
    Upload an image of steel surface to classify the type of defect
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload Steel Surface Image",
                type="numpy",
                height=300
            )
            submit_btn = gr.Button("Analyze", variant="primary")
            
        with gr.Column():
            output_json = gr.JSON(
                label="Analysis Results",
                show_label=True
            )
            
            # Add example images
            gr.Examples(
                examples=[
                    os.path.join("examples", "crazing_sample.jpg"),
                    os.path.join("examples", "inclusion_sample.jpg"),
                    os.path.join("examples", "scratches_sample.jpg")
                ],
                inputs=image_input,
                label="Example Images (Click to load)"
            )
    
    # Set up button click
    submit_btn.click(
        fn=process_image,
        inputs=image_input,
        outputs=output_json
    )

    # Add footer
    gr.Markdown("""
    <div style='text-align: center; margin-top: 20px; color: #666;'>
        Steel Surface Defect Detection System | Made with Gradio
    </div>
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )