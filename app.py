import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from huggingface_hub import hf_hub_download


CLASS_NAMES = sorted(['Patches', 'Pitted', 'Scratches', 'Rolled', 'Crazing', 'Inclusion'])


def load_model():
  
    model_path = hf_hub_download(repo_id="Ahmedhassan54/Defect_Detection_Model", filename="best_defect_model.h5")
    model = tf.keras.models.load_model(model_path)
    return model


model = load_model()

def preprocess_image(image):
    """Preprocess the image for model prediction"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_defect(image):
    """Make prediction and return results"""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    scores = tf.nn.softmax(predictions[0])
    predicted_class = CLASS_NAMES[np.argmax(scores)]
    confidence = np.max(scores) * 100
    
   
    class_probs = {CLASS_NAMES[i]: float(scores[i]) for i in range(len(CLASS_NAMES))}
    
    return predicted_class, confidence, class_probs

with gr.Blocks() as demo:
    gr.Markdown("# 🏭 Steel Surface Defect Detection")
    gr.Markdown("Upload an image of steel surface to detect defects like Patches, Pitted, Scratches, etc.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Steel Surface Image", type="numpy")
            submit_btn = gr.Button("Detect Defect")
        
        with gr.Column():
            label_output = gr.Label(label="Predicted Defect")
            confidence_output = gr.Number(label="Confidence Score (%)")
            plot_output = gr.BarPlot(
                x=CLASS_NAMES,
                y=[0]*len(CLASS_NAMES),
                label="Class Probabilities",
                vertical=False
            )
    
    submit_btn.click(
        fn=predict_defect,
        inputs=image_input,
        outputs=[label_output, confidence_output, plot_output]
    )
    
   
    gr.Examples(
        examples=[
            ["example1.jpg"],
            ["example2.jpg"],
            ["example3.jpg"]
        ],
        inputs=image_input,
        outputs=[label_output, confidence_output, plot_output],
        fn=predict_defect,
        cache_examples=True
    )

if __name__ == "__main__":
    demo.launch()