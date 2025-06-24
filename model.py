
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import files
import zipfile
import io
import shutil
from tqdm import tqdm
import time



class ProgressFileUpload:
    """Custom file upload with progress bar"""
    def __init__(self, filename, total_size):
        self.filename = filename
        self.total_size = total_size
        self.uploaded = 0
        self.start_time = time.time()
        
    def update(self, chunk_size):
        self.uploaded += chunk_size
        percent = (self.uploaded / self.total_size) * 100
        elapsed = time.time() - self.start_time
        speed = self.uploaded / (1024 * 1024 * elapsed) if elapsed > 0 else 0
        print(f"\rUploading {self.filename}: {percent:.1f}% ({self.uploaded/(1024*1024):.1f}MB/{self.total_size/(1024*1024):.1f}MB) {speed:.1f}MB/s", end='')
        if percent >= 100:
            print()

class DefectDetectorTF:
    def __init__(self, input_shape=(256, 256, 3)):
        self.input_shape = input_shape
        self.class_names = sorted(['Patches', 'Pitted', 'Scratches', 'Rolled', 'Crazing', 'Inclusion'])
        self.num_classes = len(self.class_names)
        self.model = self.build_model()
    
    def build_model(self):
        """Build an enhanced CNN model for 6 defect types"""
        model = models.Sequential([
            layers.Rescaling(1./255, input_shape=self.input_shape),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def load_dataset(self, dataset_dir):
        """Load all image formats from each class folder"""
        images = []
        labels = []
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(dataset_dir, class_name)
            if not os.path.exists(class_dir):
                raise ValueError(f"Missing folder for class: {class_name}")
                
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(supported_formats):
                    img_path = os.path.join(class_dir, img_file)
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, self.input_shape[:2])
                        images.append(img)
                        labels.append(class_idx)
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")
                        continue
        
        if len(images) == 0:
            raise ValueError("No valid images found in dataset directory")
        
        return np.array(images), np.array(labels)
    
    def save_model_with_progress(self, filepath):
        """Save model with progress tracking"""
       
        class ProgressTracker(tf.keras.callbacks.Callback):
            def __init__(self, total_size):
                self.total_size = total_size
                self.progress = 0
                self.start_time = time.time()
                
            def on_batch_end(self, batch, logs=None):
                
                self.progress += 1
                percent = min(100, (self.progress / 100) * 100)
                elapsed = time.time() - self.start_time
                speed = (self.progress * self.total_size / 100) / (1024 * 1024 * elapsed) if elapsed > 0 else 0
                print(f"\rSaving model: {percent:.1f}% ({percent/100*self.total_size/(1024*1024):.1f}MB/{self.total_size/(1024*1024):.1f}MB) {speed:.1f}MB/s", end='')
        
       
        model_size = sum(layer.count_params() for layer in self.model.layers) * 4 
        model_size = max(model_size, 50 * 1024 * 1024)  
        
        progress_tracker = ProgressTracker(model_size)
        
      
        print(f"\nStarting model save (estimated size: {model_size/(1024*1024):.1f}MB)")
        self.model.save(filepath)
        print("\nModel saved successfully!")
    
    def train(self, dataset_dir, epochs=30, batch_size=32):
        """Train with data augmentation"""
        X, y = self.load_dataset(dataset_dir)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                'best_defect_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            )
        ]
        
        print("\nTraining Summary:")
        print(f"Classes: {self.class_names}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}\n")
        
        history = self.model.fit(
            train_generator,
            steps_per_epoch=len(X_train) // batch_size,
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        
        plt.show()
        return history
    
    def predict(self, image_path):
        """Predict on any image format"""
        img = cv2.imread(image_path)
        if img is None:
            return None, "Failed to load image"
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        display_img = img.copy()
        img = cv2.resize(img, self.input_shape[:2])
        img_array = tf.expand_dims(tf.keras.utils.img_to_array(img), 0) / 255.0
        
        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        class_id = np.argmax(score)
        confidence = np.max(score)
        
        cv2.putText(
            display_img,
            f"{self.class_names[class_id]}: {confidence:.2f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        
        debug_text = "\n".join([
            f"{name}: {prob:.4f}" 
            for name, prob in zip(self.class_names, score)
        ])
        
        return display_img, debug_text



print("#" * 60)
print("### Steel Surface Defect Detection - Training ###")
print("#" * 60)




print("\nPlease upload your dataset as a zip file:")
uploaded = files.upload()
dataset_path = None

for filename in uploaded.keys():
    if filename.endswith('.zip'):
        print(f"\nExtracting {filename}...")
        total_size = os.path.getsize(filename)
        with zipfile.ZipFile(io.BytesIO(uploaded[filename]), 'r') as zip_ref:
           
            file_list = zip_ref.infolist()
            for file in tqdm(file_list, desc="Extracting", unit="file"):
                zip_ref.extract(file, '/content/project_files/dataset')
        dataset_path = '/content/project_files/dataset'
        print("\nDataset extracted successfully!")
        break

if not dataset_path:
    print("Warning: No dataset zip file uploaded. Training will fail without a dataset.")


print("\nOptionally upload an existing model (h5 file) to continue training:")
model_uploaded = files.upload()
model_path = None

for filename in model_uploaded.keys():
    if filename.endswith('.h5'):
      
        total_size = os.path.getsize(filename)
        progress = ProgressFileUpload(filename, total_size)
        with open(filename, 'rb') as f:
            while True:
                chunk = f.read(1024*1024)  
                if not chunk:
                    break
                progress.update(len(chunk))
        
        model_path = f'/content/project_files/{filename}'
        shutil.move(filename, model_path)
        print(f"\nModel uploaded to: {model_path}")
        break


detector = DefectDetectorTF()


if model_path:
    print(f"\nLoading existing model from {model_path}")
    detector.model = tf.keras.models.load_model(model_path)


if dataset_path:
    print("\nStarting training...")
    detector.train(dataset_path, epochs=10)
    
    #
    model_save_path = '/content/project_files/defect_detection_model.h5'
    detector.save_model_with_progress(model_save_path)
    
   
    print("\nUpload an image for testing:")
    test_uploaded = files.upload()
    if test_uploaded:
        test_image = list(test_uploaded.keys())[0]
       
        total_size = os.path.getsize(test_image)
        progress = ProgressFileUpload(test_image, total_size)
        with open(test_image, 'rb') as f:
            while True:
                chunk = f.read(1024*1024)  
                if not chunk:
                    break
                progress.update(len(chunk))
        
        result, debug = detector.predict(test_image)
        
        plt.imshow(result)
        plt.axis('off')
        plt.show()
        print(debug)

print("\nTraining process completed!")
print("You can find your trained model in the 'project_files' folder.")
print("To download the model manually:")
print("1. Click on the folder icon on the left")
print("2. Navigate to 'project_files' folder")
print("3. Right-click on 'defect_detection_model.h5' and select 'Download'")



print("\n\n" + "#" * 60)
print("### Steel Surface Defect Detection - Testing ###")
print("#" * 60)

def test_model_with_zip(test_zip_path, model_path, detector):
    """Test the model with a zip file containing test images"""
   
    test_dir = '/content/project_files/test_dataset'
    os.makedirs(test_dir, exist_ok=True)
    
    
    print(f"\nExtracting test dataset from {test_zip_path}...")
    with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
        zip_ref.extractall(test_dir)
    print("Test dataset extracted successfully!")
    
    
    print(f"\nLoading model from {model_path}")
    detector.model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    
   
    correct = 0
    total = 0
    class_correct = {class_name: 0 for class_name in detector.class_names}
    class_total = {class_name: 0 for class_name in detector.class_names}
    
   
    print("\nStarting evaluation on test set...")
    for class_name in detector.class_names:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: No test images found for class {class_name}")
            continue
            
       
        for img_file in tqdm(os.listdir(class_dir), desc=f"Testing {class_name}", unit="image"):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(class_dir, img_file)
                try:
                  
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Could not read image: {img_path}")
                        continue
                        
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, detector.input_shape[:2])
                    img_array = tf.expand_dims(tf.keras.utils.img_to_array(img), 0) / 255.0
                    
                    predictions = detector.model.predict(img_array, verbose=0)
                    predicted_class = np.argmax(predictions[0])
                    
                   
                    total += 1
                    class_total[class_name] += 1
                    
                    if detector.class_names[predicted_class] == class_name:
                        correct += 1
                        class_correct[class_name] += 1
                        
                except Exception as e:
                    print(f"\nError processing {img_path}: {str(e)}")
                    continue
    
    
    if total > 0:
        print("\n\n=== Test Results ===")
        print(f"Overall Accuracy: {correct/total:.2%} ({correct}/{total})")
        
        print("\nClass-wise Accuracy:")
        for class_name in detector.class_names:
            if class_total[class_name] > 0:
                acc = class_correct[class_name]/class_total[class_name]
                print(f"{class_name}: {acc:.2%} ({class_correct[class_name]}/{class_total[class_name]})")
            else:
                print(f"{class_name}: No test images found")
    else:
        print("\nNo valid test images found in the provided zip file")


print("\n\n=== Model Testing ===")
print("Please upload your test dataset as a zip file (with same folder structure as training):")
uploaded = files.upload()
test_zip_path = None

for filename in uploaded.keys():
    if filename.endswith('.zip'):
        test_zip_path = f'/content/{filename}'
        print(f"\nTest dataset {filename} uploaded successfully!")
        break

if test_zip_path:
    
    model_path = '/content/project_files/defect_detection_model.h5'
    
   
    if not os.path.exists(model_path):
        print(f"\nError: Model not found at {model_path}")
        print("Please ensure the model was saved correctly during training.")
    else:
        
        test_model_with_zip(test_zip_path, model_path, detector)
else:
    print("\nNo test zip file provided. Skipping evaluation.")

print("\nTesting process completed!")