import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from transformers import ViTFeatureExtractor, ViTModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# Define the model creation functions for ViT and EfficientNetV2

def create_vit_model():
    vit_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    def vit_feature_extractor(image):
        inputs = vit_extractor(images=image, return_tensors="tf")
        outputs = vit_model(**inputs)
        features = outputs.last_hidden_state[:, 0, :]  # CLS token features
        return features.numpy().flatten()

    return vit_feature_extractor

def create_efficientnetv2_model():
    base_model = EfficientNetV2B0(weights='imagenet', include_top=False)
    
   
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    
    return model

def extract_classical_features(image):
  
    features = []
    # Shape-based features
    features.append(cv2.contourArea(image))  # Area
    features.append(cv2.arcLength(image, True))  # Perimeter

    # Color-based features (mean and standard deviation for each channel in LAB color space)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    for i in range(3):
        features.append(np.mean(lab_image[:, :, i]))
        features.append(np.std(lab_image[:, :, i]))

    return np.array(features)

def extract_deep_features(image, vit_model, efficientnet_model):

    resized_image = cv2.resize(image, (224, 224))
    resized_image = np.expand_dims(resized_image, axis=0)  
    
   
    efficientnet_features = efficientnet_model.predict(resized_image).flatten()

    vit_features = vit_model(image)
    
    # Combine the features
    deep_features = np.concatenate([efficientnet_features, vit_features])
    return deep_features

def build_feature_matrix(image_folder, vit_model, efficientnet_model):
    feature_matrix = []
    for file_name in os.listdir(image_folder):
        image = cv2.imread(os.path.join(image_folder, file_name))
        
        classical_features = extract_classical_features(image)
        deep_features = extract_deep_features(image, vit_model, efficientnet_model)
        
        combined_features = np.concatenate([classical_features, deep_features])
        feature_matrix.append(combined_features)
    
    return np.array(feature_matrix)

if __name__ == "__main__":
    # Initialize the deep models
    vit_model = create_vit_model()
    efficientnet_model = create_efficientnetv2_model()
    
    # Build the feature matrix from the images
    features = build_feature_matrix("processed_images", vit_model, efficientnet_model)
    
    # Save the feature matrix
    np.save("feature_matrix.npy", features)
