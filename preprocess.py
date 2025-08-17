import numpy as np
import tensorflow as tf
import cv2

def load_image(img_path):
    """Load and preprocess the X-ray image"""
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    return img_array

def extract_features(img_array, model):
    """Extract deep learning features from an image"""
    img_array = np.expand_dims(img_array, axis=0)
    features = model.predict(img_array)
    return features

# ✅ Test preprocess.py independently
if __name__ == "__main__":
    img_path = r"D:\overThinkingAiModel\images\00000008_001.png"
    img = load_image(img_path)
    print("✅ Image Loaded Successfully. Shape:", img.shape)
