
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import os

# ✅ Paths to models and image
directory = "D:/overThinkingAiModel/"
feature_extractor_path = directory + "model/feature_extractor.h5"
kmeans_path = directory + "model/kmeans.pkl"
bayesian_ridge_path = directory + "model/bayesian_ridge.pkl"
# image_path = directory + "images/00001335_005.png"
image_path = "D:/overThinkingAiModel/00001336_000.png"

# ✅ Load the models
def load_models():
    print("Loading models...")
    feature_extractor = tf.keras.models.load_model(feature_extractor_path)
    with open(kmeans_path, 'rb') as file:
        kmeans = pickle.load(file)
    with open(bayesian_ridge_path, 'rb') as file:
        bayesian_ridge = pickle.load(file)
    print("Models loaded successfully!")
    return feature_extractor, kmeans, bayesian_ridge

# ✅ Preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Could not open image at {image_path}")
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# ✅ Extract features (Fixed dtype issue)
def extract_features(model, image):
    features = model.predict(image).flatten()
    features = np.asarray(features, dtype=np.float32)  # 🔹 Convert to float32
    return features.reshape(1, -1)  # Ensure correct shape

# ✅ Predict cluster & Bayesian Ridge score
def predict_analysis(kmeans, bayesian_ridge, features):
    cluster = kmeans.predict(features.astype(np.float32))[0]  # 🔹 Ensure dtype consistency
    score = bayesian_ridge.predict(features)[0]
    return cluster, score

if __name__ == "__main__":
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    feature_extractor, kmeans, bayesian_ridge = load_models()
    image = preprocess_image(image_path)
    features = extract_features(feature_extractor, image)
    cluster, score = predict_analysis(kmeans, bayesian_ridge, features)
    
    # ✅ Overthinking Analysis
    overthinking_threshold = 0.5  # Adjust threshold based on data
    classification = "Abnormal" if score > overthinking_threshold else "Normal"
    
    print(" Overthinking Analysis Complete!")
    print(f" Predicted Cluster: {cluster}")
    print(f" Bayesian Ridge Score: {score:.4f}")
    print(f"Classification: {classification}")
    
    # ✅ Visualize Results
    plt.figure(figsize=(6, 3))
    
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(image[0])
    plt.axis("off")
    plt.title("Input X-ray Image")
    
    # Bayesian Score & Clustering
    plt.subplot(1, 2, 2)
    plt.bar(["Cluster"], [cluster], color="blue")
    plt.bar(["Score"], [score], color="red")
    plt.ylim(0, max(2, score + 1))
    plt.title("Bayesian Score & Clustering")
    
    plt.show()
    
    print("✅ Prediction complete!")
