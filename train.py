import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans
from sklearn.linear_model import BayesianRidge
import pickle
import os
import matplotlib.pyplot as plt
from preprocess import load_image

# ✅ Load VGG16 as Feature Extractor
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

# ✅ Load training images
img_dir = "images/"
img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")]

# ✅ Extract Features
features = np.array([load_image(img) for img in img_paths])
features = np.array([feature_extractor.predict(np.expand_dims(img, axis=0)).flatten() for img in features])

# ✅ Apply K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(features)

# ✅ Train Bayesian Ridge Regression
bayesian_model = BayesianRidge()
labels = kmeans.labels_
bayesian_model.fit(features, labels)

# ✅ Save Models
os.makedirs("model", exist_ok=True)
feature_extractor.save("model/feature_extractor.h5")
pickle.dump(kmeans, open("model/kmeans.pkl", "wb"))
pickle.dump(bayesian_model, open("model/bayesian_ridge.pkl", "wb"))

# ✅ Visualize Clustering Results
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='coolwarm', marker='o')
plt.title("K-Means Clustering of X-ray Images")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="Cluster")
plt.savefig("model/clustering_result.png")
plt.show()

print("✅ Training Complete! Models saved in 'model/' directory.")
