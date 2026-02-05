# 🩻 X-Ray Abnormality Detection

![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Dataset](https://img.shields.io/badge/dataset-NIH%20ChestXray-orange.svg)
![Deep Learning](https://img.shields.io/badge/deep%20learning-tensorflow%20%7C%20keras-red.svg)

An end-to-end deep learning pipeline for detecting abnormalities in chest X-ray images using state-of-the-art computer vision techniques. This project leverages the NIH Chest X-ray dataset to build robust models capable of identifying various thoracic pathologies with high accuracy.

## 🎯 Project Overview

This repository implements a comprehensive machine learning solution for automated chest X-ray analysis, featuring:

- **Multi-label classification** of 14 common thoracic pathologies
- **Data preprocessing** pipeline with augmentation techniques
- **Transfer learning** using pre-trained CNN architectures
- **Model evaluation** with detailed metrics and visualizations
- **Inference pipeline** for real-time abnormality detection

## 📂 Project Structure

```
X-Ray-Abnormality-Detection/
│
├── 📁 data/
│   ├── images/                    # Raw chest X-ray images (not tracked)
│   ├── Data_Entry_2017.csv        # Dataset metadata
│   └── processed/                 # (generated) Preprocessed data
│
├── 📁 src/
│   ├── preprocess.py              # Data preprocessing pipeline
│   ├── train.py                   # Model training script
│   ├── predict.py                 # Inference script
│   └── utils/                     # Utility functions
│       ├── data_loader.py
│       ├── model_utils.py
│       └── visualization.py
│
├── 📁 models/                     # (generated) Saved model weights
│   ├── best_model.h5
│   └── model_config.json
│
├── 📁 outputs/                    # (generated) Training results
│   ├── training_logs/
│   ├── plots/
│   └── evaluation_metrics/
│
├── 📁 notebooks/                  # Jupyter notebooks for analysis
│   ├── data_exploration.ipynb
│   └── model_evaluation.ipynb
│
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment file
├── README.md                      # Project documentation
└── LICENSE                       # MIT License
```

## 📊 Dataset Information

This project utilizes the **NIH Chest X-ray Dataset**, one of the largest publicly available chest radiograph datasets.

### Dataset Details
- **Total Images**: 112,120 frontal-view X-ray images
- **Unique Patients**: 30,805 patients
- **Image Resolution**: 1024×1024 pixels
- **Format**: PNG
- **Pathologies**: 14 common thoracic conditions

### Pathology Classes
The dataset includes the following 14 pathology labels:
- Atelectasis
- Consolidation  
- Infiltration
- Pneumothorax
- Edema
- Emphysema
- Fibrosis
- Effusion
- Pneumonia
- Pleural Thickening
- Cardiomegaly
- Nodule
- Mass
- Hernia
