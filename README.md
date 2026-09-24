# NeuroScan-MRI: Brain MRI Tumor Analysis 🧠

NeuroScan-MRI is an end-to-end deep learning research prototype for brain MRI tumor segmentation, multi-class tumor classification, and model explainability.

The system combines an Attention U-Net segmentation model with an EfficientNetV2B0 classification model and uses Grad-CAM to provide visual explanations of classification predictions.

The application provides an interactive Streamlit dashboard and a FastAPI inference layer for processing MRI scans.

> **Note:** NeuroScan-MRI is a research/academic prototype and is not a clinically validated diagnostic system.

---

## 🧠 Project Overview

The system processes a brain MRI image through two complementary deep learning branches:

1. **Segmentation Branch**
   - Uses Attention U-Net.
   - Produces a binary tumor-region mask.
   - Helps visualize the detected tumor region.

2. **Classification Branch**
   - Uses EfficientNetV2B0 with transfer learning.
   - Classifies the MRI into four categories:
     - Glioma
     - Meningioma
     - Pituitary
     - No Tumor

3. **Explainability**
   - Uses Grad-CAM to visualize image regions that contributed to the classifier's prediction.

4. **Web Application**
   - Streamlit dashboard for interactive MRI analysis.
   - FastAPI inference endpoint for programmatic predictions.

---

## 🏗️ Architecture

```text
                         ┌──────────────────────┐
                         │      MRI IMAGE       │
                         └──────────┬───────────┘
                                    │
                           Image Validation
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
          ┌───────────────────┐           ┌────────────────────┐
          │   SEGMENTATION    │           │   CLASSIFICATION   │
          │      BRANCH       │           │       BRANCH       │
          └─────────┬─────────┘           └──────────┬─────────┘
                    │                                │
                    ▼                                ▼
          Grayscale Preprocessing            RGB Preprocessing
                    │                                │
                    ▼                                ▼
          ┌───────────────────┐           ┌────────────────────┐
          │   Attention U-Net │           │  EfficientNetV2B0  │
          └─────────┬─────────┘           └──────────┬─────────┘
                    │                                │
                    ▼                                ▼
             Binary Tumor Mask                4-Class Prediction
                    │                                │
                    │                       ┌────────┴─────────┐
                    │                       │                  │
                    │                       ▼                  ▼
                    │                 Predicted Class       Grad-CAM
                    │                       │             Explanation
                    │                       │                  │
                    └───────────────┬───────┴──────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Streamlit Dashboard  │
                         │      + FastAPI       │
                         └──────────────────────┘
                         Important Architecture Detail

The segmentation and classification models operate as separate branches.

The segmentation mask is used for tumor-region visualization and is not directly passed as input to the EfficientNetV2B0 classifier.

The classifier receives the original MRI image through its RGB preprocessing pipeline.

🖥️ Dashboard Preview
MRI Scan and Segmentation

Raw MRI scan with predicted segmentation mask overlay.

Region Visualization and Grad-CAM

Tumor-region visualization and Grad-CAM model-focus heatmap.

Classification Probability and Details

Classification confidence distribution and analysis information.

🚀 Live Demo

NeuroScan Analytics Dashboard:

https://neuroscan-analytics.streamlit.app/

The live dashboard provides an interactive interface for uploading MRI scans and viewing:

Tumor segmentation
Tumor classification
Confidence probabilities
Grad-CAM visualization
Analysis information
🌟 Core Features
🔬 1. Brain Tumor Segmentation

The segmentation branch uses an Attention U-Net architecture based on convolutional layers and attention mechanisms.

The model predicts a binary tumor-region mask from the MRI image.

Training loss:

BCE + Dice Loss

The Dice component helps address the imbalance between tumor and background pixels.

🧠 2. Multi-Class Tumor Classification

The classification branch uses EfficientNetV2B0 with transfer learning.

The model predicts one of four classes:

Glioma
Meningioma
Pituitary
No Tumor

The classifier uses the original MRI image after RGB preprocessing.

🔎 3. Grad-CAM Explainability

Grad-CAM is used to generate a heatmap showing regions of the MRI image that contributed to the classification prediction.

The heatmap is intended as an interpretability aid rather than a clinically validated tumor boundary.

🖥️ 4. Interactive Dashboard

The Streamlit application provides:

MRI image upload
Segmentation visualization
Classification prediction
Confidence/probability display
Grad-CAM visualization
Analysis history/interface components
⚡ 5. FastAPI Inference Layer

A FastAPI application provides an inference endpoint for integrating the trained models with other applications.

/api/v1/predict
📊 Model Performance

The following results are reported for the held-out evaluation sets used for the current project results.

Classification — EfficientNetV2B0

Overall Test Accuracy: 94.06%

Test set: 1,600 held-out MRI images

Number of classes: 4

Class	Precision	Recall	F1-Score
Glioma	0.98	0.80	0.88
Meningioma	0.90	0.97	0.93
Pituitary	0.98	1.00	0.99
No Tumor	0.91	1.00	0.95

The class-level metrics show that performance varies across tumor categories, with Glioma having lower recall than the other reported classes.

Segmentation — Attention U-Net
Metric	Result
Pixel Accuracy	99.29%
Dice Coefficient	76.41%
Mean IoU	49.14%

Pixel accuracy should be interpreted carefully because MRI segmentation contains a large number of background pixels.

For tumor-region overlap, the Dice coefficient provides a more informative measure of segmentation quality.

🛠️ System Components
Subsystem	Technology	Purpose
Segmentation	Attention U-Net	Predicts binary tumor-region mask
Classification	EfficientNetV2B0	Four-class MRI classification
Explainability	Grad-CAM	Visualizes classifier focus
Dashboard	Streamlit	Interactive MRI analysis interface
API	FastAPI	Prediction/inference endpoint
Visualization	Plotly	Charts and confidence visualization
Computer Vision	OpenCV	Image preprocessing
Deep Learning	TensorFlow / Keras	Model development and inference
Testing	pytest	Automated testing
Containerization	Docker	Application containerization
🧰 Technology Stack
🤖 Deep Learning & Computer Vision
Category	Technology
Deep Learning Framework	TensorFlow
Neural Network API	Keras
Segmentation Model	Attention U-Net
Classification Model	EfficientNetV2B0
Explainability	Grad-CAM
Computer Vision	OpenCV
Image Processing	Pillow
Numerical Computing	NumPy
Data Processing	pandas
Machine Learning Utilities	scikit-learn
🖥️ Application Layer
Category	Technology
Web Dashboard	Streamlit
Backend API	FastAPI
ASGI Server	Uvicorn
Visualization	Plotly
Configuration	YAML / PyYAML
Environment Variables	python-dotenv
⚙️ Infrastructure & Testing
Category	Technology
Containerization	Docker
Service Orchestration	Docker Compose
Testing	pytest
Model Format	Keras .keras
🚀 Getting Started
1. Clone the Repository
git clone https://github.com/omkarDhamgunde766/NeuroScan_MRI.git
cd NeuroScan_MRI
2. Create a Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
Linux / macOS
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Configure the Project

The project uses:

config.yaml

Dataset paths are configured through the project configuration.

Expected dataset structure:

data/
└── raw/
    ├── classification/
    │   ├── training/
    │   │   ├── Glioma/
    │   │   ├── Meningioma/
    │   │   ├── NoTumor/
    │   │   └── Pituitary/
    │   │
    │   └── testing/
    │       ├── Glioma/
    │       ├── Meningioma/
    │       ├── NoTumor/
    │       └── Pituitary/
    │
    └── segmentation/
        ├── Glioma/
        ├── Meningioma/
        └── Pituitary/
🧪 Model Training
Train the Segmentation Model
python -m scripts.train_segmenter --config config.yaml

The trained segmentation model is saved under:

checkpoints/neuroscan_seg.keras
Train the Classification Model
python -m scripts.train_classifier --config config.yaml

The trained classification model is saved under:

checkpoints/neuroscan_cls.keras
🖥️ Running the Dashboard

Activate the virtual environment first.

python -m streamlit run app/dashboard.py

The application will open in the browser.

The dashboard allows the user to upload an MRI image and view:

MRI Image
   ↓
Preprocessing
   ↓
Segmentation + Classification
   ↓
Tumor Mask + Predicted Class
   ↓
Grad-CAM Explanation
   ↓
Dashboard Visualization
⚡ Running the API

The FastAPI application is located at:

app/api.py

The API exposes the prediction endpoint:

/api/v1/predict

The exact server command depends on the application configuration.

🐳 Docker

The project also contains:

Dockerfile
docker-compose.yml

To build and start the containerized services:

docker-compose up --build

Typical local service ports are:

Dashboard: http://localhost:8501
API:       http://localhost:8000
🧪 Testing

Activate the virtual environment and run:

pytest tests/ -v

The repository contains tests for major components including:

Segmentation model
Classification model
Data pipeline
Inference pipeline
Explainability
Visualization
FastAPI
📁 Repository Structure
NeuroScan_MRI/
│
├── app/
│   ├── dashboard.py
│   └── api.py
│
├── checkpoints/
│   ├── neuroscan_cls.keras
│   └── neuroscan_seg.keras
│
├── data/
│   └── raw/
│       ├── classification/
│       └── segmentation/
│
├── docs/
│   ├── dashboard1.png
│   ├── dashboard2.png
│   ├── dashboard3.png
│   └── dashboard4.png
│
├── neuroscan/
│   ├── __init__.py
│   ├── config_loader.py
│   ├── data_pipeline.py
│   ├── seg_model.py
│   ├── cls_model.py
│   ├── trainer.py
│   ├── inference.py
│   ├── explainability.py
│   └── visualizer.py
│
├── notebooks/
│   └── neuroscan_kaggle_training.ipynb
│
├── scripts/
│   ├── convert_tif_to_png.py
│   ├── evaluate_models.py
│   ├── generate_synthetic_data.py
│   ├── kaggle_train_script.py
│   ├── train_classifier.py
│   └── train_segmenter.py
│
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_cls_model.py
│   ├── test_data_pipeline.py
│   ├── test_explainability.py
│   ├── test_inference.py
│   ├── test_seg_model.py
│   └── test_visualizer.py
│
├── config.yaml
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── run_neuroscan.bat
├── runtime.txt
└── README.md
🔬 Research Workflow
             MRI Image
                 │
                 ▼
        Image Validation
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
  Segmentation      Classification
   Attention U-Net   EfficientNetV2B0
        │                 │
        ▼                 ▼
   Tumor Mask       Tumor Class
        │                 │
        │             Grad-CAM
        │                 │
        └────────┬────────┘
                 ▼
        Streamlit Dashboard
🔮 Future Improvements

Possible future extensions include:

3D MRI volume processing using NIfTI data
Multi-modal MRI analysis using T1, T1Gd, T2 and FLAIR sequences
Improved tumor-region segmentation
Additional model validation on independent datasets
More robust external validation
Model calibration and uncertainty estimation
Scalable deployment infrastructure
Privacy-preserving/federated learning research
⚠️ Research Disclaimer

NeuroScan-MRI is developed as an academic/research project for brain MRI image analysis.

The model predictions, segmentation masks, confidence values, and Grad-CAM visualizations should not be interpreted as a medical diagnosis.

Clinical deployment would require appropriate medical validation, independent testing, regulatory approval, data governance, and evaluation by qualified medical professionals.

👤 Author

Omkar Dhamgunde

GitHub:

https://github.com/omkarDhamgunde766

📄 Project Repository

NeuroScan-MRI

GitHub Repository:

https://github.com/omkarDhamgunde766/NeuroScan_MRI

⭐ Acknowledgement

This project was developed as an academic/final-year project focused on applying deep learning techniques to brain MRI tumor segmentation, classification, and explainability.


### One correction I intentionally made

Your old README had this flow:

```text
MRI
 ↓
Attention U-Net
 ↓
ROI Extraction
 ↓
EfficientNetV2B0

That doesn't match your current implementation. Your current project has:

                    MRI
                     ↓
             ┌───────┴───────┐
             ↓               ↓
        Attention U-Net   EfficientNetV2B0
             ↓               ↓
        Tumor Mask       Classification
                             ↓
                          Grad-CAM