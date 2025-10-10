# Acetowhite Vision AI: High-Sensitivity Cervical Cancer Screening

A deep learning-powered, smartphone-based tool to empower frontline health workers and revolutionize cervical cancer screening in low-resource settings like Nigeria.

This project provides an end-to-end, production-ready MLOps pipeline for training, evaluating, and deploying a high-sensitivity AI model for the early detection of cervical pre-cancerous lesions from VIA (Visual Inspection with Acetic Acid) images.

# Table of Contents
1. The Problem
2. Our Solution
3. Live Application
4. Methodology
5. Technology Stack
6. Project Structure
7. Setup and Installation
8. How to Run
9. Future Work
10. 📜 License
11. 🤝 Contributors

## 1. The Problem: The Cervical Cancer Crisis in LMICs

Cervical cancer is a preventable disease, yet it claims the lives of over 350,000 women annually, with a staggering 90% of these deaths occurring in Low- and Middle-Income Countries (LMICs). In these regions, the most common screening method remains Visual Inspection with Acetic Acid (VIA), despite WHO calls for more sensitive technologies. The effectiveness of VIA, however, is severely limited by a shortage of trained gynecologists and its subjective nature, which leads to high inter-observer variability and poor accuracy (with sensitivity as low as 36.6%). This critical gap between the need for mass screening and the availability of specialist care results in countless preventable deaths.

## 2. Our Solution: Acetowhite Vision AI

Acetowhite Vision AI is designed to bridge this gap. It's a powerful yet simple tool that puts specialist-level accuracy into the hands of frontline health workers.

### Key Features:

* Predicts and identifies potential cervical pre-cancerous lesions from standard VIA images.
* Integrates a high-sensitivity deep learning model with clinical explainability (Grad-CAM).
* Ensures patient data privacy and is designed for offline, on-device processing.
* Outputs: A clear VIA Positive/Negative prediction, a confidence score, an AI explainability heatmap, and a full clinical report.

### How It Works:

The system analyzes a smartphone image of the cervix after acetic acid application to detect early signs of pre-cancerous lesions. It then delivers an instant, objective screening result and a visual explanation — empowering frontline health workers in low-resource settings to make immediate and accurate clinical decisions.

## 3. Live Application

The application is deployed and accessible via a web interface. The frontend provides a simple workflow for uploading an image and receiving a complete clinical analysis in real-time.

* Home Page: Provides an overview of the project.
* Dashboard: The main tool for uploading images and viewing prediction results.
* FAQ Page: Answers common questions about the technology and its use.

## 4. Methodology
This project leverages an advanced, two-stage training approach to build a model that is both highly accurate and clinically safe.

### a. Data Strategy: Semi-Supervised Learning
Unlike traditional supervised learning that requires all data to be labeled, this project leverages a large dataset of both labeled and unlabeled images.

* Data Sources: A small, high-quality labeled dataset (75 VIA positive, 92 VIA negative, and 20 VIA Suspicious cancer) from the `International Research Agency for Cancer (IARC)` is combined with over 6,000 unlabeled VIA images from Kaggle.
* Data Restructuring: Upon ingestion, the dataset is programmatically divided. The IARC images form the labeled set. The Kaggle images form the unlabeled set. This setup allows the model to learn rich, generalizable features from the large pool of unlabeled data, significantly improving its performance.

### b. Stage 1: Semi-Supervised Training

The efficientnet_b3 model is first trained using a semi-supervised learning (SSL) methodology. It learns from the small labeled set while also leveraging the large unlabeled set through pseudo-labeling and consistency regularization. This allows the model to develop a robust understanding of cervical tissue features without needing thousands of expert-annotated images.

### c. Stage 2: High-Sensitivity Fine-Tuning

After the initial SSL training, the model is fine-tuned exclusively on the high-quality labeled dataset. This stage uses a very low learning rate with a critical objective: to maximize sensitivity and minimize false negatives. The final prediction logic uses a low inference threshold of 0.3, ensuring that the model is biased towards catching all potential positive cases, which is the safest approach for a clinical screening tool.

## 5. Technology Stack

* Backend: FastAPI, Uvicorn
* ML Framework: PyTorch
* Model Architecture: timm (PyTorch Image Models)
* MLOps & Deployment: Docker, GitHub Actions, Render
* Frontend: HTML, CSS, JavaScript
* Core Python Libraries: pandas, scikit-learn, opencv-python, pytorch-grad-cam

### 6. Project Structure

```
Acetowhite_Vision/
├── .github/                # CI/CD workflows
│   └── workflows/
│       └── ci-cd.yml       # GitHub Actions workflow for CI/CD
├── app/
│   ├── main.py             # FastAPI application
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css   # Styles for the frontend
│   │   ├── images/
│   │   │       └── ahfid_logo.png
│   │   └── js/
│   │       └── script.js   # JavaScript for frontend interaction
│   └── templates/
│       ├── index.html      # Home page
│       ├── prediction.html # Prediction dashboard
│       └── faq.html        # FAQ page
├── artifacts/              # To store model files, evaluation reports, etc.
│   └── .gitkeep
├── config/
│   ├── config.yaml
│   └── params.yaml
├── data/                   # To store raw and processed data
│   ├── via_cervix.zip   (Local fallback data)
│   └── .gitkeep
├── logs/                   # For storing runtime logs
│   └── running_logs.log
├── notebooks/              # Original Jupyter notebooks
│   ├── 01_data_ingestion.ipynb
│   ├── 02_prepare_base_model.ipynb
│   ├── 03_high_sensitivity_training.ipynb
│   ├── 04_high_sensitivity_evaluation.ipynb
│   └── 05_high_sensitivity_inference.ipynb
├── src/
│   ├── Acetowhite_Vision/
│   │   ├── __init__.py
│   │   ├── components/
│   │   │   ├── __init__.py
│   │   │   ├── data_ingestion.py
│   │   │   ├── model_evaluation.py
│   │   │   ├── model_trainer.py
│   │   │   └── prepare_base_model.py
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   └── configuration.py
│   │   ├── entity/
│   │   │   ├── __init__.py
│   │   │   └── config_entity.py
│   │   ├── pipeline/
│   │   │   ├── __init__.py
│   │   │   ├── predict.py
│   │   │   ├── stage_01_data_ingestion.py
│   │   │   ├── stage_02_prepare_base_model.py
│   │   │   ├── stage_03_model_trainer.py
│   │   │   └── stage_04_model_evaluation.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── common.py
│   │   │   └── logger.py
│   │   └── constants/
│   │       └── __init__.py
├── tests/                  # Automated tests for the application
│   └── test_predict_pipeline.py
├── Dockerfile              # Docker configuration for deployment
├── requirements.txt        # Project dependencies
├── setup.py                # Setup script for installing the project as a package
├── main.py                 # Main script to run the training pipeline
├── pytest.ini              # Configuration for pytest
└── render.yaml             # Deployment configuration for Render
```

## 7. Setup and Installation

Prerequisites:
* Python 3.9+
* A virtual environment tool (venv, conda)

### Step 1: Clone the Repository

```
git clone [https://github.com/AHFIDAILabs/Acetowhite_Vision.git](https://github.com/AHFIDAILabs/Acetowhite_Vision.git)

cd Acetowhite_Vision
```
### Step 2: Create and Activate a Virtual Environment

```
# For Windows
python -m venv venv
venv\Scripts\activate

# For Git Bash
python -m venv venv
source venv/Scripts/activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```
pip install -r requirements.txt
```

### Step 4: Install the Project as an Editable Package

This step is crucial for the application to find all the necessary modules.

```
pip install -e .
```

### Step 5: Set Up Data

The pipeline is configured to download data from Google Drive. If this fails (e.g., due to permission issues), it will use a local fallback.
* Option A (Recommended): Ensure the Google Drive file specified in config/config.yaml is shared with "Anyone with the link".
* Option B (Fallback): Place your local via_cervix.zip file inside a data/ directory at the project root.

## 8. How to Run

There are two main ways to run this project:

* a. Run the Full MLOps Training Pipeline
This command will execute all the stages defined in main.py: Data Ingestion, Base Model Preparation, Model Training, and Evaluation.

```
python main.py
```

* b. Run the Web Application
This command starts the FastAPI server, allowing you to interact with the trained model via a web browser.

```
uvicorn app.main:app --reload
```
Navigate to http://127.0.0.1:8000 in your web browser.

* c. Run with Docker
This is the recommended method for a stable, production-like deployment. It ensures the application runs in a clean, isolated environment.

    Step 1: Build the Docker Image

    ```
    docker build -t acetowhite-vision-app .
    ```
    Step 2: Run the Docker Container

    ```
    docker run -p 8000:8000 acetowhite-vision-app
    ```
Navigate to http://127.0.0.1:8000 in your web browser.

## License
This project is distributed under the MIT License.

## Contributors
* Lead Developer: ’Wale Ogundeji
* Contributors: AHFID AI Team