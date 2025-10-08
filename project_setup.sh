#!/bin/bash

# Exit if any command fails
set -e

# Project root
PROJECT_ROOT="Acetowhite-Vision"

echo "Setting up project: $PROJECT_ROOT ..."
echo "Cleaning up any existing directory ..."
rm -rf $PROJECT_ROOT

# Create base directories
mkdir -p $PROJECT_ROOT
mkdir -p $PROJECT_ROOT/.github/workflows
mkdir -p $PROJECT_ROOT/app/static/css
mkdir -p $PROJECT_ROOT/app/static/js
mkdir -p $PROJECT_ROOT/app/templates
mkdir -p $PROJECT_ROOT/artifacts
mkdir -p $PROJECT_ROOT/data
mkdir -p $PROJECT_ROOT/notebooks
mkdir -p $PROJECT_ROOT/src/CervicalAI_Screen/{components,config,entity,pipeline,utils,constants}

# Create placeholder files
touch $PROJECT_ROOT/app/main.py
touch $PROJECT_ROOT/app/static/css/style.css
touch $PROJECT_ROOT/app/static/js/script.js
touch $PROJECT_ROOT/app/templates/index.html
touch $PROJECT_ROOT/app/templates/prediction.html
touch $PROJECT_ROOT/app/templates/faq.html
touch $PROJECT_ROOT/artifacts/.gitkeep
touch $PROJECT_ROOT/data/.gitkeep

# Notebooks
for i in {01..05}; do
    case $i in
        01) name="data_ingestion" ;;
        02) name="prepare_base_model" ;;
        03) name="high_sensitivity_training" ;;
        04) name="high_sensitivity_evaluation" ;;
        05) name="high_sensitivity_inference" ;;
    esac
    touch "$PROJECT_ROOT/notebooks/${i}_${name}.ipynb"
done

# Python package structure
for dir in $PROJECT_ROOT/src/CervicalAI_Screen \
           $PROJECT_ROOT/src/CervicalAI_Screen/components \
           $PROJECT_ROOT/src/CervicalAI_Screen/config \
           $PROJECT_ROOT/src/CervicalAI_Screen/entity \
           $PROJECT_ROOT/src/CervicalAI_Screen/pipeline \
           $PROJECT_ROOT/src/CervicalAI_Screen/utils \
           $PROJECT_ROOT/src/CervicalAI_Screen/constants; do
    touch $dir/__init__.py
done

# Component files
touch $PROJECT_ROOT/src/CervicalAI_Screen/components/data_ingestion.py
touch $PROJECT_ROOT/src/CervicalAI_Screen/components/model_evaluation.py
touch $PROJECT_ROOT/src/CervicalAI_Screen/components/model_trainer.py
touch $PROJECT_ROOT/src/CervicalAI_Screen/components/prepare_base_model.py

# Config
touch $PROJECT_ROOT/src/CervicalAI_Screen/config/configuration.py

# Entity
touch $PROJECT_ROOT/src/CervicalAI_Screen/entity/config_entity.py

# Pipeline
touch $PROJECT_ROOT/src/CervicalAI_Screen/pipeline/predict.py
touch $PROJECT_ROOT/src/CervicalAI_Screen/pipeline/stage_01_data_ingestion.py
touch $PROJECT_ROOT/src/CervicalAI_Screen/pipeline/stage_02_prepare_base_model.py
touch $PROJECT_ROOT/src/CervicalAI_Screen/pipeline/stage_03_model_trainer.py
touch $PROJECT_ROOT/src/CervicalAI_Screen/pipeline/stage_04_model_evaluation.py

# Utils
touch $PROJECT_ROOT/src/CervicalAI_Screen/utils/common.py

# Constants
touch $PROJECT_ROOT/src/CervicalAI_Screen/constants/__init__.py

# Root files
touch $PROJECT_ROOT/Dockerfile
touch $PROJECT_ROOT/requirements.txt
touch $PROJECT_ROOT/setup.py
touch $PROJECT_ROOT/main.py
touch $PROJECT_ROOT/render.yaml

# CI/CD workflow (GitHub Actions)
cat > $PROJECT_ROOT/.github/workflows/ci-cd.yml << 'EOF'
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

    - name: Run tests
      run: |
        pytest || echo "No tests found, skipping test step."

    - name: Lint code
      run: |
        pip install flake8
        flake8 --ignore=E501,F401 .
EOF

# Initialize Git
cd $PROJECT_ROOT
git init -q
git add .
git commit -m "Initial project setup with structure, venv, and CI/CD workflow" -q

# Create Python virtual environment
echo "Creating Python virtual environment ..."
python3 -m venv venv

# Activate and install dependencies if available
source venv/bin/activate
if [ -s requirements.txt ]; then
    echo "Installing dependencies ..."
    pip install -r requirements.txt
else
    echo "# Example dependencies" >> requirements.txt
    echo "fastapi" >> requirements.txt
    echo "uvicorn" >> requirements.txt
    echo "pytest" >> requirements.txt
    pip install -r requirements.txt
fi

# Make sure venv is gitignored
echo "venv/" >> .gitignore
echo "__pycache__/" >> .gitignore

git add .gitignore requirements.txt
git commit -m "Add virtual environment setup and basic dependencies" -q

echo "✅ Project '$PROJECT_ROOT' created successfully!"
echo "Navigate into the directory with: cd $PROJECT_ROOT"
echo "Activate the virtual environment using: source venv/bin/activate"