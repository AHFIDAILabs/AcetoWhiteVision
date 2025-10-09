


##  project structure
`
Acetowhite_Vision/
├── .github/                # CI/CD workflows
│   └── workflows/
│       └── ci-cd.yml       # GitHub Actions workflow for CI/CD
├── app/
│   ├── main.py             # FastAPI application
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css   # Styles for the frontend
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
`