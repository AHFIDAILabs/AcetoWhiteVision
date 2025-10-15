import sys
from pathlib import Path

# Add the 'src' directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import logging
from Acetowhite_Vision.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from Acetowhite_Vision.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from Acetowhite_Vision.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from Acetowhite_Vision.pipeline.stage_04_model_evaluation import EvaluationPipeline
from Acetowhite_Vision.utils.logger import logger

# Configure logging first
logging.basicConfig(
    filename='logfile.log',
    filemode='w',
    encoding='utf-8',
    level=logging.INFO,
    format='%(asctime)s: %(levelname)s: %(message)s'
)

STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Prepare base model"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_base_model = PrepareBaseModelTrainingPipeline()
   prepare_base_model.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Training"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_trainer = ModelTrainingPipeline()
   model_trainer.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Evaluation stage"
try:
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_evaluation = EvaluationPipeline()
   model_evaluation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise e