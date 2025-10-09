from pathlib import Path
from Acetowhite_Vision.config.configuration import ConfigurationManager
from Acetowhite_Vision.components.model_trainer import ModelTrainer
from Acetowhite_Vision.utils.logger import logger

STAGE_NAME = "Model Training Stage"


class ModelTrainingPipeline:
    """
    Orchestrates the model training process by configuring and running
    the ModelTrainer component.
    """

    def __init__(self):
        pass

    def main(self):
        """
        Executes the model training pipeline, skipping if a trained model already exists.
        """
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            trained_model_path = Path(model_trainer_config.trained_model_path)

            # ✅ Skip if model already trained
            if trained_model_path.exists():
                logger.info(f"Trained model already exists at {trained_model_path.resolve()}, skipping training stage.")
            else:
                logger.info("Starting model training...")
                model_trainer = ModelTrainer(config=model_trainer_config)
                model_trainer.train()
                logger.info(f"Model training completed. Model saved to {trained_model_path.resolve()}.")

        except Exception as e:
            logger.error(f"Error in Model Training Pipeline: {e}")
            raise


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e