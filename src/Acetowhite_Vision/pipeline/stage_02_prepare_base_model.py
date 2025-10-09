from pathlib import Path
from Acetowhite_Vision.config.configuration import ConfigurationManager
from Acetowhite_Vision.components.prepare_base_model import PrepareBaseModel
from Acetowhite_Vision.utils.logger import logger

STAGE_NAME = "Prepare base model"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)

        updated_model_path = Path(prepare_base_model_config.updated_base_model_path)

        # ✅ Skip stage if model already exists
        if updated_model_path.exists():
            logger.info(f"Base model already prepared at {updated_model_path.resolve()}, skipping stage.")
        else:
            logger.info("Preparing base model...")
            prepare_base_model.get_base_model()
            prepare_base_model.update_base_model()
            logger.info(f"Base model prepared and saved to {updated_model_path.resolve()}.")


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e