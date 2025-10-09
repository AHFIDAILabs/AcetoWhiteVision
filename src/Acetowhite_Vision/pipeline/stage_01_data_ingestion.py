from pathlib import Path
from Acetowhite_Vision.config.configuration import ConfigurationManager
from Acetowhite_Vision.components.data_ingestion import DataIngestion
from Acetowhite_Vision.utils.logger import logger

STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)

            final_data_dir = Path(data_ingestion_config.final_data_dir)

            # ✅ Skip stage if data already exists
            if final_data_dir.exists() and any(final_data_dir.iterdir()):
                logger.info(f"Data already exists at {final_data_dir.resolve()}, skipping data ingestion stage.")
            else:
                logger.info("Running data ingestion...")
                data_ingestion.run_ingestion()
                logger.info("Data ingestion completed successfully.")
        except Exception as e:
            logger.exception(e)
            raise e


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} finished <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e