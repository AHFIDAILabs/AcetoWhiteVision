import glob
from pathlib import Path
from Acetowhite_Vision.constants import *
from Acetowhite_Vision.utils.common import read_yaml, create_directories
from Acetowhite_Vision.entity.config_entity import (DataIngestionConfig,
                                                      PrepareBaseModelConfig,
                                                      TrainingConfig,
                                                      EvaluationConfig)
from Acetowhite_Vision.utils.logger import logger

class ConfigurationManager:
    """
    Manages the reading and validation of configuration files.
    Provides configuration settings for different pipeline stages.
    """
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])
        
    def get_params_config(self):
        """
        Returns the parsed parameters from params.yaml as a dictionary-like object.
        This ensures PredictionPipeline and other components can access model params directly.
        """
        return self.params

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Extracts and returns the data ingestion configuration.
        """
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        
        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            gdrive_id=config.gdrive_id,  # <-- CORRECTED
            local_archive_file=Path(config.local_archive_file),
            unzip_dir=Path(config.unzip_dir),
            final_data_dir=Path(config.final_data_dir)
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        """
        Extracts and returns the base model preparation configuration.
        """
        config = self.config.prepare_base_model
        create_directories([config.root_dir])

        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_model_name=self.params.MODEL_NAME,
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_pretrained=self.params.PRETRAINED,
            params_num_classes=self.params.NUM_CLASSES
        )


    def get_model_trainer_config(self) -> TrainingConfig:
        """
        Extracts and returns the model training configuration.
        """
        config = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        create_directories([config.root_dir])

        return TrainingConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(self.config.data_ingestion.final_data_dir),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=False,  # or params.AUGMENTATION if you define it later
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE,
            params_model_name=params.MODEL_NAME,
            params_classes=params.CLASSES
        )

    def get_evaluation_config(self) -> EvaluationConfig:
        """
        Extracts and returns the model evaluation configuration.
        Automatically locates the latest fine-tuned model in artifacts/training.
        """
        config = self.config.evaluation
        root_dir = Path("artifacts/evaluation")
        create_directories([root_dir])

        # ✅ Find latest fine-tuned model in artifacts/training
        training_dir = Path("artifacts/training")
        model_files = sorted(
            training_dir.glob("*.h5"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )

        if model_files:
            latest_model_path = model_files[0]
            logger.info(f"Auto-detected latest fine-tuned model: {latest_model_path.name}")
        else:
            # Fallback: use config path if no model found
            latest_model_path = Path(self.config.training.trained_model_path)
            logger.warning(f"No model files found in {training_dir}, using default: {latest_model_path}")

        return EvaluationConfig(
            root_dir=root_dir,
            path_of_model=latest_model_path,
            training_data=Path(self.config.data_ingestion.final_data_dir),
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )