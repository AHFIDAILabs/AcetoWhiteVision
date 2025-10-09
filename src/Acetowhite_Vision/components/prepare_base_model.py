import torch
import timm
from pathlib import Path
from Acetowhite_Vision.entity.config_entity import PrepareBaseModelConfig
from Acetowhite_Vision.utils.logger import logger

class PrepareBaseModel:
    """
    Prepares and saves the base model architecture using a pre-trained model
    from the 'timm' library.
    """
    def __init__(self, config: PrepareBaseModelConfig):
        """
        Initializes the PrepareBaseModel component.
        
        Args:
            config (PrepareBaseModelConfig): Configuration for the base model.
        """
        self.config = config
        self.model = None  # Initialize placeholder for the model object

    def get_base_model(self):
        """
        Creates and saves the base model using the configuration specified
        in params.yaml (e.g., 'efficientnet_b3').
        """
        # ✅ Skip if model file already exists
        if self.config.base_model_path.exists():
            logger.info(f"Base model already exists at {self.config.base_model_path}, skipping creation.")
            self.model = timm.create_model(
                self.config.params_model_name,
                pretrained=self.config.params_pretrained,
                num_classes=self.config.params_num_classes
            )
            return self.model

        # ✅ Otherwise, create and save the base model
        logger.info(f"Creating base model '{self.config.params_model_name}' using 'timm'...")
        self.model = timm.create_model(
            self.config.params_model_name,
            pretrained=self.config.params_pretrained,
            num_classes=self.config.params_num_classes
        )
        self.save_model(path=self.config.base_model_path, model=self.model)
        return self.model

    def update_base_model(self):
        """
        Saves the structured base model again to the updated_base_model_path
        for consistency with the pipeline design.
        """
        if self.model is None:
            raise ValueError("Model not found. Call get_base_model() first.")
        
        logger.info("Saving the final, untrained model structure...")
        self.save_model(path=self.config.updated_base_model_path, model=self.model)
        
    @staticmethod
    def save_model(path: Path, model: torch.nn.Module):
        """Saves the PyTorch model's state dictionary to the specified path."""
        torch.save(model.state_dict(), path)
        logger.info(f"Model state dictionary saved to {path}")