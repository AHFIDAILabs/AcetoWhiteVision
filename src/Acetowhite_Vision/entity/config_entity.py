from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    hf_repo_id: str            # Hugging Face repo ID
    hf_filename: str           # Filename to download from Hugging Face
    local_archive_file: Path
    unzip_dir: Path
    final_data_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_model_name: str
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_pretrained: bool
    params_num_classes: int

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
    params_learning_rate: float
    params_model_name: str
    params_classes: int
    
@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path                # ✅ Added this line
    path_of_model: Path
    training_data: Path
    all_params: dict
    params_image_size: list
    params_batch_size: int
   