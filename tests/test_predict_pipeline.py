import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch
from PIL import Image

from Acetowhite_Vision.pipeline.predict import PredictionPipeline


def create_dummy_image(image_path: Path) -> None:
    """
    Creates a dummy RGB image for testing purposes.
    
    Args:
        image_path: Path where the image should be saved
    """
    image_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_image = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(dummy_image)
    img.save(image_path)


def create_dummy_pytorch_model(model_path: Path, num_classes: int = 2) -> None:
    """
    Creates and saves a simple PyTorch model checkpoint for testing.
    
    Args:
        model_path: Path where the model should be saved
        num_classes: Number of output classes (1 or 2)
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a simple state dict that mimics an EfficientNet structure
    state_dict = {
        'conv_stem.weight': torch.randn(32, 3, 3, 3),
        'bn1.weight': torch.randn(32),
        'bn1.bias': torch.randn(32),
        'classifier.weight': torch.randn(num_classes, 1536),
        'classifier.bias': torch.randn(num_classes),
    }
    
    torch.save(state_dict, model_path)


@pytest.fixture
def mock_config():
    """Mock the configuration manager and its configs."""
    with patch('Acetowhite_Vision.pipeline.predict.ConfigurationManager') as mock_config_manager:
        # Mock params config
        mock_params = Mock()
        mock_params.MODEL_NAME = 'efficientnet_b0'
        mock_params.NUM_CLASSES = 2
        mock_params.IMAGE_SIZE = (224, 224, 3)
        mock_params.INFERENCE_THRESHOLD = 0.5
        
        # Mock training config
        mock_training = Mock()
        
        # Setup the config manager to return mocked configs
        config_instance = mock_config_manager.return_value
        config_instance.get_params_config.return_value = mock_params
        config_instance.get_model_trainer_config.return_value = mock_training
        
        yield mock_config_manager, mock_params, mock_training


@pytest.fixture
def test_image_path(tmp_path):
    """Create a test image and return its path."""
    image_path = tmp_path / "test_image.jpg"
    create_dummy_image(image_path)
    return image_path


@pytest.fixture
def test_model_path(tmp_path):
    """Create a dummy model and return its path."""
    model_path = tmp_path / "artifacts" / "training" / "model.h5"
    create_dummy_pytorch_model(model_path, num_classes=2)
    return model_path


def test_prediction_pipeline_initialization(test_image_path, test_model_path, mock_config):
    """Test if the PredictionPipeline initializes correctly."""
    _, mock_params, mock_training = mock_config
    mock_training.trained_model_path = test_model_path
    
    with patch('Acetowhite_Vision.pipeline.predict.timm.create_model') as mock_create_model:
        # Mock the model
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_create_model.return_value = mock_model
        
        pipeline = PredictionPipeline(filename=test_image_path)
        
        assert pipeline.filename == test_image_path
        assert pipeline.device in ["cuda", "cpu"]
        assert pipeline.model is not None
        assert pipeline.transform is not None


def test_model_loading_with_correct_num_classes(test_image_path, test_model_path, mock_config):
    """Test that model loads with the correct number of classes from checkpoint."""
    _, mock_params, mock_training = mock_config
    mock_training.trained_model_path = test_model_path
    
    with patch('Acetowhite_Vision.pipeline.predict.timm.create_model') as mock_create_model:
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_create_model.return_value = mock_model
        
        pipeline = PredictionPipeline(filename=test_image_path)
        
        # Verify that timm.create_model was called with num_classes=2
        # (since we created a model with 2 classes)
        mock_create_model.assert_called_once()
        call_kwargs = mock_create_model.call_args[1]
        assert call_kwargs['num_classes'] == 2


def test_preprocess_image(test_image_path, test_model_path, mock_config):
    """Test image preprocessing returns correct format."""
    _, mock_params, mock_training = mock_config
    mock_training.trained_model_path = test_model_path
    
    with patch('Acetowhite_Vision.pipeline.predict.timm.create_model') as mock_create_model:
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_create_model.return_value = mock_model
        
        pipeline = PredictionPipeline(filename=test_image_path)
        input_tensor, original_image = pipeline._preprocess_image()
        
        # Check tensor properties
        assert isinstance(input_tensor, torch.Tensor)
        assert input_tensor.shape == (1, 3, 224, 224)
        
        # Check original image properties
        assert isinstance(original_image, np.ndarray)
        assert original_image.shape == (224, 224, 3)
        assert original_image.max() <= 1.0


def test_predict_with_explanation_returns_valid_format(test_image_path, test_model_path, mock_config):
    """Test if predict_with_explanation returns the expected dictionary format."""
    _, mock_params, mock_training = mock_config
    mock_training.trained_model_path = test_model_path
    
    with patch('Acetowhite_Vision.pipeline.predict.timm.create_model') as mock_create_model:
        # Mock the model
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        
        # Mock model output
        mock_output = torch.tensor([[0.7, 0.3]])  # 2 classes output
        mock_model.return_value = mock_output
        
        mock_create_model.return_value = mock_model
        
        # Mock GradCAM to avoid complications
        with patch('Acetowhite_Vision.pipeline.predict.GradCAM'):
            pipeline = PredictionPipeline(filename=test_image_path)
            result = pipeline.predict_with_explanation()
            
            # Verify result structure
            assert isinstance(result, dict)
            assert "prediction" in result
            assert "confidence" in result
            assert "uncertainty_score" in result
            assert "uncertainty_classification" in result
            assert "clinical_report" in result
            
            # Verify prediction values
            assert result["prediction"] in ["Positive", "Negative"]
            assert 0 <= result["confidence"] <= 1
            assert 0 <= result["uncertainty_score"] <= 1
            assert result["uncertainty_classification"] in ["High", "Low"]
            assert isinstance(result["clinical_report"], str)


def test_calculate_uncertainty(test_image_path, test_model_path, mock_config):
    """Test uncertainty calculation logic."""
    _, mock_params, mock_training = mock_config
    mock_training.trained_model_path = test_model_path
    
    with patch('Acetowhite_Vision.pipeline.predict.timm.create_model') as mock_create_model:
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_create_model.return_value = mock_model
        
        pipeline = PredictionPipeline(filename=test_image_path)
        
        # Test high confidence case (prob close to 0 or 1)
        uncertainty_score, uncertainty_class = pipeline._calculate_uncertainty(0.9)
        assert uncertainty_class == "Low"
        
        # Test low confidence case (prob close to 0.5)
        uncertainty_score, uncertainty_class = pipeline._calculate_uncertainty(0.5)
        assert uncertainty_class == "High"


def test_prediction_handles_binary_output(test_image_path, test_model_path, mock_config):
    """Test prediction handles single output (binary classification)."""
    _, mock_params, mock_training = mock_config
    mock_training.trained_model_path = test_model_path
    
    # Create model with 1 class
    create_dummy_pytorch_model(test_model_path, num_classes=1)
    
    with patch('Acetowhite_Vision.pipeline.predict.timm.create_model') as mock_create_model:
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        
        # Mock single output
        mock_output = torch.tensor([[0.7]])
        mock_model.return_value = mock_output
        
        mock_create_model.return_value = mock_model
        
        pipeline = PredictionPipeline(filename=test_image_path)
        
        # Mock the model in pipeline
        pipeline.model = mock_model
        
        input_tensor = torch.randn(1, 3, 224, 224)
        prediction, confidence, prob = pipeline._calculate_prediction(input_tensor)
        
        assert prediction in ["Positive", "Negative"]
        assert 0 <= confidence <= 1
        assert 0 <= prob <= 1


def test_prediction_handles_softmax_output(test_image_path, test_model_path, mock_config):
    """Test prediction handles two outputs (softmax classification)."""
    _, mock_params, mock_training = mock_config
    mock_training.trained_model_path = test_model_path
    
    with patch('Acetowhite_Vision.pipeline.predict.timm.create_model') as mock_create_model:
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        
        # Mock two-class output
        mock_output = torch.tensor([[0.3, 0.7]])
        mock_model.return_value = mock_output
        
        mock_create_model.return_value = mock_model
        
        pipeline = PredictionPipeline(filename=test_image_path)
        pipeline.model = mock_model
        
        input_tensor = torch.randn(1, 3, 224, 224)
        prediction, confidence, prob = pipeline._calculate_prediction(input_tensor)
        
        assert prediction in ["Positive", "Negative"]
        assert 0 <= confidence <= 1
        assert 0 <= prob <= 1


def test_model_loading_failure(test_image_path, tmp_path, mock_config):
    """Test that pipeline handles model loading failure gracefully."""
    _, mock_params, mock_training = mock_config
    # Point to non-existent model
    mock_training.trained_model_path = tmp_path / "non_existent" / "model.h5"
    
    with pytest.raises(Exception):
        PredictionPipeline(filename=test_image_path)


def test_invalid_image_path(test_model_path, mock_config):
    """Test that pipeline handles invalid image path gracefully."""
    _, mock_params, mock_training = mock_config
    mock_training.trained_model_path = test_model_path
    
    with patch('Acetowhite_Vision.pipeline.predict.timm.create_model') as mock_create_model:
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_create_model.return_value = mock_model
        
        invalid_path = Path("non_existent_image.jpg")
        pipeline = PredictionPipeline(filename=invalid_path)
        
        result = pipeline.predict_with_explanation()
        
        # Should return error in result
        assert "error" in result
        assert isinstance(result["error"], str)


def test_clinical_report_generation(test_image_path, test_model_path, mock_config):
    """Test that clinical report is generated correctly."""
    _, mock_params, mock_training = mock_config
    mock_training.trained_model_path = test_model_path
    
    with patch('Acetowhite_Vision.pipeline.predict.timm.create_model') as mock_create_model:
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_create_model.return_value = mock_model
        
        pipeline = PredictionPipeline(filename=test_image_path)
        
        report = pipeline._generate_clinical_report("Positive", 0.85, "Low")
        report_upper = report.upper()
        
        assert isinstance(report, str)
        assert "ACETOWHITE VISION AI" in report
        assert "POSITIVE" in report_upper
        assert "85.00%" in report or "0.85" in report
        assert "LOW" in report_upper
        assert "DISCLAIMER" in report_upper