import torch
import base64
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import cv2
import timm
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from Acetowhite_Vision.utils.logger import logger
from Acetowhite_Vision.config.configuration import ConfigurationManager


class PredictionPipeline:
    """
    Handles the complete inference process, including prediction,
    uncertainty quantification, and Grad-CAM explainability.
    """
    
    def __init__(self, filename: Path):
        """
        Initialize the prediction pipeline.
        
        Args:
            filename: Path to the input image file
        """
        self.filename = filename
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load configurations
        config_manager = ConfigurationManager()
        self.params = config_manager.get_params_config()
        self.training_config = config_manager.get_model_trainer_config()
        
        # Initialize model and transforms
        self.model = self._load_model()
        self.transform = self._get_transform()
    
    def _load_model(self) -> torch.nn.Module:
        """
        Load the trained PyTorch model.
        
        Returns:
            Loaded and initialized model
            
        Raises:
            Exception: If model loading fails
        """
        try:
            model_path = self.training_config.trained_model_path
            logger.info(f"Loading model from: {model_path}")
            
            # Load checkpoint to inspect it
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Determine actual number of classes from checkpoint
            if 'classifier.weight' in checkpoint:
                actual_num_classes = checkpoint['classifier.weight'].shape[0]
            elif 'fc.weight' in checkpoint:
                actual_num_classes = checkpoint['fc.weight'].shape[0]
            else:
                # Fallback to config value
                actual_num_classes = self.params.NUM_CLASSES
            
            logger.info(f"Detected {actual_num_classes} classes in saved model")
            
            # Create model architecture with correct number of classes
            model = timm.create_model(
                self.params.MODEL_NAME,
                pretrained=False,
                num_classes=actual_num_classes
            )
            
            # Load trained weights
            model.load_state_dict(checkpoint)
            model.to(self.device)
            model.eval()
            
            logger.info("Model loaded successfully.")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _get_transform(self) -> transforms.Compose:
        """
        Define the image transformation pipeline.
        
        Returns:
            Composed transformation pipeline
        """
        return transforms.Compose([
            transforms.Resize(self.params.IMAGE_SIZE[:-1]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _get_target_layer(self) -> List[torch.nn.Module]:
        """
        Get the appropriate target layer for GradCAM based on model architecture.
        
        Returns:
            List containing the target layer for GradCAM
        """
        model_name = self.params.MODEL_NAME.lower()
        
        # Handle different timm model architectures
        if 'resnet' in model_name:
            return [self.model.layer4[-1]]
        
        elif 'efficientnet' in model_name:
            return [self.model.blocks[-1][-1]]
        
        elif 'vit' in model_name or 'vision_transformer' in model_name:
            return [self.model.blocks[-1].norm1]
        
        elif 'convnext' in model_name:
            return [self.model.stages[-1].blocks[-1]]
        
        elif 'densenet' in model_name:
            return [self.model.features[-1]]
        
        else:
            # Fallback: try to find the last convolutional layer
            try:
                all_modules = list(self.model.named_modules())
                
                # Find last Conv2d layer
                for name, module in reversed(all_modules):
                    if isinstance(module, torch.nn.Conv2d):
                        logger.info(f"Using layer: {name}")
                        return [module]
            except Exception:
                pass
            
            # Last resort
            logger.warning(
                "Could not identify target layer, GradCAM may not work properly"
            )
            return [self.model]
    
    def _preprocess_image(self) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Load and preprocess the input image.
        
        Returns:
            Tuple of (preprocessed tensor, original RGB image as numpy array)
        """
        img = Image.open(self.filename).convert("RGB")
        
        # Define preprocessing transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        # Apply transforms
        input_tensor = transform(img).unsqueeze(0).to(self.device)
        
        # Keep original image for visualization
        original_image_rgb = np.array(img.resize((224, 224))) / 255.0
        
        return input_tensor, original_image_rgb
    
    def _calculate_prediction(
        self, 
        input_tensor: torch.Tensor
    ) -> Tuple[str, float, float]:
        """
        Calculate prediction and confidence from model output.
        
        Args:
            input_tensor: Preprocessed image tensor
            
        Returns:
            Tuple of (prediction label, confidence score, raw probability)
        """
        with torch.no_grad():
            output = self.model(input_tensor)
            
            # Handle both binary classification approaches
            if output.shape[1] == 1:
                # Single output (binary)
                prob = torch.sigmoid(output).item()
            else:
                # Two outputs (softmax)
                probs = torch.softmax(output, dim=1)
                prob = probs[0, 1].item()  # Probability of positive class
        
        prediction = "Positive" if prob > 0.5 else "Negative"
        confidence = prob if prediction == "Positive" else 1 - prob
        
        return prediction, confidence, prob
    
    def _calculate_uncertainty(self, prob: float) -> Tuple[float, str]:
        """
        Calculate uncertainty score and classification.
        
        Args:
            prob: Raw probability from model
            
        Returns:
            Tuple of (uncertainty score, uncertainty classification)
        """
        uncertainty_score = 1 - (2 * abs(prob - 0.5))
        uncertainty_class = "High" if uncertainty_score > 0.4 else "Low"
        
        return uncertainty_score, uncertainty_class
    
    def _generate_grad_cam(
        self, 
        input_tensor: torch.Tensor, 
        original_image: np.ndarray
    ) -> Optional[str]:
        """
        Generate Grad-CAM visualization.
        
        Args:
            input_tensor: Preprocessed image tensor
            original_image: Original image as numpy array
            
        Returns:
            Base64 encoded Grad-CAM image or None if generation fails
        """
        try:
            # Get appropriate target layer
            target_layers = self._get_target_layer()
            
            # Initialize GradCAM
            cam = GradCAM(model=self.model, target_layers=target_layers)
            
            # Generate CAM for the positive class (index 0)
            targets = [ClassifierOutputTarget(0)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            # Overlay CAM on original image
            visualization = show_cam_on_image(
                original_image, 
                grayscale_cam, 
                use_rgb=True
            )
            
            # Encode to base64
            is_success, buffer = cv2.imencode(".jpg", visualization)
            if not is_success:
                raise Exception("Failed to encode Grad-CAM image.")
            
            return base64.b64encode(buffer).decode("utf-8")
            
        except Exception as e:
            logger.error(f"GradCAM generation failed: {e}", exc_info=True)
            return None
    
    def _generate_clinical_report(
        self, 
        prediction: str, 
        confidence: float, 
        uncertainty: str
    ) -> str:
        """
        Generate a structured clinical report.
        
        Args:
            prediction: Prediction label (Positive/Negative)
            confidence: Confidence score
            uncertainty: Uncertainty classification (High/Low)
            
        Returns:
            Formatted clinical report string
        """
        report = f"""
ACETOWHITE VISION AI - CLINICAL ANALYSIS REPORT
-------------------------------------------------
Patient ID:      [Not Provided]
Image Filename:  {self.filename.name}
Date Analyzed:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
-------------------------------------------------

FINDING:
========
The AI model predicts this case as: {prediction.upper()}

CONFIDENCE & UNCERTAINTY:
=========================
- Confidence in Prediction: {confidence:.2%}
- Model Uncertainty Level:  {uncertainty.upper()}

RECOMMENDATION:
===============
- If POSITIVE: Immediate review by a qualified colposcopist is strongly 
  recommended. This finding suggests the potential presence of acetowhite 
  changes that warrant further investigation.
  
- If NEGATIVE: Routine follow-up is advised. While the AI did not detect 
  significant features, this result does not replace a comprehensive 
  clinical evaluation.

DISCLAIMER:
===========
This is an AI-generated report for screening purposes only. It is not a 
diagnosis. All findings must be correlated with clinical history and 
confirmed by a qualified medical professional.
        """
        return report.strip()
    
    def predict_with_explanation(self) -> Dict:
        """
        Run prediction and generate a full clinical report with explainability.
        
        Returns:
            Dictionary containing prediction results, confidence, uncertainty,
            Grad-CAM visualization, and clinical report
        """
        try:
            # 1. Preprocess image
            input_tensor, original_image_rgb = self._preprocess_image()
            
            # 2. Get prediction and confidence
            prediction, confidence, prob = self._calculate_prediction(input_tensor)
            
            # 3. Calculate uncertainty
            uncertainty_score, uncertainty_class = self._calculate_uncertainty(prob)
            
            # 4. Generate Grad-CAM
            grad_cam_b64 = self._generate_grad_cam(
                input_tensor, 
                original_image_rgb
            )
            
            # 5. Generate clinical report
            report = self._generate_clinical_report(
                prediction, 
                confidence, 
                uncertainty_class
            )
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "uncertainty_score": uncertainty_score,
                "uncertainty_classification": uncertainty_class,
                "grad_cam_image_b64": grad_cam_b64,
                "clinical_report": report
            }
            
        except Exception as e:
            logger.error(f"Prediction pipeline failed: {e}", exc_info=True)
            return {"error": str(e)}