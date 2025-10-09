import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import json
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from Acetowhite_Vision.entity.config_entity import EvaluationConfig
from Acetowhite_Vision.utils.common import save_json, create_directories
from Acetowhite_Vision.utils.logger import logger


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval_dir = Path(self.config.root_dir) if hasattr(self.config, "root_dir") else Path("artifacts/evaluation")
        create_directories([self.eval_dir])

        logger.info(f"Evaluation artifacts will be saved in: {self.eval_dir.resolve()}")

        # ✅ Load model
        self.model = self._load_model()

    def _load_model(self):
        """Loads the fine-tuned model for evaluation."""
        model_path = Path(self.config.path_of_model)
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {model_path.resolve()}")

        logger.info(f"Loading fine-tuned model from: {model_path}")
        model = timm.create_model(
            self.config.all_params.MODEL_NAME,
            pretrained=False,
            num_classes=self.config.all_params.NUM_CLASSES
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def _get_test_generator(self):
        """Creates a data loader for the test set."""
        transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # ✅ Look for labeled folder (case-insensitive)
        base_dir = Path(self.config.training_data)
        candidates = [base_dir / "labeled", base_dir / "Labeled"]
        test_data_path = next((p for p in candidates if p.exists()), base_dir)
        logger.info(f"Loading test data from: {test_data_path}")

        test_dataset = datasets.ImageFolder(root=test_data_path, transform=transform)
        self.class_names = test_dataset.classes

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False
        )
        logger.info(f"Test data loader created successfully with {len(test_dataset)} images.")

    def evaluate(self):
        """Evaluates the model on the test data."""
        self._get_test_generator()
        self.y_true = []
        self.y_pred_probs = []

        logger.info("Starting model evaluation...")
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc="Evaluating"):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()

                self.y_true.extend(labels.numpy())
                self.y_pred_probs.extend(probs)

        self.y_true = np.array(self.y_true)
        self.y_pred_probs = np.array(self.y_pred_probs)
        self.y_pred = np.argmax(self.y_pred_probs, axis=1)
        logger.info("Model evaluation complete.")

    def _calculate_metrics_with_ci(self):
        """Computes key metrics with 95% confidence intervals."""
        cm = confusion_matrix(self.y_true, self.y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = cm[0, 0] if cm.shape[0] > 0 else 0
            fp = cm[0, 1] if cm.shape[1] > 1 else 0
            fn = cm[1, 0] if cm.shape[0] > 1 else 0
            tp = cm[1, 1] if cm.shape[1] > 1 else 0

        sensitivity = tp / (tp + fn + 1e-6)
        specificity = tn / (tn + fp + 1e-6)
        npv = tn / (tn + fn + 1e-6)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)

        def wilson_ci(p, n):
            if n == 0: return (0, 1)
            z = 1.96
            denom = 1 + z**2 / n
            center = (p + z**2 / (2 * n)) / denom
            margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
            return max(0, center - margin), min(1, center + margin)

        self.metrics = {
            "accuracy": float(accuracy),
            "sensitivity": float(sensitivity),
            "sensitivity_ci": wilson_ci(sensitivity, tp + fn),
            "specificity": float(specificity),
            "specificity_ci": wilson_ci(specificity, tn + fp),
            "negative_predictive_value": float(npv),
            "npv_ci": wilson_ci(npv, tn + fn),
        }

    def _plot_roc_curve(self, path):
        """Generates and saves ROC curve plot."""
        try:
            if self.y_pred_probs.shape[1] < 2:
                logger.warning("ROC curve skipped: only one class predicted.")
                return

            fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_probs[:, 1])
            roc_auc = auc(fpr, tpr)
            self.metrics["roc_auc"] = float(roc_auc)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            plt.savefig(path)
            logger.info(f"ROC curve saved at: {path}")
        except Exception as e:
            logger.warning(f"ROC curve generation failed: {e}")

    def save_evaluation_results(self):
        """Saves metrics, ROC curve, classification report, and confusion matrix."""
        metrics_path = self.eval_dir / "evaluation_metrics.json"
        roc_path = self.eval_dir / "roc_curve.png"
        report_path = self.eval_dir / "classification_report.json"
        cm_path = self.eval_dir / "confusion_matrix.png"

        self._calculate_metrics_with_ci()
        save_json(path=metrics_path, data=self.metrics)
        logger.info(f"Evaluation metrics saved to: {metrics_path}")

        self._plot_roc_curve(roc_path)

        report = classification_report(self.y_true, self.y_pred, target_names=self.class_names, output_dict=True)
        save_json(path=report_path, data=report)
        logger.info(f"Classification report saved to: {report_path}")

        cm = confusion_matrix(self.y_true, self.y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title("Confusion Matrix")
        plt.savefig(cm_path)
        logger.info(f"Confusion matrix saved to: {cm_path}")

    def run_evaluation(self):
        """Runs full evaluation pipeline."""
        try:
            self.evaluate()
            self.save_evaluation_results()
            logger.info("Model evaluation completed successfully.")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            raise