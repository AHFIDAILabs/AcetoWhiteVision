import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import timm
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from Acetowhite_Vision.entity.config_entity import TrainingConfig
from Acetowhite_Vision.utils.logger import logger


class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Hyperparameters
        self.consistency_weight = 1.0
        self.pseudo_label_threshold = 0.8
        self.learning_rate = 0.001
        self.hs_learning_rate = 0.0001
        self.hs_epochs = 10
        self.inference_threshold = 0.3
        self.num_classes = 2  # Binary classification: Positive / Negative

    def _get_dataloaders(self):
        """Creates loaders for labeled and unlabeled data."""
        data_dir = Path(self.config.training_data)

        # --- Auto-organize directory structure ---
        labeled_path = data_dir / "labeled"
        unlabeled_path = data_dir / "Unlabeled"

        if not labeled_path.exists():
            logger.info("Structuring data into labeled/unlabeled folders...")
            labeled_path.mkdir(exist_ok=True)
            for cls in ["Positive", "Negative"]:
                src = data_dir / cls
                dst = labeled_path / cls
                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if not dst.exists():
                        src.rename(dst)
                        logger.info(f"Moved {cls}/ into labeled/.")
            unlabeled_src = data_dir / "Unlabeled"
            if unlabeled_src.exists() and not unlabeled_path.exists():
                unlabeled_src.rename(unlabeled_path)
                logger.info("Renamed Unlabeled folder for consistency.")

        # --- Define transforms ---
        labeled_transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        unlabeled_transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # --- Load datasets ---
        labeled_dataset = datasets.ImageFolder(root=labeled_path, transform=labeled_transform)
        unlabeled_dataset = datasets.ImageFolder(root=unlabeled_path, transform=unlabeled_transform) if unlabeled_path.exists() else None

        # --- Train/validation split ---
        train_size = int(0.8 * len(labeled_dataset))
        val_size = len(labeled_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(labeled_dataset, [train_size, val_size])

        self.train_loader = DataLoader(train_dataset, batch_size=self.config.params_batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.params_batch_size, shuffle=False)
        self.unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=self.config.params_batch_size, shuffle=True) if unlabeled_dataset else None

        logger.info(f"Labeled samples: {len(labeled_dataset)}, Unlabeled samples: {len(unlabeled_dataset) if unlabeled_dataset else 0}")

    def _load_model(self):
        """Loads base model from TIMM with correct architecture."""
        logger.info(f"Loading base model from: {self.config.updated_base_model_path}")
        self.model = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=self.num_classes
        )
        self.model.load_state_dict(torch.load(self.config.updated_base_model_path, map_location=self.device))
        self.model.to(self.device)
        logger.info(f"Model loaded on {self.device}")

    def train(self):
        """Runs supervised + semi-supervised training, then fine-tuning."""
        self._get_dataloaders()
        self._load_model()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(self.config.params_epochs):
            self.model.train()
            total_loss, correct, total = 0, 0, 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.params_epochs} [Supervised]")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                pbar.set_postfix({"Loss": f"{(total_loss/total):.4f}", "Acc": f"{(correct/total):.4f}"})

            train_loss = total_loss / total
            train_acc = correct / total

            # --- semi-supervised phase ---
            if self.unlabeled_loader is not None:
                self._train_unlabeled(optimizer, criterion)

            val_loss, val_acc = self._validate(criterion)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            logger.info(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

        self._save_results(history, phase="ssl")
        self._high_sensitivity_finetune(criterion)

    def _train_unlabeled(self, optimizer, criterion):
        """Trains on unlabeled data using pseudo-labels."""
        self.model.train()
        for inputs, _ in tqdm(self.unlabeled_loader, desc="Unlabeled (SSL)"):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                conf, pseudo_labels = torch.max(probs, dim=1)
                mask = conf > self.pseudo_label_threshold

            if mask.sum() == 0:
                continue

            selected_inputs = inputs[mask]
            selected_labels = pseudo_labels[mask]
            outputs = self.model(selected_inputs)
            ssl_loss = criterion(outputs, selected_labels) * self.consistency_weight

            optimizer.zero_grad()
            ssl_loss.backward()
            optimizer.step()

    def _high_sensitivity_finetune(self, criterion):
        """Fine-tunes the model to improve sensitivity (recall for Positive)."""
        logger.info("Starting high-sensitivity fine-tuning...")
        optimizer = optim.Adam(self.model.parameters(), lr=self.hs_learning_rate)

        for epoch in range(self.hs_epochs):
            self.model.train()
            total_loss, correct, total = 0, 0, 0

            pbar = tqdm(self.train_loader, desc=f"HS Fine-Tune Epoch {epoch+1}/{self.hs_epochs}")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)

                positive_class = 1
                weights = torch.ones_like(labels, dtype=torch.float32, device=self.device)
                weights[labels == positive_class] = 2.0  # emphasize positives

                loss = (nn.functional.cross_entropy(outputs, labels, reduction='none') * weights).mean()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                pbar.set_postfix({"Loss": f"{(total_loss/total):.4f}", "Acc": f"{(correct/total):.4f}"})

            logger.info(f"HS Epoch {epoch+1}: Loss {total_loss/total:.4f}, Acc {correct/total:.4f}")

        # Replace base model with fine-tuned version
        final_model_path = Path(self.config.trained_model_path)
        torch.save(self.model.state_dict(), final_model_path)
        logger.info(f"Fine-tuned model saved and replaced main model at {final_model_path}")

    def _validate(self, criterion):
        """Runs validation step."""
        self.model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return val_loss / total, correct / total

    def _save_results(self, history, phase="ssl"):
        """Saves metrics and plots."""
        out_dir = Path("artifacts/training")
        out_dir.mkdir(parents=True, exist_ok=True)
        hist_path = out_dir / f"{phase}_history.json"

        with open(hist_path, "w") as f:
            json.dump(history, f, indent=4)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history["train_acc"], label="Train Acc")
        plt.plot(history["val_acc"], label="Val Acc")
        plt.legend(); plt.title(f"Accuracy ({phase})")

        plt.subplot(1, 2, 2)
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.legend(); plt.title(f"Loss ({phase})")

        plt.savefig(out_dir / f"{phase}_curves.png")
        logger.info(f"{phase.capitalize()} training curves saved.")