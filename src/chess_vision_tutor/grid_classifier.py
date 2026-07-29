"""Train an initial full-board 64-square chess classifier.

The model receives one normalized full-board image and predicts one of
13 classes for each of the 64 chessboard squares.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchgen import model
from torchvision import models, transforms

from chess_vision_tutor.chessred_processing import (
    BOARD_SIZE,
    CHESSRED_ROOT,
    build_target_matrix,
    find_corner_record,
    find_image_record,
    load_annotations,
    load_chessred_image,
    warp_board_from_annotations,
)



PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"

NUMBER_OF_SQUARES = 64
NUMBER_OF_CLASSES = 13

IMAGE_SIZE = 256

BATCH_SIZE = 8
LEARNING_RATE = 0.0001
EPOCHS = 10

TRAIN_SAMPLE_LIMIT: int | None = None
VALIDATION_SAMPLE_LIMIT: int | None = None

RANDOM_SEED = 42


class ChessReDGridDataset(Dataset):
    """Load warped ChessReD boards with 64 square-class targets."""

    def __init__(
        self,
        annotations: dict[str, Any],
        image_ids: list[int],
    ) -> None:
        self.annotations = annotations

        self.image_ids = [
            image_id
            for image_id in image_ids
            if self._image_exists(image_id)
        ]

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _image_exists(self, image_id: int) -> bool:
        """Check whether an image from the JSON exists in ChessReD2K."""

        try:
            image_record = find_image_record(
                self.annotations,
                image_id,
            )
        except ValueError:
            return False

        image_path = CHESSRED_ROOT / image_record["path"]

        return image_path.exists()

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_id = self.image_ids[index]

        image_record = find_image_record(
            self.annotations,
            image_id,
        )

        corner_record = find_corner_record(
            self.annotations,
            image_id,
        )

        source_image = load_chessred_image(image_record)

        warped_board = warp_board_from_annotations(
            source_image,
            corner_record,
            board_size=BOARD_SIZE,
        )

        # OpenCV uses BGR, while torchvision expects RGB.
        warped_board_rgb = cv2.cvtColor(
            warped_board,
            cv2.COLOR_BGR2RGB,
        )

        image_tensor = self.transform(warped_board_rgb)

        target_matrix = build_target_matrix(
            self.annotations,
            image_id,
            verbose=False,
        )

        target_tensor = torch.from_numpy(
            target_matrix.reshape(NUMBER_OF_SQUARES)
        ).long()

        return image_tensor, target_tensor


class GridClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT,
        )

        self.feature_extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.classifier = nn.Conv2d(
            in_channels=512,
            out_channels=NUMBER_OF_CLASSES,
            kernel_size=1,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(images)

        logits = self.classifier(features)

        logits = logits.permute(
            0,
            2,
            3,
            1,
        )

        return logits.reshape(
            -1,
            NUMBER_OF_SQUARES,
            NUMBER_OF_CLASSES,
        )

def calculate_square_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Calculate accuracy across all board squares."""

    predictions = logits.argmax(dim=2)

    correct = (predictions == targets).sum().item()
    total = targets.numel()

    return correct / total

def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, int]:
    correct_mask = predictions == targets

    occupied_mask = targets != 12
    empty_mask = targets == 12

    exact_board_mask = correct_mask.all(dim=1)

    return {
        "total_correct": int(correct_mask.sum().item()),
        "total_squares": int(targets.numel()),
        "occupied_correct": int(
            (correct_mask & occupied_mask).sum().item()
        ),
        "occupied_total": int(
            occupied_mask.sum().item()
        ),
        "empty_correct": int(
            (correct_mask & empty_mask).sum().item()
        ),
        "empty_total": int(
            empty_mask.sum().item()
        ),
        "exact_boards_correct": int(
            exact_board_mask.sum().item()
        ),
        "total_boards": int(
            targets.size(0)
        ),
    }

def calculate_class_weights(
    annotations: dict[str, Any],
    image_ids: list[int],
) -> torch.Tensor:
    class_counts = np.zeros(
        NUMBER_OF_CLASSES,
        dtype=np.int64,
    )

    for image_id in image_ids:
        target_matrix = build_target_matrix(
            annotations,
            image_id,
            verbose=False,
        )

        flattened_targets = target_matrix.reshape(-1)

        for class_id in flattened_targets:
            class_counts[int(class_id)] += 1

    class_counts = np.maximum(
        class_counts,
        1,
    )

    total_count = class_counts.sum()

    class_weights = total_count / (
        NUMBER_OF_CLASSES * class_counts
    )

    class_weights = class_weights / class_weights.mean()

    print("Class counts:")
    print(class_counts)
    print("Class weights:")
    print(class_weights)

    return torch.tensor(
        class_weights,
        dtype=torch.float32,
    )

def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, int]]:
    model.train()

    total_loss = 0.0

    accumulated_metrics = {
        "total_correct": 0,
        "total_squares": 0,
        "occupied_correct": 0,
        "occupied_total": 0,
        "empty_correct": 0,
        "empty_total": 0,
        "exact_boards_correct": 0,
        "total_boards": 0,
    }

    for images, targets in data_loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = loss_function(
            logits.reshape(-1, NUMBER_OF_CLASSES),
            targets.reshape(-1),
        )

        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=2)

        batch_metrics = calculate_metrics(
            predictions,
            targets,
        )

        total_loss += loss.item() * images.size(0)

        for metric_name, metric_value in batch_metrics.items():
            accumulated_metrics[metric_name] += metric_value

    average_loss = total_loss / len(data_loader.dataset)

    return average_loss, accumulated_metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, int]]:
    model.eval()

    total_loss = 0.0

    accumulated_metrics = {
        "total_correct": 0,
        "total_squares": 0,
        "occupied_correct": 0,
        "occupied_total": 0,
        "empty_correct": 0,
        "empty_total": 0,
        "exact_boards_correct": 0,
        "total_boards": 0,
    }

    for images, targets in data_loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)

        loss = loss_function(
            logits.reshape(-1, NUMBER_OF_CLASSES),
            targets.reshape(-1),
        )

        predictions = logits.argmax(dim=2)

        batch_metrics = calculate_metrics(
            predictions,
            targets,
        )

        total_loss += loss.item() * images.size(0)

        for metric_name, metric_value in batch_metrics.items():
            accumulated_metrics[metric_name] += metric_value

    average_loss = total_loss / len(data_loader.dataset)

    return average_loss, accumulated_metrics


def prepare_image_ids(
    annotations: dict[str, Any],
) -> tuple[list[int], list[int]]:
    """Select locally available images from the official ChessReD splits."""

    train_ids = list(
        annotations["splits"]["train"]["image_ids"]
    )

    validation_split = annotations["splits"].get(
        "val",
        annotations["splits"].get("validation"),
    )

    if validation_split is None:
        raise KeyError(
            "No validation split named 'val' or 'validation' was found."
        )

    validation_ids = list(validation_split["image_ids"])

    # Find which annotation IDs actually exist in ChessReD2K.
    available_image_ids = {
        image_record["id"]
        for image_record in annotations["images"]
        if (CHESSRED_ROOT / image_record["path"]).exists()
    }

    # Filter first, then shuffle and limit.
    train_ids = [
        image_id
        for image_id in train_ids
        if image_id in available_image_ids
    ]

    validation_ids = [
        image_id
        for image_id in validation_ids
        if image_id in available_image_ids
    ]

    random.Random(RANDOM_SEED).shuffle(train_ids)
    random.Random(RANDOM_SEED).shuffle(validation_ids)

    print(f"Available official training images: {len(train_ids)}")
    print(f"Available official validation images: {len(validation_ids)}")

    selected_train_ids = (
        train_ids
        if TRAIN_SAMPLE_LIMIT is None
        else train_ids[:TRAIN_SAMPLE_LIMIT]
    )

    selected_validation_ids = (
        validation_ids
        if VALIDATION_SAMPLE_LIMIT is None
        else validation_ids[:VALIDATION_SAMPLE_LIMIT]
    )

    return selected_train_ids, selected_validation_ids

def train_smoke_test() -> None:
    """Train a small model to verify the full learning pipeline."""

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    annotations = load_annotations()

    train_ids, validation_ids = prepare_image_ids(annotations)

    train_dataset = ChessReDGridDataset(
        annotations,
        train_ids,
    )

    validation_dataset = ChessReDGridDataset(
        annotations,
        validation_ids,
    )

    if not train_dataset:
        raise RuntimeError(
            "No training images were found in the ChessReD2K folder."
        )

    if not validation_dataset:
        raise RuntimeError(
            "No validation images were found in the ChessReD2K folder."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = GridClassifier().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    class_weights = calculate_class_weights(
        annotations,
        train_ids,
    ).to(device)

    loss_function = nn.CrossEntropyLoss(
        weight=class_weights,
    )

    print(f"Device: {device}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(validation_dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            device,
        )

        validation_loss, validation_metrics = evaluate(
            model,
            validation_loader,
            loss_function,
            device,
        )

        train_square_accuracy = (
            train_metrics["total_correct"]
            / train_metrics["total_squares"]
        )

        validation_square_accuracy = (
            validation_metrics["total_correct"]
            / validation_metrics["total_squares"]
        )

        validation_occupied_accuracy = (
            validation_metrics["occupied_correct"]
            / validation_metrics["occupied_total"]
        )

        validation_empty_accuracy = (
            validation_metrics["empty_correct"]
            / validation_metrics["empty_total"]
        )

        validation_exact_board_accuracy = (
            validation_metrics["exact_boards_correct"]
            / validation_metrics["total_boards"]
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"train square accuracy: {train_square_accuracy:.2%} | "
            f"validation loss: {validation_loss:.4f} | "
            f"validation square accuracy: {validation_square_accuracy:.2%} | "
            f"occupied: {validation_occupied_accuracy:.2%} | "
            f"empty: {validation_empty_accuracy:.2%} | "
            f"exact boards: {validation_exact_board_accuracy:.2%}"
        )

    MODEL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = MODEL_OUTPUT_DIR / "grid_classifier_spatial.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "number_of_classes": NUMBER_OF_CLASSES,
            "number_of_squares": NUMBER_OF_SQUARES,
            "image_size": IMAGE_SIZE,
        },
        model_path,
    )

    print()
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    train_smoke_test()