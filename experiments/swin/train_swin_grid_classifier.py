"""Train Model 2: a Swin-T full-board chess grid classifier."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from chess_vision_tutor.chessred_processing import (
    load_annotations,
)
from chess_vision_tutor.grid_classifier import (
    ChessReDGridDataset,
    IMAGE_SIZE,
    NUMBER_OF_CLASSES,
    NUMBER_OF_SQUARES,
    calculate_class_weights,
    calculate_metrics,
    prepare_image_ids,
)
from experiments.swin.swin_grid_classifier import (
    SwinGridClassifier,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"
MODEL_OUTPUT_PATH = (
    MODEL_OUTPUT_DIR
    / "swin_grid_classifier.pt"
)

BATCH_SIZE = 4
LEARNING_RATE = 0.00001
EPOCHS = 3
RANDOM_SEED = 123


def create_empty_metrics() -> dict[str, int]:
    return {
        "total_correct": 0,
        "total_squares": 0,
        "occupied_correct": 0,
        "occupied_total": 0,
        "empty_correct": 0,
        "empty_total": 0,
        "exact_boards_correct": 0,
        "total_boards": 0,
    }


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, int]]:
    model.train()

    total_loss = 0.0
    accumulated_metrics = create_empty_metrics()

    for batch_number, (images, targets) in enumerate(
        data_loader,
        start=1,
    ):
        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad()

        logits = model(images)

        loss = loss_function(
            logits.reshape(
                -1,
                NUMBER_OF_CLASSES,
            ),
            targets.reshape(-1),
        )

        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=2)

        batch_metrics = calculate_metrics(
            predictions,
            targets,
        )

        total_loss += (
            loss.item()
            * images.size(0)
        )

        for metric_name, metric_value in batch_metrics.items():
            accumulated_metrics[metric_name] += metric_value

        if batch_number % 25 == 0:
            print(
                f"  Batch {batch_number}/"
                f"{len(data_loader)} | "
                f"loss: {loss.item():.4f}"
            )

    average_loss = (
        total_loss
        / len(data_loader.dataset)
    )

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
    accumulated_metrics = create_empty_metrics()

    for images, targets in data_loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        logits = model(images)

        loss = loss_function(
            logits.reshape(
                -1,
                NUMBER_OF_CLASSES,
            ),
            targets.reshape(-1),
        )

        predictions = logits.argmax(dim=2)

        batch_metrics = calculate_metrics(
            predictions,
            targets,
        )

        total_loss += (
            loss.item()
            * images.size(0)
        )

        for metric_name, metric_value in batch_metrics.items():
            accumulated_metrics[metric_name] += metric_value

    average_loss = (
        total_loss
        / len(data_loader.dataset)
    )

    return average_loss, accumulated_metrics


def calculate_accuracy(
    correct: int,
    total: int,
) -> float:
    if total == 0:
        return 0.0

    return correct / total


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    validation_metrics: dict[str, int],
) -> None:
    MODEL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "number_of_classes": NUMBER_OF_CLASSES,
            "number_of_squares": NUMBER_OF_SQUARES,
            "image_size": IMAGE_SIZE,
            "architecture": "swin_t",
            "validation_metrics": validation_metrics,
        },
        MODEL_OUTPUT_PATH,
    )

    print(
        f"Saved best Model 2 checkpoint: "
        f"{MODEL_OUTPUT_PATH}"
    )


def train_swin_model() -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            RANDOM_SEED
        )

    annotations = load_annotations()

    train_ids, validation_ids = prepare_image_ids(
        annotations
    )

    train_dataset = ChessReDGridDataset(
        annotations,
        train_ids,
    )

    validation_dataset = ChessReDGridDataset(
        annotations,
        validation_ids,
    )

    if len(train_dataset) == 0:
        raise RuntimeError(
            "No training images were found."
        )

    if len(validation_dataset) == 0:
        raise RuntimeError(
            "No validation images were found."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    model = SwinGridClassifier(
        use_pretrained_weights=True,
    ).to(device)

    class_weights = calculate_class_weights(
        annotations,
        train_ids,
    ).to(device)

    loss_function = nn.CrossEntropyLoss(
        weight=class_weights,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    best_validation_accuracy = 0.0

    print()
    print("Model 2: Swin Transformer Tiny")
    print(f"Device: {device}")
    print(
        f"Training images: "
        f"{len(train_dataset)}"
    )
    print(
        f"Validation images: "
        f"{len(validation_dataset)}"
    )
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            device,
        )

        (
            validation_loss,
            validation_metrics,
        ) = evaluate(
            model,
            validation_loader,
            loss_function,
            device,
        )

        train_accuracy = calculate_accuracy(
            train_metrics["total_correct"],
            train_metrics["total_squares"],
        )

        validation_accuracy = calculate_accuracy(
            validation_metrics[
                "total_correct"
            ],
            validation_metrics[
                "total_squares"
            ],
        )

        occupied_accuracy = calculate_accuracy(
            validation_metrics[
                "occupied_correct"
            ],
            validation_metrics[
                "occupied_total"
            ],
        )

        empty_accuracy = calculate_accuracy(
            validation_metrics[
                "empty_correct"
            ],
            validation_metrics[
                "empty_total"
            ],
        )

        exact_board_accuracy = calculate_accuracy(
            validation_metrics[
                "exact_boards_correct"
            ],
            validation_metrics[
                "total_boards"
            ],
        )

        print(
            f"train loss: {train_loss:.4f} | "
            f"train accuracy: "
            f"{train_accuracy:.2%}"
        )

        print(
            f"validation loss: "
            f"{validation_loss:.4f} | "
            f"square accuracy: "
            f"{validation_accuracy:.2%} | "
            f"occupied: "
            f"{occupied_accuracy:.2%} | "
            f"empty: "
            f"{empty_accuracy:.2%} | "
            f"exact boards: "
            f"{exact_board_accuracy:.2%}"
        )

        if (
            validation_accuracy
            > best_validation_accuracy
        ):
            best_validation_accuracy = (
                validation_accuracy
            )

            save_checkpoint(
                model,
                epoch,
                validation_metrics,
            )

        print()

    print(
        "Best validation square accuracy: "
        f"{best_validation_accuracy:.2%}"
    )

    print(
        f"Model 2 checkpoint: "
        f"{MODEL_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    train_swin_model()