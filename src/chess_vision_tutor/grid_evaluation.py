
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from chess_vision_tutor.chessred_processing import (
    EMPTY_CLASS_ID,
    draw_target_overlay,
    find_corner_record,
    find_image_record,
    load_annotations,
    load_chessred_image,
    warp_board_from_annotations,
)
from chess_vision_tutor.grid_classifier import (
    NUMBER_OF_CLASSES,
    ChessReDGridDataset,
    GridClassifier,
    prepare_image_ids,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "grid_classifier_spatial.pt"
EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "grid_evaluation"

BATCH_SIZE = 8
NUMBER_OF_PREVIEWS = 5


def build_class_names(
    annotations: dict,
) -> list[str]:
    class_names = [
        f"class_{class_id}"
        for class_id in range(NUMBER_OF_CLASSES)
    ]

    for category in annotations.get("categories", []):
        class_id = int(category["id"])

        if 0 <= class_id < NUMBER_OF_CLASSES:
            class_names[class_id] = category.get(
                "name",
                f"class_{class_id}",
            )

    class_names[EMPTY_CLASS_ID] = "empty"

    return class_names


def save_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: list[str],
) -> Path:
    output_path = (
        EVALUATION_OUTPUT_DIR
        / "confusion_matrix.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.writer(output_file)

        writer.writerow(
            ["actual/predicted", *class_names]
        )

        for class_id, row in enumerate(confusion_matrix):
            writer.writerow(
                [class_names[class_id], *row.tolist()]
            )

    return output_path


def save_per_class_metrics(
    confusion_matrix: np.ndarray,
    class_names: list[str],
) -> Path:
    output_path = (
        EVALUATION_OUTPUT_DIR
        / "per_class_accuracy.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.writer(output_file)

        writer.writerow(
            [
                "class_id",
                "class_name",
                "correct",
                "total",
                "accuracy",
            ]
        )

        for class_id in range(NUMBER_OF_CLASSES):
            correct = int(
                confusion_matrix[class_id, class_id]
            )

            total = int(
                confusion_matrix[class_id].sum()
            )

            accuracy = (
                correct / total
                if total
                else 0.0
            )

            writer.writerow(
                [
                    class_id,
                    class_names[class_id],
                    correct,
                    total,
                    accuracy,
                ]
            )

    return output_path


def print_per_class_metrics(
    confusion_matrix: np.ndarray,
    class_names: list[str],
) -> None:
    print()
    print("Per-class accuracy:")

    for class_id in range(NUMBER_OF_CLASSES):
        correct = int(
            confusion_matrix[class_id, class_id]
        )

        total = int(
            confusion_matrix[class_id].sum()
        )

        accuracy = (
            correct / total
            if total
            else 0.0
        )

        print(
            f"{class_id:2d} | "
            f"{class_names[class_id]:20s} | "
            f"{accuracy:7.2%} | "
            f"{correct}/{total}"
        )


def print_top_confusions(
    confusion_matrix: np.ndarray,
    class_names: list[str],
    limit: int = 15,
) -> None:
    confusions: list[tuple[int, int, int]] = []

    for actual_class in range(NUMBER_OF_CLASSES):
        for predicted_class in range(NUMBER_OF_CLASSES):
            if actual_class == predicted_class:
                continue

            count = int(
                confusion_matrix[
                    actual_class,
                    predicted_class,
                ]
            )

            if count > 0:
                confusions.append(
                    (
                        count,
                        actual_class,
                        predicted_class,
                    )
                )

    confusions.sort(reverse=True)

    print()
    print(f"Top {limit} class confusions:")

    for count, actual_class, predicted_class in confusions[:limit]:
        print(
            f"{class_names[actual_class]} "
            f"→ {class_names[predicted_class]}: "
            f"{count}"
        )


def save_preview(
    annotations: dict,
    image_id: int,
    prediction: np.ndarray,
    target: np.ndarray,
    preview_number: int,
) -> tuple[Path, Path]:
    image_record = find_image_record(
        annotations,
        image_id,
    )

    corner_record = find_corner_record(
        annotations,
        image_id,
    )

    source_image = load_chessred_image(
        image_record
    )

    warped_board = warp_board_from_annotations(
        source_image,
        corner_record,
    )

    predicted_overlay = draw_target_overlay(
        warped_board,
        prediction,
    )

    target_overlay = draw_target_overlay(
        warped_board,
        target,
    )

    prediction_path = (
        EVALUATION_OUTPUT_DIR
        / (
            f"preview_{preview_number}_"
            f"image_{image_id}_prediction.jpg"
        )
    )

    target_path = (
        EVALUATION_OUTPUT_DIR
        / (
            f"preview_{preview_number}_"
            f"image_{image_id}_ground_truth.jpg"
        )
    )

    cv2.imwrite(
        str(prediction_path),
        predicted_overlay,
    )

    cv2.imwrite(
        str(target_path),
        target_overlay,
    )

    return prediction_path, target_path


@torch.no_grad()
def evaluate_model() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    annotations = load_annotations()
    class_names = build_class_names(annotations)

    _, validation_ids = prepare_image_ids(
        annotations
    )

    validation_dataset = ChessReDGridDataset(
        annotations,
        validation_ids,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model = GridClassifier().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    confusion_matrix = np.zeros(
        (
            NUMBER_OF_CLASSES,
            NUMBER_OF_CLASSES,
        ),
        dtype=np.int64,
    )

    total_correct = 0
    total_squares = 0

    occupied_correct = 0
    occupied_total = 0

    empty_correct = 0
    empty_total = 0

    exact_boards_correct = 0
    total_boards = 0

    incorrect_previews: list[
        tuple[int, np.ndarray, np.ndarray]
    ] = []

    processed_images = 0

    for images, targets in validation_loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        predictions = logits.argmax(dim=2)

        correct_mask = predictions == targets

        total_correct += correct_mask.sum().item()
        total_squares += targets.numel()

        occupied_mask = targets != EMPTY_CLASS_ID
        empty_mask = targets == EMPTY_CLASS_ID

        occupied_correct += (
            correct_mask & occupied_mask
        ).sum().item()

        occupied_total += occupied_mask.sum().item()

        empty_correct += (
            correct_mask & empty_mask
        ).sum().item()

        empty_total += empty_mask.sum().item()

        board_correct = correct_mask.all(dim=1)

        exact_boards_correct += (
            board_correct.sum().item()
        )

        total_boards += targets.size(0)

        targets_cpu = targets.cpu().numpy()
        predictions_cpu = predictions.cpu().numpy()

        flattened_targets = targets_cpu.reshape(-1)
        flattened_predictions = (
            predictions_cpu.reshape(-1)
        )

        encoded_pairs = (
            flattened_targets * NUMBER_OF_CLASSES
            + flattened_predictions
        )

        batch_confusion = np.bincount(
            encoded_pairs,
            minlength=(
                NUMBER_OF_CLASSES
                * NUMBER_OF_CLASSES
            ),
        ).reshape(
            NUMBER_OF_CLASSES,
            NUMBER_OF_CLASSES,
        )

        confusion_matrix += batch_confusion

        for batch_index in range(
            targets.size(0)
        ):
            if len(incorrect_previews) >= NUMBER_OF_PREVIEWS:
                break

            if board_correct[batch_index].item():
                continue

            image_id = validation_dataset.image_ids[
                processed_images + batch_index
            ]

            prediction_matrix = (
                predictions_cpu[batch_index]
                .reshape(8, 8)
            )

            target_matrix = (
                targets_cpu[batch_index]
                .reshape(8, 8)
            )

            incorrect_previews.append(
                (
                    image_id,
                    prediction_matrix,
                    target_matrix,
                )
            )

        processed_images += targets.size(0)

    overall_accuracy = (
        total_correct / total_squares
    )

    occupied_accuracy = (
        occupied_correct / occupied_total
        if occupied_total
        else 0.0
    )

    empty_accuracy = (
        empty_correct / empty_total
        if empty_total
        else 0.0
    )

    exact_board_accuracy = (
        exact_boards_correct / total_boards
        if total_boards
        else 0.0
    )

    EVALUATION_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    confusion_path = save_confusion_matrix(
        confusion_matrix,
        class_names,
    )

    per_class_path = save_per_class_metrics(
        confusion_matrix,
        class_names,
    )

    print()
    print(f"Device: {device}")
    print(f"Validation boards: {total_boards}")
    print()
    print(
        f"Overall square accuracy: "
        f"{overall_accuracy:.2%}"
    )
    print(
        f"Occupied-square accuracy: "
        f"{occupied_accuracy:.2%}"
    )
    print(
        f"Empty-square accuracy: "
        f"{empty_accuracy:.2%}"
    )
    print(
        f"Exact-board accuracy: "
        f"{exact_board_accuracy:.2%}"
    )
    print(
        f"Exact boards correct: "
        f"{exact_boards_correct}/{total_boards}"
    )

    print_per_class_metrics(
        confusion_matrix,
        class_names,
    )

    print_top_confusions(
        confusion_matrix,
        class_names,
    )

    print()
    print(
        f"Saved confusion matrix: "
        f"{confusion_path}"
    )
    print(
        f"Saved per-class metrics: "
        f"{per_class_path}"
    )

    for preview_number, (
        image_id,
        prediction,
        target,
    ) in enumerate(
        incorrect_previews,
        start=1,
    ):
        prediction_path, target_path = save_preview(
            annotations,
            image_id,
            prediction,
            target,
            preview_number,
        )

        print()
        print(
            f"Preview {preview_number} "
            f"image ID: {image_id}"
        )
        print("Predicted matrix:")
        print(prediction)
        print("Ground-truth matrix:")
        print(target)
        print(
            f"Saved prediction: "
            f"{prediction_path}"
        )
        print(
            f"Saved ground truth: "
            f"{target_path}"
        )


if __name__ == "__main__":
    evaluate_model()