from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from chess_vision_tutor.board_inference import (
    predict_board,
)
from chess_vision_tutor.ensemble_inference import (
    DEFAULT_YOLO_MODEL,
    compare_predictions,
)
from chess_vision_tutor.yolo_board_inference import (
    predict_yolo_board,
)


BOARD_SIZE = 8
EMPTY_CLASS = 12

PIECE_TO_CLASS = {
    "P": 0,
    "R": 1,
    "N": 2,
    "B": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "r": 7,
    "n": 8,
    "b": 9,
    "q": 10,
    "k": 11,
}


STRESS_TEST_ROOT = Path(
    "data/stress_test"
)

IMAGES_DIR = (
    STRESS_TEST_ROOT
    / "images"
)

GROUND_TRUTH_PATH = (
    STRESS_TEST_ROOT
    / "ground_truth.json"
)

OUTPUT_PATH = Path(
    "outputs/evaluation/"
    "stress_test_results.csv"
)


def fen_to_matrix(
    fen: str,
) -> np.ndarray:
    ranks = fen.strip().split("/")

    if len(ranks) != BOARD_SIZE:
        raise ValueError(
            "FEN must contain exactly 8 ranks."
        )

    matrix = np.full(
        (BOARD_SIZE, BOARD_SIZE),
        EMPTY_CLASS,
        dtype=np.int64,
    )

    for row, rank in enumerate(ranks):
        column = 0

        for token in rank:
            if token.isdigit():
                column += int(token)
                continue

            if token not in PIECE_TO_CLASS:
                raise ValueError(
                    f"Unsupported FEN token: {token}"
                )

            if column >= BOARD_SIZE:
                raise ValueError(
                    "FEN rank contains more than 8 squares."
                )

            matrix[
                row,
                column,
            ] = PIECE_TO_CLASS[
                token
            ]

            column += 1

        if column != BOARD_SIZE:
            raise ValueError(
                "Each FEN rank must describe exactly 8 squares."
            )

    return matrix


def rotation_candidates(
    matrix: np.ndarray,
) -> list[
    tuple[int, np.ndarray]
]:
    return [
        (
            0,
            matrix,
        ),
        (
            90,
            np.rot90(
                matrix,
                k=3,
            ),
        ),
        (
            180,
            np.rot90(
                matrix,
                k=2,
            ),
        ),
        (
            270,
            np.rot90(
                matrix,
                k=1,
            ),
        ),
    ]


def best_orientation_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[
    float,
    bool,
    int,
]:
    best_accuracy = -1.0
    best_exact = False
    best_rotation = 0

    for (
        rotation,
        rotated,
    ) in rotation_candidates(
        prediction
    ):
        accuracy = float(
            np.mean(
                rotated
                == ground_truth
            )
        )

        exact = bool(
            np.array_equal(
                rotated,
                ground_truth,
            )
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_exact = exact
            best_rotation = rotation

    return (
        best_accuracy,
        best_exact,
        best_rotation,
    )


def load_ground_truth() -> dict[
    str,
    str,
]:
    with GROUND_TRUTH_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(
            file
        )

    return {
        str(file_name):
            str(fen)
        for file_name, fen
        in data.items()
    }


def evaluate_image(
    image_path: Path,
    ground_truth_fen: str,
) -> dict[str, object]:
    ground_truth = fen_to_matrix(
        ground_truth_fen
    )

    started = time.perf_counter()

    (
        resnet18_prediction,
        resnet18_confidence,
        resnet18_top_predictions,
        resnet18_top_confidences,
    ) = predict_board(
        image_path
    )

    (
        yolo_prediction,
        yolo_confidence,
    ) = predict_yolo_board(
        image_path,
        DEFAULT_YOLO_MODEL,
    )

    (
        ensemble_prediction,
        _status_matrix,
        warning_squares,
        required_review_squares,
    ) = compare_predictions(
        resnet18_prediction,
        resnet18_confidence,
        resnet18_top_predictions,
        resnet18_top_confidences,
        yolo_prediction,
        yolo_confidence,
    )

    elapsed_seconds = (
        time.perf_counter()
        - started
    )

    (
        resnet18_accuracy,
        resnet18_exact,
        resnet18_rotation,
    ) = best_orientation_metrics(
        resnet18_prediction,
        ground_truth,
    )

    (
        yolo_accuracy,
        yolo_exact,
        yolo_rotation,
    ) = best_orientation_metrics(
        yolo_prediction,
        ground_truth,
    )

    (
        ensemble_accuracy,
        ensemble_exact,
        ensemble_rotation,
    ) = best_orientation_metrics(
        ensemble_prediction,
        ground_truth,
    )

    return {
        "processing_success":
            True,
        "resnet18_accuracy":
            resnet18_accuracy,
        "resnet18_exact":
            resnet18_exact,
        "resnet18_rotation":
            resnet18_rotation,
        "yolo_accuracy":
            yolo_accuracy,
        "yolo_exact":
            yolo_exact,
        "yolo_rotation":
            yolo_rotation,
        "ensemble_accuracy":
            ensemble_accuracy,
        "ensemble_exact":
            ensemble_exact,
        "ensemble_rotation":
            ensemble_rotation,
        "warning_count":
            len(
                warning_squares
            ),
        "required_review_count":
            len(
                required_review_squares
            ),
        "inference_seconds":
            elapsed_seconds,
    }


def print_summary(
    rows: list[
        dict[str, object]
    ],
) -> None:
    total = len(
        rows
    )

    successful_rows = [
        row
        for row in rows
        if bool(
            row.get(
                "processing_success",
                False,
            )
        )
    ]

    success_count = len(
        successful_rows
    )

    print(
        "\n"
        + "=" * 64
    )

    print(
        "Stress Test Evaluation"
    )

    print(
        "=" * 64
    )

    print(
        f"Total images: {total}"
    )

    print(
        "Processing success: "
        f"{success_count}/{total} "
        f"({success_count / total:.2%})"
    )

    if not successful_rows:
        return

    def mean(
        key: str,
    ) -> float:
        return sum(
            float(
                row[key]
            )
            for row
            in successful_rows
        ) / success_count

    def rate(
        key: str,
    ) -> float:
        return sum(
            bool(
                row[key]
            )
            for row
            in successful_rows
        ) / success_count

    print(
        "\nSquare accuracy"
    )

    print(
        "  ResNet18: "
        f"{mean('resnet18_accuracy'):.2%}"
    )

    print(
        "  YOLO11: "
        f"{mean('yolo_accuracy'):.2%}"
    )

    print(
        "  Final ensemble: "
        f"{mean('ensemble_accuracy'):.2%}"
    )

    print(
        "\nExact full-board accuracy"
    )

    print(
        "  ResNet18: "
        f"{rate('resnet18_exact'):.2%}"
    )

    print(
        "  YOLO11: "
        f"{rate('yolo_exact'):.2%}"
    )

    print(
        "  Final ensemble: "
        f"{rate('ensemble_exact'):.2%}"
    )

    print(
        "\nHuman-review load"
    )

    print(
        "  Average warnings: "
        f"{mean('warning_count'):.2f}"
    )

    print(
        "  Average required reviews: "
        f"{mean('required_review_count'):.2f}"
    )

    print(
        "\nAverage inference time"
    )

    print(
        f"  {mean('inference_seconds'):.2f} seconds/image"
    )


def main() -> None:
    ground_truth = (
        load_ground_truth()
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results: list[
        dict[str, object]
    ] = []

    for (
        index,
        (
            file_name,
            fen,
        ),
    ) in enumerate(
        ground_truth.items(),
        start=1,
    ):
        image_path = (
            IMAGES_DIR
            / file_name
        )

        print(
            f"[{index}/"
            f"{len(ground_truth)}] "
            f"{file_name}"
        )

        row: dict[
            str,
            object,
        ] = {
            "file_name":
                file_name,
            "image_path":
                str(
                    image_path
                ),
            "ground_truth_fen":
                fen,
        }

        try:
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            if not fen.strip():
                raise ValueError(
                    "Ground-truth FEN is empty."
                )

            metrics = (
                evaluate_image(
                    image_path,
                    fen,
                )
            )

            row.update(
                metrics
            )

            print(
                "  ResNet18="
                f"{float(metrics['resnet18_accuracy']):.2%}"
                " | YOLO11="
                f"{float(metrics['yolo_accuracy']):.2%}"
                " | ensemble="
                f"{float(metrics['ensemble_accuracy']):.2%}"
                " | warnings="
                f"{metrics['warning_count']}"
                " | review="
                f"{metrics['required_review_count']}"
            )

        except Exception as error:
            row[
                "processing_success"
            ] = False

            row[
                "error"
            ] = (
                f"{type(error).__name__}: "
                f"{error}"
            )

            print(
                f"  ERROR: {row['error']}"
            )

        results.append(
            row
        )

    fieldnames = [
        "file_name",
        "image_path",
        "ground_truth_fen",
        "processing_success",
        "resnet18_accuracy",
        "resnet18_exact",
        "resnet18_rotation",
        "yolo_accuracy",
        "yolo_exact",
        "yolo_rotation",
        "ensemble_accuracy",
        "ensemble_exact",
        "ensemble_rotation",
        "warning_count",
        "required_review_count",
        "inference_seconds",
        "error",
    ]

    present_fields = {
        key
        for row in results
        for key in row
    }

    final_fieldnames = [
        field
        for field in fieldnames
        if field in present_fields
    ]

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                final_fieldnames
            ),
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print_summary(
        results
    )

    print(
        "\nSaved detailed results to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()