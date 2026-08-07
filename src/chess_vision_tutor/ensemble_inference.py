from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chess_vision_tutor.board_inference import (
    predict_board,
)
from chess_vision_tutor.yolo_board_inference import (
    predict_yolo_board,
)


EMPTY_CLASS = 12
CONFIDENCE_THRESHOLD = 0.70

VERY_HIGH_CONFIDENCE = 0.991
TOP_THREE_SUPPORT_MIN_CONFIDENCE = 0.10

DEFAULT_YOLO_MODEL = Path(
    "runs/detect/outputs/yolo_15_epochs/"
    "chess_piece_detector/weights/best.pt"
)


def compare_predictions(
    model_1_prediction: np.ndarray,
    model_1_confidence: np.ndarray,
    model_1_top_predictions: np.ndarray,
    model_1_top_confidences: np.ndarray,
    model_2_prediction: np.ndarray,
    model_2_confidence: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[dict[str, object]],
    list[dict[str, object]],
]:
    proposed_matrix = model_1_prediction.copy()

    status_matrix = np.zeros(
        (8, 8),
        dtype=np.int64,
    )

    warning_squares: list[
        dict[str, object]
    ] = []

    required_review_squares: list[
        dict[str, object]
    ] = []

    for row in range(8):
        for column in range(8):
            model_1_class = int(
                model_1_prediction[row, column]
            )

            model_2_class = int(
                model_2_prediction[row, column]
            )

            model_1_score = float(
                model_1_confidence[row, column]
            )

            model_2_score = float(
                model_2_confidence[row, column]
            )

            top_classes = [
                int(value)
                for value in model_1_top_predictions[
                    row,
                    column,
                ]
            ]

            top_confidences = [
                float(value)
                for value in model_1_top_confidences[
                    row,
                    column,
                ]
            ]

            model_1_empty = (
                model_1_class == EMPTY_CLASS
            )

            model_2_empty = (
                model_2_class == EMPTY_CLASS
            )

            model_1_very_confident = (
                model_1_score
                >= VERY_HIGH_CONFIDENCE
            )

            model_2_very_confident = (
                not model_2_empty
                and model_2_score
                >= VERY_HIGH_CONFIDENCE
            )

            model_2_top_three_support = 0.0

            if model_2_class in top_classes:
                model_2_index = top_classes.index(
                    model_2_class
                )

                model_2_top_three_support = (
                    top_confidences[
                        model_2_index
                    ]
                )

            model_2_matches_supported_candidate = (
                not model_2_empty
                and model_2_class in top_classes
                and model_2_score
                >= CONFIDENCE_THRESHOLD
                and model_2_top_three_support
                >= TOP_THREE_SUPPORT_MIN_CONFIDENCE
            )

            item = {
                "row": row,
                "column": column,
                "model_1_class": model_1_class,
                "model_1_confidence": model_1_score,
                "model_1_top_classes": top_classes,
                "model_1_top_confidences": (
                    top_confidences
                ),
                "model_2_class": model_2_class,
                "model_2_confidence": model_2_score,
            }

            if model_1_empty and model_2_empty:
                if (
                    model_1_score
                    < CONFIDENCE_THRESHOLD
                ):
                    item["reasons"] = [
                        "low Model 1 empty confidence"
                    ]

                    warning_squares.append(
                        item
                    )

                    status_matrix[
                        row,
                        column,
                    ] = 1

                continue

            if (
                model_1_class
                == model_2_class
            ):
                proposed_matrix[
                    row,
                    column,
                ] = model_1_class

                continue

            if (
                model_1_very_confident
                and model_2_very_confident
            ):
                item["reasons"] = [
                    "both models are extremely "
                    "confident but disagree"
                ]

                required_review_squares.append(
                    item
                )

                status_matrix[
                    row,
                    column,
                ] = 2

                continue

            if model_1_very_confident:
                proposed_matrix[
                    row,
                    column,
                ] = model_1_class

                continue

            if model_2_very_confident:
                proposed_matrix[
                    row,
                    column,
                ] = model_2_class

                continue

            if (
                not model_1_empty
                and model_2_empty
            ):
                proposed_matrix[
                    row,
                    column,
                ] = model_1_class

                if (
                    model_1_score
                    < CONFIDENCE_THRESHOLD
                ):
                    item["reasons"] = [
                        "Model 2 missed the piece "
                        "and Model 1 confidence is low"
                    ]

                    warning_squares.append(
                        item
                    )

                    status_matrix[
                        row,
                        column,
                    ] = 1

                continue

            if model_2_matches_supported_candidate:
                proposed_matrix[
                    row,
                    column,
                ] = model_2_class

                item["reasons"] = [
                    "Model 2 matches a supported "
                    "Model 1 top-three candidate"
                ]

                warning_squares.append(
                    item
                )

                status_matrix[
                    row,
                    column,
                ] = 1

                continue

            if (
                model_1_empty
                and not model_2_empty
            ):
                item["reasons"] = [
                    "Model 2 detected a piece "
                    "without meaningful Model 1 support"
                ]

                required_review_squares.append(
                    item
                )

                status_matrix[
                    row,
                    column,
                ] = 2

                continue

            item["reasons"] = [
                "models support different "
                "occupied-piece classes"
            ]

            required_review_squares.append(
                item
            )

            status_matrix[
                row,
                column,
            ] = 2

    return (
        proposed_matrix,
        status_matrix,
        warning_squares,
        required_review_squares,
    )


def run_ensemble(
    image_path: Path,
    yolo_model_path: Path = DEFAULT_YOLO_MODEL,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[dict[str, object]],
    list[dict[str, object]],
]:
    (
        model_1_prediction,
        model_1_confidence,
        model_1_top_predictions,
        model_1_top_confidences,
    ) = predict_board(
        image_path
    )

    model_2_prediction, model_2_confidence = (
        predict_yolo_board(
            image_path,
            yolo_model_path,
        )
    )

    return compare_predictions(
        model_1_prediction,
        model_1_confidence,
        model_1_top_predictions,
        model_1_top_confidences,
        model_2_prediction,
        model_2_confidence,
    )


def print_ensemble_results(
    image_path: Path,
    yolo_model_path: Path,
) -> None:
    (
        proposed_matrix,
        status_matrix,
        warning_squares,
        required_review_squares,
    ) = run_ensemble(
        image_path,
        yolo_model_path,
    )

    print("\nProposed matrix:")
    print(proposed_matrix)

    print("\nStatus matrix:")
    print(status_matrix)

    print(
        f"\nWarning squares: "
        f"{len(warning_squares)}"
    )

    for item in warning_squares:
        reasons = ", ".join(
            item["reasons"]
        )

        print(
            f"row={item['row']}, "
            f"column={item['column']} | "
            f"Model 1={item['model_1_class']} "
            f"({item['model_1_confidence']:.2%}) | "
            f"Model 1 second="
            f"{item['model_1_top_classes'][1]} "
            f"({item['model_1_top_confidences'][1]:.2%}) | "
            f"Model 2={item['model_2_class']} "
            f"({item['model_2_confidence']:.2%}) | "
            f"{reasons}"
        )

    print(
        f"\nRequired review squares: "
        f"{len(required_review_squares)}"
    )

    for item in required_review_squares:
        reasons = ", ".join(
            item["reasons"]
        )

        top_classes = item[
            "model_1_top_classes"
        ]

        top_confidences = item[
            "model_1_top_confidences"
        ]

        print(
            f"\nrow={item['row']}, "
            f"column={item['column']}"
        )

        print(
            "Model 1 top 3:"
        )

        for rank, (
            piece_class,
            confidence,
        ) in enumerate(
            zip(
                top_classes,
                top_confidences,
            ),
            start=1,
        ):
            print(
                f"  {rank}. class={piece_class} | "
                f"confidence={confidence:.2%}"
            )

        print(
            f"Model 2: "
            f"class={item['model_2_class']} | "
            f"confidence="
            f"{item['model_2_confidence']:.2%}"
        )

        print(
            f"Reason: {reasons}"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image_path",
        type=Path,
    )

    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=DEFAULT_YOLO_MODEL,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    print_ensemble_results(
        arguments.image_path,
        arguments.yolo_model,
    )


if __name__ == "__main__":
    main()