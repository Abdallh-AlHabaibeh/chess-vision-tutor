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

SECOND_CHOICE_MIN_CONFIDENCE = 0.20
TOP_TWO_MAX_CONFIDENCE_GAP = 0.25

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
    accepted_matrix = model_1_prediction.copy()

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

            model_1_second_class = int(
                model_1_top_predictions[
                    row,
                    column,
                    1,
                ]
            )

            model_1_second_score = float(
                model_1_top_confidences[
                    row,
                    column,
                    1,
                ]
            )

            top_two_gap = (
                model_1_score
                - model_1_second_score
            )

            model_1_empty = (
                model_1_class == EMPTY_CLASS
            )

            model_2_empty = (
                model_2_class == EMPTY_CLASS
            )

            model_2_matches_strong_second_choice = (
                not model_2_empty
                and model_2_class
                == model_1_second_class
                and model_2_score
                >= CONFIDENCE_THRESHOLD
                and model_1_second_score
                >= SECOND_CHOICE_MIN_CONFIDENCE
                and top_two_gap
                <= TOP_TWO_MAX_CONFIDENCE_GAP
            )

            item = {
                "row": row,
                "column": column,
                "model_1_class": model_1_class,
                "model_1_confidence": model_1_score,
                "model_1_top_classes": [
                    int(value)
                    for value in model_1_top_predictions[
                        row,
                        column,
                    ]
                ],
                "model_1_top_confidences": [
                    float(value)
                    for value in model_1_top_confidences[
                        row,
                        column,
                    ]
                ],
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
                not model_1_empty
                and not model_2_empty
            ):
                if model_1_class != model_2_class:
                    if (
                        model_2_matches_strong_second_choice
                    ):
                        item["reasons"] = [
                            "Model 2 matches strong "
                            "Model 1 second choice"
                        ]

                        warning_squares.append(
                            item
                        )

                        status_matrix[
                            row,
                            column,
                        ] = 1

                    else:
                        item["reasons"] = [
                            "models disagree"
                        ]

                        required_review_squares.append(
                            item
                        )

                        status_matrix[
                            row,
                            column,
                        ] = 2

                    continue

                low_confidence_reasons: list[str] = []

                if (
                    model_1_score
                    < CONFIDENCE_THRESHOLD
                ):
                    low_confidence_reasons.append(
                        "low Model 1 confidence"
                    )

                if (
                    model_2_score
                    < CONFIDENCE_THRESHOLD
                ):
                    low_confidence_reasons.append(
                        "low Model 2 confidence"
                    )

                if low_confidence_reasons:
                    item["reasons"] = (
                        low_confidence_reasons
                    )

                    warning_squares.append(
                        item
                    )

                    status_matrix[
                        row,
                        column,
                    ] = 1

                continue

            if (
                not model_1_empty
                and model_2_empty
            ):
                item["reasons"] = [
                    "Model 1 detected a piece but "
                    "Model 2 did not"
                ]

                required_review_squares.append(
                    item
                )

                status_matrix[
                    row,
                    column,
                ] = 2

                continue

            if model_2_matches_strong_second_choice:
                item["reasons"] = [
                    "Model 2 matches strong "
                    "Model 1 second choice"
                ]

                warning_squares.append(
                    item
                )

                status_matrix[
                    row,
                    column,
                ] = 1

                continue

            item["reasons"] = [
                "Model 2 detected a piece but "
                "Model 1 predicted empty"
            ]

            required_review_squares.append(
                item
            )

            status_matrix[
                row,
                column,
            ] = 2

    return (
        accepted_matrix,
        status_matrix,
        warning_squares,
        required_review_squares,
    )


def run_ensemble(
    image_path: Path,
    yolo_model_path: Path,
) -> None:
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

    (
        accepted_matrix,
        status_matrix,
        warning_squares,
        required_review_squares,
    ) = compare_predictions(
        model_1_prediction,
        model_1_confidence,
        model_1_top_predictions,
        model_1_top_confidences,
        model_2_prediction,
        model_2_confidence,
    )

    print("\nModel 1 matrix:")
    print(model_1_prediction)

    print("\nModel 2 matrix:")
    print(model_2_prediction)

    print("\nCurrent accepted matrix:")
    print(accepted_matrix)

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

    run_ensemble(
        arguments.image_path,
        arguments.yolo_model,
    )


if __name__ == "__main__":
    main()