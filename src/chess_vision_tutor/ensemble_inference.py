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

YOLO_CONFIDENCE_THRESHOLD = 0.70
RESNET18_VERY_HIGH_CONFIDENCE = 0.991
RESNET18_TOP_THREE_SUPPORT_MIN_CONFIDENCE = 0.10

DEFAULT_YOLO_MODEL = Path(
    "runs/detect/outputs/yolo_15_epochs/"
    "chess_piece_detector/weights/best.pt"
)


def compare_predictions(
    resnet18_prediction: np.ndarray,
    resnet18_confidence: np.ndarray,
    resnet18_top_predictions: np.ndarray,
    resnet18_top_confidences: np.ndarray,
    yolo_prediction: np.ndarray,
    yolo_confidence: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Combine YOLO and ResNet18 predictions.

    YOLO is the primary recognition model.

    ResNet18 is retained as a secondary support and disagreement
    signal. It does not automatically override YOLO predictions.

    Status values:

    0 = accepted
    1 = warning
    2 = requires human review
    """

    # YOLO is the primary prediction source.
    proposed_matrix = yolo_prediction.copy()

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
            resnet18_class = int(
                resnet18_prediction[
                    row,
                    column,
                ]
            )

            yolo_class = int(
                yolo_prediction[
                    row,
                    column,
                ]
            )

            resnet18_score = float(
                resnet18_confidence[
                    row,
                    column,
                ]
            )

            yolo_score = float(
                yolo_confidence[
                    row,
                    column,
                ]
            )

            resnet18_top_classes = [
                int(value)
                for value in resnet18_top_predictions[
                    row,
                    column,
                ]
            ]

            resnet18_top_confidence_values = [
                float(value)
                for value in resnet18_top_confidences[
                    row,
                    column,
                ]
            ]

            resnet18_empty = (
                resnet18_class
                == EMPTY_CLASS
            )

            yolo_empty = (
                yolo_class
                == EMPTY_CLASS
            )

            resnet18_very_confident = (
                resnet18_score
                >= RESNET18_VERY_HIGH_CONFIDENCE
            )

            yolo_confident = (
                not yolo_empty
                and yolo_score
                >= YOLO_CONFIDENCE_THRESHOLD
            )

            yolo_resnet18_support = 0.0

            if (
                yolo_class
                in resnet18_top_classes
            ):
                support_index = (
                    resnet18_top_classes.index(
                        yolo_class
                    )
                )

                yolo_resnet18_support = (
                    resnet18_top_confidence_values[
                        support_index
                    ]
                )

            resnet18_supports_yolo = (
                yolo_class
                in resnet18_top_classes
                and yolo_resnet18_support
                >= RESNET18_TOP_THREE_SUPPORT_MIN_CONFIDENCE
            )

            item = {
                "row":
                    row,
                "column":
                    column,
                "resnet18_class":
                    resnet18_class,
                "resnet18_confidence":
                    resnet18_score,
                "resnet18_top_classes":
                    resnet18_top_classes,
                "resnet18_top_confidences":
                    resnet18_top_confidence_values,
                "yolo_class":
                    yolo_class,
                "yolo_confidence":
                    yolo_score,
            }

            if (
                resnet18_class
                == yolo_class
            ):
                continue

            
            if yolo_confident:
                if resnet18_supports_yolo:
                    # ResNet18 disagrees, but still gives meaningful support to YOLO's class.
                    continue

                if resnet18_very_confident:
                    item["reasons"] = [
                        "YOLO is confident, but "
                        "ResNet18 strongly disagrees"
                    ]

                    warning_squares.append(
                        item
                    )

                    status_matrix[
                        row,
                        column,
                    ] = 1

                continue

            # YOLO detected a piece, but with lower confidence.
            if not yolo_empty:
                if resnet18_supports_yolo:
                    item["reasons"] = [
                        "YOLO confidence is below "
                        "the primary threshold, but "
                        "ResNet18 supports the class"
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
                        "YOLO detected a low-confidence "
                        "piece without meaningful "
                        "ResNet18 support"
                    ]

                    required_review_squares.append(
                        item
                    )

                    status_matrix[
                        row,
                        column,
                    ] = 2

                continue

            # YOLO predicts empty while ResNet18 detects a piece.
            # Surface the disagreement for review instead of overriding YOLO.
            if (
                yolo_empty
                and not resnet18_empty
            ):
                if resnet18_very_confident:
                    item["reasons"] = [
                        "YOLO detected no piece, but "
                        "ResNet18 strongly predicts one"
                    ]

                    required_review_squares.append(
                        item
                    )

                    status_matrix[
                        row,
                        column,
                    ] = 2

                else:
                    item["reasons"] = [
                        "YOLO detected no piece while "
                        "ResNet18 predicts a piece"
                    ]

                    warning_squares.append(
                        item
                    )

                    status_matrix[
                        row,
                        column,
                    ] = 1

                continue

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
        yolo_model_path,
    )

    return compare_predictions(
        resnet18_prediction,
        resnet18_confidence,
        resnet18_top_predictions,
        resnet18_top_confidences,
        yolo_prediction,
        yolo_confidence,
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

    print(
        "\nProposed matrix:"
    )
    print(
        proposed_matrix
    )

    print(
        "\nStatus matrix:"
    )
    print(
        status_matrix
    )

    print(
        f"\nWarning squares: "
        f"{len(warning_squares)}"
    )

    for item in warning_squares:
        reasons = ", ".join(
            item["reasons"]
        )

        print(
            f"\nrow={item['row']}, "
            f"column={item['column']}"
        )

        print(
            "ResNet18 top 3:"
        )

        for rank, (
            piece_class,
            confidence,
        ) in enumerate(
            zip(
                item[
                    "resnet18_top_classes"
                ],
                item[
                    "resnet18_top_confidences"
                ],
            ),
            start=1,
        ):
            print(
                f"  {rank}. "
                f"class={piece_class} | "
                f"confidence="
                f"{confidence:.2%}"
            )

        print(
            "YOLO11: "
            f"class={item['yolo_class']} | "
            f"confidence="
            f"{item['yolo_confidence']:.2%}"
        )

        print(
            f"Reason: {reasons}"
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
            f"\nrow={item['row']}, "
            f"column={item['column']}"
        )

        print(
            "ResNet18 top 3:"
        )

        for rank, (
            piece_class,
            confidence,
        ) in enumerate(
            zip(
                item[
                    "resnet18_top_classes"
                ],
                item[
                    "resnet18_top_confidences"
                ],
            ),
            start=1,
        ):
            print(
                f"  {rank}. "
                f"class={piece_class} | "
                f"confidence="
                f"{confidence:.2%}"
            )

        print(
            "YOLO11: "
            f"class={item['yolo_class']} | "
            f"confidence="
            f"{item['yolo_confidence']:.2%}"
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