from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from chess_vision_tutor.board_inference import predict_board
from chess_vision_tutor.chess_logic import validate_position
from chess_vision_tutor.ensemble_inference import (
    DEFAULT_YOLO_MODEL,
    compare_predictions,
)
from chess_vision_tutor.yolo_board_inference import (
    predict_yolo_board,
)


EMPTY_CLASS = 12
BOARD_SIZE = 8


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the ResNet18 grid classifier, "
            "YOLO11 detector, and final ensemble "
            "on the official ChessReD2K test split."
        )
    )

    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(
            "data/dataset/chessred2k/annotations.json"
        ),
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "data/dataset/chessred2k"
        ),
    )

    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=DEFAULT_YOLO_MODEL,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help=(
            "Number of official ChessReD2K test "
            "images to evaluate. Default: 100."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/evaluation/"
            "chessred_test_results.csv"
        ),
    )

    return parser.parse_args()


def load_dataset(
    annotations_path: Path,
) -> tuple[
    dict[int, dict[str, object]],
    dict[int, np.ndarray],
    list[int],
]:
    with annotations_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    images_by_id = {
        int(item["id"]): item
        for item in data["images"]
    }

    ground_truth_by_id: dict[
        int,
        np.ndarray,
    ] = {}

    for image_id in images_by_id:
        ground_truth_by_id[
            image_id
        ] = np.full(
            (BOARD_SIZE, BOARD_SIZE),
            EMPTY_CLASS,
            dtype=np.int64,
        )

    for piece in data["annotations"]["pieces"]:
        image_id = int(
            piece["image_id"]
        )

        position = str(
            piece["chessboard_position"]
        )

        category_id = int(
            piece["category_id"]
        )

        file_index = (
            ord(position[0].lower())
            - ord("a")
        )

        rank = int(
            position[1]
        )

        row = (
            BOARD_SIZE
            - rank
        )

        column = file_index

        ground_truth_by_id[
            image_id
        ][
            row,
            column,
        ] = category_id

    test_ids = [
        int(value)
        for value in data[
            "splits"
        ][
            "chessred2k"
        ][
            "test"
        ][
            "image_ids"
        ]
    ]

    return (
        images_by_id,
        ground_truth_by_id,
        test_ids,
    )


def sample_test_ids(
    *,
    test_ids: list[int],
    images_by_id: dict[
        int,
        dict[str, object],
    ],
    limit: int,
    seed: int,
) -> list[int]:
    if limit <= 0:
        raise ValueError(
            "--limit must be greater than zero."
        )

    available_ids = [
        image_id
        for image_id in test_ids
        if image_id in images_by_id
    ]

    if limit >= len(
        available_ids
    ):
        return available_ids

    by_game: dict[
        int,
        list[int],
    ] = defaultdict(list)

    for image_id in available_ids:
        game_id = int(
            images_by_id[
                image_id
            ][
                "game_id"
            ]
        )

        by_game[
            game_id
        ].append(
            image_id
        )

    rng = random.Random(
        seed
    )

    for game_ids in by_game.values():
        rng.shuffle(
            game_ids
        )

    game_keys = list(
        by_game
    )

    rng.shuffle(
        game_keys
    )

    selected: list[int] = []

    while (
        len(selected) < limit
        and any(
            by_game[
                game_id
            ]
            for game_id
            in game_keys
        )
    ):
        for game_id in game_keys:
            if (
                len(selected)
                >= limit
            ):
                break

            if by_game[
                game_id
            ]:
                selected.append(
                    by_game[
                        game_id
                    ].pop()
                )

    return selected


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
    np.ndarray,
]:
    best_accuracy = -1.0
    best_exact = False
    best_rotation = 0
    best_matrix = prediction

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

        if (
            accuracy
            > best_accuracy
        ):
            best_accuracy = (
                accuracy
            )

            best_exact = (
                exact
            )

            best_rotation = (
                rotation
            )

            best_matrix = (
                rotated
            )

    return (
        best_accuracy,
        best_exact,
        best_rotation,
        best_matrix,
    )


def structural_validity(
    board_matrix: np.ndarray,
) -> bool:
    white_result = (
        validate_position(
            board_matrix.tolist(),
            white_to_move=True,
        )
    )

    black_result = (
        validate_position(
            board_matrix.tolist(),
            white_to_move=False,
        )
    )

    return bool(
        white_result.is_valid
        or black_result.is_valid
    )


def evaluate_image(
    *,
    image_path: Path,
    ground_truth: np.ndarray,
    yolo_model_path: Path,
) -> dict[str, object]:
    started = (
        time.perf_counter()
    )

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
        _,
    ) = best_orientation_metrics(
        resnet18_prediction,
        ground_truth,
    )

    (
        yolo_accuracy,
        yolo_exact,
        yolo_rotation,
        _,
    ) = best_orientation_metrics(
        yolo_prediction,
        ground_truth,
    )

    (
        ensemble_accuracy,
        ensemble_exact,
        ensemble_rotation,
        aligned_ensemble,
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
        "structurally_valid":
            structural_validity(
                aligned_ensemble
            ),
        "inference_seconds":
            elapsed_seconds,
    }


def print_summary(
    rows: list[
        dict[str, object]
    ],
) -> None:
    if not rows:
        print(
            "No images were evaluated."
        )

        return

    count = len(
        rows
    )

    def mean(
        selected_rows: list[
            dict[str, object]
        ],
        key: str,
    ) -> float:
        return sum(
            float(
                row[key]
            )
            for row
            in selected_rows
        ) / len(
            selected_rows
        )

    def rate(
        selected_rows: list[
            dict[str, object]
        ],
        key: str,
    ) -> float:
        return sum(
            bool(
                row[key]
            )
            for row
            in selected_rows
        ) / len(
            selected_rows
        )

    print(
        "\n"
        + "=" * 68
    )

    print(
        "ChessReD2K Official Test Evaluation"
    )

    print(
        "=" * 68
    )

    print(
        f"Images successfully evaluated: "
        f"{count}"
    )

    print(
        "\nOverall recognition"
    )

    print(
        "  Square accuracy:"
    )

    print(
        "    ResNet18 grid classifier: "
        f"{mean(rows, 'resnet18_accuracy'):.2%}"
    )

    print(
        "    YOLO11 detector: "
        f"{mean(rows, 'yolo_accuracy'):.2%}"
    )

    print(
        "    Final ensemble: "
        f"{mean(rows, 'ensemble_accuracy'):.2%}"
    )

    print(
        "  Exact full-board accuracy:"
    )

    print(
        "    ResNet18 grid classifier: "
        f"{rate(rows, 'resnet18_exact'):.2%}"
    )

    print(
        "    YOLO11 detector: "
        f"{rate(rows, 'yolo_exact'):.2%}"
    )

    print(
        "    Final ensemble: "
        f"{rate(rows, 'ensemble_exact'):.2%}"
    )

    print(
        "\nHuman-review load"
    )

    print(
        "  Average warnings: "
        f"{mean(rows, 'warning_count'):.2f}"
    )

    print(
        "  Average required reviews: "
        f"{mean(rows, 'required_review_count'):.2f}"
    )

    print(
        "\nStructurally valid "
        "final ensemble boards"
    )

    print(
        f"  "
        f"{rate(rows, 'structurally_valid'):.2%}"
    )

    print(
        "\nAverage inference time"
    )

    print(
        f"  "
        f"{mean(rows, 'inference_seconds'):.2f} "
        f"seconds/image"
    )


def main() -> None:
    arguments = (
        parse_arguments()
    )

    (
        images_by_id,
        ground_truth_by_id,
        test_ids,
    ) = load_dataset(
        arguments.annotations
    )

    selected_ids = (
        sample_test_ids(
            test_ids=test_ids,
            images_by_id=(
                images_by_id
            ),
            limit=(
                arguments.limit
            ),
            seed=(
                arguments.seed
            ),
        )
    )

    arguments.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results: list[
        dict[str, object]
    ] = []

    for (
        index,
        image_id,
    ) in enumerate(
        selected_ids,
        start=1,
    ):
        image_info = (
            images_by_id[
                image_id
            ]
        )

        relative_path = Path(
            str(
                image_info[
                    "path"
                ]
            )
        )

        image_path = (
            arguments.dataset_root
            / relative_path
        )

        print(
            f"[{index}/"
            f"{len(selected_ids)}] "
            f"{image_path}"
        )

        row: dict[
            str,
            object,
        ] = {
            "image_id":
                image_id,
            "game_id":
                int(
                    image_info[
                        "game_id"
                    ]
                ),
            "move_id":
                int(
                    image_info[
                        "move_id"
                    ]
                ),
            "file_name":
                str(
                    image_info[
                        "file_name"
                    ]
                ),
            "image_path":
                str(
                    image_path
                ),
        }

        try:
            metrics = (
                evaluate_image(
                    image_path=(
                        image_path
                    ),
                    ground_truth=(
                        ground_truth_by_id[
                            image_id
                        ]
                    ),
                    yolo_model_path=(
                        arguments.yolo_model
                    ),
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
                f"  ERROR: "
                f"{row['error']}"
            )

        results.append(
            row
        )

    preferred_fieldnames = [
        "file_name",
        "game_id",
        "move_id",
        "image_id",
        "image_path",
        "processing_success",
        "inference_seconds",
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
        "structurally_valid",
        "error",
    ]

    present_fields = {
        key
        for row in results
        for key in row
    }

    fieldnames = [
        field
        for field
        in preferred_fieldnames
        if field
        in present_fields
    ]

    with arguments.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    successful_rows = [
        row
        for row in results
        if "error" not in row
    ]

    print_summary(
        successful_rows
    )

    print(
        "\nSaved detailed results to:"
    )

    print(
        arguments.output
    )


if __name__ == "__main__":
    main()