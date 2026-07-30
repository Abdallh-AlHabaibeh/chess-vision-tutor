"""Convert ChessReD bounding boxes into Ultralytics YOLO format."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from chess_vision_tutor.chessred_processing import (
    CHESSRED_ROOT,
    load_annotations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

YOLO_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "dataset"
    / "chessred_yolo"
)

TRAIN_LIST_PATH = YOLO_OUTPUT_DIR / "train.txt"
VALIDATION_LIST_PATH = YOLO_OUTPUT_DIR / "val.txt"
DATASET_YAML_PATH = YOLO_OUTPUT_DIR / "dataset.yaml"

NUMBER_OF_CLASSES = 12

CLASS_NAMES = [
    "white_pawn",
    "white_rook",
    "white_knight",
    "white_bishop",
    "white_queen",
    "white_king",
    "black_pawn",
    "black_rook",
    "black_knight",
    "black_bishop",
    "black_queen",
    "black_king",
]


def group_piece_annotations(
    annotations: dict[str, Any],
) -> dict[int, list[dict[str, Any]]]:
    pieces_by_image: dict[
        int,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for piece in annotations["annotations"]["pieces"]:
        pieces_by_image[
            int(piece["image_id"])
        ].append(piece)

    return pieces_by_image


def convert_bbox_to_yolo(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x, y, width, height = bbox

    center_x = x + width / 2.0
    center_y = y + height / 2.0

    normalized_center_x = center_x / image_width
    normalized_center_y = center_y / image_height
    normalized_width = width / image_width
    normalized_height = height / image_height

    return (
        normalized_center_x,
        normalized_center_y,
        normalized_width,
        normalized_height,
    )


def write_label_file(
    image_record: dict[str, Any],
    pieces: list[dict[str, Any]],
) -> Path:
    relative_image_path = Path(
        image_record["path"]
    )

    relative_label_path = (
        Path("labels")
        / relative_image_path.relative_to("images")
    ).with_suffix(".txt")

    label_path = (
        CHESSRED_ROOT
        / relative_label_path
    )

    label_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_width = int(
        image_record["width"]
    )

    image_height = int(
        image_record["height"]
    )

    label_lines: list[str] = []

    for piece in pieces:
        class_id = int(
            piece["category_id"]
        )

        if not 0 <= class_id < NUMBER_OF_CLASSES:
            raise ValueError(
                f"Unexpected category ID: {class_id}"
            )

        (
            center_x,
            center_y,
            width,
            height,
        ) = convert_bbox_to_yolo(
            piece["bbox"],
            image_width,
            image_height,
        )

        label_lines.append(
            f"{class_id} "
            f"{center_x:.6f} "
            f"{center_y:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    label_path.write_text(
        "\n".join(label_lines),
        encoding="utf-8",
    )

    return label_path


def collect_available_images(
    annotations: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    available_images: dict[
        int,
        dict[str, Any],
    ] = {}

    for image_record in annotations["images"]:
        image_path = (
            CHESSRED_ROOT
            / image_record["path"]
        )

        if image_path.exists():
            available_images[
                int(image_record["id"])
            ] = image_record

    return available_images


def write_image_list(
    image_ids: list[int],
    available_images: dict[
        int,
        dict[str, Any],
    ],
    output_path: Path,
) -> int:
    image_paths: list[str] = []

    for image_id in image_ids:
        image_record = available_images.get(
            int(image_id)
        )

        if image_record is None:
            continue

        image_path = (
            CHESSRED_ROOT
            / image_record["path"]
        ).resolve()

        image_paths.append(
            image_path.as_posix()
        )

    output_path.write_text(
        "\n".join(image_paths),
        encoding="utf-8",
    )

    return len(image_paths)


def write_dataset_yaml() -> None:
    names_lines = "\n".join(
        f"  {index}: {class_name}"
        for index, class_name in enumerate(
            CLASS_NAMES
        )
    )

    yaml_content = (
        f"path: {YOLO_OUTPUT_DIR.resolve().as_posix()}\n"
        f"train: {TRAIN_LIST_PATH.resolve().as_posix()}\n"
        f"val: {VALIDATION_LIST_PATH.resolve().as_posix()}\n"
        "\n"
        "names:\n"
        f"{names_lines}\n"
    )

    DATASET_YAML_PATH.write_text(
        yaml_content,
        encoding="utf-8",
    )


def prepare_yolo_dataset() -> None:
    annotations = load_annotations()

    YOLO_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pieces_by_image = group_piece_annotations(
        annotations
    )

    available_images = collect_available_images(
        annotations
    )

    total_labels = 0

    for image_id, image_record in available_images.items():
        pieces = pieces_by_image.get(
            image_id,
            [],
        )

        write_label_file(
            image_record,
            pieces,
        )

        total_labels += len(pieces)

    train_ids = list(
        annotations["splits"]["train"]["image_ids"]
    )

    validation_split = annotations["splits"].get(
        "val",
        annotations["splits"].get("validation"),
    )

    if validation_split is None:
        raise KeyError(
            "No validation split was found."
        )

    validation_ids = list(
        validation_split["image_ids"]
    )

    train_count = write_image_list(
        train_ids,
        available_images,
        TRAIN_LIST_PATH,
    )

    validation_count = write_image_list(
        validation_ids,
        available_images,
        VALIDATION_LIST_PATH,
    )

    write_dataset_yaml()

    print(
        f"Training images: {train_count}"
    )

    print(
        f"Validation images: {validation_count}"
    )

    print(
        f"Piece labels: {total_labels}"
    )

    print(
        f"Dataset YAML: {DATASET_YAML_PATH}"
    )


if __name__ == "__main__":
    prepare_yolo_dataset()