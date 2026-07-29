"""Prepare and visualize one ChessReD training sample.

This module:

1. Loads ChessReD annotations.
2. Loads one source image.
3. Uses the annotated board corners to create an 800×800 top-down board.
4. Builds an 8×8 target matrix with 13 possible classes.
5. Saves a labeled preview for manual verification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHESSRED_ROOT = PROJECT_ROOT / "data" / "dataset" / "chessred2k"
ANNOTATIONS_PATH = CHESSRED_ROOT / "annotations.json"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "chessred_sample"

BOARD_SIZE = 800
GRID_SIZE = 8
SQUARE_SIZE = BOARD_SIZE // GRID_SIZE

EMPTY_CLASS_ID = 12

CLASS_NAMES = {
    0: "white-pawn",
    1: "white-rook",
    2: "white-knight",
    3: "white-bishop",
    4: "white-queen",
    5: "white-king",
    6: "black-pawn",
    7: "black-rook",
    8: "black-knight",
    9: "black-bishop",
    10: "black-queen",
    11: "black-king",
    12: "empty",
}

CLASS_LABELS = {
    0: "wP",
    1: "wR",
    2: "wN",
    3: "wB",
    4: "wQ",
    5: "wK",
    6: "bP",
    7: "bR",
    8: "bN",
    9: "bB",
    10: "bQ",
    11: "bK",
    12: "--",
}


def load_annotations(path: Path = ANNOTATIONS_PATH) -> dict[str, Any]:
    """Load the ChessReD annotation file."""

    if not path.exists():
        raise FileNotFoundError(f"Annotations file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    required_keys = {"images", "annotations", "categories", "splits"}

    missing_keys = required_keys.difference(data.keys())

    if missing_keys:
        raise ValueError(
            f"ChessReD annotations are missing required keys: {missing_keys}"
        )

    return data


def find_image_record(
    annotations: dict[str, Any],
    image_id: int,
) -> dict[str, Any]:

    for image_record in annotations["images"]:
        if image_record["id"] == image_id:
            return image_record

    raise ValueError(f"No image record found for image_id={image_id}")


def find_corner_record(
    annotations: dict[str, Any],
    image_id: int,
) -> dict[str, Any]:
    """Find the four annotated board corners for an image."""

    corner_records = annotations["annotations"]["corners"]

    for corner_record in corner_records:
        if corner_record["image_id"] == image_id:
            return corner_record

    raise ValueError(f"No corner annotation found for image_id={image_id}")


def load_chessred_image(
    image_record: dict[str, Any],
) -> np.ndarray:

    image_path = CHESSRED_ROOT / image_record["path"]

    if not image_path.exists():
        raise FileNotFoundError(f"ChessReD image not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"OpenCV could not load image: {image_path}")

    return image


def warp_board_from_annotations(
    image: np.ndarray,
    corner_record: dict[str, Any],
    board_size: int = BOARD_SIZE,
) -> np.ndarray:
    """Warp the annotated board into a normalized top-down square."""

    corners = corner_record["corners"]

    source_points = np.array(
        [
            corners["top_left"],
            corners["top_right"],
            corners["bottom_right"],
            corners["bottom_left"],
        ],
        dtype=np.float32,
    )

    destination_points = np.array(
        [
            [0, 0],
            [board_size - 1, 0],
            [board_size - 1, board_size - 1],
            [0, board_size - 1],
        ],
        dtype=np.float32,
    )

    perspective_matrix = cv2.getPerspectiveTransform(
        source_points,
        destination_points,
    )

    warped_board = cv2.warpPerspective(
        image,
        perspective_matrix,
        (board_size, board_size),
    )

    return warped_board


def chess_square_to_matrix_position(square_name: str) -> tuple[int, int]:
    """Convert a chess square such as a8 into a matrix row and column.

    Matrix orientation:

    row 0 -> rank 8
    row 7 -> rank 1
    col 0 -> file a
    col 7 -> file h
    """

    if len(square_name) != 2:
        raise ValueError(f"Invalid chess square: {square_name}")

    file_character = square_name[0].lower()
    rank_character = square_name[1]

    if file_character not in "abcdefgh":
        raise ValueError(f"Invalid chess file: {square_name}")

    if rank_character not in "12345678":
        raise ValueError(f"Invalid chess rank: {square_name}")

    column = ord(file_character) - ord("a")
    rank = int(rank_character)
    row = 8 - rank

    return row, column


def build_target_matrix(
    annotations: dict[str, Any],
    image_id: int,
    verbose: bool = True,
) -> np.ndarray:
    """Build an 8×8 matrix containing one class ID per square."""

    target_matrix = np.full(
        (GRID_SIZE, GRID_SIZE),
        EMPTY_CLASS_ID,
        dtype=np.int64,
    )

    piece_annotations = annotations["annotations"]["pieces"]

    image_piece_count = 0

    for piece in piece_annotations:
        if piece["image_id"] != image_id:
            continue

        square_name = piece["chessboard_position"]
        category_id = int(piece["category_id"])

        if category_id not in CLASS_NAMES:
            raise ValueError(
                f"Unknown category_id={category_id} "
                f"for image_id={image_id}"
            )

        row, column = chess_square_to_matrix_position(square_name)

        if target_matrix[row, column] != EMPTY_CLASS_ID:
            raise ValueError(
                f"Multiple pieces assigned to square {square_name} "
                f"for image_id={image_id}"
            )

        target_matrix[row, column] = category_id
        image_piece_count += 1

    if verbose:
        print(f"Annotated pieces: {image_piece_count}")
        print(f"Empty squares: {64 - image_piece_count}")
    return target_matrix


def draw_target_overlay(
    warped_board: np.ndarray,
    target_matrix: np.ndarray,
) -> np.ndarray:
    """Draw the 8×8 grid, square names, and target labels."""

    preview = warped_board.copy()

    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            x1 = column * SQUARE_SIZE
            y1 = row * SQUARE_SIZE
            x2 = x1 + SQUARE_SIZE
            y2 = y1 + SQUARE_SIZE

            class_id = int(target_matrix[row, column])
            class_label = CLASS_LABELS[class_id]

            file_character = chr(ord("a") + column)
            rank = 8 - row
            square_name = f"{file_character}{rank}"

            cv2.rectangle(
                preview,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.rectangle(
                preview,
                (x1 + 3, y1 + 3),
                (x1 + 55, y1 + 43),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                preview,
                square_name,
                (x1 + 7, y1 + 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                preview,
                class_label,
                (x1 + 7, y1 + 37),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return preview


def save_target_matrix(
    target_matrix: np.ndarray,
    output_path: Path,
) -> None:

    matrix_as_list = target_matrix.tolist()

    output_path.write_text(
        json.dumps(matrix_as_list, indent=2),
        encoding="utf-8",
    )


def process_sample(image_id: int = 0) -> None:

    annotations = load_annotations()

    image_record = find_image_record(
        annotations,
        image_id,
    )

    corner_record = find_corner_record(
        annotations,
        image_id,
    )

    source_image = load_chessred_image(image_record)

    warped_board = warp_board_from_annotations(
        source_image,
        corner_record,
    )

    target_matrix = build_target_matrix(
        annotations,
        image_id,
    )

    labeled_preview = draw_target_overlay(
        warped_board,
        target_matrix,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    warped_output_path = OUTPUT_DIR / f"image_{image_id}_warped.jpg"
    preview_output_path = OUTPUT_DIR / f"image_{image_id}_labels.jpg"
    target_output_path = OUTPUT_DIR / f"image_{image_id}_target.json"

    if not cv2.imwrite(str(warped_output_path), warped_board):
        raise RuntimeError(
            f"Failed to save warped board: {warped_output_path}"
        )

    if not cv2.imwrite(str(preview_output_path), labeled_preview):
        raise RuntimeError(
            f"Failed to save labeled preview: {preview_output_path}"
        )

    save_target_matrix(
        target_matrix,
        target_output_path,
    )

    print()
    print(f"Image ID: {image_id}")
    print(f"Source: {image_record['path']}")
    print(f"Camera: {image_record.get('camera', 'unknown')}")
    print(f"Warped shape: {warped_board.shape}")
    print()
    print("Target matrix:")
    print(target_matrix)
    print()
    print(f"Saved warped board: {warped_output_path}")
    print(f"Saved labeled preview: {preview_output_path}")
    print(f"Saved target matrix: {target_output_path}")


if __name__ == "__main__":
    process_sample(image_id=0)