from pathlib import Path

import cv2
import numpy as np


GRID_SIZE = 8
FILES = "abcdefgh"
RANKS = "87654321"


def extract_board_squares(
    playable_board: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Split a normalized playable chessboard into 64 labeled square crops.

    The board is assumed to be oriented with:
        top-left = a8
        bottom-right = h1
    """
    if playable_board is None or playable_board.size == 0:
        raise ValueError(
            "Cannot extract squares from an empty board."
        )

    board_height, board_width = playable_board.shape[:2]

    if board_height % GRID_SIZE != 0:
        raise ValueError(
            "Board height must be divisible by eight."
        )

    if board_width % GRID_SIZE != 0:
        raise ValueError(
            "Board width must be divisible by eight."
        )

    square_height = board_height // GRID_SIZE
    square_width = board_width // GRID_SIZE

    extracted_squares: dict[str, np.ndarray] = {}

    for row_index in range(GRID_SIZE):
        for column_index in range(GRID_SIZE):
            top = row_index * square_height
            bottom = (row_index + 1) * square_height

            left = column_index * square_width
            right = (column_index + 1) * square_width

            square_image = playable_board[
                top:bottom,
                left:right,
            ].copy()

            square_name = (
                f"{FILES[column_index]}"
                f"{RANKS[row_index]}"
            )

            extracted_squares[square_name] = square_image

    return extracted_squares


def save_square_crops(
    squares: dict[str, np.ndarray],
    output_directory: str | Path,
) -> None:
    """
    Save extracted squares for visual debugging.
    """
    if len(squares) != 64:
        raise ValueError(
            "Exactly 64 square crops are required."
        )

    output_path = Path(output_directory)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    for square_name, square_image in squares.items():
        output_file = output_path / f"{square_name}.jpg"

        saved_successfully = cv2.imwrite(
            str(output_file),
            square_image,
        )

        if not saved_successfully:
            raise RuntimeError(
                f"Failed to save square: {output_file}"
            )