from pathlib import Path

import cv2
import numpy as np


GRID_SIZE = 8
FILES = "abcdefgh"
RANKS = "87654321"


def extract_board_squares(
    playable_board: np.ndarray,
    padding: int = 20,
) -> dict[str, np.ndarray]:
    """
    Split a normalized playable chessboard into 64 padded square crops.

    The board is assumed to be oriented with:
        top-left = a8
        bottom-right = h1

    Each crop contains the logical square plus additional surrounding
    context. Border padding is added first so edge and corner squares
    still produce crops with identical dimensions.
    """
    if playable_board is None or playable_board.size == 0:
        raise ValueError(
            "Cannot extract squares from an empty board."
        )

    if padding < 0:
        raise ValueError(
            "Padding cannot be negative."
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

    padded_board = cv2.copyMakeBorder(
        playable_board,
        padding,
        padding,
        padding,
        padding,
        borderType=cv2.BORDER_REPLICATE,
    )

    extracted_squares: dict[str, np.ndarray] = {}

    for row_index in range(GRID_SIZE):
        for column_index in range(GRID_SIZE):
            original_top = row_index * square_height
            original_left = column_index * square_width

            padded_top = original_top
            padded_left = original_left

            padded_bottom = (
                original_top
                + square_height
                + (2 * padding)
            )

            padded_right = (
                original_left
                + square_width
                + (2 * padding)
            )

            square_image = padded_board[
                padded_top:padded_bottom,
                padded_left:padded_right,
            ].copy()

            expected_height = square_height + (2 * padding)
            expected_width = square_width + (2 * padding)

            if square_image.shape[:2] != (
                expected_height,
                expected_width,
            ):
                raise RuntimeError(
                    "Unexpected padded crop size for "
                    f"row {row_index}, column {column_index}: "
                    f"{square_image.shape[:2]}"
                )

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