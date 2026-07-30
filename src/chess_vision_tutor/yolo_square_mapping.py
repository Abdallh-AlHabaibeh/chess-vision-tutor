from __future__ import annotations

import cv2
import numpy as np


def transform_point_to_playable_board(
    point: tuple[float, float],
    transform_matrix: np.ndarray,
    playable_bounds: tuple[int, int, int, int],
    output_size: int = 800,
) -> tuple[float, float]:
    if transform_matrix.shape != (3, 3):
        raise ValueError(
            "Transform matrix must have shape (3, 3)."
        )

    if output_size <= 0:
        raise ValueError(
            "Output size must be greater than zero."
        )

    left, right, top, bottom = playable_bounds

    crop_width = right - left + 1
    crop_height = bottom - top + 1

    if crop_width <= 0 or crop_height <= 0:
        raise ValueError(
            "Playable-board bounds are invalid."
        )

    source_point = np.array(
        [[[point[0], point[1]]]],
        dtype=np.float32,
    )

    warped_point = cv2.perspectiveTransform(
        source_point,
        transform_matrix,
    )[0, 0]

    playable_x = (
        warped_point[0] - left
    ) * output_size / crop_width

    playable_y = (
        warped_point[1] - top
    ) * output_size / crop_height

    return float(playable_x), float(playable_y)


def map_box_to_square(
    box: tuple[float, float, float, float],
    transform_matrix: np.ndarray,
    playable_bounds: tuple[int, int, int, int],
    output_size: int = 800,
) -> tuple[int, int, float, float] | None:
    x_min, y_min, x_max, y_max = box

    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            "Bounding box coordinates are invalid."
        )

    anchor_x = (x_min + x_max) / 2.0
    anchor_y = y_max

    playable_x, playable_y = (
        transform_point_to_playable_board(
            (anchor_x, anchor_y),
            transform_matrix,
            playable_bounds,
            output_size,
        )
    )

    if not (
        0 <= playable_x < output_size
        and 0 <= playable_y < output_size
    ):
        return None

    square_size = output_size / 8

    column = int(playable_x // square_size)
    row = int(playable_y // square_size)

    return row, column, playable_x, playable_y