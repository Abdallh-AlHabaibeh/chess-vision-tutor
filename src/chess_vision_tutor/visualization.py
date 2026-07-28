"""Debug visualization utilities for board processing.

These functions draw intermediate detection results on image copies.

They do not modify board geometry or perform any recognition.
They are intended only for debugging, validation, and documentation.
"""

import cv2
import numpy as np


def draw_board_contour(
    image: np.ndarray,
    contour: np.ndarray,
) -> np.ndarray:
    """Draw a detected outer board contour."""

    if image is None or image.size == 0:
        raise ValueError(
            "Cannot draw on an empty image."
        )

    if contour is None or contour.size == 0:
        raise ValueError(
            "Cannot draw an empty contour."
        )

    output_image = image.copy()

    cv2.drawContours(
        output_image,
        [contour],
        contourIdx=-1,
        color=(0, 255, 0),
        thickness=3,
    )

    return output_image


def draw_checkerboard_corner_candidates(
    image: np.ndarray,
    corners: np.ndarray,
) -> np.ndarray:
    """Draw detected checkerboard-corner candidates."""

    if image is None or image.size == 0:
        raise ValueError(
            "Cannot draw corners on an empty image."
        )

    if corners is None:
        raise ValueError(
            "Corner candidates cannot be None."
        )

    output_image = image.copy()

    for x_coordinate, y_coordinate in corners:
        center = (
            int(round(float(x_coordinate))),
            int(round(float(y_coordinate))),
        )

        cv2.circle(
            output_image,
            center,
            radius=5,
            color=(0, 0, 255),
            thickness=-1,
        )

    return output_image


def draw_complete_grid_boundaries(
    image: np.ndarray,
    vertical_boundaries: list[int],
    horizontal_boundaries: list[int],
) -> np.ndarray:
    """Draw all nine vertical and horizontal board boundaries."""

    if image is None or image.size == 0:
        raise ValueError(
            "Cannot draw on an empty image."
        )

    output_image = image.copy()
    image_height, image_width = image.shape[:2]

    for x_coordinate in vertical_boundaries:
        cv2.line(
            output_image,
            (x_coordinate, 0),
            (x_coordinate, image_height - 1),
            color=(255, 0, 0),
            thickness=3,
        )

    for y_coordinate in horizontal_boundaries:
        cv2.line(
            output_image,
            (0, y_coordinate),
            (image_width - 1, y_coordinate),
            color=(0, 255, 255),
            thickness=3,
        )

    return output_image


def draw_supported_grid_positions(
    image: np.ndarray,
    vertical_positions: list[int],
    horizontal_positions: list[int],
) -> np.ndarray:
    """Draw all supported candidate grid positions.

    Vertical positions are red.
    Horizontal positions are green.
    """

    if image is None or image.size == 0:
        raise ValueError(
            "Cannot draw grid positions on an empty image."
        )

    output_image = image.copy()
    image_height, image_width = image.shape[:2]

    for x_coordinate in vertical_positions:
        cv2.line(
            output_image,
            (x_coordinate, 0),
            (x_coordinate, image_height - 1),
            color=(0, 0, 255),
            thickness=2,
        )

    for y_coordinate in horizontal_positions:
        cv2.line(
            output_image,
            (0, y_coordinate),
            (image_width - 1, y_coordinate),
            color=(0, 255, 0),
            thickness=2,
        )

    return output_image


def draw_regular_grid_positions(
    image: np.ndarray,
    vertical_positions: list[int],
    horizontal_positions: list[int],
) -> np.ndarray:
    """Draw the selected regularly spaced grid positions."""

    if image is None or image.size == 0:
        raise ValueError(
            "Cannot draw grid positions on an empty image."
        )

    output_image = image.copy()
    image_height, image_width = image.shape[:2]

    for x_coordinate in vertical_positions:
        cv2.line(
            output_image,
            (x_coordinate, 0),
            (x_coordinate, image_height - 1),
            color=(0, 0, 255),
            thickness=3,
        )

    for y_coordinate in horizontal_positions:
        cv2.line(
            output_image,
            (0, y_coordinate),
            (image_width - 1, y_coordinate),
            color=(0, 255, 0),
            thickness=3,
        )

    return output_image