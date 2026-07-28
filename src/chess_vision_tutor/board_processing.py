"""Core chessboard detection and perspective-warping utilities.

This module is responsible only for:

- Loading an image.
- Resizing an image.
- Preprocessing an image for contour detection.
- Detecting the outer chessboard contour.
- Ordering the four board corners.
- Warping the detected board into a square top-down view.

Internal playable-grid reconstruction is handled in
`grid_reconstruction.py`.

Debug drawing utilities are handled in `visualization.py`.
"""

from pathlib import Path

import cv2
import numpy as np


def load_image(image_path: str | Path) -> np.ndarray:
    """Load an image from disk.

    Args:
        image_path:
            Path to the image file.

    Returns:
        The loaded OpenCV BGR image.

    Raises:
        FileNotFoundError:
            If the file does not exist.

        ValueError:
            If OpenCV cannot decode the image.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Failed to load image: {image_path}"
        )

    return image


def resize_image(
    image: np.ndarray,
    max_dimension: int = 1200,
) -> np.ndarray:
    """Resize an image while preserving its aspect ratio.

    Images smaller than `max_dimension` are returned unchanged.
    """
    if image is None or image.size == 0:
        raise ValueError(
            "Cannot resize an empty image."
        )

    if max_dimension <= 0:
        raise ValueError(
            "Maximum dimension must be greater than zero."
        )

    height, width = image.shape[:2]
    largest_dimension = max(height, width)

    if largest_dimension <= max_dimension:
        return image

    scale = max_dimension / largest_dimension

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def preprocess_image(
    image: np.ndarray,
) -> np.ndarray:
    """Prepare an image for chessboard contour detection.

    Processing steps:

    1. Convert to grayscale.
    2. Apply Gaussian blur.
    3. Detect edges using Canny.
    4. Apply morphological closing to connect nearby edges.
    """
    if image is None or image.size == 0:
        raise ValueError(
            "Cannot preprocess an empty image."
        )

    grayscale_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    blurred_image = cv2.GaussianBlur(
        grayscale_image,
        (5, 5),
        0,
    )

    edge_image = cv2.Canny(
        blurred_image,
        threshold1=50,
        threshold2=150,
    )

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8,
    )

    closed_edge_image = cv2.morphologyEx(
        edge_image,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return closed_edge_image


def detect_board_contour(
    edge_image: np.ndarray,
) -> np.ndarray | None:
    """Detect a large four-sided contour representing the chessboard.

    The function first tests the contour approximation directly.

    If the contour itself does not produce a valid quadrilateral, its
    convex hull is tested as a fallback.
    """
    if edge_image is None or edge_image.size == 0:
        raise ValueError(
            "Cannot detect a board in an empty image."
        )

    image_height, image_width = edge_image.shape[:2]
    image_area = image_height * image_width

    contours, _ = cv2.findContours(
        edge_image,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    sorted_contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )

    for contour in sorted_contours:
        contour_area = cv2.contourArea(contour)

        # Reject pieces, individual squares, and other small contours.
        if contour_area < 0.10 * image_area:
            continue

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        approximation = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True,
        )

        if (
            len(approximation) == 4
            and cv2.isContourConvex(approximation)
        ):
            return approximation

        convex_hull = cv2.convexHull(contour)

        hull_perimeter = cv2.arcLength(
            convex_hull,
            True,
        )

        hull_approximation = cv2.approxPolyDP(
            convex_hull,
            0.02 * hull_perimeter,
            True,
        )

        if (
            len(hull_approximation) == 4
            and cv2.isContourConvex(hull_approximation)
        ):
            return hull_approximation

    return None


def order_board_corners(
    contour: np.ndarray,
) -> np.ndarray:
    """Order four board points consistently.

    Returned order:

    1. Top-left.
    2. Top-right.
    3. Bottom-right.
    4. Bottom-left.
    """
    if contour is None or contour.size != 8:
        raise ValueError(
            "Board contour must contain exactly four points."
        )

    points = contour.reshape(4, 2).astype(
        np.float32
    )

    ordered_points = np.zeros(
        (4, 2),
        dtype=np.float32,
    )

    coordinate_sums = points.sum(axis=1)

    coordinate_differences = np.diff(
        points,
        axis=1,
    ).reshape(-1)

    ordered_points[0] = points[
        np.argmin(coordinate_sums)
    ]

    ordered_points[2] = points[
        np.argmax(coordinate_sums)
    ]

    ordered_points[1] = points[
        np.argmin(coordinate_differences)
    ]

    ordered_points[3] = points[
        np.argmax(coordinate_differences)
    ]

    return ordered_points


def warp_board(
    image: np.ndarray,
    contour: np.ndarray,
    output_size: int = 800,
) -> np.ndarray:
    """Transform a detected chessboard into a square top-down view."""
    if image is None or image.size == 0:
        raise ValueError(
            "Cannot warp an empty image."
        )

    if contour is None or contour.size == 0:
        raise ValueError(
            "Cannot warp without a valid contour."
        )

    if output_size <= 0:
        raise ValueError(
            "Output size must be greater than zero."
        )

    source_points = order_board_corners(
        contour
    )

    destination_points = np.array(
        [
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1],
        ],
        dtype=np.float32,
    )

    transform_matrix = cv2.getPerspectiveTransform(
        source_points,
        destination_points,
    )

    warped_board = cv2.warpPerspective(
        image,
        transform_matrix,
        (output_size, output_size),
    )

    return warped_board