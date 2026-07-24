from pathlib import Path

import cv2
import numpy as np


def load_image(image_path: str | Path) -> np.ndarray:
    """
    Load a chessboard image.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    return image


def resize_image(
    image: np.ndarray,
    max_dimension: int = 1200,
) -> np.ndarray:
    """
    Resize an image while preserving its aspect ratio.
    """
    height, width = image.shape[:2]
    largest_dimension = max(height, width)

    if largest_dimension <= max_dimension:
        return image

    scale = max_dimension / largest_dimension

    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Prepare an image for chessboard contour detection.

    Convert the image to grayscale.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot preprocess an empty image.")

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
    """
    Detect a large four sided contour representing the chessboard.
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


def draw_board_contour(
    image: np.ndarray,
    contour: np.ndarray,
) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Cannot draw on an empty image.")

    if contour is None or contour.size == 0:
        raise ValueError("Cannot draw an empty contour.")

    output_image = image.copy()

    cv2.drawContours(
        output_image,
        [contour],
        contourIdx=-1,
        color=(0, 255, 0),
        thickness=3,
    )

    return output_image


def order_board_corners(
    contour: np.ndarray,
) -> np.ndarray:
    if contour is None or contour.size != 8:
        raise ValueError("Board contour must contain exactly four points.")

    points = contour.reshape(4, 2).astype(np.float32)

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
    """
    Transform a detected chessboard into a square top-down view.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot warp an empty image.")

    if contour is None or contour.size == 0:
        raise ValueError("Cannot warp without a valid contour.")

    if output_size <= 0:
        raise ValueError("Output size must be greater than zero.")

    source_points = order_board_corners(contour)

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


def detect_checkerboard_corner_candidates(
    warped_board: np.ndarray,
    maximum_corners: int = 150,
    quality_level: float = 0.01,
    minimum_distance: int = 20,
) -> np.ndarray:
    if warped_board is None or warped_board.size == 0:
        raise ValueError(
            "Cannot detect corners in an empty warped board."
        )

    if maximum_corners <= 0:
        raise ValueError(
            "Maximum corners must be greater than zero."
        )

    if not 0 < quality_level <= 1:
        raise ValueError(
            "Quality level must be between zero and one."
        )

    if minimum_distance <= 0:
        raise ValueError(
            "Minimum distance must be greater than zero."
        )

    grayscale_image = cv2.cvtColor(
        warped_board,
        cv2.COLOR_BGR2GRAY,
    )

    blurred_image = cv2.GaussianBlur(
        grayscale_image,
        (5, 5),
        0,
    )

    corners = cv2.goodFeaturesToTrack(
        blurred_image,
        maxCorners=maximum_corners,
        qualityLevel=quality_level,
        minDistance=minimum_distance,
        blockSize=7,
        useHarrisDetector=False,
    )

    if corners is None:
        return np.empty(
            (0, 2),
            dtype=np.float32,
        )

    return corners.reshape(-1, 2)


def draw_checkerboard_corner_candidates(
    image: np.ndarray,
    corners: np.ndarray,
) -> np.ndarray:
    """
    Draw detected corner candidates on a copy of the warped board.
    """
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


def cluster_coordinate_values(
    values: np.ndarray,
    tolerance: float = 12.0,
    minimum_support: int = 3,
) -> list[int]:
    """
    Group nearby coordinate values into shared grid positions.

    A cluster is retained only when enough detected corner candidates
    support approximately the same coordinate.
    """
    if values.size == 0:
        return []

    if tolerance <= 0:
        raise ValueError("Tolerance must be greater than zero.")

    if minimum_support <= 0:
        raise ValueError(
            "Minimum support must be greater than zero."
        )

    sorted_values = np.sort(
        values.astype(np.float32)
    )

    clusters: list[list[float]] = [
        [float(sorted_values[0])]
    ]

    for value in sorted_values[1:]:
        current_cluster = clusters[-1]

        current_center = float(
            np.median(current_cluster)
        )

        if abs(float(value) - current_center) <= tolerance:
            current_cluster.append(float(value))
        else:
            clusters.append([float(value)])

    supported_positions: list[int] = []

    for cluster in clusters:
        if len(cluster) < minimum_support:
            continue

        cluster_position = int(
            round(float(np.median(cluster)))
        )

        supported_positions.append(
            cluster_position
        )

    return supported_positions


def detect_supported_grid_positions(
    corners: np.ndarray,
    tolerance: float = 12.0,
    minimum_support: int = 3,
) -> tuple[list[int], list[int]]:
    if corners is None or corners.size == 0:
        return [], []

    x_coordinates = corners[:, 0]
    y_coordinates = corners[:, 1]

    vertical_positions = cluster_coordinate_values(
        x_coordinates,
        tolerance=tolerance,
        minimum_support=minimum_support,
    )

    horizontal_positions = cluster_coordinate_values(
        y_coordinates,
        tolerance=tolerance,
        minimum_support=minimum_support,
    )

    return vertical_positions, horizontal_positions


def generate_regular_sequences(
    positions: list[int],
    expected_count: int = 7,
    match_tolerance: float = 14.0,
    minimum_spacing: float = 50.0,
    maximum_spacing: float = 130.0,
    minimum_matches: int = 5,
) -> list[tuple[list[int], float, int, float]]:
    """
    Generate plausible regularly spaced coordinate sequences.

    Each result contains:
        sequence positions,
        estimated spacing,
        number of supported positions,
        total matching error.
    """
    if expected_count < 2:
        raise ValueError(
            "Expected count must be at least two."
        )

    if match_tolerance <= 0:
        raise ValueError(
            "Match tolerance must be greater than zero."
        )

    if minimum_spacing <= 0:
        raise ValueError(
            "Minimum spacing must be greater than zero."
        )

    if maximum_spacing <= minimum_spacing:
        raise ValueError(
            "Maximum spacing must exceed minimum spacing."
        )

    if minimum_matches <= 0:
        raise ValueError(
            "Minimum matches must be greater than zero."
        )

    sorted_positions = sorted(set(positions))

    if len(sorted_positions) < 2:
        return []

    sequence_results: dict[
        tuple[int, ...],
        tuple[list[int], float, int, float],
    ] = {}

    for first_index in range(len(sorted_positions)):
        for second_index in range(
            first_index + 1,
            len(sorted_positions),
        ):
            first_position = sorted_positions[first_index]
            second_position = sorted_positions[second_index]

            coordinate_difference = (
                second_position - first_position
            )

            for grid_index_difference in range(
                1,
                expected_count,
            ):
                spacing = (
                    coordinate_difference
                    / grid_index_difference
                )

                if not (
                    minimum_spacing
                    <= spacing
                    <= maximum_spacing
                ):
                    continue

                for first_grid_index in range(
                    expected_count
                    - grid_index_difference
                ):
                    sequence_start = (
                        first_position
                        - first_grid_index * spacing
                    )

                    predicted_positions = [
                        sequence_start + index * spacing
                        for index in range(expected_count)
                    ]

                    match_count = 0
                    total_error = 0.0

                    for predicted_position in predicted_positions:
                        nearest_error = min(
                            abs(
                                candidate_position
                                - predicted_position
                            )
                            for candidate_position
                            in sorted_positions
                        )

                        if nearest_error <= match_tolerance:
                            match_count += 1
                            total_error += nearest_error

                    if match_count < minimum_matches:
                        continue

                    rounded_sequence = [
                        int(round(position))
                        for position in predicted_positions
                    ]

                    sequence_key = tuple(rounded_sequence)

                    existing_result = sequence_results.get(
                        sequence_key
                    )

                    new_result = (
                        rounded_sequence,
                        float(spacing),
                        match_count,
                        total_error,
                    )

                    if (
                        existing_result is None
                        or total_error < existing_result[3]
                    ):
                        sequence_results[
                            sequence_key
                        ] = new_result

    sorted_results = sorted(
        sequence_results.values(),
        key=lambda result: (
            -result[2],
            result[3],
        ),
    )

    return sorted_results[:50]


def count_supported_grid_intersections(
    corners: np.ndarray,
    vertical_sequence: list[int],
    horizontal_sequence: list[int],
    point_tolerance: float = 16.0,
) -> int:
    if corners is None or corners.size == 0:
        return 0

    if not vertical_sequence or not horizontal_sequence:
        return 0

    vertical_array = np.asarray(
        vertical_sequence,
        dtype=np.float32,
    )

    horizontal_array = np.asarray(
        horizontal_sequence,
        dtype=np.float32,
    )

    supported_intersections: set[
        tuple[int, int]
    ] = set()

    for x_coordinate, y_coordinate in corners:
        vertical_index = int(
            np.argmin(
                np.abs(
                    vertical_array - x_coordinate
                )
            )
        )

        horizontal_index = int(
            np.argmin(
                np.abs(
                    horizontal_array - y_coordinate
                )
            )
        )

        vertical_error = abs(
            float(
                vertical_array[vertical_index]
                - x_coordinate
            )
        )

        horizontal_error = abs(
            float(
                horizontal_array[horizontal_index]
                - y_coordinate
            )
        )

        if (
            vertical_error <= point_tolerance
            and horizontal_error <= point_tolerance
        ):
            supported_intersections.add(
                (
                    vertical_index,
                    horizontal_index,
                )
            )

    return len(supported_intersections)


def select_best_grid_pair(
    corners: np.ndarray,
    vertical_sequences: list[
        tuple[list[int], float, int, float]
    ],
    horizontal_sequences: list[
        tuple[list[int], float, int, float]
    ],
    maximum_spacing_difference_ratio: float = 0.18,
    point_tolerance: float = 16.0,
) -> tuple[list[int], list[int]]:
    best_vertical: list[int] = []
    best_horizontal: list[int] = []
    best_score = float("-inf")

    for (
        vertical_sequence,
        vertical_spacing,
        vertical_matches,
        vertical_error,
    ) in vertical_sequences:
        for (
            horizontal_sequence,
            horizontal_spacing,
            horizontal_matches,
            horizontal_error,
        ) in horizontal_sequences:
            average_spacing = (
                vertical_spacing
                + horizontal_spacing
            ) / 2.0

            if average_spacing <= 0:
                continue

            spacing_difference_ratio = (
                abs(
                    vertical_spacing
                    - horizontal_spacing
                )
                / average_spacing
            )

            if (
                spacing_difference_ratio
                > maximum_spacing_difference_ratio
            ):
                continue

            supported_intersections = (
                count_supported_grid_intersections(
                    corners,
                    vertical_sequence,
                    horizontal_sequence,
                    point_tolerance=point_tolerance,
                )
            )

            score = (
                supported_intersections * 10.0
                + vertical_matches
                + horizontal_matches
                - spacing_difference_ratio * 30.0
                - (vertical_error + horizontal_error) * 0.05
            )

            if score > best_score:
                best_score = score
                best_vertical = vertical_sequence
                best_horizontal = horizontal_sequence

    return best_vertical, best_horizontal


def detect_joint_regular_grid_positions(
    corners: np.ndarray,
    cluster_tolerance: float = 12.0,
    minimum_support: int = 3,
    sequence_tolerance: float = 14.0,
) -> tuple[
    list[int],
    list[int],
    list[int],
    list[int],
]:
    """
    Detect the best vertical and horizontal grid sequences jointly.

    Neither axis is assumed to be correct first. Candidate sequences from
    both axes are compared using square spacing and 2D corner support.
    """
    supported_vertical, supported_horizontal = (
        detect_supported_grid_positions(
            corners,
            tolerance=cluster_tolerance,
            minimum_support=minimum_support,
        )
    )

    vertical_sequences = generate_regular_sequences(
        supported_vertical,
        expected_count=7,
        match_tolerance=sequence_tolerance,
    )

    horizontal_sequences = generate_regular_sequences(
        supported_horizontal,
        expected_count=7,
        match_tolerance=sequence_tolerance,
    )

    regular_vertical, regular_horizontal = (
        select_best_grid_pair(
            corners,
            vertical_sequences,
            horizontal_sequences,
        )
    )

    return (
        supported_vertical,
        supported_horizontal,
        regular_vertical,
        regular_horizontal,
    )


def estimate_grid_spacing(
    positions: list[int],
) -> float:
    if len(positions) < 2:
        raise ValueError(
            "At least two grid positions are required."
        )

    sorted_positions = np.asarray(
        sorted(positions),
        dtype=np.float32,
    )

    differences = np.diff(sorted_positions)

    spacing = float(
        np.median(differences)
    )

    if spacing <= 0:
        raise ValueError(
            "Grid spacing must be greater than zero."
        )

    return spacing


def count_axis_boundary_support(
    coordinate_values: np.ndarray,
    boundaries: list[int],
    tolerance: float = 16.0,
) -> int:
    if coordinate_values.size == 0:
        return 0

    support_count = 0

    for boundary in boundaries:
        nearby_coordinates = np.abs(
            coordinate_values - boundary
        ) <= tolerance

        support_count += int(
            np.count_nonzero(nearby_coordinates)
        )

    return support_count


def infer_full_grid_boundaries(
    detected_positions: list[int],
    coordinate_values: np.ndarray,
    image_length: int,
    expected_boundary_count: int = 9,
    tolerance: float = 16.0,
) -> list[int]:
    """
    Infer all nine chessboard boundaries from seven detected positions.

    The seven detected positions may correspond to boundaries:
        0 through 6,
        1 through 7,
        or 2 through 8.

    Each possible placement is tested, and the one with the strongest
    corner support inside the image is retained.
    """
    if len(detected_positions) != 7:
        raise ValueError(
            "Exactly seven detected grid positions are required."
        )

    if image_length <= 0:
        raise ValueError(
            "Image length must be greater than zero."
        )

    sorted_positions = sorted(
        detected_positions
    )

    spacing = estimate_grid_spacing(
        sorted_positions
    )

    missing_boundary_count = (
        expected_boundary_count
        - len(sorted_positions)
    )

    if missing_boundary_count < 0:
        raise ValueError(
            "Detected positions exceed expected boundary count."
        )

    best_boundaries: list[int] = []
    best_score = float("-inf")

    for starting_index in range(
        missing_boundary_count + 1
    ):
        grid_start = (
            sorted_positions[0]
            - starting_index * spacing
        )

        predicted_boundaries = [
            int(
                round(
                    grid_start
                    + index * spacing
                )
            )
            for index in range(
                expected_boundary_count
            )
        ]

        first_boundary = predicted_boundaries[0]
        last_boundary = predicted_boundaries[-1]

        # Reject grids extending substantially outside the image.
        if first_boundary < -tolerance:
            continue

        if last_boundary > image_length - 1 + tolerance:
            continue

        boundary_support = count_axis_boundary_support(
            coordinate_values,
            predicted_boundaries,
            tolerance=tolerance,
        )

        outside_penalty = (
            max(0, -first_boundary)
            + max(
                0,
                last_boundary - (image_length - 1),
            )
        )

        score = (
            boundary_support
            - outside_penalty * 5.0
        )

        if score > best_score:
            best_score = score
            best_boundaries = predicted_boundaries

    return best_boundaries


def infer_complete_chessboard_grid(
    corners: np.ndarray,
    regular_vertical_positions: list[int],
    regular_horizontal_positions: list[int],
    image_shape: tuple[int, ...],
) -> tuple[list[int], list[int]]:
    """
    Infer the complete 9×9 boundary grid.
    """
    if corners is None or corners.size == 0:
        raise ValueError(
            "Corner candidates are required."
        )

    image_height, image_width = image_shape[:2]

    vertical_boundaries = (
        infer_full_grid_boundaries(
            regular_vertical_positions,
            corners[:, 0],
            image_width,
        )
    )

    horizontal_boundaries = (
        infer_full_grid_boundaries(
            regular_horizontal_positions,
            corners[:, 1],
            image_height,
        )
    )

    if len(vertical_boundaries) != 9:
        raise ValueError(
            "Could not infer nine vertical boundaries."
        )

    if len(horizontal_boundaries) != 9:
        raise ValueError(
            "Could not infer nine horizontal boundaries."
        )

    return (
        vertical_boundaries,
        horizontal_boundaries,
    )


def crop_playable_board(
    warped_board: np.ndarray,
    vertical_boundaries: list[int],
    horizontal_boundaries: list[int],
    output_size: int = 800,
) -> np.ndarray:
    """
    Crop the exact playable 8×8 region from the warped board.
    """
    if warped_board is None or warped_board.size == 0:
        raise ValueError(
            "Cannot crop an empty warped board."
        )

    if len(vertical_boundaries) != 9:
        raise ValueError(
            "Nine vertical boundaries are required."
        )

    if len(horizontal_boundaries) != 9:
        raise ValueError(
            "Nine horizontal boundaries are required."
        )

    image_height, image_width = warped_board.shape[:2]

    left = max(
        0,
        int(round(vertical_boundaries[0])),
    )

    right = min(
        image_width - 1,
        int(round(vertical_boundaries[-1])),
    )

    top = max(
        0,
        int(round(horizontal_boundaries[0])),
    )

    bottom = min(
        image_height - 1,
        int(round(horizontal_boundaries[-1])),
    )

    if right <= left:
        raise ValueError(
            "Invalid horizontal playable-board boundaries."
        )

    if bottom <= top:
        raise ValueError(
            "Invalid vertical playable-board boundaries."
        )

    playable_board = warped_board[
        top:bottom + 1,
        left:right + 1,
    ]

    normalized_board = cv2.resize(
        playable_board,
        (output_size, output_size),
        interpolation=cv2.INTER_LINEAR,
    )

    return normalized_board


def draw_complete_grid_boundaries(
    image: np.ndarray,
    vertical_boundaries: list[int],
    horizontal_boundaries: list[int],
) -> np.ndarray:
    """
    Draw all nine vertical and nine horizontal boundaries.
    """
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
    """
    Draw all supported candidate grid positions.

    Vertical positions are drawn in red.
    Horizontal positions are drawn in green.
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
    """
    Draw only the selected regularly spaced grid positions.
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