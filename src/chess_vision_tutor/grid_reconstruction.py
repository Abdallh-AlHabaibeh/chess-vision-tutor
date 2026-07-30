"""Playable chessboard-grid reconstruction utilities.

This module is responsible for:

- Detecting internal checkerboard corner candidates.
- Identifying supported vertical and horizontal grid positions.
- Finding regularly spaced grid-line sequences.
- Inferring all nine board boundaries.
- Cropping the exact playable 8×8 board region.

Debug drawing functions are kept separately in `visualization.py`.
"""

import cv2
import numpy as np


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


def cluster_coordinate_values(
    values: np.ndarray,
    tolerance: float = 12.0,
    minimum_support: int = 3,
) -> list[int]:
    """Group nearby coordinate values into shared grid positions.

    A cluster is retained only when enough detected corner candidates
    support approximately the same coordinate.
    """
    if values.size == 0:
        return []

    if tolerance <= 0:
        raise ValueError(
            "Tolerance must be greater than zero."
        )

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
    """Generate plausible regularly spaced coordinate sequences.

    Each result contains:

    - sequence positions;
    - estimated spacing;
    - number of supported positions;
    - total matching error.
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

    sorted_positions = sorted(
        set(positions)
    )

    if len(sorted_positions) < 2:
        return []

    sequence_results: dict[
        tuple[int, ...],
        tuple[list[int], float, int, float],
    ] = {}

    for first_index in range(
        len(sorted_positions)
    ):
        for second_index in range(
            first_index + 1,
            len(sorted_positions),
        ):
            first_position = sorted_positions[
                first_index
            ]

            second_position = sorted_positions[
                second_index
            ]

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
                        sequence_start
                        + index * spacing
                        for index in range(
                            expected_count
                        )
                    ]

                    match_count = 0
                    total_error = 0.0

                    for predicted_position in (
                        predicted_positions
                    ):
                        nearest_error = min(
                            abs(
                                candidate_position
                                - predicted_position
                            )
                            for candidate_position
                            in sorted_positions
                        )

                        if (
                            nearest_error
                            <= match_tolerance
                        ):
                            match_count += 1
                            total_error += (
                                nearest_error
                            )

                    if match_count < minimum_matches:
                        continue

                    rounded_sequence = [
                        int(round(position))
                        for position
                        in predicted_positions
                    ]

                    sequence_key = tuple(
                        rounded_sequence
                    )

                    existing_result = (
                        sequence_results.get(
                            sequence_key
                        )
                    )

                    new_result = (
                        rounded_sequence,
                        float(spacing),
                        match_count,
                        total_error,
                    )

                    if (
                        existing_result is None
                        or total_error
                        < existing_result[3]
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

    if (
        not vertical_sequence
        or not horizontal_sequence
    ):
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
                    vertical_array
                    - x_coordinate
                )
            )
        )

        horizontal_index = int(
            np.argmin(
                np.abs(
                    horizontal_array
                    - y_coordinate
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
                horizontal_array[
                    horizontal_index
                ]
                - y_coordinate
            )
        )

        if (
            vertical_error <= point_tolerance
            and horizontal_error
            <= point_tolerance
        ):
            supported_intersections.add(
                (
                    vertical_index,
                    horizontal_index,
                )
            )

    return len(
        supported_intersections
    )


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
                    point_tolerance=(
                        point_tolerance
                    ),
                )
            )

            score = (
                supported_intersections * 10.0
                + vertical_matches
                + horizontal_matches
                - spacing_difference_ratio
                * 30.0
                - (
                    vertical_error
                    + horizontal_error
                )
                * 0.05
            )

            if score > best_score:
                best_score = score
                best_vertical = (
                    vertical_sequence
                )
                best_horizontal = (
                    horizontal_sequence
                )

    return (
        best_vertical,
        best_horizontal,
    )


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
    """Detect the best vertical and horizontal sequences jointly.

    Neither axis is assumed to be correct first. Candidate sequences
    from both axes are compared using square spacing and two-dimensional
    corner support.
    """
    (
        supported_vertical,
        supported_horizontal,
    ) = detect_supported_grid_positions(
        corners,
        tolerance=cluster_tolerance,
        minimum_support=minimum_support,
    )

    vertical_sequences = (
        generate_regular_sequences(
            supported_vertical,
            expected_count=7,
            match_tolerance=(
                sequence_tolerance
            ),
        )
    )

    horizontal_sequences = (
        generate_regular_sequences(
            supported_horizontal,
            expected_count=7,
            match_tolerance=(
                sequence_tolerance
            ),
        )
    )

    (
        regular_vertical,
        regular_horizontal,
    ) = select_best_grid_pair(
        corners,
        vertical_sequences,
        horizontal_sequences,
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

    differences = np.diff(
        sorted_positions
    )

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
        nearby_coordinates = (
            np.abs(
                coordinate_values - boundary
            )
            <= tolerance
        )

        support_count += int(
            np.count_nonzero(
                nearby_coordinates
            )
        )

    return support_count


def infer_full_grid_boundaries(
    detected_positions: list[int],
    coordinate_values: np.ndarray,
    image_length: int,
    expected_boundary_count: int = 9,
    tolerance: float = 16.0,
) -> list[int]:
    """Infer all nine boundaries from seven detected positions.

    The seven detected positions may correspond to boundaries:

    - 0 through 6;
    - 1 through 7;
    - 2 through 8.

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

        first_boundary = (
            predicted_boundaries[0]
        )

        last_boundary = (
            predicted_boundaries[-1]
        )

        if first_boundary < -tolerance:
            continue

        if (
            last_boundary
            > image_length - 1 + tolerance
        ):
            continue

        boundary_support = (
            count_axis_boundary_support(
                coordinate_values,
                predicted_boundaries,
                tolerance=tolerance,
            )
        )

        outside_penalty = (
            max(0, -first_boundary)
            + max(
                0,
                last_boundary
                - (image_length - 1),
            )
        )

        score = (
            boundary_support
            - outside_penalty * 5.0
        )

        if score > best_score:
            best_score = score
            best_boundaries = (
                predicted_boundaries
            )

    return best_boundaries


def infer_complete_chessboard_grid(
    corners: np.ndarray,
    regular_vertical_positions: list[int],
    regular_horizontal_positions: list[int],
    image_shape: tuple[int, ...],
) -> tuple[list[int], list[int]]:
    """Infer the complete 9×9 boundary grid."""
    if corners is None or corners.size == 0:
        raise ValueError(
            "Corner candidates are required."
        )

    image_height, image_width = (
        image_shape[:2]
    )

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

def get_playable_board_bounds(
    warped_board: np.ndarray,
    vertical_boundaries: list[int],
    horizontal_boundaries: list[int],
) -> tuple[int, int, int, int]:
    """Return the playable-board crop as left, right, top, bottom."""
    if warped_board is None or warped_board.size == 0:
        raise ValueError(
            "Cannot calculate bounds for an empty warped board."
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

    return left, right, top, bottom

def crop_playable_board(
    warped_board: np.ndarray,
    vertical_boundaries: list[int],
    horizontal_boundaries: list[int],
    output_size: int = 800,
) -> np.ndarray:
    """Crop the exact playable 8×8 region from the warped board."""
    if output_size <= 0:
        raise ValueError(
            "Output size must be greater than zero."
        )

    left, right, top, bottom = (
        get_playable_board_bounds(
            warped_board,
            vertical_boundaries,
            horizontal_boundaries,
        )
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