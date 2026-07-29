from pathlib import Path

import cv2
import numpy as np

from chess_vision_tutor.board_processing import (
    crop_playable_board,
    detect_board_contour,
    detect_checkerboard_corner_candidates,
    detect_joint_regular_grid_positions,
    infer_complete_chessboard_grid,
    load_image,
    order_board_corners,
    preprocess_image,
)
from chess_vision_tutor.square_extraction import (
    extract_board_squares,
)


OUTPUT_SIZE = 800
GRID_SIZE = 8
CLASS_NAMES = [
    "B_b",
    "B_w",
    "K_b",
    "K_w",
    "N_b",
    "N_w",
    "P_b",
    "P_w",
    "Q_b",
    "Q_w",
    "R_b",
    "R_w",
]


def create_output_directories(
    output_root: Path,
) -> None:
    """
    Create one directory for empty squares and each chess-piece class.
    """
    class_names = ["empty", *CLASS_NAMES]

    for class_name in class_names:
        class_directory = output_root / class_name
        class_directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def create_board_transform(
    image: np.ndarray,
    contour: np.ndarray,
    output_size: int = OUTPUT_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Warp the detected outer board and return its perspective matrix.
    """
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

    return warped_board, transform_matrix


def transform_point(
    point: tuple[float, float],
    transform_matrix: np.ndarray,
) -> tuple[float, float]:
    """
    Apply a perspective transformation to one image point.
    """
    point_array = np.array(
        [[[point[0], point[1]]]],
        dtype=np.float32,
    )

    transformed_point = cv2.perspectiveTransform(
        point_array,
        transform_matrix,
    )[0, 0]

    return (
        float(transformed_point[0]),
        float(transformed_point[1]),
    )


def read_yolo_labels(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> list[tuple[int, float, float]]:
    """
    Read YOLO labels and return class ID and bounding-box bottom center.

    YOLO format:
        class_id center_x center_y width height
    """
    if not label_path.exists():
        return []

    pieces: list[tuple[int, float, float]] = []

    label_lines = label_path.read_text(
        encoding="utf-8",
    ).splitlines()

    for line in label_lines:
        values = line.strip().split()

        if len(values) != 5:
            continue

        class_id = int(values[0])
        center_x = float(values[1]) * image_width
        center_y = float(values[2]) * image_height
        box_height = float(values[4]) * image_height

        bottom_center_y = (
            center_y
            + box_height / 2.0
        )

        pieces.append(
            (
                class_id,
                center_x,
                bottom_center_y,
            )
        )

    return pieces


def transform_to_playable_board(
    warped_point: tuple[float, float],
    vertical_boundaries: list[int],
    horizontal_boundaries: list[int],
    output_size: int = OUTPUT_SIZE,
) -> tuple[float, float] | None:
    """
    Convert a point on the outer warped board into playable-board space.
    """
    left = float(vertical_boundaries[0])
    right = float(vertical_boundaries[-1])
    top = float(horizontal_boundaries[0])
    bottom = float(horizontal_boundaries[-1])

    x_coordinate, y_coordinate = warped_point

    if not (
        left <= x_coordinate <= right
        and top <= y_coordinate <= bottom
    ):
        return None

    playable_width = right - left
    playable_height = bottom - top

    if playable_width <= 0 or playable_height <= 0:
        return None

    normalized_x = (
        (x_coordinate - left)
        / playable_width
        * output_size
    )

    normalized_y = (
        (y_coordinate - top)
        / playable_height
        * output_size
    )

    return normalized_x, normalized_y


def point_to_square_name(
    point: tuple[float, float],
    board_size: int = OUTPUT_SIZE,
) -> str | None:
    """
    Convert a playable-board point into its square crop name.
    """
    x_coordinate, y_coordinate = point
    square_size = board_size / GRID_SIZE

    column_index = int(
        x_coordinate // square_size
    )

    row_index = int(
        y_coordinate // square_size
    )

    if not (
        0 <= column_index < GRID_SIZE
        and 0 <= row_index < GRID_SIZE
    ):
        return None

    files = "abcdefgh"
    ranks = "87654321"

    return (
        f"{files[column_index]}"
        f"{ranks[row_index]}"
    )


def process_dataset_image(
    image_path: Path,
    label_path: Path,
    output_root: Path,
) -> bool:
    """
    Convert one labeled full-board image into classified square crops.
    """
    image = load_image(image_path)
    image_height, image_width = image.shape[:2]

    edge_image = preprocess_image(image)

    board_contour = detect_board_contour(
        edge_image,
    )

    if board_contour is None:
        print(
            f"Skipped {image_path.name}: "
            "board contour not detected."
        )
        return False

    warped_board, transform_matrix = (
        create_board_transform(
            image,
            board_contour,
        )
    )

    corner_candidates = (
        detect_checkerboard_corner_candidates(
            warped_board,
        )
    )

    (
        _,
        _,
        regular_vertical,
        regular_horizontal,
    ) = detect_joint_regular_grid_positions(
        corner_candidates,
    )

    if (
        len(regular_vertical) != 7
        or len(regular_horizontal) != 7
    ):
        print(
            f"Skipped {image_path.name}: "
            "regular grid not detected."
        )
        return False

    try:
        (
            vertical_boundaries,
            horizontal_boundaries,
        ) = infer_complete_chessboard_grid(
            corner_candidates,
            regular_vertical,
            regular_horizontal,
            warped_board.shape,
        )
    except ValueError as error:
        print(
            f"Skipped {image_path.name}: {error}"
        )
        return False

    playable_board = crop_playable_board(
        warped_board,
        vertical_boundaries,
        horizontal_boundaries,
        output_size=OUTPUT_SIZE,
    )

    padded_squares = extract_board_squares(
        playable_board,
        padding=20,
    )

    square_classes: dict[str, str] = {
        square_name: "empty"
        for square_name in padded_squares
    }

    yolo_pieces = read_yolo_labels(
        label_path,
        image_width,
        image_height,
    )

    for (
        class_id,
        bottom_center_x,
        bottom_center_y,
    ) in yolo_pieces:
        if not 0 <= class_id < len(CLASS_NAMES):
            continue

        warped_point = transform_point(
            (
                bottom_center_x,
                bottom_center_y,
            ),
            transform_matrix,
        )

        playable_point = transform_to_playable_board(
            warped_point,
            vertical_boundaries,
            horizontal_boundaries,
        )

        if playable_point is None:
            continue

        square_name = point_to_square_name(
            playable_point,
        )

        if square_name is None:
            continue

        square_classes[square_name] = (
            CLASS_NAMES[class_id]
        )

    image_stem = image_path.stem

    for square_name, square_image in padded_squares.items():
        class_name = square_classes[square_name]

        output_directory = (
            output_root / class_name
        )

        output_file = output_directory / (
            f"{image_stem}_{square_name}.jpg"
        )

        saved_successfully = cv2.imwrite(
            str(output_file),
            square_image,
        )

        if not saved_successfully:
            raise RuntimeError(
                f"Failed to save: {output_file}"
            )

    return True


def build_pilot_dataset(
    maximum_images: int = 20,
) -> None:
    """
    Build a small pilot classification dataset before full conversion.
    """
    image_directory = Path(
        "data/dataset/chess_pieces/train/images"
    )

    label_directory = Path(
        "data/dataset/chess_pieces/train/labels"
    )

    output_root = Path(
        "data/classification_dataset/pilot"
    )

    create_output_directories(
        output_root,
    )

    image_paths = sorted(
        path
        for path in image_directory.iterdir()
        if path.suffix.lower()
        in {".jpg", ".jpeg", ".png"}
    )

    successful_images = 0
    skipped_images = 0

    for image_path in image_paths[
        :maximum_images
    ]:
        label_path = (
            label_directory
            / f"{image_path.stem}.txt"
        )

        try:
            processed_successfully = (
                process_dataset_image(
                    image_path,
                    label_path,
                    output_root,
                )
            )
        except Exception as error:
            print(
                f"Skipped {image_path.name}: {error}"
            )
            processed_successfully = False

        if processed_successfully:
            successful_images += 1
        else:
            skipped_images += 1

    print(
        "Pilot dataset conversion complete."
    )
    print(
        f"Successful boards: {successful_images}"
    )
    print(
        f"Skipped boards: {skipped_images}"
    )
    print(
        f"Output directory: {output_root}"
    )


if __name__ == "__main__":
    build_pilot_dataset()