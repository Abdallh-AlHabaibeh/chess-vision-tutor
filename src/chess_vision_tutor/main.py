from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from chess_vision_tutor.board_processing import (
    create_board_transform,
    detect_board_contour,
    load_image,
    preprocess_image,
    resize_image,
    warp_board,
)
from chess_vision_tutor.grid_reconstruction import (
    crop_playable_board,
    detect_checkerboard_corner_candidates,
    detect_joint_regular_grid_positions,
    get_playable_board_bounds,
    infer_complete_chessboard_grid,
)
from chess_vision_tutor.square_extraction import (
    extract_board_squares,
    save_square_crops,
)
from chess_vision_tutor.visualization import (
    draw_board_contour,
    draw_checkerboard_corner_candidates,
    draw_complete_grid_boundaries,
    draw_regular_grid_positions,
    draw_supported_grid_positions,
)


DEFAULT_SQUARE_OUTPUT_DIR = Path("data/processed/squares")


def display_debug_images(
    images: dict[str, np.ndarray],
) -> None:
    for window_name, image in images.items():
        cv2.imshow(window_name, image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_board_image(
    image_path: str | Path,
    *,
    debug: bool = False,
    save_squares: bool = False,
    square_output_dir: str | Path = DEFAULT_SQUARE_OUTPUT_DIR,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    image = load_image(image_path)
    resized_image = resize_image(image)
    processed_image = preprocess_image(resized_image)

    board_contour = detect_board_contour(
        processed_image
    )

    if board_contour is None:
        raise RuntimeError(
            "No chessboard contour was detected."
        )

    transform_matrix = create_board_transform(
        board_contour
    )   

    warped_board = warp_board(
        resized_image,
        board_contour,
        transform_matrix=transform_matrix,
    )

    corner_candidates = (
        detect_checkerboard_corner_candidates(
            warped_board
        )
    )

    (
        supported_vertical_positions,
        supported_horizontal_positions,
        regular_vertical_positions,
        regular_horizontal_positions,
    ) = detect_joint_regular_grid_positions(
        corner_candidates,
        cluster_tolerance=12.0,
        minimum_support=3,
        sequence_tolerance=14.0,
    )

    has_reliable_internal_grid = (
        len(regular_vertical_positions) == 7
        and len(regular_horizontal_positions) == 7
    )

    if not has_reliable_internal_grid:
        if debug:
            debug_images = {
                "Original Image": resized_image,
                "Detected Edges": processed_image,
                "Detected Chessboard Contour": (
                    draw_board_contour(
                        resized_image,
                        board_contour,
                    )
                ),
                "Warped Chessboard": warped_board,
                "Checkerboard Corner Candidates": (
                    draw_checkerboard_corner_candidates(
                        warped_board,
                        corner_candidates,
                    )
                ),
            }

            display_debug_images(debug_images)

        raise RuntimeError(
            "Could not detect a reliable 7x7 internal grid."
        )

    (
        complete_vertical_boundaries,
        complete_horizontal_boundaries,
    ) = infer_complete_chessboard_grid(
        corner_candidates,
        regular_vertical_positions,
        regular_horizontal_positions,
        warped_board.shape,
    )

    playable_board = crop_playable_board(
        warped_board,
        complete_vertical_boundaries,
        complete_horizontal_boundaries,
    )

    playable_bounds = get_playable_board_bounds(
    warped_board,
    complete_vertical_boundaries,
    complete_horizontal_boundaries,
    )

    board_squares: dict[str, np.ndarray] | None = None

    if save_squares:
        board_squares = extract_board_squares(
            playable_board
        )

        save_square_crops(
            board_squares,
            square_output_dir,
        )

    if debug:
        debug_images = {
            "Original Image": resized_image,
            "Detected Edges": processed_image,
            "Detected Chessboard Contour": (
                draw_board_contour(
                    resized_image,
                    board_contour,
                )
            ),
            "Warped Chessboard": warped_board,
            "Checkerboard Corner Candidates": (
                draw_checkerboard_corner_candidates(
                    warped_board,
                    corner_candidates,
                )
            ),
            "Supported Grid Positions": (
                draw_supported_grid_positions(
                    warped_board,
                    supported_vertical_positions,
                    supported_horizontal_positions,
                )
            ),
            "Regular Grid Positions": (
                draw_regular_grid_positions(
                    warped_board,
                    regular_vertical_positions,
                    regular_horizontal_positions,
                )
            ),
            "Complete 9x9 Grid": (
                draw_complete_grid_boundaries(
                    warped_board,
                    complete_vertical_boundaries,
                    complete_horizontal_boundaries,
                )
            ),
            "Cropped Playable Board": playable_board,
        }

        if board_squares is not None:
            debug_images.update(
                {
                    "Square a8": board_squares["a8"],
                    "Square h8": board_squares["h8"],
                    "Square a1": board_squares["a1"],
                    "Square h1": board_squares["h1"],
                }
            )

        print(f"Input image: {image_path}")
        print(f"Original shape: {image.shape}")
        print(f"Resized shape: {resized_image.shape}")
        print(
            "Detected outer corners: "
            f"{board_contour.reshape(4, 2)}"
        )
        print(
            "Checkerboard corner candidates: "
            f"{len(corner_candidates)}"
        )
        print(
            "Complete vertical boundaries: "
            f"{complete_vertical_boundaries}"
        )
        print(
            "Complete horizontal boundaries: "
            f"{complete_horizontal_boundaries}"
        )
        print(
            f"Playable board shape: "
            f"{playable_board.shape}"
        )

        if board_squares is not None:
            print(
                f"Saved square crops: "
                f"{len(board_squares)}"
            )

        display_debug_images(debug_images)

    if return_metadata:
        metadata = {
            "resized_image": resized_image,
            "transform_matrix": transform_matrix,
            "playable_bounds": playable_bounds,
        }

        return playable_board, metadata

    return playable_board


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect and normalize a chessboard from an image."
        )
    )

    parser.add_argument(
        "image_path",
        type=Path,
    )

    parser.add_argument(
        "--debug",
        action="store_true",
    )

    parser.add_argument(
        "--save-squares",
        action="store_true",
    )

    parser.add_argument(
        "--square-output-dir",
        type=Path,
        default=DEFAULT_SQUARE_OUTPUT_DIR,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    try:
        playable_board = process_board_image(
            arguments.image_path,
            debug=arguments.debug,
            save_squares=arguments.save_squares,
            square_output_dir=(
                arguments.square_output_dir
            ),
        )
    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
    ) as error:
        raise SystemExit(
            f"Board processing failed: {error}"
        ) from error

    print(
        "Board processed successfully. "
        f"Normalized shape: {playable_board.shape}"
    )


if __name__ == "__main__":
    main()