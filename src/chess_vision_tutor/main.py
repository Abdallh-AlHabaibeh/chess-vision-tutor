from pathlib import Path

import cv2
import numpy as np

from chess_vision_tutor.board_processing import (
    crop_playable_board,
    detect_board_contour,
    detect_checkerboard_corner_candidates,
    detect_joint_regular_grid_positions,
    draw_board_contour,
    draw_checkerboard_corner_candidates,
    draw_complete_grid_boundaries,
    draw_regular_grid_positions,
    draw_supported_grid_positions,
    infer_complete_chessboard_grid,
    load_image,
    preprocess_image,
    resize_image,
    warp_board,
)


def display_image(
    window_name: str,
    image: np.ndarray,
) -> None:
    cv2.imshow(
        window_name,
        image,
    )


def print_image_information(
    original_image: np.ndarray,
    resized_image: np.ndarray,
    processed_image: np.ndarray,
) -> None:

    print("Image loaded and preprocessed successfully.")
    print(f"Original shape: {original_image.shape}")
    print(f"Resized shape: {resized_image.shape}")
    print(f"Processed shape: {processed_image.shape}")
    print(f"Processed data type: {processed_image.dtype}")


def print_board_information(
    board_contour: np.ndarray,
    warped_board: np.ndarray,
    corner_candidates: np.ndarray,
    supported_vertical_positions: list[int],
    supported_horizontal_positions: list[int],
    regular_vertical_positions: list[int],
    regular_horizontal_positions: list[int],
    complete_vertical_boundaries: list[int],
    complete_horizontal_boundaries: list[int],
    playable_board: np.ndarray,
) -> None:
    
    print("Chessboard contour detected successfully.")
    print(
        "Detected corners: "
        f"{board_contour.reshape(4, 2)}"
    )
    print(f"Warped board shape: {warped_board.shape}")
    print(
        "Detected corner candidates: "
        f"{len(corner_candidates)}"
    )
    print(
        "Supported vertical positions: "
        f"{supported_vertical_positions}"
    )
    print(
        "Supported horizontal positions: "
        f"{supported_horizontal_positions}"
    )
    print(
        "Regular vertical positions: "
        f"{regular_vertical_positions}"
    )
    print(
        "Regular horizontal positions: "
        f"{regular_horizontal_positions}"
    )
    print(
        "Complete vertical boundaries: "
        f"{complete_vertical_boundaries}"
    )
    print(
        "Complete horizontal boundaries: "
        f"{complete_horizontal_boundaries}"
    )
    print(f"Playable board shape: {playable_board.shape}")


def wait_for_windows() -> None:
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    image_path = Path("data/raw/board5_test.jpg")

    image = load_image(image_path)
    resized_image = resize_image(image)
    processed_image = preprocess_image(resized_image)

    print_image_information(
        image,
        resized_image,
        processed_image,
    )

    display_image(
        "Original Image",
        resized_image,
    )

    display_image(
        "Detected Edges",
        processed_image,
    )

    board_contour = detect_board_contour(
        processed_image,
    )

    if board_contour is None:
        print("No chessboard contour was detected.")
        wait_for_windows()
        return

    contour_image = draw_board_contour(
        resized_image,
        board_contour,
    )

    warped_board = warp_board(
        resized_image,
        board_contour,
    )

    corner_candidates = (
        detect_checkerboard_corner_candidates(
            warped_board,
        )
    )

    corner_candidate_image = (
        draw_checkerboard_corner_candidates(
            warped_board,
            corner_candidates,
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
        print(
            "Could not detect a reliable 7x7 internal grid."
        )

        display_image(
            "Detected Chessboard Contour",
            contour_image,
        )

        display_image(
            "Warped Chessboard",
            warped_board,
        )

        display_image(
            "Checkerboard Corner Candidates",
            corner_candidate_image,
        )

        wait_for_windows()
        return

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

    supported_grid_image = (
        draw_supported_grid_positions(
            warped_board,
            supported_vertical_positions,
            supported_horizontal_positions,
        )
    )

    regular_grid_image = (
        draw_regular_grid_positions(
            warped_board,
            regular_vertical_positions,
            regular_horizontal_positions,
        )
    )

    complete_grid_image = (
        draw_complete_grid_boundaries(
            warped_board,
            complete_vertical_boundaries,
            complete_horizontal_boundaries,
        )
    )

    print_board_information(
        board_contour,
        warped_board,
        corner_candidates,
        supported_vertical_positions,
        supported_horizontal_positions,
        regular_vertical_positions,
        regular_horizontal_positions,
        complete_vertical_boundaries,
        complete_horizontal_boundaries,
        playable_board,
    )

    display_image(
        "Detected Chessboard Contour",
        contour_image,
    )

    display_image(
        "Warped Chessboard",
        warped_board,
    )

    display_image(
        "Checkerboard Corner Candidates",
        corner_candidate_image,
    )

    display_image(
        "Supported Grid Positions",
        supported_grid_image,
    )

    display_image(
        "Regular Grid Positions",
        regular_grid_image,
    )

    display_image(
        "Complete 9x9 Grid",
        complete_grid_image,
    )

    display_image(
        "Cropped Playable Board",
        playable_board,
    )

    wait_for_windows()


if __name__ == "__main__":
    main()