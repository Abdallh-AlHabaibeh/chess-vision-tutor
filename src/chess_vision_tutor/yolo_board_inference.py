from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from chess_vision_tutor.main import process_board_image
from chess_vision_tutor.yolo_square_mapping import (
    map_box_to_square,
)


EMPTY_CLASS = 12


def predict_yolo_board(
    image_path: str | Path,
    model_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    _, metadata = process_board_image(
        image_path,
        return_metadata=True,
    )

    resized_image = metadata["resized_image"]
    transform_matrix = metadata["transform_matrix"]
    playable_bounds = metadata["playable_bounds"]

    debug_image = resized_image.copy()

    model = YOLO(str(model_path))

    results = model.predict(
        source=resized_image,
        imgsz=640,
        conf=0.25,
        iou=0.5,
        agnostic_nms=True,
        device=0,
        verbose=False,
    )

    prediction_matrix = np.full(
        (8, 8),
        EMPTY_CLASS,
        dtype=np.int64,
    )

    confidence_matrix = np.zeros(
        (8, 8),
        dtype=np.float32,
    )

    boxes = results[0].boxes

    if boxes is None:
        return prediction_matrix, confidence_matrix

    for coordinates, class_id, confidence in zip(
        boxes.xyxy.cpu().numpy(),
        boxes.cls.cpu().numpy(),
        boxes.conf.cpu().numpy(),
        strict=True,
    ):
        mapped_square = map_box_to_square(
            tuple(float(value) for value in coordinates),
            transform_matrix,
            playable_bounds,
        )

        if mapped_square is None:
            continue

        row, column, playable_x, playable_y = (
            mapped_square
        )

        if confidence <= confidence_matrix[row, column]:
            continue

        prediction_matrix[row, column] = int(class_id)
        confidence_matrix[row, column] = float(confidence)

        x_min, y_min, x_max, y_max = (
            int(value) for value in coordinates
        )

        anchor_x = int(
            (x_min + x_max) / 2
        )
        anchor_y = y_max

        cv2.rectangle(
            debug_image,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2,
        )

        cv2.circle(
            debug_image,
            (anchor_x, anchor_y),
            6,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            debug_image,
            f"r{row} c{column}",
            (
                x_min,
                max(20, y_min - 8),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

        print(
            f"class={int(class_id):2d} | "
            f"confidence={confidence:6.2%} | "
            f"board_point=({playable_x:7.2f}, "
            f"{playable_y:7.2f}) | "
            f"row={row}, column={column}"
        )

    output_path = Path(
        "outputs/yolo_square_mapping_debug.jpg"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        debug_image,
    )

    print(
        f"\nSaved mapping debug image: "
        f"{output_path}"
    )

    return prediction_matrix, confidence_matrix


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image_path",
        type=Path,
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "runs/detect/outputs/yolo_15_epochs/"
            "chess_piece_detector/weights/best.pt"
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    prediction_matrix, confidence_matrix = (
        predict_yolo_board(
            arguments.image_path,
            arguments.model,
        )
    )

    print("\nYOLO square matrix:")
    print(prediction_matrix)

    print("\nYOLO confidence matrix:")
    print(
        np.round(
            confidence_matrix,
            3,
        )
    )


if __name__ == "__main__":
    main()