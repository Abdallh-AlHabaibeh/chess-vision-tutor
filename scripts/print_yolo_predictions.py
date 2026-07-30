from __future__ import annotations

import argparse
from pathlib import Path


CLASS_NAMES = [
    "white_pawn",
    "white_rook",
    "white_knight",
    "white_bishop",
    "white_queen",
    "white_king",
    "black_pawn",
    "black_rook",
    "black_knight",
    "black_bishop",
    "black_queen",
    "black_king",
]


def print_predictions(
    label_path: Path,
) -> None:
    if not label_path.exists():
        raise FileNotFoundError(
            f"Label file not found: {label_path}"
        )

    predictions: list[
        tuple[float, int, float, float, float, float]
    ] = []

    for line in label_path.read_text(
        encoding="utf-8"
    ).splitlines():
        values = line.split()

        if len(values) != 6:
            raise ValueError(
                f"Unexpected YOLO line: {line}"
            )

        class_id = int(values[0])

        x_center = float(values[1])
        y_center = float(values[2])
        width = float(values[3])
        height = float(values[4])
        confidence = float(values[5])

        predictions.append(
            (
                confidence,
                class_id,
                x_center,
                y_center,
                width,
                height,
            )
        )

    predictions.sort(
        key=lambda prediction: prediction[0],
        reverse=True,
    )

    print(
        f"Total detections: {len(predictions)}"
    )
    print()

    for index, prediction in enumerate(
        predictions,
        start=1,
    ):
        (
            confidence,
            class_id,
            x_center,
            y_center,
            width,
            height,
        ) = prediction

        print(
            f"{index:02d}. "
            f"{CLASS_NAMES[class_id]:<14} | "
            f"confidence={confidence:6.2%} | "
            f"center=({x_center:.3f}, {y_center:.3f}) | "
            f"size=({width:.3f}, {height:.3f})"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "label_path",
        type=Path,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    print_predictions(
        arguments.label_path
    )


if __name__ == "__main__":
    main()