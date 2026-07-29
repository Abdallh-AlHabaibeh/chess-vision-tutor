from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from chess_vision_tutor.chessred_processing import draw_target_overlay
from chess_vision_tutor.grid_classifier import (
    IMAGE_SIZE,
    GridClassifier,
)
from chess_vision_tutor.main import process_board_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "grid_classifier_spatial.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "local_inference"
)


def load_model(
    device: torch.device,
) -> GridClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model = GridClassifier().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def prepare_board_tensor(
    board_image: np.ndarray,
) -> torch.Tensor:
    board_rgb = cv2.cvtColor(
        board_image,
        cv2.COLOR_BGR2RGB,
    )

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return transform(board_rgb).unsqueeze(0)


@torch.no_grad()
def predict_board(
    image_path: Path,
) -> np.ndarray:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    normalized_board = process_board_image(
        image_path,
        debug=False,
        save_squares=False,
    )

    model = load_model(device)

    board_tensor = prepare_board_tensor(
        normalized_board
    ).to(device)

    logits = model(board_tensor)

    prediction = (
        logits.argmax(dim=2)
        .reshape(8, 8)
        .cpu()
        .numpy()
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    overlay = draw_target_overlay(
        normalized_board,
        prediction,
    )

    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_prediction.jpg"
    )

    cv2.imwrite(
        str(output_path),
        overlay,
    )

    print(f"Device: {device}")
    print(f"Input image: {image_path}")
    print()
    print("Predicted matrix:")
    print(prediction)
    print()
    print(f"Saved prediction: {output_path}")

    return prediction


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image_path",
        type=Path,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    predict_board(
        arguments.image_path
    )


if __name__ == "__main__":
    main()