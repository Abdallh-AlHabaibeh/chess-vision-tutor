from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    Swin_T_Weights,
    swin_t,
)


NUMBER_OF_BOARD_ROWS = 8
NUMBER_OF_BOARD_COLUMNS = 8
NUMBER_OF_CLASSES = 13
SWIN_FEATURE_CHANNELS = 768


class SwinGridClassifier(nn.Module):
    """
    Model 2: Swin Transformer Tiny spatial grid classifier.

    Input:
        [batch_size, 3, image_height, image_width]

    Output:
        [batch_size, 64, 13]
    """

    def __init__(
        self,
        number_of_classes: int = NUMBER_OF_CLASSES,
        use_pretrained_weights: bool = True,
    ) -> None:
        super().__init__()

        weights = (
            Swin_T_Weights.DEFAULT
            if use_pretrained_weights
            else None
        )

        backbone = swin_t(
            weights=weights,
        )

        # Keep only the Swin spatial feature extractor.
        # The original ImageNet pooling and classification
        # layers are not used.
        self.features = backbone.features
        self.norm = backbone.norm

        self.spatial_head = nn.Sequential(
            nn.Conv2d(
                in_channels=SWIN_FEATURE_CHANNELS,
                out_channels=256,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.Dropout(p=0.20),
            nn.Conv2d(
                in_channels=256,
                out_channels=number_of_classes,
                kernel_size=1,
            ),
        )

        self.board_pool = nn.AdaptiveAvgPool2d(
            (
                NUMBER_OF_BOARD_ROWS,
                NUMBER_OF_BOARD_COLUMNS,
            )
        )

        self.number_of_classes = number_of_classes

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        # Torchvision Swin feature layout:
        # [batch, height, width, channels]
        features = self.features(images)
        features = self.norm(features)

        # Convert to convolution layout:
        # [batch, channels, height, width]
        features = features.permute(
            0,
            3,
            1,
            2,
        )

        square_logits = self.spatial_head(
            features
        )

        square_logits = self.board_pool(
            square_logits
        )

        # [batch, 13, 8, 8]
        # becomes [batch, 8, 8, 13]
        square_logits = square_logits.permute(
            0,
            2,
            3,
            1,
        )

        return square_logits.reshape(
            images.size(0),
            64,
            self.number_of_classes,
        )


def test_model_shape() -> None:
    model = SwinGridClassifier(
        use_pretrained_weights=False,
    )

    dummy_batch = torch.randn(
        2,
        3,
        256,
        256,
    )

    with torch.no_grad():
        output = model(dummy_batch)

    expected_shape = (
        2,
        64,
        NUMBER_OF_CLASSES,
    )

    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected output shape: "
            f"{tuple(output.shape)}. "
            f"Expected: {expected_shape}"
        )

    print(
        "Swin Model 2 output shape:",
        tuple(output.shape),
    )


if __name__ == "__main__":
    test_model_shape()