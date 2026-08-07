from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import (
    streamlit_image_coordinates,
)


PIECE_SYMBOLS = {
    0: "♙",
    1: "♖",
    2: "♘",
    3: "♗",
    4: "♕",
    5: "♔",
    6: "♟",
    7: "♜",
    8: "♞",
    9: "♝",
    10: "♛",
    11: "♚",
    12: "",
}

BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8

LIGHT_SQUARE = "#f0d9b5"
DARK_SQUARE = "#b58863"
WARNING_COLOR = "#f4c430"
REVIEW_COLOR = "#d62828"


def load_piece_font(
    size: int,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/seguisym.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path(
            "/usr/share/fonts/truetype/"
            "dejavu/DejaVuSans.ttf"
        ),
    ]

    for font_path in candidates:
        if font_path.exists():
            return ImageFont.truetype(
                str(font_path),
                size=size,
            )

    return ImageFont.load_default()


def create_board_image(
    board_matrix: list[list[int]],
    status_matrix: list[list[int]] | None = None,
) -> Image.Image:
    image = Image.new(
        "RGB",
        (BOARD_SIZE, BOARD_SIZE),
        "white",
    )

    draw = ImageDraw.Draw(image)
    font = load_piece_font(58)

    for row in range(8):
        for column in range(8):
            left = column * SQUARE_SIZE
            top = row * SQUARE_SIZE
            right = left + SQUARE_SIZE
            bottom = top + SQUARE_SIZE

            square_color = (
                LIGHT_SQUARE
                if (row + column) % 2 == 0
                else DARK_SQUARE
            )

            draw.rectangle(
                [left, top, right, bottom],
                fill=square_color,
            )

            status = (
                status_matrix[row][column]
                if status_matrix is not None
                else 0
            )

            if status == 1:
                draw.rectangle(
                    [left + 3, top + 3, right - 3, bottom - 3],
                    outline=WARNING_COLOR,
                    width=6,
                )

            elif status == 2:
                draw.rectangle(
                    [left + 3, top + 3, right - 3, bottom - 3],
                    outline=REVIEW_COLOR,
                    width=6,
                )

            piece_class = int(
                board_matrix[row][column]
            )

            piece_symbol = PIECE_SYMBOLS[
                piece_class
            ]

            if not piece_symbol:
                continue

            bounding_box = draw.textbbox(
                (0, 0),
                piece_symbol,
                font=font,
            )

            text_width = (
                bounding_box[2]
                - bounding_box[0]
            )

            text_height = (
                bounding_box[3]
                - bounding_box[1]
            )

            x = (
                left
                + (SQUARE_SIZE - text_width) / 2
            )

            y = (
                top
                + (SQUARE_SIZE - text_height) / 2
                - bounding_box[1]
            )

            piece_color = (
                "white"
                if piece_class <= 5
                else "black"
            )

            stroke_color = (
                "black"
                if piece_class <= 5
                else None
            )

            draw.text(
                (x, y),
                piece_symbol,
                font=font,
                fill=piece_color,
                stroke_width=1 if stroke_color else 0,
                stroke_fill=stroke_color,
            )

    draw.rectangle(
        [0, 0, BOARD_SIZE - 1, BOARD_SIZE - 1],
        outline="#2f2f2f",
        width=4,
    )

    return image


def render_board(
    board_matrix: list[list[int]],
    status_matrix: list[list[int]] | None = None,
) -> tuple[int, int] | None:
    board_image = create_board_image(
        board_matrix,
        status_matrix,
    )

    click = streamlit_image_coordinates(
        board_image,
        key="digital_chessboard",
    )

    if click is None:
        return None

    column = min(
        int(click["x"] // SQUARE_SIZE),
        7,
    )

    row = min(
        int(click["y"] // SQUARE_SIZE),
        7,
    )

    return row, column