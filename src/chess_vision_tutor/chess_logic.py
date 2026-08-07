from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np


CLASS_TO_FEN = {
    0: "P",
    1: "R",
    2: "N",
    3: "B",
    4: "Q",
    5: "K",
    6: "p",
    7: "r",
    8: "n",
    9: "b",
    10: "q",
    11: "k",
    12: None,
}


@dataclass
class ValidationResult:
    fen: str
    is_valid: bool
    status: int
    issues: list[str]


def matrix_to_board_fen(
    board_matrix: np.ndarray | list[list[int]],
) -> str:
    matrix = np.asarray(
        board_matrix,
        dtype=np.int64,
    )

    if matrix.shape != (8, 8):
        raise ValueError(
            "Board matrix must have shape (8, 8)."
        )

    fen_rows: list[str] = []

    for row in matrix:
        empty_count = 0
        fen_row = ""

        for value in row:
            piece_class = int(value)

            if piece_class not in CLASS_TO_FEN:
                raise ValueError(
                    f"Unknown piece class: {piece_class}"
                )

            fen_piece = CLASS_TO_FEN[
                piece_class
            ]

            if fen_piece is None:
                empty_count += 1
                continue

            if empty_count > 0:
                fen_row += str(empty_count)
                empty_count = 0

            fen_row += fen_piece

        if empty_count > 0:
            fen_row += str(empty_count)

        fen_rows.append(fen_row)

    return "/".join(fen_rows)


def build_fen(
    board_matrix: np.ndarray | list[list[int]],
    white_to_move: bool = True,
) -> str:
    board_fen = matrix_to_board_fen(
        board_matrix
    )

    active_color = (
        "w"
        if white_to_move
        else "b"
    )

    return (
        f"{board_fen} "
        f"{active_color} "
        f"- - 0 1"
    )


def describe_status(
    status: int,
) -> list[str]:
    status_messages = {
        chess.STATUS_NO_WHITE_KING:
            "The board has no white king.",
        chess.STATUS_NO_BLACK_KING:
            "The board has no black king.",
        chess.STATUS_TOO_MANY_KINGS:
            "The board has too many kings.",
        chess.STATUS_TOO_MANY_WHITE_PAWNS:
            "The board has more than eight white pawns.",
        chess.STATUS_TOO_MANY_BLACK_PAWNS:
            "The board has more than eight black pawns.",
        chess.STATUS_PAWNS_ON_BACKRANK:
            "A pawn is located on the first or eighth rank.",
        chess.STATUS_TOO_MANY_WHITE_PIECES:
            "The board has too many white pieces.",
        chess.STATUS_TOO_MANY_BLACK_PIECES:
            "The board has too many black pieces.",
        chess.STATUS_OPPOSITE_CHECK:
            "Both kings appear to be in check.",
        chess.STATUS_TOO_MANY_CHECKERS:
            "The position contains too many checking pieces.",
        chess.STATUS_IMPOSSIBLE_CHECK:
            "The position contains an impossible check.",
    }

    issues: list[str] = []

    for flag, message in status_messages.items():
        if status & flag:
            issues.append(message)

    if (
        status != chess.STATUS_VALID
        and not issues
    ):
        issues.append(
            f"Unknown validation status: {status}"
        )

    return issues


def validate_position(
    board_matrix: np.ndarray | list[list[int]],
    white_to_move: bool = True,
) -> ValidationResult:
    fen = build_fen(
        board_matrix,
        white_to_move=white_to_move,
    )

    board = chess.Board(
        fen
    )

    status = board.status()

    return ValidationResult(
        fen=fen,
        is_valid=(
            status == chess.STATUS_VALID
        ),
        status=int(status),
        issues=describe_status(
            status
        ),
    )


def main() -> None:
    starting_position = np.array(
        [
            [7, 8, 9, 10, 11, 9, 8, 7],
            [6, 6, 6, 6, 6, 6, 6, 6],
            [12, 12, 12, 12, 12, 12, 12, 12],
            [12, 12, 12, 12, 12, 12, 12, 12],
            [12, 12, 12, 12, 12, 12, 12, 12],
            [12, 12, 12, 12, 12, 12, 12, 12],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 2, 3, 4, 5, 3, 2, 1],
        ],
        dtype=np.int64,
    )

    result = validate_position(
        starting_position
    )

    print(f"FEN: {result.fen}")
    print(f"Valid: {result.is_valid}")
    print(f"Status: {result.status}")
    print(f"Issues: {result.issues}")


if __name__ == "__main__":
    main()