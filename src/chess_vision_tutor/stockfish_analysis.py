from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine


DEFAULT_STOCKFISH_PATH = Path(
    "engines/stockfish/"
    "stockfish-windows-x86-64-avx2.exe"
)


@dataclass
class StockfishAnalysis:
    best_move_uci: str
    best_move_san: str
    evaluation_cp: int | None
    mate_in: int | None
    principal_variation_uci: list[str]
    principal_variation_san: list[str]


def score_to_values(
    score: chess.engine.PovScore,
    turn: chess.Color,
) -> tuple[int | None, int | None]:
    relative_score = score.pov(
        turn
    )

    mate_in = relative_score.mate()

    if mate_in is not None:
        return None, mate_in

    centipawns = relative_score.score(
        mate_score=100000
    )

    return centipawns, None


def variation_to_san(
    board: chess.Board,
    moves: list[chess.Move],
) -> list[str]:
    variation_board = board.copy(
        stack=False
    )

    san_moves: list[str] = []

    for move in moves:
        if move not in variation_board.legal_moves:
            break

        san_moves.append(
            variation_board.san(
                move
            )
        )

        variation_board.push(
            move
        )

    return san_moves


def analyze_fen(
    fen: str,
    engine_path: Path = DEFAULT_STOCKFISH_PATH,
    depth: int = 16,
    variation_length: int = 8,
) -> StockfishAnalysis:
    if not engine_path.exists():
        raise FileNotFoundError(
            f"Stockfish executable not found: "
            f"{engine_path}"
        )

    board = chess.Board(
        fen
    )

    if not board.is_valid():
        raise ValueError(
            "Stockfish analysis requires a valid "
            "chess position."
        )

    engine = chess.engine.SimpleEngine.popen_uci(
        str(engine_path)
    )

    try:
        analysis = engine.analyse(
            board,
            chess.engine.Limit(
                depth=depth
            ),
        )

        principal_variation = list(
            analysis.get(
                "pv",
                [],
            )
        )

        if not principal_variation:
            raise RuntimeError(
                "Stockfish returned no principal "
                "variation."
            )

        best_move = principal_variation[0]

        best_move_san = board.san(
            best_move
        )

        evaluation_cp, mate_in = score_to_values(
            analysis["score"],
            board.turn,
        )

        limited_variation = principal_variation[
            :variation_length
        ]

        return StockfishAnalysis(
            best_move_uci=best_move.uci(),
            best_move_san=best_move_san,
            evaluation_cp=evaluation_cp,
            mate_in=mate_in,
            principal_variation_uci=[
                move.uci()
                for move in limited_variation
            ],
            principal_variation_san=variation_to_san(
                board,
                limited_variation,
            ),
        )

    finally:
        engine.quit()


def format_evaluation(
    analysis: StockfishAnalysis,
) -> str:
    if analysis.mate_in is not None:
        return f"Mate in {abs(analysis.mate_in)}"

    if analysis.evaluation_cp is None:
        return "Unknown"

    pawns = analysis.evaluation_cp / 100

    return f"{pawns:+.2f}"


def main() -> None:
    starting_fen = (
        "rnbqkbnr/pppppppp/8/8/8/8/"
        "PPPPPPPP/RNBQKBNR w - - 0 1"
    )

    analysis = analyze_fen(
        starting_fen
    )

    print(
        f"Best move (SAN): "
        f"{analysis.best_move_san}"
    )

    print(
        f"Best move (UCI): "
        f"{analysis.best_move_uci}"
    )

    print(
        f"Evaluation: "
        f"{format_evaluation(analysis)}"
    )

    print(
        "Principal variation (SAN): "
        + " ".join(
            analysis.principal_variation_san
        )
    )


if __name__ == "__main__":
    main()