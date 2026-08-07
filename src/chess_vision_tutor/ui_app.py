from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from chess_vision_tutor.chess_logic import (
    validate_position,
)
from chess_vision_tutor.ensemble_inference import (
    run_ensemble,
)
from chess_vision_tutor.stockfish_analysis import (
    analyze_fen,
    format_evaluation,
)
from chess_vision_tutor.tutor_chat import (
    ask_gemini_tutor,
)
from chess_vision_tutor.ui_board import (
    render_board,
)


PIECE_NAMES = {
    0: "White pawn",
    1: "White rook",
    2: "White knight",
    3: "White bishop",
    4: "White queen",
    5: "White king",
    6: "Black pawn",
    7: "Black rook",
    8: "Black knight",
    9: "Black bishop",
    10: "Black queen",
    11: "Black king",
    12: "Empty",
}


ORIENTATION_OPTIONS = {
    "a1": "a1 — White side is at the bottom",
    "h8": "h8 — Black side is at the bottom",
    "a8": "a8 — White side is on the left",
    "h1": "h1 — White side is on the right",
}


def orient_board_matrix(
    board_matrix: list[list[int]],
    bottom_left_square: str,
) -> list[list[int]]:
    """Rotate the photographed board into standard FEN orientation."""
    matrix = [
        list(row)
        for row in board_matrix
    ]

    if bottom_left_square == "a1":
        return matrix

    if bottom_left_square == "h8":
        return [
            list(reversed(row))
            for row in reversed(matrix)
        ]

    if bottom_left_square == "a8":
        return [
            [
                matrix[
                    len(matrix) - 1 - column
                ][row]
                for column in range(len(matrix))
            ]
            for row in range(len(matrix))
        ]

    if bottom_left_square == "h1":
        return [
            [
                matrix[column][
                    len(matrix) - 1 - row
                ]
                for column in range(len(matrix))
            ]
            for row in range(len(matrix))
        ]

    raise ValueError(
        f"Unsupported board orientation: {bottom_left_square}"
    )


def initialize_inference(
    uploaded_image,
) -> None:
    uploaded_bytes = uploaded_image.getvalue()

    image_hash = hashlib.sha256(
        uploaded_bytes
    ).hexdigest()

    if (
        st.session_state.get("image_hash")
        == image_hash
    ):
        return

    temporary_path: Path | None = None

    try:
        suffix = Path(
            uploaded_image.name
        ).suffix

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temporary_file:
            temporary_file.write(
                uploaded_bytes
            )

            temporary_path = Path(
                temporary_file.name
            )

        with st.spinner(
            "Analyzing the chessboard..."
        ):
            (
                proposed_matrix,
                status_matrix,
                warning_squares,
                required_review_squares,
            ) = run_ensemble(
                temporary_path
            )

        st.session_state.image_hash = image_hash
        st.session_state.board_matrix = (
            proposed_matrix.tolist()
        )
        st.session_state.status_matrix = (
            status_matrix.tolist()
        )
        st.session_state.warning_squares = (
            warning_squares
        )
        st.session_state.required_review_squares = (
            required_review_squares
        )
        st.session_state.selected_square = None
        st.session_state.pop(
            "validation_result",
            None,
        )
        st.session_state.pop(
            "stockfish_result",
            None,
        )
        st.session_state.tutor_messages = []

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()


def find_review_item(
    row: int,
    column: int,
) -> dict[str, object] | None:
    all_items = (
        st.session_state.warning_squares
        + st.session_state.required_review_squares
    )

    for item in all_items:
        if (
            int(item["row"]) == row
            and int(item["column"]) == column
        ):
            return item

    return None


def clear_analysis_state() -> None:
    st.session_state.pop(
        "validation_result",
        None,
    )
    st.session_state.pop(
        "stockfish_result",
        None,
    )
    st.session_state.tutor_messages = []


def clear_square_review(
    row: int,
    column: int,
) -> None:
    st.session_state.warning_squares = [
        item
        for item in st.session_state.warning_squares
        if not (
            int(item["row"]) == row
            and int(item["column"]) == column
        )
    ]

    st.session_state.required_review_squares = [
        item
        for item in st.session_state.required_review_squares
        if not (
            int(item["row"]) == row
            and int(item["column"]) == column
        )
    ]



def confirm_all_warnings() -> None:
    warning_positions = {
        (
            int(item["row"]),
            int(item["column"]),
        )
        for item in st.session_state.warning_squares
    }

    for row, column in warning_positions:
        st.session_state.status_matrix[
            row
        ][column] = 0

    st.session_state.warning_squares = []
    clear_analysis_state()
    st.session_state.selected_square = None
    st.session_state.pop(
        "manual_review_square",
        None,
    )



def confidence_text(
    value: object,
) -> str:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return "—"

    return f"{confidence:.2%}"


def render_model_details(
    review_item: dict[str, object] | None,
) -> None:
    if review_item is None:
        return

    model_1_class = int(
        review_item["model_1_class"]
    )
    model_2_class = int(
        review_item["model_2_class"]
    )

    first_column, second_column = st.columns(2)

    with first_column:
        st.write(
            "Model 1: "
            f"**{PIECE_NAMES[model_1_class]}**"
        )
        st.caption(
            confidence_text(
                review_item[
                    "model_1_confidence"
                ]
            )
        )

    with second_column:
        if model_2_class == 12:
            st.write("Model 2: **No detection**")
        else:
            st.write(
                "Model 2: "
                f"**{PIECE_NAMES[model_2_class]}**"
            )

        st.caption(
            confidence_text(
                review_item[
                    "model_2_confidence"
                ]
            )
        )

    top_classes = review_item.get(
        "model_1_top_classes",
        [],
    )
    top_confidences = review_item.get(
        "model_1_top_confidences",
        [],
    )

    if top_classes and top_confidences:
        alternatives = []

        for piece_class, confidence in zip(
            top_classes,
            top_confidences,
        ):
            alternatives.append(
                f"{PIECE_NAMES[int(piece_class)]} "
                f"({float(confidence):.1%})"
            )

        st.caption(
            "Model 1 top predictions: "
            + " · ".join(alternatives)
        )


def create_piece_options(
    review_item: dict[str, object] | None,
    current_class: int,
) -> list[int]:
    recommended: list[int] = []

    if review_item is not None:
        model_1_class = int(
            review_item["model_1_class"]
        )
        model_2_class = int(
            review_item["model_2_class"]
        )

        candidate_classes = [
            model_1_class,
            model_2_class,
            *[
                int(value)
                for value in review_item.get(
                    "model_1_top_classes",
                    [],
                )
            ],
        ]

        for piece_class in candidate_classes:
            if piece_class not in recommended:
                recommended.append(piece_class)

    if current_class not in recommended:
        recommended.insert(
            0,
            current_class,
        )

    for piece_class in range(13):
        if piece_class not in recommended:
            recommended.append(piece_class)

    return recommended


def finish_square_action(
    row: int,
    column: int,
) -> None:
    st.session_state.selected_square = None
    st.session_state.pop(
        "manual_review_square",
        None,
    )
    st.session_state.suppressed_board_click = (
        row,
        column,
    )


def render_square_editor() -> None:
    selected_square = st.session_state.get(
        "selected_square"
    )

    if selected_square is None:
        st.info(
            "Click any square on the digital board "
            "to confirm or edit it."
        )
        return

    row, column = selected_square

    current_class = int(
        st.session_state.board_matrix[
            row
        ][
            column
        ]
    )

    current_status = int(
        st.session_state.status_matrix[
            row
        ][
            column
        ]
    )

    review_item = find_review_item(
        row,
        column,
    )

    manual_review_square = (
        st.session_state.get(
            "manual_review_square"
        )
    )

    show_compact_warning = (
        current_status == 1
        and manual_review_square
        != (row, column)
    )

    if show_compact_warning:
        st.markdown(
            f"#### Warning — Row {row + 1}, "
            f"Column {column + 1}"
        )

        st.write(
            f"Current piece: "
            f"**{PIECE_NAMES[current_class]}**"
        )

        confirm_column, change_column = st.columns(
            2
        )

        with confirm_column:
            if st.button(
                "✓ Confirm current piece",
                type="primary",
                width="stretch",
                key=f"confirm_{row}_{column}",
            ):
                st.session_state.status_matrix[
                    row
                ][
                    column
                ] = 0

                clear_square_review(
                    row,
                    column,
                )
                clear_analysis_state()

                finish_square_action(
                    row,
                    column,
                )
                st.rerun()

        with change_column:
            if st.button(
                "Change piece",
                width="stretch",
                key=f"show_change_{row}_{column}",
            ):
                st.session_state.manual_review_square = (
                    row,
                    column,
                )
                st.rerun()

        return

    status_name = {
        0: "Accepted",
        1: "Warning",
        2: "Required review",
    }[current_status]

    st.markdown(
        f"### Selected square — Row {row + 1}, "
        f"Column {column + 1}"
    )

    st.write(
        f"Current piece: "
        f"**{PIECE_NAMES[current_class]}**"
    )

    st.caption(
        f"Status: {status_name}"
    )

    render_model_details(
        review_item
    )

    piece_options = create_piece_options(
        review_item,
        current_class,
    )

    selected_class = st.selectbox(
        "Change piece",
        options=piece_options,
        index=piece_options.index(
            current_class
        ),
        format_func=lambda value: PIECE_NAMES[
            value
        ],
        key=(
            f"piece_editor_"
            f"{st.session_state.image_hash}_"
            f"{row}_{column}"
        ),
    )

    button_column, cancel_column = st.columns(2)

    with button_column:
        if st.button(
            "Apply change",
            type="primary",
            width="stretch",
            key=f"apply_{row}_{column}",
        ):
            st.session_state.board_matrix[
                row
            ][
                column
            ] = int(selected_class)

            st.session_state.status_matrix[
                row
            ][
                column
            ] = 0

            clear_square_review(
                row,
                column,
            )
            clear_analysis_state()

            finish_square_action(
                row,
                column,
            )
            st.rerun()

    with cancel_column:
        if st.button(
            "Cancel",
            width="stretch",
            key=f"cancel_{row}_{column}",
        ):
            finish_square_action(
                row,
                column,
            )
            st.rerun()




def render_tutor_chat(
    *,
    validation_result: dict[str, object],
    stockfish_result: dict[str, object],
) -> None:
    st.divider()
    st.subheader("Chess Tutor")

    if "tutor_messages" not in st.session_state:
        st.session_state.tutor_messages = []

    for message in st.session_state.tutor_messages:
        with st.chat_message(
            message["role"]
        ):
            st.write(
                message["content"]
            )

    user_message = st.chat_input(
        "Ask for a hint or ask about the position..."
    )

    if not user_message:
        return

    st.session_state.tutor_messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    with st.chat_message("user"):
        st.write(user_message)

    try:
        api_key = st.secrets[
            "GEMINI_API_KEY"
        ]
        model_name = st.secrets[
            "GEMINI_MODEL"
        ]

        with st.chat_message("assistant"):
            with st.spinner(
                "Tutor is thinking..."
            ):
                answer = ask_gemini_tutor(
                    api_key=api_key,
                    model_name=model_name,
                    fen=str(
                        validation_result["fen"]
                    ),
                    best_move_san=str(
                        stockfish_result[
                            "best_move_san"
                        ]
                    ),
                    best_move_uci=str(
                        stockfish_result[
                            "best_move_uci"
                        ]
                    ),
                    evaluation=str(
                        stockfish_result[
                            "evaluation"
                        ]
                    ),
                    principal_variation=list(
                        stockfish_result[
                            "principal_variation_san"
                        ]
                    ),
                    chat_history=(
                        st.session_state.tutor_messages[
                            :-1
                        ]
                    ),
                    user_message=user_message,
                )

                st.write(answer)

        st.session_state.tutor_messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

    except KeyError:
        st.error(
            "Gemini secrets are missing. Add "
            "GEMINI_API_KEY and GEMINI_MODEL."
        )

    except Exception as error:
        st.error(
            f"Tutor request failed: {error}"
        )

def render_tutor_setup() -> None:
    st.divider()

    with st.expander(
        "Board setup",
        expanded=True,
    ):
        first_column, second_column = st.columns(2)

        with first_column:
            bottom_left_square = st.selectbox(
                "Bottom-left square in the photo",
                options=list(
                    ORIENTATION_OPTIONS
                ),
                format_func=lambda value: (
                    ORIENTATION_OPTIONS[value]
                ),
                key="bottom_left_square",
            )

        with second_column:
            side_to_move = st.selectbox(
                "Who moves next?",
                options=[
                    "White",
                    "Black",
                ],
                key="side_to_move",
            )

        start_tutor = st.button(
            "Start tutor",
            type="primary",
            width="stretch",
        )

    if start_tutor:
        oriented_matrix = orient_board_matrix(
            st.session_state.board_matrix,
            bottom_left_square,
        )

        result = validate_position(
            oriented_matrix,
            white_to_move=(
                side_to_move == "White"
            ),
        )

        st.session_state.validation_result = {
            "fen": result.fen,
            "is_valid": result.is_valid,
            "status": result.status,
            "issues": result.issues,
        }

        st.session_state.pop(
            "stockfish_result",
            None,
        )
        st.session_state.tutor_messages = []

        if result.is_valid:
            try:
                with st.spinner(
                    "Preparing the tutor..."
                ):
                    analysis = analyze_fen(
                        result.fen
                    )

                st.session_state.stockfish_result = {
                    "best_move_san":
                        analysis.best_move_san,
                    "best_move_uci":
                        analysis.best_move_uci,
                    "evaluation":
                        format_evaluation(
                            analysis
                        ),
                    "principal_variation_san":
                        analysis.principal_variation_san,
                }

            except Exception as error:
                st.error(
                    f"Could not prepare the tutor: {error}"
                )
                return

    validation_result = st.session_state.get(
        "validation_result"
    )

    if validation_result is None:
        return

    if not validation_result["is_valid"]:
        st.error(
            "The position needs correction before tutoring."
        )

        for issue in validation_result["issues"]:
            st.write(
                f"• {issue}"
            )

        return

    stockfish_result = st.session_state.get(
        "stockfish_result"
    )

    if stockfish_result is None:
        return


    render_tutor_chat(
        validation_result=validation_result,
        stockfish_result=stockfish_result,
    )

def main() -> None:
    st.set_page_config(
        page_title="Chess Vision Tutor",
        page_icon="♟️",
        layout="wide",
    )

    st.title("Chess Vision Tutor")

    st.write(
        "Upload a board photo, correct uncertain squares, "
        "and ask the tutor for guidance."
    )

    uploaded_image = st.file_uploader(
        "Upload chessboard image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
    )

    if uploaded_image is None:
        st.info(
            "Upload an image to begin."
        )
        return

    try:
        initialize_inference(
            uploaded_image
        )

    except Exception as error:
        st.error(
            f"Could not process the image: {error}"
        )
        return

    left_column, right_column = st.columns(
        2,
        gap="large",
    )

    with left_column:
        st.subheader("Original Image")

        st.image(
            uploaded_image.getvalue(),
            width="stretch",
        )

    with right_column:
        st.subheader("Digital Board")

        clicked_square = render_board(
            st.session_state.board_matrix,
            st.session_state.status_matrix,
        )

        if clicked_square is not None:
            clicked_row, clicked_column = (
                clicked_square
            )

            clicked_position = (
                int(clicked_row),
                int(clicked_column),
            )

            suppressed_click = (
                st.session_state.get(
                    "suppressed_board_click"
                )
            )

            if clicked_position != suppressed_click:
                st.session_state.pop(
                    "suppressed_board_click",
                    None,
                )

                if (
                    st.session_state.get(
                        "selected_square"
                    )
                    != clicked_position
                ):
                    st.session_state.selected_square = (
                        clicked_position
                    )
                    st.session_state.pop(
                        "manual_review_square",
                        None,
                    )
                    st.rerun()

        warning_count = len(
            st.session_state.warning_squares
        )

        if warning_count > 0:
            if st.button(
                f"Confirm all {warning_count} warnings",
                width="stretch",
                key="confirm_all_warnings",
            ):
                confirm_all_warnings()
                st.rerun()

        render_square_editor()

    render_tutor_setup()

    

if __name__ == "__main__":
    main()