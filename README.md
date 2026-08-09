# Chess Vision Tutor

Chess Vision Tutor is a multimodal AI system that converts a photograph of a physical chessboard into a digital chess position, validates the reconstructed board, analyzes it with Stockfish, and provides guided explanations through an LLM-based chess tutor.

The project combines classical computer vision, two neural recognition architectures, chess-specific validation, engine analysis, and human-in-the-loop correction in one end-to-end application.

---

## Overview

The final pipeline is:

```text
Chessboard photo
    ↓
OpenCV board localization
    ↓
Perspective correction
    ↓
Playable 8×8 board reconstruction
    ↓
ResNet18 + YOLO11 recognition
    ↓
YOLO-primary disagreement/review routing
    ↓
Interactive user correction
    ↓
Board orientation + side-to-move selection
    ↓
FEN generation and python-chess validation
    ↓
Stockfish analysis
    ↓
Gemini-based conversational chess tutor
```

The application is designed around a simple principle:

> Use the strongest model as the primary prediction source, preserve independent disagreement signals, and require user confirmation when the vision system is uncertain.

---

## Features

- Physical chessboard image upload
- OpenCV chessboard contour detection
- Perspective correction
- Playable-board reconstruction
- 64-square coordinate mapping
- ResNet18 spatial full-board classifier
- YOLO11 chess-piece detector
- YOLO-primary multi-model review routing
- Per-square warning and required-review states
- Interactive digital board correction
- Board orientation selection
- Side-to-move selection
- FEN generation
- Chess-position validation with `python-chess`
- Stockfish engine analysis
- Conversational chess tutoring with Gemini
- Formal held-out evaluation
- External stress testing and failure analysis

---

## Vision Architecture

### Board Processing

The preprocessing pipeline uses OpenCV to isolate and normalize the chessboard before neural inference.

Main stages:

1. Detect the outer chessboard contour.
2. Order the board corners consistently.
3. Apply a perspective transform.
4. Detect internal grid structure.
5. Infer the complete 9×9 square-boundary layout.
6. Crop the playable 8×8 board.
7. Pass the normalized board to both recognition models.

The geometry pipeline was intentionally kept deterministic and explainable rather than replaced with a large end-to-end vision model.

---

## Recognition Models

### ResNet18 Spatial Grid Classifier

The ResNet18 model receives the normalized full-board image and predicts all 64 squares directly.

Output:

```text
64 × 13 classes
```

The 13 classes are:

- 6 White piece classes
- 6 Black piece classes
- Empty square

ResNet18 is retained as an independent secondary signal in the final system.

---

### YOLO11 Detector

YOLO11 detects and classifies physical chess pieces on the normalized board.

Each detection is mapped to a logical chess square using its board-relative position.

YOLO11 is the primary recognition model in the final architecture because it produced the strongest formal held-out results.

---

## Final Multi-Model Routing

The first version of the ensemble was ResNet18-heavy and allowed the weaker model to influence too many predictions.

Formal evaluation showed that this reduced accuracy.

The final system was therefore redesigned around a YOLO-primary strategy:

```text
YOLO11 provides the default board prediction.

ResNet18 acts as an independent second opinion.

Agreement:
    accept the prediction

Strong YOLO prediction + strong ResNet disagreement:
    keep YOLO
    raise a warning

Low-confidence disagreement:
    require user review

YOLO predicts empty while ResNet predicts a piece:
    do not silently overwrite
    flag the square for review
```

The resulting architecture is better described as:

> **Primary-model + secondary-review routing**

rather than a traditional voting ensemble.

---

## Human-in-the-Loop Review

The application displays the reconstructed position as an interactive digital board.

Users can:

- click any square,
- inspect uncertain model predictions,
- compare ResNet18 and YOLO11 outputs,
- confirm the current piece,
- replace an incorrect piece,
- confirm all warnings,
- correct the final board before chess analysis begins.

This review layer is intentional.

The system does not assume that arbitrary real-world chess photographs can always be reconstructed perfectly.

---

## Chess Logic

After the board is confirmed, the user selects:

- board orientation,
- side to move.

The system then:

1. Converts the board into standard orientation.
2. Generates a FEN position.
3. Validates the reconstructed position with `python-chess`.
4. Rejects invalid positions before engine analysis.

---

## Stockfish Analysis

Valid positions are analyzed using Stockfish.

The engine provides internal context including:

- best move,
- evaluation,
- principal variation.

The UI does not immediately expose the engine answer as the main experience.

Instead, the engine output is passed to the tutor as hidden chess context.

---

## Conversational Tutor

The tutor uses Gemini to explain the position conversationally.

The goal is not simply to reveal the best engine move.

Instead, the tutor can:

- provide hints,
- explain tactical ideas,
- discuss candidate moves,
- answer questions about the current position,
- use Stockfish analysis as hidden supporting context.

This keeps the system closer to a chess tutor than a move-prediction interface.

---

# Evaluation

## Dataset

Formal evaluation uses the nested ChessReD2K test split.

Local split sizes:

```text
Training:    1,442
Validation:    330
Test:          306
```

The formal test set contains 306 held-out images.

Recognition metrics are calculated only on boards that successfully pass preprocessing.

Evaluation is orientation-normalized by comparing predictions under four possible rotations.

This corresponds to measuring recognition under the assumption that the correct orientation is provided by the user.

---

## Formal ChessReD2K Results

### Board Processing

```text
Successfully processed: 290 / 306
Processing success:      94.77%
```

### Recognition on Successfully Processed Boards

| Model | Square Accuracy | Exact Full-Board Accuracy |
|---|---:|---:|
| ResNet18 | 87.16% | 3.45% |
| YOLO11 | 98.18% | 77.59% |
| Final Ensemble | 98.18% | 77.59% |

Additional final-system statistics:

```text
Average warnings:          0.93
Average required reviews:  0.20
Structurally valid boards: 92.41%
Average inference time:    6.93 s/image
```

### Interpretation

YOLO11 substantially outperformed the ResNet18 grid classifier on the held-out ChessReD2K benchmark.

The original ResNet-heavy ensemble achieved only:

```text
91.82% square accuracy
10.00% exact-board accuracy
```

Because the ensemble was worse than YOLO11 alone, the architecture was redesigned.

The final YOLO-primary routing system preserved YOLO11's full recognition performance while keeping ResNet18 as an independent warning/review signal.

---

# External Stress Test

A separate small external stress set was created to examine robustness under domain shift.

The stress images include different:

- physical piece styles,
- board styles,
- camera angles,
- backgrounds,
- positions,
- visual conditions.

This set is used as practical robustness and failure analysis, not as a second formal held-out benchmark.

## Final Locked Stress-Test Results

```text
Total images:              11
Successfully processed:     9
Processing success:        81.82%
```

Recognition on successfully processed images:

| Model | Square Accuracy | Exact Full-Board Accuracy |
|---|---:|---:|
| ResNet18 | 82.12% | 0.00% |
| YOLO11 | 71.88% | 11.11% |
| Final Ensemble | 71.88% | 11.11% |

Additional stress-test statistics:

```text
Average warnings:          5.11
Average required reviews:  6.11
Average inference time:    ~2.63 s/image
```

### Stress-Test Finding

An interesting architecture reversal appeared under external domain shift.

Formal ChessReD2K:

```text
ResNet18: 87.16%
YOLO11:   98.18%
```

External stress test:

```text
ResNet18: 82.12%
YOLO11:   71.88%
```

On this small external set, ResNet18 achieved higher average square accuracy than YOLO11.

Because the stress set is small, this should not be interpreted as proof that ResNet18 is universally more robust.

Instead, it suggests that the two architectures have complementary generalization behavior.

YOLO11 remains the primary model because it is substantially stronger on the larger formal held-out benchmark and achieves far stronger exact-board reconstruction.

---

## Failure Analysis

The main remaining weaknesses are not limited to piece recognition.

Observed failure modes include:

- chessboard contour not detected,
- extreme perspective,
- incorrect board boundaries,
- internal-grid reconstruction failure,
- square-mapping errors,
- unusual physical piece designs,
- unfamiliar board styles,
- distracting backgrounds,
- domain shift outside the training/evaluation distribution.

Some severe low-accuracy examples were caused by upstream geometry failures even when YOLO piece detections themselves looked visually strong.

This distinction is important:

```text
Correct piece recognition
does not guarantee
correct logical square reconstruction.
```

---

## Evaluation Caveat

The reported `98.18%` formal recognition accuracy is **not** a full end-to-end success rate over all 306 images.

The correct interpretation is:

> Board preprocessing succeeded on 94.77% of held-out ChessReD2K images. On those successfully normalized boards, YOLO11 and the final ensemble achieved 98.18% square accuracy and 77.59% exact-board accuracy.

---

# Installation

## Requirements

- Python 3.12
- `uv`
- Stockfish
- Gemini API key

Clone the repository:

```bash
git clone https://github.com/Abdallh-AlHabaibeh/chess-vision-tutor.git
cd chess-vision-tutor
```

Install dependencies:

```bash
uv sync
```

---

## Gemini Secrets

Create:

```text
.streamlit/secrets.toml
```

with:

```toml
GEMINI_API_KEY = "your-api-key"
GEMINI_MODEL = "your-model-name"
```

---

## Stockfish

The local Windows development version expects a Stockfish executable under the configured local engine path.

For deployment, the application can fall back to a system-installed Stockfish binary.

Local engine binaries should not be committed to the repository.

---

# Run the Application

From the project root:

```bash
uv run streamlit run src/chess_vision_tutor/ui_app.py
```

Then open the local Streamlit URL, usually:

```text
http://localhost:8501
```

---

# Typical Workflow

```text
1. Upload a chessboard photo
2. Wait for board processing and recognition
3. Inspect the reconstructed digital board
4. Confirm warnings or correct pieces manually
5. Select board orientation
6. Select side to move
7. Start the tutor
8. Position is validated
9. Stockfish analyzes the board
10. Ask the tutor for hints or explanations
```

---
# Project Structure

```text
chess-vision-tutor/
│
├── src/
│   └── chess_vision_tutor/
│       ├── __init__.py
│       ├── board_inference.py
│       ├── board_processing.py
│       ├── chess_logic.py
│       ├── chessred_processing.py
│       ├── ensemble_inference.py
│       ├── grid_classifier.py
│       ├── grid_evaluation.py
│       ├── grid_reconstruction.py
│       ├── main.py
│       ├── square_extraction.py
│       ├── stockfish_analysis.py
│       ├── tutor_chat.py
│       ├── ui_app.py
│       ├── ui_board.py
│       ├── visualization.py
│       ├── yolo_board_inference.py
│       └── yolo_square_mapping.py
│
├── scripts/
│   ├── evaluate_chessred_test.py
│   └── evaluate_stress_test.py
│
├── models/
│   └── grid_classifier_spatial.pt
│
├── data/
│   └── stress_test/
│
├── docs/
│
├── engines/
│
├── experiments/
│
├── outputs/
│   └── evaluation/
│
├── runs/
│
├── tests/
│
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

Some generated files, local binaries, raw datasets, secrets, and debug outputs are intentionally excluded from version control.

---

# Design Decisions

Several important design decisions changed during development.

### Board-first geometry

The system first detects and reconstructs the chessboard rather than trying to infer the entire position directly from the original photograph.

This makes square assignment explicit and debuggable.

### Full-board recognition

The project moved away from isolated square crops after observing that physical chess pieces can visually extend outside their logical square boundaries.

Both final recognition architectures operate with full-board context.

### Two independent model families

ResNet18 and YOLO11 provide different visual representations and different failure behavior.

The final architecture preserves both rather than relying on a single neural model.

### Evaluation-driven ensemble redesign

The original ensemble was not kept simply because it had already been implemented.

Formal metrics showed that it damaged YOLO11's performance, so the model authority was redesigned around quantitative evidence.

### Human confirmation

Human correction is treated as part of the system architecture rather than as an afterthought.

The final reconstructed position is confirmed before downstream chess analysis.

---

# Limitations

Current limitations include:

- preprocessing does not succeed on every arbitrary chessboard image,
- extreme perspective can break board localization or grid reconstruction,
- out-of-domain piece and board styles reduce recognition accuracy,
- orientation is user-provided rather than automatically inferred,
- some incorrect predictions may still be high confidence,
- the external robustness benchmark is currently small,
- runtime is heavier than a lightweight web-only classifier because the system includes OpenCV, PyTorch, YOLO, Stockfish, and an LLM tutor.

---

# Future Work

Potential future improvements include:

- more robust board-localization and grid-confidence checks,
- learned chessboard keypoint detection,
- larger external robustness benchmark,
- retraining ResNet18 on a larger dataset,
- confidence calibration,
- improved automatic detection of geometry failures,
- video/frame aggregation,
- broader deployment optimization,
- optional automatic orientation estimation.

These are intentionally left as future work rather than added as late-stage heuristics to the current release.

---

# Tech Stack

- Python
- OpenCV
- PyTorch
- torchvision
- Ultralytics YOLO
- NumPy
- python-chess
- Stockfish
- Streamlit
- Gemini API

---

# Repository

GitHub:

```text
https://github.com/Abdallh-AlHabaibeh/chess-vision-tutor
```

---

## Status

The complete local pipeline is implemented and tested end-to-end:

```text
Image upload
→ board reconstruction
→ dual-model recognition
→ interactive correction
→ FEN validation
→ Stockfish analysis
→ conversational tutoring
```

A public Streamlit deployment is planned as a separate deployment step.
