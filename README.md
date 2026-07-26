# Chess Vision Tutor

Chess Vision Tutor is a multimodal system that converts an image of a physical chessboard into a digital position for analysis and explanation.

## Pipeline

- Detect the chessboard using OpenCV
- Correct perspective distortion
- Reconstruct the playable 8 × 8 grid
- Detect and classify chess pieces
- Map each piece to its board square
- Generate and validate FEN
- Analyze the position with Stockfish
- Explain moves through an LLM tutor

## Current Implementation

The current pipeline supports:

- Chessboard contour detection
- Perspective correction
- Playable-board reconstruction
- Square extraction
- Chess-coordinate mapping from `a8` to `h1`

Piece detection and position reconstruction are the next development stage.