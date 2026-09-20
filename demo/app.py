import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import streamlit as st

from chess_vision_tutor.ui_app import main

main()


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(59, 130, 246, 0.10),
                transparent 30%
            ),
            radial-gradient(
                circle at 85% 20%,
                rgba(139, 92, 246, 0.08),
                transparent 28%
            ),
            #0d1117;
        color: #f1f5f9;
    }

    [data-testid="stHeader"] {
        display: none !important;
    }

    [data-testid="stToolbar"] {
        display: none !important;
    }

    h1,
    h2,
    h3 {
        color: #f8fafc !important;
    }

    p,
    label,
    .stMarkdown,
    [data-testid="stCaptionContainer"] {
        color: #cbd5e1;
    }

    hr {
        border-color: rgba(148, 163, 184, 0.15) !important;
    }

    [data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.70);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 0.35rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #111827 !important;
        border: 1px dashed rgba(96, 165, 250, 0.45) !important;
        border-radius: 12px;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #e2e8f0 !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: #1e293b !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(96, 165, 250, 0.45) !important;
        border-radius: 9px !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        background: #263449 !important;
        border-color: #3b82f6 !important;
        color: #ffffff !important;
    }

    [data-testid="stFileUploaderFile"] {
        background: #1e293b !important;
        border: 1px solid rgba(148, 163, 184, 0.28) !important;
        border-radius: 10px !important;
    }

    [data-testid="stFileUploaderFile"] * {
        color: #e2e8f0 !important;
    }

    [data-testid="stFileUploaderFile"] button {
        color: #94a3b8 !important;
    }

    [data-testid="stFileUploader"] section + div {
        background: #1e293b !important;
        color: #e2e8f0 !important;
    }

    [data-testid="stAlert"] {
        background: #17243a !important;
        border: 1px solid rgba(96, 165, 250, 0.25) !important;
        border-radius: 12px;
        color: #e2e8f0 !important;
    }

    [data-testid="stAlert"] * {
        color: #e2e8f0 !important;
    }

    [data-testid="stExpander"] {
        background: #0f172a !important;
        border: 1px solid rgba(148, 163, 184, 0.20) !important;
        border-radius: 12px !important;
        overflow: hidden;
    }

    [data-testid="stExpander"] details {
        background: #0f172a !important;
    }

    [data-testid="stExpander"] summary {
        background: #0f172a !important;
        color: #e2e8f0 !important;
    }

    [data-testid="stExpander"] summary * {
        color: #e2e8f0 !important;
    }

    [data-baseweb="select"] > div {
        background: #111827 !important;
        border-color: rgba(148, 163, 184, 0.28) !important;
        color: #f1f5f9 !important;
    }

    [data-baseweb="select"] * {
        color: #f1f5f9 !important;
    }

    [data-baseweb="popover"] {
        background: #111827 !important;
    }

    [role="listbox"] {
        background: #111827 !important;
    }

    [role="option"] {
        background: #111827 !important;
        color: #f1f5f9 !important;
    }

    [role="option"]:hover {
        background: #1e293b !important;
    }

    .stButton > button {
        background: #172033;
        color: #f1f5f9;
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 10px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: #1e293b;
        border-color: rgba(96, 165, 250, 0.65);
        color: #ffffff;
    }

    .stButton > button[kind="primary"] {
        background: #2563eb !important;
        border-color: #3b82f6 !important;
        color: #ffffff !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: #3b82f6 !important;
    }

    [data-testid="stChatMessage"] {
        background: #111827 !important;
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        color: #e2e8f0 !important;
    }

    [data-testid="stChatMessage"] * {
        color: #e2e8f0 !important;
    }

    [data-testid="stBottomBlockContainer"] {
        background: #0d1117 !important;
    }

    [data-testid="stBottomBlockContainer"] > div {
        background: #0d1117 !important;
    }

    [data-testid="stChatInput"] {
        background: #111827 !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 12px !important;
        box-shadow: none !important;
        outline: none !important;
    }

    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div {
        background: #111827 !important;
        border-color: #3b82f6 !important;
        box-shadow: none !important;
        outline: none !important;
    }

    [data-testid="stChatInput"]:focus-within,
    [data-testid="stChatInput"] > div:focus-within,
    [data-testid="stChatInput"] > div > div:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px rgba(59, 130, 246, 0.35) !important;
        outline: none !important;
    }

    [data-testid="stChatInput"] textarea {
        background: #111827 !important;
        color: #f8fafc !important;
        caret-color: #f8fafc !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"] textarea:focus {
        outline: none !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #94a3b8 !important;
        opacity: 1;
    }

    [data-testid="stChatInput"] button {
        background: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"] button:hover {
        background: #3b82f6 !important;
    }

    @media (max-width: 768px) {
        h1 {
            font-size: 2rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)