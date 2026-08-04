"""Sanity unit tests for Streamlit app and README.md documentation integrity."""

from pathlib import Path
import pytest


def test_streamlit_importable():
    """Verify Streamlit is correctly installed and importable."""
    import streamlit as st
    assert st.__version__ is not None


def test_app_py_file_exists():
    """Verify app.py Streamlit dashboard file exists and contains main entry point."""
    app_path = Path("app.py")
    assert app_path.exists()

    content = app_path.read_text(encoding="utf-8")
    assert "st.set_page_config" in content
    assert "Multi-Agent Content Pipeline" in content
    assert "build_graph" in content


def test_readme_md_integrity():
    """Verify README.md exists and contains all required interview & documentation sections."""
    readme_path = Path("README.md")
    assert readme_path.exists()

    content = readme_path.read_text(encoding="utf-8")

    assert "# ✍️ ScribeGraph — Evaluated Multi-Agent Content Pipeline" in content
    assert "```mermaid" in content
    assert "Benchmark Trade-Off Analysis" in content
    assert "Quickstart Guide" in content
    assert "Technical Interview Defense Q&A Cheatsheet" in content
    assert "MAX_REVISIONS = 3" in content
