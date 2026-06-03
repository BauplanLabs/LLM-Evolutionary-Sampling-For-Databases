"""Shared pytest configuration and fixtures for the test suite."""

from pathlib import Path

import pytest
from dotenv import load_dotenv


def pytest_configure(config):
    # Load .env from the repo root so AWS credentials etc. are available.
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)

    config.addinivalue_line(
        "markers",
        "integration: tests that make real Modal/S3 calls (deselect with -m 'not integration')",
    )
    config.addinivalue_line(
        "markers",
        "modal: real-Modal integration tests (deselect with -m 'not modal', run with -m modal)",
    )
    config.addinivalue_line(
        "markers",
        "heavy: slow tests that generate benchmark data locally (deselect with "
        "-m 'not heavy', run with -m heavy)",
    )
