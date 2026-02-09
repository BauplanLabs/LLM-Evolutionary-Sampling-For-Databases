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
