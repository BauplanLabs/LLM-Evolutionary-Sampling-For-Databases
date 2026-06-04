"""Shared pytest configuration and fixtures for the test suite."""

from pathlib import Path

import pytest
from dotenv import load_dotenv


def pytest_configure(config):
    # Load .env from the repo root so AWS credentials etc. are available.
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)

    # Test taxonomy. Unmarked tests are fast, in-process unit tests with no
    # external dependencies and run by default. The markers below classify the
    # tests that need something extra, so they can be selected or deselected:
    #
    #   default fast suite :  pytest -m "not modal and not heavy"
    #   real Modal ($)     :  pytest -m modal
    #   local-engine tests :  pytest -m local
    #   heavy data-gen     :  pytest -m heavy
    for marker in (
        "modal: needs real Modal sandboxes and AWS/S3 credentials (costs money); "
        "deselect with -m 'not modal', run with -m modal",
        "local: needs the optional DataFusion engine (uv sync --extra local); runs "
        "in-process with no cloud, and auto-skips if datafusion is not installed",
        "heavy: slow — generates benchmark data locally via DuckDB; implies local; "
        "deselect with -m 'not heavy', run with -m heavy",
    ):
        config.addinivalue_line("markers", marker)
