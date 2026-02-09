"""Tests for query_gen.query_filterer — pure filtering logic (no Modal/LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from query_gen.query_filterer import filter_evaluation_queries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_query(query_id: int, exec_time_min: float) -> dict:
    """Build a query entry with evaluation_stats."""
    return {
        "id": query_id,
        "query": f"SELECT {query_id}",
        "evaluation_stats": {
            "execution_time": {"min": exec_time_min, "max": exec_time_min * 1.2},
        },
    }


def _write_queries(path: Path, queries: list[dict]) -> None:
    path.write_text(json.dumps(queries, indent=2))


# ===================================================================
# filter_evaluation_queries
# ===================================================================

class TestFilterEvaluationQueries:
    def test_all_pass(self, tmp_path: Path):
        queries = [_make_query(1, 1.0), _make_query(2, 2.0)]
        infile = tmp_path / "input.json"
        outfile = tmp_path / "output.json"
        _write_queries(infile, queries)

        filter_evaluation_queries(
            str(infile), str(outfile), dead_letter_file=None,
            lowerbound_seconds=0.5, upperbound_seconds=3.0, verbose=False,
        )
        result = json.loads(outfile.read_text())
        assert len(result) == 2

    def test_all_filtered_out(self, tmp_path: Path):
        queries = [_make_query(1, 0.1), _make_query(2, 0.2)]
        infile = tmp_path / "input.json"
        outfile = tmp_path / "output.json"
        _write_queries(infile, queries)

        filter_evaluation_queries(
            str(infile), str(outfile), dead_letter_file=None,
            lowerbound_seconds=1.0, upperbound_seconds=5.0, verbose=False,
        )
        result = json.loads(outfile.read_text())
        assert len(result) == 0

    def test_partial_filter(self, tmp_path: Path):
        queries = [_make_query(1, 0.5), _make_query(2, 2.0), _make_query(3, 10.0)]
        infile = tmp_path / "input.json"
        outfile = tmp_path / "output.json"
        _write_queries(infile, queries)

        filter_evaluation_queries(
            str(infile), str(outfile), dead_letter_file=None,
            lowerbound_seconds=1.0, upperbound_seconds=5.0, verbose=False,
        )
        result = json.loads(outfile.read_text())
        assert len(result) == 1
        assert result[0]["id"] == 2

    def test_dead_letter_file(self, tmp_path: Path):
        queries = [_make_query(1, 0.5), _make_query(2, 2.0), _make_query(3, 10.0)]
        infile = tmp_path / "input.json"
        outfile = tmp_path / "output.json"
        dlfile = tmp_path / "dead_letter.json"
        _write_queries(infile, queries)

        filter_evaluation_queries(
            str(infile), str(outfile), dead_letter_file=str(dlfile),
            lowerbound_seconds=1.0, upperbound_seconds=5.0, verbose=False,
        )
        kept = json.loads(outfile.read_text())
        dropped = json.loads(dlfile.read_text())
        assert len(kept) == 1
        assert len(dropped) == 2
        assert {d["id"] for d in dropped} == {1, 3}

    def test_boundary_values_inclusive(self, tmp_path: Path):
        queries = [_make_query(1, 1.0), _make_query(2, 5.0)]
        infile = tmp_path / "input.json"
        outfile = tmp_path / "output.json"
        _write_queries(infile, queries)

        filter_evaluation_queries(
            str(infile), str(outfile), dead_letter_file=None,
            lowerbound_seconds=1.0, upperbound_seconds=5.0, verbose=False,
        )
        result = json.loads(outfile.read_text())
        assert len(result) == 2

    def test_empty_input(self, tmp_path: Path):
        infile = tmp_path / "input.json"
        outfile = tmp_path / "output.json"
        _write_queries(infile, [])

        filter_evaluation_queries(
            str(infile), str(outfile), dead_letter_file=None,
            lowerbound_seconds=0.0, upperbound_seconds=10.0, verbose=False,
        )
        result = json.loads(outfile.read_text())
        assert result == []

    def test_missing_evaluation_stats_dropped(self, tmp_path: Path):
        queries = [
            {"id": 1, "query": "SELECT 1"},  # no evaluation_stats
            _make_query(2, 2.0),
        ]
        infile = tmp_path / "input.json"
        outfile = tmp_path / "output.json"
        dlfile = tmp_path / "dead_letter.json"
        _write_queries(infile, queries)

        filter_evaluation_queries(
            str(infile), str(outfile), dead_letter_file=str(dlfile),
            lowerbound_seconds=1.0, upperbound_seconds=5.0, verbose=False,
        )
        kept = json.loads(outfile.read_text())
        dropped = json.loads(dlfile.read_text())
        assert len(kept) == 1
        assert kept[0]["id"] == 2
        assert len(dropped) == 1
        assert dropped[0]["id"] == 1

    def test_custom_runtime_metric(self, tmp_path: Path):
        queries = [
            {
                "id": 1, "query": "SELECT 1",
                "evaluation_stats": {
                    "execution_time": {"min": 0.1, "max": 10.0},
                },
            },
        ]
        infile = tmp_path / "input.json"
        outfile = tmp_path / "output.json"
        _write_queries(infile, queries)

        # Filter on max instead of min
        filter_evaluation_queries(
            str(infile), str(outfile), dead_letter_file=None,
            lowerbound_seconds=5.0, upperbound_seconds=15.0,
            runtime_metric="execution_time.max", verbose=False,
        )
        result = json.loads(outfile.read_text())
        assert len(result) == 1

    def test_creates_parent_dirs(self, tmp_path: Path):
        queries = [_make_query(1, 1.0)]
        infile = tmp_path / "input.json"
        outfile = tmp_path / "sub" / "dir" / "output.json"
        _write_queries(infile, queries)

        filter_evaluation_queries(
            str(infile), str(outfile), dead_letter_file=None,
            lowerbound_seconds=0.0, upperbound_seconds=5.0, verbose=False,
        )
        assert outfile.exists()

    def test_deterministic(self, tmp_path: Path):
        queries = [_make_query(i, float(i)) for i in range(10)]
        infile = tmp_path / "input.json"
        out1 = tmp_path / "output1.json"
        out2 = tmp_path / "output2.json"
        _write_queries(infile, queries)

        filter_evaluation_queries(
            str(infile), str(out1), dead_letter_file=None,
            lowerbound_seconds=3.0, upperbound_seconds=7.0, verbose=False,
        )
        filter_evaluation_queries(
            str(infile), str(out2), dead_letter_file=None,
            lowerbound_seconds=3.0, upperbound_seconds=7.0, verbose=False,
        )
        assert json.loads(out1.read_text()) == json.loads(out2.read_text())
