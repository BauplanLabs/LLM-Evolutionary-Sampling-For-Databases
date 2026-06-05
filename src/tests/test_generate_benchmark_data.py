"""Basic (no-download) tests for the benchmark data generator: the JOB CSV->Parquet
conversion, the download wrapper, and the dispatch — all offline. The real JOB
download and DuckDB TPC generation are not exercised here."""

from __future__ import annotations

import urllib.error
import urllib.request
from unittest.mock import patch

import pytest

import pandas as pd

from modal_controller.generate_benchmark_data import (
    convert_job_csvs_to_parquet,
    generate_benchmark_data,
    _download_file,
    JOB_SCHEMA,
)


class TestJobSchema:
    def test_has_21_tables(self):
        assert len(JOB_SCHEMA) == 21

    def test_columns_are_typed(self):
        for table, cols in JOB_SCHEMA.items():
            assert cols, f"{table} has no columns"
            for name, dtype in cols:
                assert isinstance(name, str) and dtype in ("Int64", "string"), f"{table}.{name}"


class TestConvertJobCsvs:
    def _write_csv(self, d, name, text):
        (d / f"{name}.csv").write_text(text)

    def test_converts_with_schema_columns_and_dtypes(self, tmp_path):
        csv_dir = tmp_path / "csv"; csv_dir.mkdir()
        out_dir = tmp_path / "out"
        self._write_csv(csv_dir, "kind_type", "1,movie\n2,tv series\n")
        self._write_csv(csv_dir, "movie_keyword", "1,100,5\n2,101,6\n")

        written = convert_job_csvs_to_parquet(str(csv_dir), str(out_dir))
        assert "kind_type" in written and "movie_keyword" in written

        kt = pd.read_parquet(out_dir / "kind_type.parquet")
        assert list(kt.columns) == ["id", "kind"]
        assert pd.api.types.is_integer_dtype(kt["id"])
        assert kt["id"].tolist() == [1, 2]
        assert kt["kind"].tolist() == ["movie", "tv series"]

        mk = pd.read_parquet(out_dir / "movie_keyword.parquet")
        assert list(mk.columns) == ["id", "movie_id", "keyword_id"]
        for c in ("id", "movie_id", "keyword_id"):
            assert pd.api.types.is_integer_dtype(mk[c])

    def test_empty_int_becomes_null(self, tmp_path):
        csv_dir = tmp_path / "csv"; csv_dir.mkdir()
        out_dir = tmp_path / "out"
        self._write_csv(csv_dir, "movie_keyword", "1,100,5\n2,,6\n")

        convert_job_csvs_to_parquet(str(csv_dir), str(out_dir))

        mk = pd.read_parquet(out_dir / "movie_keyword.parquet")
        assert mk["movie_id"].tolist()[0] == 100
        assert pd.isna(mk["movie_id"].tolist()[1])

    def test_missing_csv_skipped_without_error(self, tmp_path):
        csv_dir = tmp_path / "csv"; csv_dir.mkdir()
        out_dir = tmp_path / "out"
        self._write_csv(csv_dir, "kind_type", "1,movie\n")
        written = convert_job_csvs_to_parquet(str(csv_dir), str(out_dir))
        assert written == ["kind_type"]
        assert not (out_dir / "title.parquet").exists()

    def test_existing_parquet_not_overwritten(self, tmp_path):
        csv_dir = tmp_path / "csv"; csv_dir.mkdir()
        out_dir = tmp_path / "out"; out_dir.mkdir()
        self._write_csv(csv_dir, "kind_type", "1,movie\n2,tv\n")
        pd.DataFrame({"id": [99], "kind": ["sentinel"]}).to_parquet(out_dir / "kind_type.parquet")
        written = convert_job_csvs_to_parquet(str(csv_dir), str(out_dir))
        assert "kind_type" in written
        kt = pd.read_parquet(out_dir / "kind_type.parquet")
        assert kt["kind"].tolist() == ["sentinel"]


class TestDispatch:
    def test_job_dispatches_to_generate_job(self, tmp_path):
        with patch("modal_controller.generate_benchmark_data._generate_job") as gj:
            generate_benchmark_data("job", factor=3, base_data_dir=str(tmp_path))
        gj.assert_called_once_with(str(tmp_path))  # factor is ignored for job

    def test_tpc_dispatches_to_generate_tpc(self, tmp_path):
        with patch("modal_controller.generate_benchmark_data._generate_tpc") as gt:
            generate_benchmark_data("tpch", factor=2, base_data_dir=str(tmp_path), seed=0.1)
        gt.assert_called_once_with("tpch", 2, str(tmp_path), seed=0.1)

    def test_unknown_benchmark_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown benchmark"):
            generate_benchmark_data("mysql", factor=1, base_data_dir=str(tmp_path))


class TestDownloadFile:
    """The retry/timeout wrapper guarding the large JOB download (no network)."""

    def test_streams_url_to_dest(self, tmp_path):
        src = tmp_path / "src.bin"
        src.write_bytes(b"hello-job-data")
        dest = tmp_path / "out.bin"
        _download_file(src.as_uri(), str(dest))
        assert dest.read_bytes() == b"hello-job-data"

    def test_retries_then_succeeds(self, tmp_path, monkeypatch):
        src = tmp_path / "src.bin"
        src.write_bytes(b"payload")
        dest = tmp_path / "out.bin"
        real_urlopen = urllib.request.urlopen
        state = {"n": 0}

        def flaky(url, timeout=None):
            state["n"] += 1
            if state["n"] < 3:
                raise urllib.error.URLError("transient")
            return real_urlopen(url, timeout=timeout)

        monkeypatch.setattr(urllib.request, "urlopen", flaky)
        _download_file(src.as_uri(), str(dest), attempts=3, timeout=5)
        assert state["n"] == 3
        assert dest.read_bytes() == b"payload"

    def test_raises_after_exhausting_attempts(self, tmp_path, monkeypatch):
        dest = tmp_path / "out.bin"

        def always_fail(url, timeout=None):
            raise urllib.error.URLError("down")

        monkeypatch.setattr(urllib.request, "urlopen", always_fail)
        with pytest.raises(RuntimeError, match="after 3 attempts"):
            _download_file("https://example.invalid/x.tgz", str(dest), attempts=3, timeout=1)
