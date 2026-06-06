"""Generate benchmark data as Parquet files for DataFusion.

  job:          the Join Order Benchmark dataset, downloaded and converted to Parquet.
  tpch / tpcds: generated with DuckDB at a scale factor.

Used by the Modal image build and the local exec_local data path; both pass an
explicit ``--data-dir`` / ``base_data_dir``. This file is baked into the Modal
image and run standalone, so it must stay self-contained (stdlib + duckdb /
pandas / pyarrow only — no ``modal_controller`` imports).
"""

import argparse
import os
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from os.path import join

import duckdb


# ---------------------------------------------------------------------------
# Join Order Benchmark (JOB)
# ---------------------------------------------------------------------------

# Frozen Join Order Benchmark dataset (CWI mirror), ~1.2 GB.
JOB_DATA_URL = "https://event.cwi.nl/da/job/imdb.tgz"

# Schema for the 21 JOB tables. The CSVs are header-less, so column order here
# must match the CSV column order. Integer columns use nullable Int64.
JOB_SCHEMA = {
    "aka_name": [
        ("id", "Int64"), ("person_id", "Int64"), ("name", "string"),
        ("imdb_index", "string"), ("name_pcode_cf", "string"),
        ("name_pcode_nf", "string"), ("surname_pcode", "string"),
        ("md5sum", "string"),
    ],
    "aka_title": [
        ("id", "Int64"), ("movie_id", "Int64"), ("title", "string"),
        ("imdb_index", "string"), ("kind_id", "Int64"),
        ("production_year", "Int64"), ("phonetic_code", "string"),
        ("episode_of_id", "Int64"), ("season_nr", "Int64"),
        ("episode_nr", "Int64"), ("note", "string"), ("md5sum", "string"),
    ],
    "cast_info": [
        ("id", "Int64"), ("person_id", "Int64"), ("movie_id", "Int64"),
        ("person_role_id", "Int64"), ("note", "string"),
        ("nr_order", "Int64"), ("role_id", "Int64"),
    ],
    "char_name": [
        ("id", "Int64"), ("name", "string"), ("imdb_index", "string"),
        ("imdb_id", "Int64"), ("name_pcode_nf", "string"),
        ("surname_pcode", "string"), ("md5sum", "string"),
    ],
    "comp_cast_type": [
        ("id", "Int64"), ("kind", "string"),
    ],
    "company_name": [
        ("id", "Int64"), ("name", "string"), ("country_code", "string"),
        ("imdb_id", "Int64"), ("name_pcode_nf", "string"),
        ("name_pcode_sf", "string"), ("md5sum", "string"),
    ],
    "company_type": [
        ("id", "Int64"), ("kind", "string"),
    ],
    "complete_cast": [
        ("id", "Int64"), ("movie_id", "Int64"), ("subject_id", "Int64"),
        ("status_id", "Int64"),
    ],
    "info_type": [
        ("id", "Int64"), ("info", "string"),
    ],
    "keyword": [
        ("id", "Int64"), ("keyword", "string"), ("phonetic_code", "string"),
    ],
    "kind_type": [
        ("id", "Int64"), ("kind", "string"),
    ],
    "link_type": [
        ("id", "Int64"), ("link", "string"),
    ],
    "movie_companies": [
        ("id", "Int64"), ("movie_id", "Int64"), ("company_id", "Int64"),
        ("company_type_id", "Int64"), ("note", "string"),
    ],
    "movie_info": [
        ("id", "Int64"), ("movie_id", "Int64"), ("info_type_id", "Int64"),
        ("info", "string"), ("note", "string"),
    ],
    "movie_info_idx": [
        ("id", "Int64"), ("movie_id", "Int64"), ("info_type_id", "Int64"),
        ("info", "string"), ("note", "string"),
    ],
    "movie_keyword": [
        ("id", "Int64"), ("movie_id", "Int64"), ("keyword_id", "Int64"),
    ],
    "movie_link": [
        ("id", "Int64"), ("movie_id", "Int64"), ("linked_movie_id", "Int64"),
        ("link_type_id", "Int64"),
    ],
    "name": [
        ("id", "Int64"), ("name", "string"), ("imdb_index", "string"),
        ("imdb_id", "Int64"), ("gender", "string"),
        ("name_pcode_cf", "string"), ("name_pcode_nf", "string"),
        ("surname_pcode", "string"), ("md5sum", "string"),
    ],
    "person_info": [
        ("id", "Int64"), ("person_id", "Int64"), ("info_type_id", "Int64"),
        ("info", "string"), ("note", "string"),
    ],
    "role_type": [
        ("id", "Int64"), ("role", "string"),
    ],
    "title": [
        ("id", "Int64"), ("title", "string"), ("imdb_index", "string"),
        ("kind_id", "Int64"), ("production_year", "Int64"),
        ("imdb_id", "Int64"), ("phonetic_code", "string"),
        ("episode_of_id", "Int64"), ("season_nr", "Int64"),
        ("episode_nr", "Int64"), ("series_years", "string"),
        ("md5sum", "string"),
    ],
}


def _download_file(url, dest, attempts=3, timeout=60):
    """Stream *url* to *dest* with a per-read socket timeout and retries.

    Guards the large JOB download against a stalled or transient mirror
    connection, which a plain ``urlretrieve`` (no timeout, no retry) would let
    hang the caller — including a Modal image build — indefinitely.
    """
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp, open(dest, "wb") as out:
                shutil.copyfileobj(resp, out)
            return
        except (urllib.error.URLError, OSError) as err:
            last_err = err
            print(f"  download attempt {attempt}/{attempts} failed: {err}")
    raise RuntimeError(f"Failed to download {url} after {attempts} attempts: {last_err}")


def convert_job_csvs_to_parquet(csv_dir, output_dir, schema=JOB_SCHEMA):
    """Convert each JOB CSV in *csv_dir* to Parquet in *output_dir* per *schema*.

    Returns the tables written. Missing CSVs and already-written tables are
    skipped. Factored out from the download so it can be tested on small fixtures.
    """
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    written = []
    for table_name, columns in schema.items():
        col_names = [c[0] for c in columns]
        csv_path = join(csv_dir, f"{table_name}.csv")
        parquet_path = join(output_dir, f"{table_name}.parquet")

        if os.path.exists(parquet_path):
            written.append(table_name)
            continue
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(
            csv_path, header=None, names=col_names,
            escapechar="\\", encoding="utf-8", on_bad_lines="skip", low_memory=False,
        )
        for col_name, dtype in columns:
            if dtype == "Int64" and col_name in df.columns:
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce").astype("Int64")
        df.to_parquet(parquet_path, index=False, engine="pyarrow")
        written.append(table_name)
    return written


def _generate_job(base_data_dir):
    data_dir = join(base_data_dir, 'data_job')
    # JOB may be the first dataset built into the image, before any TPC step
    # creates base_data_dir — ensure it exists for mkdtemp below.
    os.makedirs(base_data_dir, exist_ok=True)
    download_dir = tempfile.mkdtemp(prefix="_job_download_", dir=base_data_dir)
    try:
        tgz_path = join(download_dir, "job.tgz")
        print(
            f"Downloading the Join Order Benchmark dataset (~1.2 GB) from {JOB_DATA_URL}. "
            "This is a one-time download and may take several minutes."
        )
        _download_file(JOB_DATA_URL, tgz_path)
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=download_dir)

        csv_dir = download_dir
        if not any(f.endswith(".csv") for f in os.listdir(download_dir)):
            for name in os.listdir(download_dir):
                sub = join(download_dir, name)
                if os.path.isdir(sub) and any(f.endswith(".csv") for f in os.listdir(sub)):
                    csv_dir = sub
                    break

        tables = convert_job_csvs_to_parquet(csv_dir, data_dir)
        print(f"Generated {len(tables)} JOB tables at {data_dir}")
        return tables
    finally:
        shutil.rmtree(download_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# TPC-H / TPC-DS
# ---------------------------------------------------------------------------

def _generate_tpc(benchmark_type, factor, base_data_dir, seed=0.42):
    config = {
        'tpch': {
            'extension': 'tpch',
            'datagen_call': f'CALL dbgen(sf={factor})',
            'data_dir': join(base_data_dir, 'data_tpch'),
            'queries_call': 'FROM tpch_queries()'
        },
        'tpcds': {
            'extension': 'tpcds',
            'datagen_call': f'CALL dsdgen(sf={factor})',
            'data_dir': join(base_data_dir, 'data_tpcds'),
            'queries_call': 'FROM tpcds_queries()'
        }
    }

    cfg = config[benchmark_type]
    data_dir = cfg['data_dir']
    os.makedirs(data_dir, exist_ok=True)

    con = duckdb.connect(database=':memory:')
    con.execute(f"INSTALL {cfg['extension']}; LOAD {cfg['extension']}")
    con.execute(f"SELECT setseed({seed})")
    con.execute(cfg['datagen_call'])

    tables = [_[0] for _ in con.execute("show tables").fetchall()]
    for t in tables:
        con.execute(f"COPY (SELECT * FROM {t}) TO '{data_dir}/{t}.parquet' (FORMAT parquet);")

    queries = [_[1] for _ in con.execute(cfg['queries_call']).fetchall()]
    for i, query in enumerate(queries):
        with open(f"{data_dir}/query_{i+1}.sql", 'w') as f:
            f.write(query)

    print(f"Generated {len(tables)} {benchmark_type.upper()} tables and {len(queries)} queries (sf={factor}) at {data_dir}")
    return tables


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_benchmark_data(benchmark_type, factor, base_data_dir, seed=0.42):
    """Generate *benchmark_type* data under *base_data_dir*. ``factor`` is the TPC
    scale factor and is ignored for ``job`` (a single fixed dataset)."""
    if benchmark_type in ("tpch", "tpcds"):
        return _generate_tpc(benchmark_type, factor, base_data_dir, seed=seed)
    if benchmark_type == "job":
        return _generate_job(base_data_dir)
    raise ValueError(f"Unknown benchmark type: {benchmark_type}")


def main():
    parser = argparse.ArgumentParser(description='Generate TPC-H / TPC-DS / JOB benchmark data')
    parser.add_argument('--benchmark', '-b', choices=['tpch', 'tpcds', 'job'], required=True,
                        help='Which benchmark to generate')
    parser.add_argument('--scale-factor', '-s', type=int, default=1,
                        help='Scale factor for TPC data (ignored for job; default: 1)')
    parser.add_argument('--data-dir', '-d', type=str, required=True,
                        help='Base directory under which data_<dataset> folders are written')
    parser.add_argument('--seed', type=float, default=0.42,
                        help='Random seed for reproducible TPC data generation')
    args = parser.parse_args()

    generate_benchmark_data(args.benchmark, factor=args.scale_factor,
                            base_data_dir=args.data_dir, seed=args.seed)


if __name__ == "__main__":
    main()
