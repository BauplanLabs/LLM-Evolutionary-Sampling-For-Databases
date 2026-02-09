"""

Standalone script to generate TPC-H Parquet files and queries using DuckDB.

"""

import duckdb
from os.path import dirname, abspath, join


def generate_tpch_files_and_queries(factor=1):
    # get the target data directory
    data_dir = join(dirname(dirname(abspath(__file__))), "data")
    # starting DuckDB connection and installing TPC-H extension
    con = duckdb.connect(database=':memory:')
    con.execute("INSTALL tpch; LOAD tpch")
    con.execute(f"CALL dbgen(sf={factor})")
    con.execute(f"SELECT setseed(0.42)")
    tables = [_[0] for _ in con.execute("show tables").fetchall()]
    # Exporting tables to Parquet files
    for t in tables:
        print(f"Exporting table {t} to Parquet file...")
        con.execute(f"COPY (SELECT * FROM {t}) TO '{data_dir}/{t}.parquet' (FORMAT parquet);")
    # Exporting queries
    queries = [_[1] for _ in con.execute('FROM tpch_queries()').fetchall()]
    for i, query in enumerate(queries):
        with open(f"{data_dir}/query_{i+1}.sql", 'w') as f:
            f.write(query)
        print(f"Exported query {i+1} to query_{i+1}.sql")
        
    return
    
    
if __name__ == "__main__":
    SCALE_FACTOR = 3  # Change this to scale the dataset
    generate_tpch_files_and_queries(factor=SCALE_FACTOR)
