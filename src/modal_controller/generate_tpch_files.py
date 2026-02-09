"""

Standalone script to generate TPC-H or TPC-DS Parquet files using DuckDB.

"""

import duckdb
import os
import argparse
from os.path import dirname, abspath, join


def generate_benchmark_data(benchmark_type='tpch', factor=1, base_data_dir=None, seed=0.42):
    if base_data_dir is None:
        base_data_dir = dirname(abspath(__file__))
    
    # Benchmark configurations
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
    
    if benchmark_type not in config:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")
    
    cfg = config[benchmark_type]
    data_dir = cfg['data_dir']
    os.makedirs(data_dir, exist_ok=True)
    
    con = duckdb.connect(database=':memory:')
    con.execute(f"INSTALL {cfg['extension']}; LOAD {cfg['extension']}")
    con.execute(f"SELECT setseed({seed})")
    print(f"Set random seed to {seed}")
    con.execute(cfg['datagen_call'])
    
    tables = [_[0] for _ in con.execute("show tables").fetchall()]
    for t in tables:
        print(f"Exporting {benchmark_type.upper()} table {t} to Parquet file...")
        con.execute(f"COPY (SELECT * FROM {t}) TO '{data_dir}/{t}.parquet' (FORMAT parquet);")
    
    # Export queries
    queries = [_[1] for _ in con.execute(cfg['queries_call']).fetchall()]
    for i, query in enumerate(queries):
        query_file = f"{data_dir}/query_{i+1}.sql"
        with open(query_file, 'w') as f:
            f.write(query)
        print(f"Exported {benchmark_type.upper()} query {i+1} to query_{i+1}.sql")
    
    print(f"Generated {len(tables)} {benchmark_type.upper()} tables and {len(queries)} queries at scale factor {factor}")
    print(f"Data directory: {data_dir}")
    return tables


def main():
    parser = argparse.ArgumentParser(description='Generate TPC-H or TPC-DS benchmark data tables using DuckDB')
    parser.add_argument('--benchmark', '-b', choices=['tpch', 'tpcds', 'both'], default='tpch',
                       help='Which benchmark to generate (default: tpch)')
    parser.add_argument('--scale-factor', '-s', type=int, default=1,
                       help='Scale factor for data generation (default: 1)')
    parser.add_argument('--data-dir', '-d', type=str, default=None,
                       help='Base directory for data_tpch and data_tpcds folders (default: project root)')
    parser.add_argument('--seed', type=float, default=0.42,
                       help='Random seed for reproducible data generation')
    
    args = parser.parse_args()
    
    if args.benchmark == 'tpch':
        generate_benchmark_data('tpch', factor=args.scale_factor, base_data_dir=args.data_dir, seed=args.seed)
    elif args.benchmark == 'tpcds':
        generate_benchmark_data('tpcds', factor=args.scale_factor, base_data_dir=args.data_dir, seed=args.seed)
    elif args.benchmark == 'both':
        print("=== Generating both TPC-H and TPC-DS ===")
        generate_benchmark_data('tpch', factor=args.scale_factor, base_data_dir=args.data_dir, seed=args.seed)
        generate_benchmark_data('tpcds', factor=args.scale_factor, base_data_dir=args.data_dir, seed=args.seed)


if __name__ == "__main__":
    main()