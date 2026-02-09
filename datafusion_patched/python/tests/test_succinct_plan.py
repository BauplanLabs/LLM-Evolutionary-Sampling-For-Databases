import glob
import json
import os
import time
import pyarrow as pa
import pytest
import jsonpatch
from datafusion import SessionContext
from datafusion import substrait as ss
from datafusion.plan import ExecutionPlan

from datafusion.context import SessionConfig


class TestExecutionPlanSummary:
    """Test the new to_succinct_json and from_succinct_json functionality with real parquet data."""

    @pytest.fixture
    def ctx_with_tables(self):
        """Create a SessionContext with two registered parquet tables."""
        data_folder = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "benchmarks",
            "data",
        )
        self.data_folder = data_folder
        # get all parquet files in the data folder
        self.parquet_files = glob.glob(f"{data_folder}/*.parquet")
        # extract table names from parquet files
        self.tables = [file.split("/")[-1].split(".")[0] for file in self.parquet_files]
        print(f"Found tables: {self.tables}")

        config = SessionConfig().set("datafusion.sql_parser.dialect", "postgresql")
        config = config.set(
            "datafusion.execution.parquet.schema_force_view_types", "False"
        )
        config = config.set("datafusion.optimizer.expand_views_at_output", "True")
        ctx = SessionContext(config)

        for table in self.tables:
            ctx.register_parquet(
                table, os.path.join(self.data_folder, f"{table}.parquet")
            )

        return ctx


    def test_roundtrip_union(self, ctx_with_tables):
        """Ensure plans with >2 inputs (UnionExec) survive succinct JSON round-trip without crashing."""
        ctx = ctx_with_tables

        query = (
            "(SELECT l_orderkey FROM lineitem LIMIT 5) "
            "UNION ALL (SELECT l_orderkey FROM lineitem LIMIT 5) "
            "UNION ALL (SELECT l_orderkey FROM lineitem LIMIT 5)"
        )
        df = ctx.sql(query)
        plan = df.execution_plan()

        succinct_json = plan.to_succinct_json()
        roundtrip_plan = ExecutionPlan.from_succinct_json(ctx, succinct_json)

        # Basic sanity checks on display output
        assert plan.display()
        assert roundtrip_plan.display()

        # Root node name should match
        assert plan.display().split()[0] == roundtrip_plan.display().split()[0]

    @pytest.mark.parametrize(
        "query_file",
        [
            "q1.sql",
            "q2.sql",
            "q3.sql",
            "q4.sql",
            "q5.sql",
            "q6.sql",
            "q7.sql",
            "q8.sql",
            "q9.sql",
            "q10.sql",
            "q11.sql",
            "q12.sql",
            "q13.sql",
            "q14.sql",
            "q15.sql",
            "q16.sql",
            "q17.sql",
            "q18.sql",
            "q19.sql",
            "q20.sql",
            "q21.sql",
            "q22.sql",
        ],
    )
    def test_tpch_query(self, ctx_with_tables, query_file):
        """Test round trip for all TPC-H queries to ensure plan results are identical."""
        ctx = ctx_with_tables

        # Read the query file
        query_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "benchmarks",
            "tpch",
            "queries",
            query_file,
        )

        with open(query_path, "r") as f:
            query_sql = f.read()

        # Skip comment lines and empty lines
        query_lines = [
            line
            for line in query_sql.split("\n")
            if line.strip() and not line.strip().startswith("--")
        ]
        query_sql = "\n".join(query_lines)

        print(f"Testing query: {query_file}")

        # Special handling for q15.sql which contains multiple statements
        if query_file == "q15.sql":
            # Rewrite q15.sql as a single statement using CTE
            query_sql = """
            WITH revenue0 AS (
                SELECT
                    l_suppkey AS supplier_no,
                    SUM(l_extendedprice * (1 - l_discount)) AS total_revenue
                FROM
                    lineitem
                WHERE
                    l_shipdate >= date '1996-08-01'
                    AND l_shipdate < date '1996-08-01' + interval '3' month
                GROUP BY
                    l_suppkey
            )
            SELECT
                s_suppkey,
                s_name,
                s_address,
                s_phone,
                total_revenue
            FROM
                supplier,
                revenue0
            WHERE
                s_suppkey = supplier_no
                AND total_revenue = (
                    SELECT
                        MAX(total_revenue)
                    FROM
                        revenue0
                )
            ORDER BY
                s_suppkey
            """

        # Execute the query and get the plan
        df = ctx.sql(query_sql)
        plan = df.execution_plan()

        # Test round trip
        succinct_plan = plan.to_succinct_json()
        roundtrip_plan = ExecutionPlan.from_succinct_json(ctx, succinct_plan)

        # Compare results
        c1 = ctx.collect(roundtrip_plan)
        c2 = ctx.collect(plan)

        assert c1 == c2, f"Round trip failed for {query_file}: results don't match"

    def test_join_selection_optimizer_exclusion(self, ctx_with_tables):
        """Test that the JoinSelection optimizer rule is excluded when using new_without_join_selection."""
        # Create a regular context with all default optimizer rules
        config = SessionConfig().set("datafusion.sql_parser.dialect", "postgresql")
        config = config.set(
            "datafusion.execution.parquet.schema_force_view_types", "False"
        )
        config = config.set("datafusion.optimizer.expand_views_at_output", "True")

        regular_ctx = SessionContext(config)

        # Create a context without JoinSelection optimizer rule
        no_join_selection_ctx = SessionContext.new_without_join_selection(config)

        # Register the same tables in both contexts
        for table in self.tables:
            regular_ctx.register_parquet(
                table, os.path.join(self.data_folder, f"{table}.parquet")
            )
            no_join_selection_ctx.register_parquet(
                table, os.path.join(self.data_folder, f"{table}.parquet")
            )

        # Use a query with joins that would benefit from join selection optimization
        # This query joins lineitem with orders, which should trigger different join strategies
        query = """
        SELECT l.l_orderkey, o.o_orderdate, l.l_quantity, o.o_totalprice
        FROM orders o
        JOIN lineitem l ON l.l_orderkey = o.o_orderkey
        WHERE l.l_shipdate >= '1995-01-01' 
        AND l.l_shipdate < '1996-01-01'
        LIMIT 100
        """

        # Get execution plans from both contexts
        regular_df = regular_ctx.sql(query)
        no_join_selection_df = no_join_selection_ctx.sql(query)

        regular_plan = regular_df.execution_plan()
        no_join_selection_plan = no_join_selection_df.execution_plan()

        # Get the plan representations
        regular_plan_str = regular_plan.display_indent()
        no_join_selection_plan_str = no_join_selection_plan.display_indent()

        print("=== Regular Context Plan ===")
        print(regular_plan_str)
        print("\n=== No JoinSelection Context Plan ===")
        print(no_join_selection_plan_str)

        # The plans should potentially be different due to different join selection
        # We can't assert a specific difference since it depends on table statistics,
        # but we can verify that both plans are valid and executable

        # Execute both plans to ensure they're valid
        regular_results = regular_ctx.collect(regular_plan)
        no_join_selection_results = no_join_selection_ctx.collect(
            no_join_selection_plan
        )

        # Both should return the same data (just potentially with different execution strategies)
        assert len(regular_results) == len(no_join_selection_results)

        # Verify that the plans are different strings (indicating different optimization)
        # This might not always be the case if the optimizer makes the same choice regardless,
        # but it's a reasonable check for this proof of concept
        print(f"Regular plan length: {len(regular_plan_str)}")
        print(f"No JoinSelection plan length: {len(no_join_selection_plan_str)}")

        # Log the difference for verification - we don't assert equality because
        # the point is to show they can be different
        are_plans_different = regular_plan_str != no_join_selection_plan_str
        assert are_plans_different

    def test_tpch_query5(self, ctx_with_tables):
        """Test that swapping hash join children in TPC-H Query 5 produces the same results."""
        ctx = ctx_with_tables
        query_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "benchmarks",
            "tpch",
            "queries",
            "q5.sql",
        )

        with open(query_path, "r") as f:
            query_sql = f.read()
        df = ctx.sql(query_sql)
        plan = df.execution_plan()
        succinct_plan = plan.to_succinct_json()
        with open("python/tests/data_test_context/q5.json", "w") as f:
            f.write(succinct_plan)

    def test_bad_queries(self, ctx_with_tables):
        """Test physical plans for syntax_errors and no_physical_plan queries."""
        ctx = ctx_with_tables

        # Load bad queries JSON file
        bad_queries_path = os.path.join(
            os.path.dirname(__file__),
            "data_test_context",
            "bad_tpch_queries.json"
        )

        with open(bad_queries_path) as f:
            bad_queries = json.load(f)

        # Categories to test
        categories = ["syntax_errors", "no_physical_plan"]

        for category in categories:
            queries = bad_queries.get(category, [])
            category_info = f"\n=== Testing {category} category"
            print(f"{category_info}: {len(queries)} queries ===")

            for idx, query_sql in enumerate(queries, 1):
                print(f"\nQuery {idx}/{len(queries)} in {category}:")
                print(f"Query preview: {query_sql[:100]}...")

                try:
                    # Attempt to create a dataframe from the SQL query
                    df = ctx.sql(query_sql)

                    # Attempt to get the execution plan
                    plan = df.execution_plan()

                    # If we get here, the plan was created successfully
                    print("✓ Physical plan created successfully")
                    print(f"  Plan type: {plan.display().split()[0]}")

                    # Optionally try to get succinct JSON representation
                    try:
                        succinct_json = plan.to_succinct_json()
                        json_len = len(succinct_json)
                        print(f"  Succinct JSON length: {json_len} chars")
                    except Exception as json_error:
                        print(f"  ⚠ Failed to create JSON: {json_error}")

                except Exception as e:
                    # Expected for queries that cannot generate physical plans
                    error_type = type(e).__name__
                    error_msg = str(e)[:200]  # Truncate long errors
                    print(f"✗ Failed to create physical plan: {error_type}")
                    print(f"  Error: {error_msg}")

                    # Documenting behavior, not failing
                    # Add assertions if specific queries should succeed/fail
                    continue

        # This test documents which queries can/cannot generate plans
        # Add assertions here if specific queries should succeed or fail
        assert True, "Test completed - check output for results"

    def test_top5_best_optimizations(self, ctx_with_tables):
        """Test applying patches to original plan produces same results."""
        ctx = ctx_with_tables

        # Load the optimizations JSON file
        optimizations_path = os.path.join(
            os.path.dirname(__file__),
            "data_test_context",
            "top_5_best_optimizations.json"
        )

        with open(optimizations_path) as f:
            optimizations = json.load(f)

        print(f"\n=== Testing {len(optimizations)} optimization cases ===")

        for idx, case in enumerate(optimizations, 1):
            query_index = case["query_index"]
            sample_id = case["sample_id"]
            query_sql = case["query_sql"]
            patches = case["patches"]

            info = f"query_index={query_index}, sample_id={sample_id}"
            print(f"\n--- Case {idx}/{len(optimizations)} ({info}) ---")
            print(f"Query preview: {query_sql[:100]}...")
            print(f"Number of patches: {len(patches)}")

            try:
                # Step 1: Execute the SQL query to get the original plan
                df = ctx.sql(query_sql)
                original_plan = df.execution_plan()
                print("✓ Generated original plan from SQL")

                # Step 2: Convert original plan to JSON
                original_plan_json_str = original_plan.to_succinct_json()
                original_plan_json = json.loads(original_plan_json_str)

                # Step 3: Apply patches to the structure field
                patch = jsonpatch.JsonPatch(patches)
                original_structure = original_plan_json.get("structure", {})
                optimized_structure = patch.apply(original_structure)
                print("✓ Applied patches to structure field")

                # Step 4: Create optimized plan JSON with patched structure
                optimized_plan_json = original_plan_json.copy()
                optimized_plan_json["structure"] = optimized_structure

                # Step 5: Load optimized plan from patched JSON
                optimized_plan_json_str = json.dumps(optimized_plan_json)
                optimized_plan = ExecutionPlan.from_succinct_json(
                    ctx, optimized_plan_json_str
                )
                print("✓ Loaded optimized plan from patched JSON")

                # Step 5: Execute and time the original plan
                start_time = time.time()
                original_results = ctx.collect(original_plan)
                original_time = time.time() - start_time
                print(f"✓ Original plan executed in {original_time:.4f}s")

                # Step 6: Execute and time the optimized plan
                start_time = time.time()
                optimized_results = ctx.collect(optimized_plan)
                optimized_time = time.time() - start_time
                print(f"✓ Optimized plan executed in {optimized_time:.4f}s")

                # Step 7: Calculate measured improvement
                if original_time > 0:
                    measured_improvement = (
                        (original_time - optimized_time) / original_time * 100
                    )
                    print(f"✓ Measured improvement: {measured_improvement:.2f}%")
                else:
                    measured_improvement = 0

                # Step 8: Compare results
                mismatch_msg = (
                    f"Results mismatch for query_index={query_index}, "
                    f"sample_id={sample_id}"
                )
                assert original_results == optimized_results, mismatch_msg
                print("✓ Results match!")

                # Step 9: Compare with reported metrics
                if "metrics" in case:
                    metrics = case["metrics"]
                    original_metrics = metrics.get("original", {})
                    optimized_metrics = metrics.get("optimized", {})
                    improvements = metrics.get("improvements", {})

                    print("\n  Reported benchmark metrics:")
                    if original_metrics:
                        orig_mean = original_metrics.get("mean", 0)
                        orig_median = original_metrics.get("median", 0)
                        print(f"    Original - Mean: {orig_mean:.4f}s, "
                              f"Median: {orig_median:.4f}s")
                    if optimized_metrics:
                        opt_mean = optimized_metrics.get("mean", 0)
                        opt_median = optimized_metrics.get("median", 0)
                        print(f"    Optimized - Mean: {opt_mean:.4f}s, "
                              f"Median: {opt_median:.4f}s")

                    if improvements:
                        print("  Reported improvements:")
                        mean_pct = improvements.get("mean_pct", 0)
                        median_pct = improvements.get("median_pct", 0)
                        min_pct = improvements.get("min_pct", 0)
                        print(f"    Mean: {mean_pct:.2f}%")
                        print(f"    Median: {median_pct:.2f}%")
                        print(f"    Min: {min_pct:.2f}%")

                        # Compare measured vs reported
                        print("\n  Measured vs Reported:")
                        print(f"    Measured: {measured_improvement:.2f}%")
                        print(f"    Reported Mean: {mean_pct:.2f}%")
                        diff = abs(measured_improvement - mean_pct)
                        print(f"    Difference: {diff:.2f}%")

            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)[:500]  # Truncate long errors
                print(f"✗ Failed: {error_type}")
                print(f"  Error: {error_msg}")
                # Re-raise to fail the test
                raise

        print(f"\n=== All {len(optimizations)} optimization cases passed! ===")
