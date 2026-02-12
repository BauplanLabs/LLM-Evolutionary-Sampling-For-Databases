import json
import re
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from dbplanbench_utils import log_line, write_json

# Default model for query generation via OpenAI
_DEFAULT_QUERY_GEN_MODEL = "gpt-5"

def _get_completion_from_open_ai(
    client,
    system_prompt: str,
    prompt: str,
    model: str = _DEFAULT_QUERY_GEN_MODEL,
) -> str | None:
    """Call OpenAI to generate a SQL query; extract from ``<query>`` tags.

    Returns the extracted SQL string, or None if parsing fails.
    """
    response = client.responses.create(
        model=model,
        input=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
    # NOTE: This indexing assumes the current GPT-5 response shape.
    # If OpenAI response schema/refusal shape changes, this should be hardened.
    text = response.output[1].content[0].text

    # Extract SQL from <query> tags
    query_match = re.search(r'<query>\s*```sql\s*(.*?)\s*```\s*</query>', text, re.DOTALL)
    if query_match:
        text = query_match.group(1).strip()
    else:
        text = None
    
    return text


def build_prompt(
    table_to_schema: dict,
    user_prompt: str,
    complexity: int,
    single_table: str | None = None,
    include_sample_rows: bool = False,
    verbose: bool = True,
):
    """Build a user prompt by formatting *user_prompt* with schema and complexity.

    Args:
        table_to_schema: Pre-fetched ``{table_name: {schema, sample_rows}}`` dict.
        user_prompt: Template string with ``{complexity}``, ``{tables}``, ``{schema}``.
        complexity: Target complexity level for generation.
        single_table: If set, restrict schema to this table only.
        include_sample_rows: Include sample data rows in the prompt.
        verbose: Unused (kept for API compatibility).

    Returns:
        Formatted prompt string.
    """
    if not table_to_schema:
        raise ValueError("table_to_schema must be provided")
    if single_table:
        table_to_schema = {single_table: table_to_schema[single_table]}
    
    table_items = list(table_to_schema.items())
    random.shuffle(table_items)
    table_to_schema = dict(table_items)
    
    prompt_schema = ""
    for table, table_info in table_to_schema.items():
        schema = table_info['schema']
        sample_rows = table_info['sample_rows']
        
        prompt_schema += f"\n==== Schema for table {table} ====\n"
        for field in schema:
            if isinstance(field, dict):
                field_name = field.get("name")
                field_type = field.get("type")
            else:
                field_name = getattr(field, "name", None)
                field_type = getattr(field, "type", None)
            prompt_schema += f"  - {field_name}: {field_type}\n"
        
        if include_sample_rows and sample_rows:
            prompt_schema += f"\nExample rows (2 random samples):\n"
            num_rows = len(next(iter(sample_rows.values())))
            for i in range(num_rows):
                row_values = [f"{col}={sample_rows[col][i]}" for col in sample_rows.keys()]
                prompt_schema += f"  Row {i+1}: {', '.join(row_values)}\n"
    
    return user_prompt.format(
            complexity=complexity,
            tables=", ".join(table_to_schema.keys()),
            schema=prompt_schema
        )   


def save_generated_queries_to_json(
    generated_queries: list,
    output_file: str,
):
    """Write *generated_queries* (id, query, complexity) to *output_file* as JSON."""
    data = [
        {"id": r['id'], "query": r['query'], "complexity": r['complexity']}
        for r in generated_queries
    ]
    write_json(Path(output_file), data)


def generate_queries(
    oai_client,
    system_prompt: str,
    user_prompt: str,
    complexity_counts: dict[int, int],
    dataset: str,
    table_to_schema: dict,
    output_file: str,
    max_workers: int = 5,
    include_sample_rows: bool = False,
    verbose: bool = True,
):
    """Generate SQL queries via concurrent OpenAI calls and save to *output_file*.

    Builds prompts per complexity level, calls the LLM concurrently, filters
    out empty or LIMIT-containing queries, and writes valid results to JSON.

    Args:
        oai_client: OpenAI client instance.
        system_prompt: System prompt for the LLM.
        user_prompt: User prompt template (formatted with schema/complexity).
        complexity_counts: ``{complexity: count}`` for this generation step.
        dataset: Dataset name ("tpch" or "tpcds").
        table_to_schema: Pre-fetched table schema dict.
        output_file: Destination JSON file.
        max_workers: Max concurrent LLM threads.
        include_sample_rows: Include sample rows in the prompt.
        verbose: Print progress.
    """
    if output_file is None:
        raise ValueError("output_file must be provided")
    if table_to_schema is None:
        raise ValueError("table_to_schema must be provided")

    request_counts = {
        int(complexity): int(count)
        for complexity, count in complexity_counts.items()
    }
    if any(count < 0 for count in request_counts.values()):
        raise ValueError("complexity_counts values must be non-negative")
    n_total = sum(request_counts.values())
    if n_total <= 0:
        raise ValueError("complexity_counts must request at least one query")
    log_line(
        verbose,
        f"Generation batch: n={n_total} dataset={dataset.upper()} workers={max_workers}",
    )

    tasks = []
    next_id = 0
    for complexity in sorted(request_counts):
        request_n = request_counts[complexity]
        for _ in range(request_n):
            prompt = build_prompt(
                table_to_schema=table_to_schema,
                user_prompt=user_prompt,
                complexity=complexity,
                include_sample_rows=include_sample_rows,
                verbose=verbose,
            )
            tasks.append({
                'id': next_id,
                'complexity': complexity,
                'prompt': prompt
            })
            next_id += 1
    
    def generate_single_query(task):
        raw = _get_completion_from_open_ai(
            client=oai_client,
            system_prompt=system_prompt,
            prompt=task['prompt'],
        )
        return {
            'id': task['id'],
            'complexity': task['complexity'],
            'query': raw.strip() if raw else None,
        }
    
    # Generate queries concurrently
    log_line(verbose, "Generating candidate queries...")
    generated_queries = []
    
    log_per_query = verbose and max_workers <= 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(generate_single_query, task): task for task in random.sample(tasks, len(tasks))}
        
        # Process completed tasks
        for future in tqdm(
            as_completed(future_to_task),
            total=len(tasks),
            desc="Generating queries",
            disable=not verbose,
        ):
            try:
                result = future.result()
                generated_queries.append(result)
            except Exception as e:
                log_line(verbose, f"Generation error: {e}")
                continue
    
    # Sort by ID to maintain order
    generated_queries.sort(key=lambda x: x['id'])
    
    # Filter out empty queries and those with 'limit'
    valid_queries = []
    for result in generated_queries:
        new_query = result['query']
        query_id = result['id']
        
        if not new_query:
            if log_per_query:
                log_line(verbose, f"Query {query_id+1} empty; skipping.")
            continue
            
        # we are asking in the prompt to not use the word "limit" in the query, so we check for it
        # if the query contains "limit", we skip it - if we remove this prompt feature,
        # we should remove this check
        if 'limit ' in new_query.lower():
            if log_per_query:
                log_line(verbose, f"Query {query_id+1} contains forbidden LIMIT; skipping.")
            continue
        
        valid_queries.append(result)
    
    log_line(verbose, f"Generation complete: valid={len(valid_queries)}/{n_total}")

    # Save the generated queries to JSON
    save_generated_queries_to_json(
        generated_queries=valid_queries,
        output_file=output_file
    )

    # Print summary statistics
    generated_complexity_counts = {}
    for result in valid_queries:
        complexity = result['complexity']
        generated_complexity_counts[complexity] = generated_complexity_counts.get(complexity, 0) + 1

    log_line(verbose, "Generated SQL by complexity:")
    for complexity, count in sorted(generated_complexity_counts.items()):
        percentage = (count / len(valid_queries)) * 100 if valid_queries else 0
        log_line(verbose, f"  {complexity}: {count} ({percentage:.1f}%)")
    
    return output_file
