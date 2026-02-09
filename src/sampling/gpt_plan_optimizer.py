"""GPT-based SQL plan optimizer using LiteLLM."""

from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import litellm
import time
import random
from tqdm import tqdm
from dataclasses import dataclass
from sampling.sql_optimization_prompts import SYSTEM_PROMPT, create_user_prompt
from sampling.utils import extract_patches_from_response
from api_utils import log_line

@dataclass
class GenerationResult:
    """Result of a single LLM optimization attempt.

    Attributes:
        model_response: Raw model response text (response + reasoning).
        error_message: Error string if generation failed, else None.
        sampled_patches: Extracted JSON Patch ops, or None on failure.
    """
    model_response: str
    error_message: Optional[str]
    sampled_patches: Optional[List[Dict]]

class GPTPlanOptimizer:
    """Generates JSON Patch optimizations for SQL execution plans via LiteLLM."""

    def __init__(self, model: str = "gpt-5", api_key: Optional[str] = None):
        self.model = model
        if api_key:
            litellm.api_key = api_key

    def optimize_plan(self, query: str, plan: Dict[str, Any], n_samples: int = 1, verbose: bool = True, n_retry: int = 2, retry_backoff_s: float = 1.0) -> List[GenerationResult]:
        """Generate *n_samples* optimization patches for a single plan.

        Args:
            query: SQL query string.
            plan: Plan dict with ``structure`` and ``succinct_table_info`` keys.
            n_samples: How many independent LLM calls to make.
            verbose: Print errors.
            n_retry: Max retries per LLM call on transient errors.
            retry_backoff_s: Base backoff between retries (doubles each attempt).

        Returns:
            List of GenerationResult (length *n_samples*).
        """
        structure = plan["structure"]
        succinct_table_info = plan["succinct_table_info"]
        
        user_prompt = create_user_prompt(query, structure, succinct_table_info)

        def generate_single_sample(sample_id: int) -> Tuple[int, GenerationResult]:
            try:
                response = None
                last_error = None
                for attempt in range(n_retry + 1):
                    try:
                        response = litellm.completion(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt}
                            ],
                            temperature=1,
                        )
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt < n_retry:
                            time.sleep(retry_backoff_s * (2 ** attempt))
                if response is None and last_error is not None:
                    raise last_error
                response_content = response.choices[0].message.content
                
                try:
                    reasoning_content = response.choices[0].message.reasoning_content
                except AttributeError:
                    reasoning_content = "<No reasoning content available>"

                sampled_patches = extract_patches_from_response(response_content, verbose=verbose)
                generation_result = GenerationResult(
                    model_response=f"Response:\n--\n{response_content}\nReasoning:\n--\n{reasoning_content}",
                    error_message=None,
                    sampled_patches=sampled_patches
                )
                
            except Exception as e:
                if verbose:
                    log_line(verbose, f"Error generating optimization patches (sample {sample_id+1}): {e}")
                    # import traceback
                    # traceback.print_exc()
                    # Re-enable traceback prints above for low-level debugging.
                generation_result = GenerationResult(
                    model_response="",
                    error_message=str(e),
                    sampled_patches=None
                )

            return (sample_id, generation_result)

        
        results = [None] * n_samples
        
        with ThreadPoolExecutor(max_workers=n_samples) as executor:
            future_to_id = {executor.submit(generate_single_sample, i): i for i in range(n_samples)}
            
            for future in as_completed(future_to_id):
                sample_id, generation_result = future.result()
                results[sample_id] = generation_result

        return results

    # TODO: make this resilient, admit n_retry and adjust implementation accordingly
    def optimize_plans_batch(self, queries: List[tuple[str, Dict[str, Any]]], n_samples: int = 1, max_workers: int = 10, verbose: bool = True, **kwargs) -> List[List[GenerationResult]]:
        """Generate optimization patches for multiple (query, plan) pairs in parallel.

        Args:
            queries: List of (query_str, plan_dict) tuples.
            n_samples: LLM samples per query.
            max_workers: Max concurrent queries to optimize.
            verbose: Print progress.
            **kwargs: Consumed: ``n_retry``, ``retry_backoff_s``.

        Returns:
            List of GenerationResult lists, aligned to *queries*.
        """
        
        n_retry = kwargs.pop("n_retry", 1)
        retry_backoff_s = kwargs.pop("retry_backoff_s", 1.0)

        def optimize_single_query(query_id: int, query: str, plan: Dict[str, Any]) -> tuple[int, List[GenerationResult]]:
            result = self.optimize_plan(query, plan, n_samples, verbose=verbose, n_retry=n_retry, retry_backoff_s=retry_backoff_s)
            return (query_id, result)
        
        results = [None] * len(queries)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(optimize_single_query, i, query, plan): i 
                for i, (query, plan) in random.sample(list(enumerate(queries)), k=len(queries))
            }
            
            for future in tqdm(as_completed(future_to_id), total=len(queries), desc="Generating optimizations", disable=not verbose):
                query_id, query_result = future.result()
                results[query_id] = query_result
        
        return results
