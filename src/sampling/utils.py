import copy
import re
from typing import Any, Dict, List, Optional
import jsonpatch
import json
from api_utils import log_line

def apply_patches_to_plan(original_plan: Dict[str, Any], patches: List[Dict]) -> Dict[str, Any]:
    """Apply RFC 6902 JSON Patch operations to *original_plan*'s structure.

    Deep-copies the plan, applies *patches* to its ``structure`` key,
    and returns the modified plan (original is not mutated).
    """
    
    new_plan = copy.deepcopy(original_plan)
    
    structure = new_plan['structure']
    patch_obj = jsonpatch.JsonPatch(patches)
    modified_structure = patch_obj.apply(structure)
    
    new_plan['structure'] = modified_structure
    
    return new_plan
        
def extract_patches_from_response(content: str, verbose: bool = True) -> Optional[List[Dict]]:
    """Parse a ``<patch>[...]</patch>`` block from an LLM response into patch ops.

    Returns the list of parsed JSON Patch operations, or None if the
    block is missing or unparseable.
    """
    patch_match = re.search(r'<patch>\s*(.*?)\s*</patch>', content, re.DOTALL)
    if not patch_match:
        log_line(verbose, "No <patch> section found in response")
        return None
    
    patch_content = patch_match.group(1).strip()
    
    # Extract JSON from the patch section
    json_match = re.search(r'\[(.*)\]', patch_content, re.DOTALL)
    if not json_match:
        log_line(verbose, "No JSON array found in patch section")
        return None
    
    try:
        patches_str = f"[{json_match.group(1)}]"
        patches = json.loads(patches_str)
        
        if not isinstance(patches, list):
            log_line(verbose, "Patches is not a list")
            return None
            
        return patches
        
    except json.JSONDecodeError as e:
        log_line(verbose, f"Failed to parse JSON patches: {e}")
        return None
