import copy
import json
from typing import Any, Dict, List, Tuple, Set


class PlanTransferError(Exception):
    """Raised when plan transfer between scale factors fails."""
    pass

def _json_dumps_canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_pointer_unescape(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _json_pointer_split(ptr: str) -> List[str]:
    if ptr == "":
        return []
    if not ptr.startswith("/"):
        raise PlanTransferError(f"patch-apply: invalid-json-pointer ptr={ptr!r}")
    return [_json_pointer_unescape(p) for p in ptr.split("/")[1:]]


def _json_pointer_resolve_parent(doc: Any, ptr: str) -> Tuple[Any, Any]:
    """
    Return (parent, last_token) for ptr. For ptr == "", parent is None and last_token is None.
    """
    parts = _json_pointer_split(ptr)
    if not parts:
        return None, None
    cur = doc
    for i, tok in enumerate(parts[:-1]):
        if isinstance(cur, dict):
            if tok not in cur:
                raise PlanTransferError(f"patch-apply: path-not-found ptr={ptr!r} at={tok!r}")
            cur = cur[tok]
        elif isinstance(cur, list):
            if tok == "-":
                raise PlanTransferError(f"patch-apply: invalid-list-index '-' in middle ptr={ptr!r}")
            try:
                idx = int(tok)
            except Exception as e:
                raise PlanTransferError(f"patch-apply: invalid-list-index ptr={ptr!r} token={tok!r}") from e
            if idx < 0 or idx >= len(cur):
                raise PlanTransferError(f"patch-apply: list-index-oob ptr={ptr!r} idx={idx}")
            cur = cur[idx]
        else:
            raise PlanTransferError(f"patch-apply: non-container ptr={ptr!r} at={tok!r}")
    return cur, parts[-1]


def _json_pointer_get(doc: Any, ptr: str) -> Any:
    parts = _json_pointer_split(ptr)
    cur = doc
    for tok in parts:
        if isinstance(cur, dict):
            if tok not in cur:
                raise PlanTransferError(f"patch-apply: get-path-not-found ptr={ptr!r} at={tok!r}")
            cur = cur[tok]
        elif isinstance(cur, list):
            if tok == "-":
                raise PlanTransferError(f"patch-apply: get-invalid-list-index '-' ptr={ptr!r}")
            try:
                idx = int(tok)
            except Exception as e:
                raise PlanTransferError(f"patch-apply: get-invalid-list-index ptr={ptr!r} token={tok!r}") from e
            if idx < 0 or idx >= len(cur):
                raise PlanTransferError(f"patch-apply: get-list-index-oob ptr={ptr!r} idx={idx}")
            cur = cur[idx]
        else:
            raise PlanTransferError(f"patch-apply: get-non-container ptr={ptr!r} at={tok!r}")
    return cur


def _json_pointer_add(doc: Any, ptr: str, value: Any) -> Any:
    if ptr == "":
        # Replace whole document
        return value
    parent, tok = _json_pointer_resolve_parent(doc, ptr)
    if isinstance(parent, dict):
        parent[tok] = value
        return doc
    if isinstance(parent, list):
        if tok == "-":
            parent.append(value)
            return doc
        try:
            idx = int(tok)
        except Exception as e:
            raise PlanTransferError(f"patch-apply: add-invalid-list-index ptr={ptr!r} token={tok!r}") from e
        if idx < 0 or idx > len(parent):
            raise PlanTransferError(f"patch-apply: add-list-index-oob ptr={ptr!r} idx={idx}")
        parent.insert(idx, value)
        return doc
    raise PlanTransferError(f"patch-apply: add-parent-non-container ptr={ptr!r}")


def _json_pointer_remove(doc: Any, ptr: str) -> Any:
    if ptr == "":
        raise PlanTransferError("patch-apply: remove-root-not-allowed ptr=''")
    parent, tok = _json_pointer_resolve_parent(doc, ptr)
    if isinstance(parent, dict):
        if tok not in parent:
            raise PlanTransferError(f"patch-apply: remove-path-not-found ptr={ptr!r}")
        del parent[tok]
        return doc
    if isinstance(parent, list):
        if tok == "-":
            raise PlanTransferError(f"patch-apply: remove-invalid-list-index '-' ptr={ptr!r}")
        try:
            idx = int(tok)
        except Exception as e:
            raise PlanTransferError(f"patch-apply: remove-invalid-list-index ptr={ptr!r} token={tok!r}") from e
        if idx < 0 or idx >= len(parent):
            raise PlanTransferError(f"patch-apply: remove-list-index-oob ptr={ptr!r} idx={idx}")
        parent.pop(idx)
        return doc
    raise PlanTransferError(f"patch-apply: remove-parent-non-container ptr={ptr!r}")


def _json_pointer_replace(doc: Any, ptr: str, value: Any) -> Any:
    if ptr == "":
        return value
    # require path exists
    _ = _json_pointer_get(doc, ptr)
    parent, tok = _json_pointer_resolve_parent(doc, ptr)
    if isinstance(parent, dict):
        parent[tok] = value
        return doc
    if isinstance(parent, list):
        if tok == "-":
            raise PlanTransferError(f"patch-apply: replace-invalid-list-index '-' ptr={ptr!r}")
        try:
            idx = int(tok)
        except Exception as e:
            raise PlanTransferError(f"patch-apply: replace-invalid-list-index ptr={ptr!r} token={tok!r}") from e
        if idx < 0 or idx >= len(parent):
            raise PlanTransferError(f"patch-apply: replace-list-index-oob ptr={ptr!r} idx={idx}")
        parent[idx] = value
        return doc
    raise PlanTransferError(f"patch-apply: replace-parent-non-container ptr={ptr!r}")


def _apply_json_patch(doc: Any, patch: List[Dict[str, Any]]) -> Any:
    if not isinstance(patch, list):
        raise PlanTransferError("patch-apply: patch-must-be-list")
    cur = doc
    for i, op in enumerate(patch):
        if not isinstance(op, dict) or "op" not in op or "path" not in op:
            raise PlanTransferError(f"patch-apply: invalid-op index={i}")
        opname = op["op"]
        path = op["path"]
        if opname == "add":
            if "value" not in op:
                raise PlanTransferError(f"patch-apply: add-missing-value index={i}")
            cur = _json_pointer_add(cur, path, copy.deepcopy(op["value"]))
        elif opname == "remove":
            cur = _json_pointer_remove(cur, path)
        elif opname == "replace":
            if "value" not in op:
                raise PlanTransferError(f"patch-apply: replace-missing-value index={i}")
            cur = _json_pointer_replace(cur, path, copy.deepcopy(op["value"]))
        elif opname == "move":
            if "from" not in op:
                raise PlanTransferError(f"patch-apply: move-missing-from index={i}")
            val = copy.deepcopy(_json_pointer_get(cur, op["from"]))
            cur = _json_pointer_remove(cur, op["from"])
            cur = _json_pointer_add(cur, path, val)
        elif opname == "copy":
            if "from" not in op:
                raise PlanTransferError(f"patch-apply: copy-missing-from index={i}")
            val = copy.deepcopy(_json_pointer_get(cur, op["from"]))
            cur = _json_pointer_add(cur, path, val)
        elif opname == "test":
            if "value" not in op:
                raise PlanTransferError(f"patch-apply: test-missing-value index={i}")
            actual = _json_pointer_get(cur, path)
            expected = op["value"]
            if actual != expected:
                raise PlanTransferError(f"patch-apply: test-failed index={i} path={path!r}")
        else:
            raise PlanTransferError(f"patch-apply: unsupported-op index={i} op={opname!r}")
    return cur





_REF_KEYS_SCALAR = {"input", "left", "right"}
_REF_KEYS_LIST = {"inputs", "children"}


def _is_intish(x: Any) -> bool:
    if isinstance(x, int):
        return True
    if isinstance(x, str) and x.isdigit():
        return True
    return False


def _to_int(x: Any) -> int:
    if isinstance(x, int):
        return x
    if isinstance(x, str) and x.isdigit():
        return int(x)
    raise ValueError(f"not-intish: {x!r}")


def _extract_operator(node: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    if not isinstance(node, dict) or len(node) != 1:
        raise PlanTransferError("validation: node-must-have-single-operator")
    op_name = next(iter(node.keys()))
    return op_name, node[op_name]


def _table_info_entries(succinct: Any) -> List[Dict[str, Any]]:
    if succinct is None:
        return []
    if isinstance(succinct, list):
        return [e for e in succinct if isinstance(e, dict) and "id" in e and "node" in e]
    # tolerate dict form {"entries":[...]} (generic)
    if isinstance(succinct, dict) and "entries" in succinct and isinstance(succinct["entries"], list):
        return [e for e in succinct["entries"] if isinstance(e, dict) and "id" in e and "node" in e]
    raise PlanTransferError("transfer: succinct_table_info must be list[{'id','node'}] or {'entries':...}")


def _scan_normalized_signature(scan_node: Dict[str, Any]) -> str:
    """
    Build a stable signature for a parquetScan-like node, excluding dataset-dependent fileGroups/stats.
    """
    if not isinstance(scan_node, dict) or "parquetScan" not in scan_node:
        return _json_dumps_canonical(scan_node)

    ps = scan_node.get("parquetScan", {})
    base = ps.get("baseConf", {})

    schema_cols = None
    if isinstance(base, dict):
        schema = base.get("schema", {})
        if isinstance(schema, dict):
            schema_cols = schema.get("columns", None)

    projection = base.get("projection", None)
    constraints = base.get("constraints", None)
    predicate = ps.get("predicate", None)

    # Normalize out dataset-dependent fields
    normalized = {
        "op": "parquetScan",
        "baseConf": {
            "schema.columns": schema_cols,
            "projection": projection,
            "constraints": constraints,
        },
        "predicate": predicate,
    }
    return _json_dumps_canonical(normalized)


def _build_scan_signature_map(succinct_table_info: Any) -> Dict[int, str]:
    sig_by_id: Dict[int, str] = {}
    for entry in _table_info_entries(succinct_table_info):
        sid = entry["id"]
        if not isinstance(sid, int):
            raise PlanTransferError("transfer: succinct_table_info entry.id must be int")
        node = entry.get("node", {})
        sig_by_id[sid] = _scan_normalized_signature(node)
    return sig_by_id


def _build_scan_id_mapping(sf1_succinct: Any, sf2_succinct: Any) -> Dict[int, int]:
    sf1_sig = _build_scan_signature_map(sf1_succinct)
    sf2_sig = _build_scan_signature_map(sf2_succinct)

    # Reverse maps: sig -> sorted list of ids
    sf1_by_sig: Dict[str, List[int]] = {}
    sf2_by_sig: Dict[str, List[int]] = {}
    for sid, sig in sf1_sig.items():
        sf1_by_sig.setdefault(sig, []).append(sid)
    for sid, sig in sf2_sig.items():
        sf2_by_sig.setdefault(sig, []).append(sid)

    for sig in sf1_by_sig:
        sf1_by_sig[sig] = sorted(sf1_by_sig[sig])
    for sig in sf2_by_sig:
        sf2_by_sig[sig] = sorted(sf2_by_sig[sig])

    mapping: Dict[int, int] = {}
    for sig, sf1_ids in sf1_by_sig.items():
        sf2_ids = sf2_by_sig.get(sig, [])
        if len(sf2_ids) < len(sf1_ids):
            raise PlanTransferError(
                f"transfer: scan-signature-unmatched sig={sig[:120]!r} sf1_count={len(sf1_ids)} sf2_count={len(sf2_ids)}"
            )
        # Pair deterministically by sorted id order
        for i, sf1_id in enumerate(sf1_ids):
            mapping[sf1_id] = sf2_ids[i]
    return mapping


def _find_internal_references(structure: Dict[str, Any]) -> Set[int]:
    internal_ids = {int(k) for k in structure.keys() if isinstance(k, str) and k.isdigit()}
    refs: Set[int] = set()

    def walk(obj: Any):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in _REF_KEYS_SCALAR and _is_intish(v):
                    refs.add(_to_int(v))
                elif k in _REF_KEYS_LIST and isinstance(v, list) and all(_is_intish(x) for x in v):
                    refs.update(_to_int(x) for x in v)
                else:
                    walk(v)
        elif isinstance(obj, list):
            for it in obj:
                walk(it)

    for node in structure.values():
        walk(node)
    # Only internal references
    return {r for r in refs if r in internal_ids}


def _find_roots(structure: Dict[str, Any]) -> List[int]:
    internal_ids = sorted(int(k) for k in structure.keys() if isinstance(k, str) and k.isdigit())
    referenced = _find_internal_references(structure)
    roots = [i for i in internal_ids if i not in referenced]
    return roots if roots else internal_ids[:1]  # fallback


def _topo_order_from_roots(structure: Dict[str, Any]) -> List[str]:
    """
    Deterministic traversal order (not necessarily strict topo for DAG with joins, but stable):
    - start from roots in increasing numeric id
    - DFS with children in a stable order (left/right/input/inputs)
    - include any unreachable nodes in increasing numeric id
    """
    nodes_by_id: Dict[int, Dict[str, Any]] = {}
    for k, v in structure.items():
        if isinstance(k, str) and k.isdigit():
            nodes_by_id[int(k)] = v

    def children_of(node_obj: Dict[str, Any]) -> List[int]:
        # Only follow internal references that exist in structure
        kids: List[int] = []
        def extract(conf: Any):
            if isinstance(conf, dict):
                for kk, vv in conf.items():
                    if kk in _REF_KEYS_SCALAR and _is_intish(vv):
                        kids.append(_to_int(vv))
                    elif kk in _REF_KEYS_LIST and isinstance(vv, list) and all(_is_intish(x) for x in vv):
                        kids.extend(_to_int(x) for x in vv)
            # Do not recurse; references are expected only directly in operator conf for these keys.
        if isinstance(node_obj, dict) and len(node_obj) == 1:
            _, conf = _extract_operator(node_obj)
            extract(conf)
        return [c for c in kids if c in nodes_by_id]

    roots = _find_roots(structure)
    visited: Set[int] = set()
    order: List[int] = []

    def dfs(nid: int):
        if nid in visited:
            return
        visited.add(nid)
        # visit children first (post-order), stable
        node_obj = nodes_by_id.get(nid)
        if node_obj is not None:
            for c in children_of(node_obj):
                dfs(c)
        order.append(nid)

    for r in sorted(roots):
        dfs(r)

    for nid in sorted(nodes_by_id.keys()):
        dfs(nid)

    # Convert to string ids
    return [str(n) for n in order]


def _assign_new_node_ids(
    sf1_opt_structure: Dict[str, Any],
    reserved_ids: Set[int],
) -> Dict[int, int]:
    """
    Map old_internal_id -> new_internal_id.
    Prefer keeping same id when possible (not reserved and not already used), otherwise assign next available.
    """
    old_ids = [int(k) for k in sf1_opt_structure.keys() if isinstance(k, str) and k.isdigit()]
    old_ids_sorted = sorted(old_ids)
    used: Set[int] = set(reserved_ids)
    mapping: Dict[int, int] = {}

    # Determine assignment order based on traversal; ensures deterministic even if keys unsorted
    traversal = _topo_order_from_roots(sf1_opt_structure)
    traversal_ids = [int(k) for k in traversal if k.isdigit()]
    # ensure all nodes included
    seen = set(traversal_ids)
    for oid in old_ids_sorted:
        if oid not in seen:
            traversal_ids.append(oid)

    def next_free(start: int) -> int:
        x = start
        while x in used:
            x += 1
        return x

    for oid in traversal_ids:
        if oid not in used and oid not in mapping.values():
            # keep same id if possible
            nid = oid
        else:
            nid = next_free(0)
        mapping[oid] = nid
        used.add(nid)

    return mapping


def _rewrite_plan_references(
    plan: Dict[str, Any],
    internal_id_map: Dict[int, int],
    scan_id_map: Dict[int, int],
    stage: str,
) -> Dict[str, Any]:
    """
    Rebuild plan dict with remapped top-level keys and remapped node references (input/left/right/inputs/children).
    """
    # Build reverse key mapping old_str -> new_str
    old_to_new_key: Dict[str, str] = {}
    for old_int, new_int in internal_id_map.items():
        old_to_new_key[str(old_int)] = str(new_int)

    def map_ref(value: Any) -> Any:
        if not _is_intish(value):
            return value
        vid = _to_int(value)
        if vid in internal_id_map:
            return int(internal_id_map[vid])
        if vid in scan_id_map:
            return int(scan_id_map[vid])
        return value

    def rewrite_obj(obj: Any) -> Any:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k in _REF_KEYS_SCALAR:
                    out[k] = map_ref(v)
                elif k in _REF_KEYS_LIST and isinstance(v, list) and all(_is_intish(x) for x in v):
                    out[k] = [map_ref(x) for x in v]
                else:
                    out[k] = rewrite_obj(v)
            return out
        if isinstance(obj, list):
            return [rewrite_obj(x) for x in obj]
        return obj

    new_plan: Dict[str, Any] = {}
    for old_key, node in plan.items():
        if not (isinstance(old_key, str) and old_key.isdigit()):
            raise PlanTransferError(f"{stage}: non-numeric-node-id key={old_key!r}")
        if int(old_key) not in internal_id_map:
            raise PlanTransferError(f"{stage}: missing-internal-id-map node_id={old_key!r}")
        new_key = old_to_new_key[old_key]
        # Deepcopy each node, then rewrite
        node_copy = copy.deepcopy(node)
        node_rewritten = rewrite_obj(node_copy)
        new_plan[new_key] = node_rewritten

    return new_plan


def _rebind_parquet_scans_in_structure(
    structure: Dict[str, Any],
    sf2_succinct_table_info: Any,
    stage: str,
) -> Dict[str, Any]:
    """
    If the structure itself contains parquetScan nodes, replace their dataset-dependent baseConf fields
    using SF2 succinct_table_info by signature matching.
    """
    sf2_entries = _table_info_entries(sf2_succinct_table_info)
    sf2_by_sig: Dict[str, List[Dict[str, Any]]] = {}
    for e in sf2_entries:
        sig = _scan_normalized_signature(e.get("node", {}))
        sf2_by_sig.setdefault(sig, []).append(e)

    for sig in sf2_by_sig:
        # deterministic ordering for duplicates
        sf2_by_sig[sig] = sorted(sf2_by_sig[sig], key=lambda x: x["id"])

    # For deterministic pairing when multiple identical scan nodes exist in structure:
    # gather structure scan nodes grouped by sig in increasing node-id order.
    struct_scan_ids_by_sig: Dict[str, List[str]] = {}
    for nid_str in sorted([k for k in structure.keys() if isinstance(k, str) and k.isdigit()], key=int):
        node = structure[nid_str]
        if isinstance(node, dict) and "parquetScan" in node:
            sig = _scan_normalized_signature(node)
            struct_scan_ids_by_sig.setdefault(sig, []).append(nid_str)

    out = copy.deepcopy(structure)
    for sig, node_ids in struct_scan_ids_by_sig.items():
        sf2_list = sf2_by_sig.get(sig, [])
        if len(sf2_list) < len(node_ids):
            raise PlanTransferError(
                f"{stage}: rebind-scan-signature-unmatched sig={sig[:120]!r} struct_count={len(node_ids)} sf2_count={len(sf2_list)}"
            )
        for i, nid_str in enumerate(node_ids):
            sf1_node = out[nid_str]
            sf2_scan_node = copy.deepcopy(sf2_list[i]["node"])
            # Preserve predicate if present on sf1_node (rare, but safe)
            try:
                sf1_pred = sf1_node.get("parquetScan", {}).get("predicate", None)
            except Exception:
                sf1_pred = None
            if sf1_pred is not None:
                sf2_scan_node.setdefault("parquetScan", {})
                sf2_scan_node["parquetScan"]["predicate"] = sf1_pred
            out[nid_str] = sf2_scan_node

    return out


def _validate_structure_dict(structure: Any, stage: str) -> None:
    if not isinstance(structure, dict):
        raise PlanTransferError(f"{stage}: structure-must-be-dict")
    for k, v in structure.items():
        if not (isinstance(k, str) and k.isdigit()):
            raise PlanTransferError(f"{stage}: invalid-node-id key={k!r}")
        if not isinstance(v, dict):
            raise PlanTransferError(f"{stage}: node-must-be-dict node_id={k!r}")
        if len(v) != 1:
            raise PlanTransferError(f"{stage}: node-must-have-single-operator node_id={k!r}")


def _validate_references(structure: Dict[str, Any], sf2_scan_ids: Set[int], stage: str) -> None:
    internal_ids = {int(k) for k in structure.keys() if isinstance(k, str) and k.isdigit()}

    bad_refs: List[Tuple[str, str, Any]] = []

    def walk(nid: str, obj: Any):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in _REF_KEYS_SCALAR and _is_intish(v):
                    rid = _to_int(v)
                    if rid not in internal_ids and rid not in sf2_scan_ids:
                        bad_refs.append((nid, k, rid))
                elif k in _REF_KEYS_LIST and isinstance(v, list) and all(_is_intish(x) for x in v):
                    for x in v:
                        rid = _to_int(x)
                        if rid not in internal_ids and rid not in sf2_scan_ids:
                            bad_refs.append((nid, k, rid))
                else:
                    walk(nid, v)
        elif isinstance(obj, list):
            for it in obj:
                walk(nid, it)

    for nid, node in structure.items():
        walk(nid, node)

    if bad_refs:
        # concise: show first few
        sample = bad_refs[:5]
        raise PlanTransferError(f"{stage}: dangling-refs count={len(bad_refs)} sample={sample!r}")




def transfer_plan(
    query,
    sf1_base_structure,
    sf1_succinct_table_info,
    sf2_base_structure,
    sf2_succinct_table_info,
    sf1_patch,
) -> dict:
    """Transfer an optimized plan from one scale factor (SF1) to another (SF2).

    Applies *sf1_patch* to *sf1_base_structure*, then remaps scan-node IDs
    and rebinds parquetScan metadata so the resulting structure is runnable
    against the SF2 dataset. Inputs are never mutated.

    Args:
        query: SQL query string (unused, kept for context).
        sf1_base_structure: Base plan structure dict at SF1.
        sf1_succinct_table_info: Scan-node metadata at SF1.
        sf2_base_structure: Base plan structure dict at SF2.
        sf2_succinct_table_info: Scan-node metadata at SF2.
        sf1_patch: JSON Patch operations to apply to SF1 base.

    Returns:
        Transferred structure dict runnable at SF2.

    Raises:
        PlanTransferError: On patch failure, unresolvable references, or
            input mutation.
    """
    # Snapshot inputs for immutability validation
    try:
        sf1_base_before = _json_dumps_canonical(sf1_base_structure)
        sf2_base_before = _json_dumps_canonical(sf2_base_structure)
        sf1_succ_before = _json_dumps_canonical(sf1_succinct_table_info)
        sf2_succ_before = _json_dumps_canonical(sf2_succinct_table_info)
        sf1_patch_before = _json_dumps_canonical(sf1_patch)
    except Exception as e:
        raise PlanTransferError("validation: inputs-not-json-serializable") from e

    try:
        sf1_base_copy = copy.deepcopy(sf1_base_structure)
        sf1_opt_structure = _apply_json_patch(sf1_base_copy, copy.deepcopy(sf1_patch))
    except Exception as e:
        raise PlanTransferError(f"patch-apply: failed err={e.__class__.__name__}") from e

    # Validate inputs were not mutated
    if _json_dumps_canonical(sf1_base_structure) != sf1_base_before:
        raise PlanTransferError("validation: input-mutated sf1_base_structure")
    if _json_dumps_canonical(sf2_base_structure) != sf2_base_before:
        raise PlanTransferError("validation: input-mutated sf2_base_structure")
    if _json_dumps_canonical(sf1_succinct_table_info) != sf1_succ_before:
        raise PlanTransferError("validation: input-mutated sf1_succinct_table_info")
    if _json_dumps_canonical(sf2_succinct_table_info) != sf2_succ_before:
        raise PlanTransferError("validation: input-mutated sf2_succinct_table_info")
    if _json_dumps_canonical(sf1_patch) != sf1_patch_before:
        raise PlanTransferError("validation: input-mutated sf1_patch")

    # Sanity validate the patched structure is a proper plan-structure dict
    _validate_structure_dict(sf1_opt_structure, stage="validation: sf1_opt_structure")

    transferred = _transfer_plan(
        sf1_base_structure=sf1_base_structure,
        sf1_succinct_table_info=sf1_succinct_table_info,
        sf2_base_structure=sf2_base_structure,
        sf2_succinct_table_info=sf2_succinct_table_info,
        sf1_opt_structure=sf1_opt_structure,
    )

    # Validate returned object is the structure dict itself
    _validate_structure_dict(transferred, stage="validation: transferred_structure")

    # Validate references are resolvable as internal nodes or SF2 scan ids
    sf2_scan_ids = set(_build_scan_signature_map(sf2_succinct_table_info).keys())
    _validate_references(transferred, sf2_scan_ids=sf2_scan_ids, stage="validation: transferred_structure")

    return transferred


def _transfer_plan(
    sf1_base_structure,
    sf1_succinct_table_info,
    sf2_base_structure,
    sf2_succinct_table_info,
    sf1_opt_structure,
) -> dict:
    """Core transfer logic: remap scan IDs, rebind parquet metadata, renumber nodes.

    Produces a runnable structure for SF2. Must NOT access sf1_patch directly;
    operates only on the already-patched *sf1_opt_structure*.
    """
    # Basic validations
    _validate_structure_dict(sf1_base_structure, stage="transfer: sf1_base_structure")
    _validate_structure_dict(sf2_base_structure, stage="transfer: sf2_base_structure")
    _validate_structure_dict(sf1_opt_structure, stage="transfer: sf1_opt_structure")

    # 1) Build scan-id mapping SF1 -> SF2 by scan signature
    scan_id_map = _build_scan_id_mapping(sf1_succinct_table_info, sf2_succinct_table_info)
    sf2_scan_ids = set(_build_scan_signature_map(sf2_succinct_table_info).keys())

    # 2) Start from SF1 optimized structure; rebind embedded parquetScan nodes (if any) to SF2 metadata
    #    (still keeping structure-only output)
    try:
        rebased_opt = _rebind_parquet_scans_in_structure(
            structure=sf1_opt_structure,
            sf2_succinct_table_info=sf2_succinct_table_info,
            stage="transfer",
        )
    except PlanTransferError:
        raise
    except Exception as e:
        raise PlanTransferError(f"transfer: rebind-parquet-scans failed err={e.__class__.__name__}") from e

    # 3) Renumber internal node IDs to avoid collision with SF2 scan ids
    reserved = set(sf2_scan_ids)
    internal_id_map = _assign_new_node_ids(rebased_opt, reserved_ids=reserved)

    # 4) Rewrite internal references and scan leaf references
    try:
        transferred = _rewrite_plan_references(
            plan=rebased_opt,
            internal_id_map=internal_id_map,
            scan_id_map=scan_id_map,
            stage="transfer",
        )
    except PlanTransferError:
        raise
    except Exception as e:
        raise PlanTransferError(f"transfer: rewrite-references failed err={e.__class__.__name__}") from e

    # 5) Validate transferred references
    _validate_structure_dict(transferred, stage="transfer: transferred_structure")
    _validate_references(transferred, sf2_scan_ids=sf2_scan_ids, stage="transfer: transferred_structure")

    # 6) Ensure that every referenced leaf (non-internal) that looks like an ID is an SF2 scan id.
    #    (This catches SF1 scan ids that weren't mapped.)
    internal_ids = {int(k) for k in transferred.keys() if k.isdigit()}

    unresolved: Set[int] = set()

    def walk(obj: Any):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in _REF_KEYS_SCALAR and _is_intish(v):
                    rid = _to_int(v)
                    if rid not in internal_ids and rid not in sf2_scan_ids:
                        unresolved.add(rid)
                elif k in _REF_KEYS_LIST and isinstance(v, list) and all(_is_intish(x) for x in v):
                    for x in v:
                        rid = _to_int(x)
                        if rid not in internal_ids and rid not in sf2_scan_ids:
                            unresolved.add(rid)
                else:
                    walk(v)
        elif isinstance(obj, list):
            for it in obj:
                walk(it)

    for node in transferred.values():
        walk(node)

    if unresolved:
        sample = sorted(unresolved)[:10]
        raise PlanTransferError(f"transfer: unresolved-leaf-refs count={len(unresolved)} sample={sample!r}")

    return transferred