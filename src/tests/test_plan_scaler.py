"""Tests for sampling.plan_scaler — pure plan-transfer utilities (no Modal/S3)."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List

import pytest

from sampling.plan_scaler import (
    PlanTransferError,
    _json_pointer_unescape,
    _json_pointer_split,
    _json_pointer_resolve_parent,
    _json_pointer_get,
    _json_pointer_add,
    _json_pointer_remove,
    _json_pointer_replace,
    _apply_json_patch,
    _is_intish,
    _to_int,
    _extract_operator,
    _table_info_entries,
    _scan_normalized_signature,
    _build_scan_signature_map,
    _build_scan_id_mapping,
    _find_internal_references,
    _find_roots,
    _topo_order_from_roots,
    _assign_new_node_ids,
    _rewrite_plan_references,
    _validate_structure_dict,
    _validate_references,
    transfer_plan,
)


# ---------------------------------------------------------------------------
# Shared fixtures — realistic plan structures
# ---------------------------------------------------------------------------

def _make_scan_node(columns, projection=None, predicate=None):
    """Build a parquetScan node with given schema columns."""
    base_conf = {
        "schema": {"columns": columns},
        "projection": projection or list(range(len(columns))),
        "constraints": None,
    }
    node = {"parquetScan": {"baseConf": base_conf}}
    if predicate is not None:
        node["parquetScan"]["predicate"] = predicate
    return node


def _make_table_info_entry(sid, columns, projection=None):
    """Build a succinct_table_info entry."""
    node = _make_scan_node(columns, projection)
    # Add dataset-dependent fields that will vary between SF1/SF2
    node["parquetScan"]["baseConf"]["fileGroups"] = [f"/data/sf/table_{sid}.parquet"]
    return {"id": sid, "node": node}


@pytest.fixture
def simple_plan_pair():
    """A simple plan with one join: projection -> hashJoin -> (scanA, scanB).

    SF1 uses scan IDs 100, 101; SF2 uses 200, 201 (same schema signatures).
    Internal nodes: 10 (projection), 5 (hashJoin).
    """
    sf1_structure = {
        "10": {"projection": {"input": 5, "expr": ["col_a"]}},
        "5": {"hashJoinExec": {"left": 100, "right": 101, "on": "key"}},
    }
    sf1_table_info = [
        _make_table_info_entry(100, ["key", "col_a"]),
        _make_table_info_entry(101, ["key", "col_b"]),
    ]
    sf2_structure = {
        "10": {"projection": {"input": 5, "expr": ["col_a"]}},
        "5": {"hashJoinExec": {"left": 200, "right": 201, "on": "key"}},
    }
    sf2_table_info = [
        _make_table_info_entry(200, ["key", "col_a"]),
        _make_table_info_entry(201, ["key", "col_b"]),
    ]
    return {
        "sf1_structure": sf1_structure,
        "sf1_table_info": sf1_table_info,
        "sf2_structure": sf2_structure,
        "sf2_table_info": sf2_table_info,
    }


# ===================================================================
# JSON Pointer primitives
# ===================================================================

class TestJsonPointerUnescape:
    def test_no_escaping(self):
        assert _json_pointer_unescape("foo") == "foo"

    def test_tilde_zero(self):
        assert _json_pointer_unescape("a~0b") == "a~b"

    def test_tilde_one(self):
        assert _json_pointer_unescape("a~1b") == "a/b"

    def test_both_escapes(self):
        assert _json_pointer_unescape("~0~1") == "~/"

    def test_order_matters(self):
        # ~1 is replaced first, then ~0
        assert _json_pointer_unescape("~01") == "~1"


class TestJsonPointerSplit:
    def test_empty_string(self):
        assert _json_pointer_split("") == []

    def test_root_key(self):
        assert _json_pointer_split("/foo") == ["foo"]

    def test_nested_path(self):
        assert _json_pointer_split("/a/b/c") == ["a", "b", "c"]

    def test_escaped_slash(self):
        assert _json_pointer_split("/a~1b") == ["a/b"]

    def test_no_leading_slash_raises(self):
        with pytest.raises(PlanTransferError, match="invalid-json-pointer"):
            _json_pointer_split("foo/bar")

    def test_numeric_index(self):
        assert _json_pointer_split("/items/0/name") == ["items", "0", "name"]


class TestJsonPointerGet:
    def test_root_key(self):
        assert _json_pointer_get({"a": 1}, "/a") == 1

    def test_nested_dict(self):
        doc = {"a": {"b": {"c": 42}}}
        assert _json_pointer_get(doc, "/a/b/c") == 42

    def test_list_index(self):
        doc = {"items": [10, 20, 30]}
        assert _json_pointer_get(doc, "/items/1") == 20

    def test_empty_pointer_returns_doc(self):
        doc = {"x": 1}
        assert _json_pointer_get(doc, "") == doc

    def test_missing_key_raises(self):
        with pytest.raises(PlanTransferError, match="get-path-not-found"):
            _json_pointer_get({"a": 1}, "/b")

    def test_list_oob_raises(self):
        with pytest.raises(PlanTransferError, match="get-list-index-oob"):
            _json_pointer_get({"x": [1]}, "/x/5")

    def test_non_container_raises(self):
        with pytest.raises(PlanTransferError, match="get-non-container"):
            _json_pointer_get({"a": 42}, "/a/b")


class TestJsonPointerAdd:
    def test_add_to_dict(self):
        doc = {"a": 1}
        result = _json_pointer_add(doc, "/b", 2)
        assert result == {"a": 1, "b": 2}

    def test_add_to_list_end(self):
        doc = {"items": [1, 2]}
        result = _json_pointer_add(doc, "/items/-", 3)
        assert result["items"] == [1, 2, 3]

    def test_insert_into_list(self):
        doc = {"items": [1, 3]}
        result = _json_pointer_add(doc, "/items/1", 2)
        assert result["items"] == [1, 2, 3]

    def test_replace_whole_document(self):
        result = _json_pointer_add({"old": True}, "", {"new": True})
        assert result == {"new": True}

    def test_nested_add(self):
        doc = {"a": {"b": {}}}
        result = _json_pointer_add(doc, "/a/b/c", 99)
        assert result["a"]["b"]["c"] == 99


class TestJsonPointerRemove:
    def test_remove_dict_key(self):
        doc = {"a": 1, "b": 2}
        result = _json_pointer_remove(doc, "/b")
        assert result == {"a": 1}

    def test_remove_list_element(self):
        doc = {"items": [1, 2, 3]}
        result = _json_pointer_remove(doc, "/items/1")
        assert result["items"] == [1, 3]

    def test_remove_root_raises(self):
        with pytest.raises(PlanTransferError, match="remove-root-not-allowed"):
            _json_pointer_remove({"a": 1}, "")

    def test_remove_missing_key_raises(self):
        with pytest.raises(PlanTransferError, match="remove-path-not-found"):
            _json_pointer_remove({"a": 1}, "/b")


class TestJsonPointerReplace:
    def test_replace_dict_value(self):
        doc = {"a": 1}
        result = _json_pointer_replace(doc, "/a", 99)
        assert result == {"a": 99}

    def test_replace_list_element(self):
        doc = {"items": [1, 2, 3]}
        result = _json_pointer_replace(doc, "/items/0", 10)
        assert result["items"] == [10, 2, 3]

    def test_replace_root(self):
        result = _json_pointer_replace({"old": 1}, "", {"new": 2})
        assert result == {"new": 2}

    def test_replace_missing_raises(self):
        with pytest.raises(PlanTransferError, match="get-path-not-found"):
            _json_pointer_replace({"a": 1}, "/b", 2)


# ===================================================================
# _apply_json_patch — RFC 6902
# ===================================================================

class TestApplyJsonPatch:
    def test_add_op(self):
        doc = {"a": 1}
        patch = [{"op": "add", "path": "/b", "value": 2}]
        assert _apply_json_patch(doc, patch) == {"a": 1, "b": 2}

    def test_remove_op(self):
        doc = {"a": 1, "b": 2}
        patch = [{"op": "remove", "path": "/b"}]
        assert _apply_json_patch(doc, patch) == {"a": 1}

    def test_replace_op(self):
        doc = {"a": 1}
        patch = [{"op": "replace", "path": "/a", "value": 99}]
        assert _apply_json_patch(doc, patch) == {"a": 99}

    def test_move_op(self):
        doc = {"a": 1, "b": 2}
        patch = [{"op": "move", "from": "/a", "path": "/c"}]
        result = _apply_json_patch(doc, patch)
        assert result == {"b": 2, "c": 1}

    def test_copy_op(self):
        doc = {"a": 1}
        patch = [{"op": "copy", "from": "/a", "path": "/b"}]
        result = _apply_json_patch(doc, patch)
        assert result == {"a": 1, "b": 1}

    def test_test_op_passes(self):
        doc = {"a": 1}
        patch = [{"op": "test", "path": "/a", "value": 1}]
        assert _apply_json_patch(doc, patch) == {"a": 1}

    def test_test_op_fails(self):
        doc = {"a": 1}
        patch = [{"op": "test", "path": "/a", "value": 2}]
        with pytest.raises(PlanTransferError, match="test-failed"):
            _apply_json_patch(doc, patch)

    def test_multi_op_sequence(self):
        doc = {"x": 1}
        patch = [
            {"op": "add", "path": "/y", "value": 2},
            {"op": "replace", "path": "/x", "value": 10},
            {"op": "remove", "path": "/y"},
        ]
        assert _apply_json_patch(doc, patch) == {"x": 10}

    def test_empty_patch(self):
        doc = {"a": 1}
        assert _apply_json_patch(doc, []) == {"a": 1}

    def test_non_list_patch_raises(self):
        with pytest.raises(PlanTransferError, match="patch-must-be-list"):
            _apply_json_patch({}, "not a list")

    def test_invalid_op_raises(self):
        with pytest.raises(PlanTransferError, match="unsupported-op"):
            _apply_json_patch({}, [{"op": "destroy", "path": "/a"}])

    def test_missing_value_for_add_raises(self):
        with pytest.raises(PlanTransferError, match="add-missing-value"):
            _apply_json_patch({}, [{"op": "add", "path": "/a"}])

    def test_missing_from_for_move_raises(self):
        with pytest.raises(PlanTransferError, match="move-missing-from"):
            _apply_json_patch({"a": 1}, [{"op": "move", "path": "/b"}])

    def test_deep_copy_isolation(self):
        """Added values must be deep copies — mutations to the patch dict must not affect the result."""
        inner = {"nested": [1, 2, 3]}
        patch = [{"op": "add", "path": "/x", "value": inner}]
        doc = _apply_json_patch({}, patch)
        inner["nested"].append(999)
        assert doc["x"]["nested"] == [1, 2, 3]


# ===================================================================
# Small helpers
# ===================================================================

class TestIsIntish:
    def test_int(self):
        assert _is_intish(42) is True

    def test_str_digit(self):
        assert _is_intish("7") is True

    def test_str_non_digit(self):
        assert _is_intish("abc") is False

    def test_float(self):
        assert _is_intish(3.14) is False

    def test_none(self):
        assert _is_intish(None) is False

    def test_zero(self):
        assert _is_intish(0) is True
        assert _is_intish("0") is True


class TestToInt:
    def test_int(self):
        assert _to_int(42) == 42

    def test_str_digit(self):
        assert _to_int("7") == 7

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="not-intish"):
            _to_int("abc")


class TestExtractOperator:
    def test_single_key(self):
        name, conf = _extract_operator({"hashJoinExec": {"left": 1}})
        assert name == "hashJoinExec"
        assert conf == {"left": 1}

    def test_empty_dict_raises(self):
        with pytest.raises(PlanTransferError, match="single-operator"):
            _extract_operator({})

    def test_multiple_keys_raises(self):
        with pytest.raises(PlanTransferError, match="single-operator"):
            _extract_operator({"a": 1, "b": 2})

    def test_non_dict_raises(self):
        with pytest.raises(PlanTransferError, match="single-operator"):
            _extract_operator("not a dict")


# ===================================================================
# Scan signature / table info helpers
# ===================================================================

class TestTableInfoEntries:
    def test_list_form(self):
        entries = [{"id": 1, "node": {}}, {"id": 2, "node": {}}]
        assert _table_info_entries(entries) == entries

    def test_dict_with_entries_key(self):
        data = {"entries": [{"id": 1, "node": {}}, {"id": 2, "node": {}}]}
        result = _table_info_entries(data)
        assert len(result) == 2

    def test_none_returns_empty(self):
        assert _table_info_entries(None) == []

    def test_filters_invalid_entries(self):
        entries = [{"id": 1, "node": {}}, {"bad": True}, {"id": 2}]
        result = _table_info_entries(entries)
        assert len(result) == 1
        assert result[0]["id"] == 1


class TestScanNormalizedSignature:
    def test_same_schema_same_signature(self):
        node_a = _make_scan_node(["x", "y"])
        node_b = _make_scan_node(["x", "y"])
        assert _scan_normalized_signature(node_a) == _scan_normalized_signature(node_b)

    def test_different_schema_different_signature(self):
        node_a = _make_scan_node(["x", "y"])
        node_b = _make_scan_node(["x", "z"])
        assert _scan_normalized_signature(node_a) != _scan_normalized_signature(node_b)

    def test_file_groups_ignored(self):
        """Dataset-dependent fields like fileGroups don't affect the signature."""
        node_a = _make_scan_node(["a"])
        node_a["parquetScan"]["baseConf"]["fileGroups"] = ["/data/sf1/table.parquet"]
        node_b = _make_scan_node(["a"])
        node_b["parquetScan"]["baseConf"]["fileGroups"] = ["/data/sf2/table.parquet"]
        assert _scan_normalized_signature(node_a) == _scan_normalized_signature(node_b)

    def test_non_parquet_node(self):
        node = {"csvScan": {"path": "/data"}}
        sig = _scan_normalized_signature(node)
        assert isinstance(sig, str)


class TestBuildScanSignatureMap:
    def test_basic(self):
        table_info = [
            _make_table_info_entry(100, ["a", "b"]),
            _make_table_info_entry(101, ["c", "d"]),
        ]
        sig_map = _build_scan_signature_map(table_info)
        assert 100 in sig_map
        assert 101 in sig_map
        assert sig_map[100] != sig_map[101]

    def test_same_schema_same_signature(self):
        table_info = [
            _make_table_info_entry(100, ["a"]),
            _make_table_info_entry(101, ["a"]),
        ]
        sig_map = _build_scan_signature_map(table_info)
        assert sig_map[100] == sig_map[101]


class TestBuildScanIdMapping:
    def test_one_to_one(self):
        sf1 = [_make_table_info_entry(100, ["a"]), _make_table_info_entry(101, ["b"])]
        sf2 = [_make_table_info_entry(200, ["a"]), _make_table_info_entry(201, ["b"])]
        mapping = _build_scan_id_mapping(sf1, sf2)
        assert mapping == {100: 200, 101: 201}

    def test_duplicate_signatures_paired_by_sorted_id(self):
        sf1 = [_make_table_info_entry(100, ["a"]), _make_table_info_entry(102, ["a"])]
        sf2 = [_make_table_info_entry(200, ["a"]), _make_table_info_entry(203, ["a"])]
        mapping = _build_scan_id_mapping(sf1, sf2)
        # Sorted SF1: [100, 102], sorted SF2: [200, 203] => paired by index
        assert mapping == {100: 200, 102: 203}

    def test_sf2_has_extra_scans_ok(self):
        sf1 = [_make_table_info_entry(100, ["a"])]
        sf2 = [_make_table_info_entry(200, ["a"]), _make_table_info_entry(201, ["a"])]
        mapping = _build_scan_id_mapping(sf1, sf2)
        assert mapping == {100: 200}

    def test_sf2_missing_signature_raises(self):
        sf1 = [_make_table_info_entry(100, ["a"])]
        sf2 = [_make_table_info_entry(200, ["b"])]  # different schema
        with pytest.raises(PlanTransferError, match="scan-signature-unmatched"):
            _build_scan_id_mapping(sf1, sf2)

    def test_sf2_too_few_of_same_signature_raises(self):
        sf1 = [_make_table_info_entry(100, ["a"]), _make_table_info_entry(101, ["a"])]
        sf2 = [_make_table_info_entry(200, ["a"])]  # only one
        with pytest.raises(PlanTransferError, match="scan-signature-unmatched"):
            _build_scan_id_mapping(sf1, sf2)


# ===================================================================
# Plan structure graph helpers
# ===================================================================

class TestFindInternalReferences:
    def test_scalar_refs(self):
        structure = {
            "10": {"projection": {"input": 5}},
            "5": {"scan": {"table": "t"}},
        }
        refs = _find_internal_references(structure)
        assert refs == {5}

    def test_list_refs(self):
        structure = {
            "10": {"union": {"inputs": [5, 6]}},
            "5": {"scan": {"table": "a"}},
            "6": {"scan": {"table": "b"}},
        }
        refs = _find_internal_references(structure)
        assert refs == {5, 6}

    def test_external_refs_excluded(self):
        """References to IDs not in structure keys are excluded."""
        structure = {
            "10": {"projection": {"input": 999}},
        }
        refs = _find_internal_references(structure)
        assert refs == set()

    def test_left_right_refs(self):
        structure = {
            "10": {"hashJoinExec": {"left": 5, "right": 6}},
            "5": {"scan": {"table": "a"}},
            "6": {"scan": {"table": "b"}},
        }
        refs = _find_internal_references(structure)
        assert refs == {5, 6}


class TestFindRoots:
    def test_single_root(self):
        structure = {
            "10": {"projection": {"input": 5}},
            "5": {"scan": {"table": "t"}},
        }
        roots = _find_roots(structure)
        assert roots == [10]

    def test_multiple_roots(self):
        structure = {
            "10": {"scan": {"table": "a"}},
            "20": {"scan": {"table": "b"}},
        }
        roots = _find_roots(structure)
        assert set(roots) == {10, 20}

    def test_chain(self):
        structure = {
            "10": {"projection": {"input": 5}},
            "5": {"filter": {"input": 3}},
            "3": {"scan": {"table": "t"}},
        }
        roots = _find_roots(structure)
        assert roots == [10]


class TestTopoOrderFromRoots:
    def test_linear_chain(self):
        structure = {
            "10": {"projection": {"input": 5}},
            "5": {"filter": {"input": 3}},
            "3": {"scan": {"table": "t"}},
        }
        order = _topo_order_from_roots(structure)
        # Post-order DFS from root 10: visits 3, then 5, then 10
        assert order == ["3", "5", "10"]

    def test_join_tree(self):
        structure = {
            "10": {"hashJoinExec": {"left": 5, "right": 6}},
            "5": {"scan": {"table": "a"}},
            "6": {"scan": {"table": "b"}},
        }
        order = _topo_order_from_roots(structure)
        # Post-order from root 10: left child 5 first, then right 6, then 10
        assert order == ["5", "6", "10"]

    def test_deterministic_on_repeated_calls(self):
        structure = {
            "10": {"hashJoinExec": {"left": 5, "right": 6}},
            "5": {"scan": {"table": "a"}},
            "6": {"scan": {"table": "b"}},
        }
        assert _topo_order_from_roots(structure) == _topo_order_from_roots(structure)


class TestAssignNewNodeIds:
    def test_no_conflicts(self):
        structure = {"5": {"scan": {}}, "10": {"proj": {"input": 5}}}
        mapping = _assign_new_node_ids(structure, reserved_ids=set())
        # No reserved IDs, so nodes keep their IDs
        assert mapping[5] == 5
        assert mapping[10] == 10

    def test_conflict_with_reserved(self):
        structure = {"5": {"scan": {}}}
        mapping = _assign_new_node_ids(structure, reserved_ids={5})
        # 5 is reserved, so must get a different ID
        assert mapping[5] != 5
        assert mapping[5] not in {5}

    def test_deterministic(self):
        structure = {
            "1": {"scan": {}},
            "2": {"proj": {"input": 1}},
            "3": {"filter": {"input": 2}},
        }
        reserved = {1, 3}
        m1 = _assign_new_node_ids(structure, reserved)
        m2 = _assign_new_node_ids(structure, reserved)
        assert m1 == m2


# ===================================================================
# _rewrite_plan_references
# ===================================================================

class TestRewritePlanReferences:
    def test_basic_rewrite(self):
        plan = {
            "10": {"projection": {"input": 5}},
            "5": {"hashJoinExec": {"left": 100, "right": 101}},
        }
        internal_map = {10: 20, 5: 15}
        scan_map = {100: 200, 101: 201}
        result = _rewrite_plan_references(plan, internal_map, scan_map, "test")
        # Keys should be remapped
        assert "20" in result
        assert "15" in result
        assert "10" not in result
        # Internal ref 5 -> 15
        assert result["20"]["projection"]["input"] == 15
        # Scan refs -> SF2 IDs
        assert result["15"]["hashJoinExec"]["left"] == 200
        assert result["15"]["hashJoinExec"]["right"] == 201

    def test_list_refs_rewritten(self):
        plan = {"10": {"union": {"inputs": [5, 6]}}, "5": {"scan": {}}, "6": {"scan": {}}}
        internal_map = {10: 100, 5: 50, 6: 60}
        result = _rewrite_plan_references(plan, internal_map, {}, "test")
        assert result["100"]["union"]["inputs"] == [50, 60]

    def test_does_not_mutate_original(self):
        plan = {"10": {"projection": {"input": 5}}, "5": {"scan": {}}}
        plan_copy = copy.deepcopy(plan)
        internal_map = {10: 20, 5: 15}
        _rewrite_plan_references(plan, internal_map, {}, "test")
        assert plan == plan_copy

    def test_missing_internal_id_raises(self):
        plan = {"10": {"scan": {}}}
        with pytest.raises(PlanTransferError, match="missing-internal-id-map"):
            _rewrite_plan_references(plan, {}, {}, "test")

    def test_non_numeric_key_raises(self):
        plan = {"root": {"scan": {}}}
        with pytest.raises(PlanTransferError, match="non-numeric-node-id"):
            _rewrite_plan_references(plan, {}, {}, "test")


# ===================================================================
# Validation helpers
# ===================================================================

class TestValidateStructureDict:
    def test_valid(self):
        _validate_structure_dict({"10": {"scan": {}}}, "test")

    def test_non_dict_raises(self):
        with pytest.raises(PlanTransferError, match="structure-must-be-dict"):
            _validate_structure_dict([1, 2], "test")

    def test_non_numeric_key_raises(self):
        with pytest.raises(PlanTransferError, match="invalid-node-id"):
            _validate_structure_dict({"abc": {"scan": {}}}, "test")

    def test_node_not_dict_raises(self):
        with pytest.raises(PlanTransferError, match="node-must-be-dict"):
            _validate_structure_dict({"10": "not a dict"}, "test")

    def test_multiple_operators_raises(self):
        with pytest.raises(PlanTransferError, match="single-operator"):
            _validate_structure_dict({"10": {"scan": {}, "filter": {}}}, "test")


class TestValidateReferences:
    def test_valid_refs(self):
        structure = {
            "10": {"projection": {"input": 5}},
            "5": {"hashJoinExec": {"left": 100, "right": 101}},
        }
        sf2_scan_ids = {100, 101}
        _validate_references(structure, sf2_scan_ids, "test")  # should not raise

    def test_dangling_ref_raises(self):
        structure = {
            "10": {"projection": {"input": 999}},
        }
        with pytest.raises(PlanTransferError, match="dangling-refs"):
            _validate_references(structure, sf2_scan_ids=set(), stage="test")

    def test_internal_ref_ok(self):
        structure = {
            "10": {"projection": {"input": 5}},
            "5": {"scan": {}},
        }
        _validate_references(structure, sf2_scan_ids=set(), stage="test")


# ===================================================================
# transfer_plan — end-to-end
# ===================================================================

class TestTransferPlan:
    def test_noop_patch(self, simple_plan_pair):
        """Empty patch should produce a valid transferred plan."""
        p = simple_plan_pair
        result = transfer_plan(
            query="SELECT ...",
            sf1_base_structure=p["sf1_structure"],
            sf1_succinct_table_info=p["sf1_table_info"],
            sf2_base_structure=p["sf2_structure"],
            sf2_succinct_table_info=p["sf2_table_info"],
            sf1_patch=[],
        )
        assert isinstance(result, dict)
        # Result should reference SF2 scan IDs (200, 201), not SF1 (100, 101)
        all_refs = set()
        for node in result.values():
            for op_conf in node.values():
                if isinstance(op_conf, dict):
                    for k, v in op_conf.items():
                        if k in ("left", "right", "input") and isinstance(v, int):
                            all_refs.add(v)
        internal_ids = {int(k) for k in result.keys()}
        leaf_refs = all_refs - internal_ids
        assert all(r in {200, 201} for r in leaf_refs), f"unexpected leaf refs: {leaf_refs}"

    def test_replace_patch(self, simple_plan_pair):
        """A replace patch on the internal structure should transfer correctly."""
        p = simple_plan_pair
        patch = [{"op": "replace", "path": "/10/projection/expr", "value": ["col_b"]}]
        result = transfer_plan(
            query="SELECT ...",
            sf1_base_structure=p["sf1_structure"],
            sf1_succinct_table_info=p["sf1_table_info"],
            sf2_base_structure=p["sf2_structure"],
            sf2_succinct_table_info=p["sf2_table_info"],
            sf1_patch=patch,
        )
        # Find the projection node and verify the expr was changed
        proj_node = None
        for node in result.values():
            if "projection" in node:
                proj_node = node
        assert proj_node is not None
        assert proj_node["projection"]["expr"] == ["col_b"]

    def test_inputs_not_mutated(self, simple_plan_pair):
        """transfer_plan must not mutate any of its inputs."""
        p = simple_plan_pair
        sf1_struct_before = copy.deepcopy(p["sf1_structure"])
        sf2_struct_before = copy.deepcopy(p["sf2_structure"])
        sf1_ti_before = copy.deepcopy(p["sf1_table_info"])
        sf2_ti_before = copy.deepcopy(p["sf2_table_info"])
        patch = [{"op": "replace", "path": "/10/projection/expr", "value": ["col_b"]}]
        patch_before = copy.deepcopy(patch)

        transfer_plan(
            query="q",
            sf1_base_structure=p["sf1_structure"],
            sf1_succinct_table_info=p["sf1_table_info"],
            sf2_base_structure=p["sf2_structure"],
            sf2_succinct_table_info=p["sf2_table_info"],
            sf1_patch=patch,
        )
        assert p["sf1_structure"] == sf1_struct_before
        assert p["sf2_structure"] == sf2_struct_before
        assert p["sf1_table_info"] == sf1_ti_before
        assert p["sf2_table_info"] == sf2_ti_before
        assert patch == patch_before

    def test_bad_patch_raises(self, simple_plan_pair):
        """A patch targeting a nonexistent path should raise PlanTransferError."""
        p = simple_plan_pair
        bad_patch = [{"op": "replace", "path": "/9999/nonexistent", "value": 1}]
        with pytest.raises(PlanTransferError, match="patch-apply"):
            transfer_plan(
                query="q",
                sf1_base_structure=p["sf1_structure"],
                sf1_succinct_table_info=p["sf1_table_info"],
                sf2_base_structure=p["sf2_structure"],
                sf2_succinct_table_info=p["sf2_table_info"],
                sf1_patch=bad_patch,
            )

    def test_deterministic(self, simple_plan_pair):
        """Same inputs must always produce the same output."""
        p = simple_plan_pair
        patch = [{"op": "replace", "path": "/10/projection/expr", "value": ["new"]}]
        r1 = transfer_plan("q", p["sf1_structure"], p["sf1_table_info"],
                           p["sf2_structure"], p["sf2_table_info"], patch)
        r2 = transfer_plan("q", p["sf1_structure"], p["sf1_table_info"],
                           p["sf2_structure"], p["sf2_table_info"], patch)
        assert r1 == r2

    def test_add_node_patch(self, simple_plan_pair):
        """A patch that adds a new internal node should transfer correctly."""
        p = simple_plan_pair
        # Add a filter node between projection and join
        patch = [
            {"op": "add", "path": "/7", "value": {"filterExec": {"input": 5, "predicate": "x > 0"}}},
            {"op": "replace", "path": "/10/projection/input", "value": 7},
        ]
        result = transfer_plan(
            query="q",
            sf1_base_structure=p["sf1_structure"],
            sf1_succinct_table_info=p["sf1_table_info"],
            sf2_base_structure=p["sf2_structure"],
            sf2_succinct_table_info=p["sf2_table_info"],
            sf1_patch=patch,
        )
        assert isinstance(result, dict)
        # Should have 3 internal nodes now
        assert len(result) == 3
        # Verify structure is valid (no dangling refs)
        _validate_structure_dict(result, "post-test")

    def test_id_collision_resolved(self):
        """When SF1 internal IDs collide with SF2 scan IDs, they get renumbered."""
        sf1_structure = {
            "200": {"projection": {"input": 100}},
        }
        sf1_table_info = [_make_table_info_entry(100, ["a"])]
        sf2_structure = {
            "200": {"projection": {"input": 200}},
        }
        sf2_table_info = [_make_table_info_entry(200, ["a"])]

        result = transfer_plan(
            query="q",
            sf1_base_structure=sf1_structure,
            sf1_succinct_table_info=sf1_table_info,
            sf2_base_structure=sf2_structure,
            sf2_succinct_table_info=sf2_table_info,
            sf1_patch=[],
        )
        # Internal node 200 collides with SF2 scan ID 200, so it should be renumbered
        internal_ids = {int(k) for k in result.keys()}
        assert 200 not in internal_ids, "ID 200 should be renumbered to avoid collision with SF2 scan ID"
        # The leaf ref should point to SF2 scan 200
        for node in result.values():
            for op_conf in node.values():
                if isinstance(op_conf, dict) and "input" in op_conf:
                    assert op_conf["input"] == 200
