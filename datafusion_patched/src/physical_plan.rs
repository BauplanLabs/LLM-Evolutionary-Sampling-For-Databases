// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use anyhow::{Context, Result as AnyhowResult};
use datafusion::{
    common::tree_node::TreeNode,
    physical_plan::{
        display::DisplayableExecutionPlan, displayable, ExecutionPlan, ExecutionPlanProperties,
    },
};
use datafusion_proto::physical_plan::{AsExecutionPlan, DefaultPhysicalExtensionCodec};
use prost::Message;
use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyBytes};

use crate::{
    context::PySessionContext,
    errors::{PyDataFusionError, PyDataFusionResult},
};

/// Type alias for the node ID used in execution plan graphs
type NodeId = usize;

/// Type alias for the parent-child relationship map in execution plan graphs
type ChildrenMap = HashMap<NodeId, Vec<NodeId>>;

/// Validates that an execution plan graph has no circular references using depth-first search.
///
/// This function performs cycle detection on a directed graph represented by a parent-child
/// relationship map. It uses the "white-gray-black" DFS algorithm where:
/// - White nodes: not yet visited (not in any set)
/// - Gray nodes: currently being processed (in `currently_visiting`)
/// - Black nodes: completely processed (in `completely_visited`)
///
/// A cycle is detected when we encounter a gray node during traversal.
///
/// # Arguments
/// * `children_map` - Map from parent node ID to vector of child node IDs
///
/// # Returns
/// * `Ok(())` if no cycles are detected
/// * `Err(anyhow::Error)` if a cycle is found, with details about the problematic node
fn validate_execution_plan_graph(children_map: &ChildrenMap) -> AnyhowResult<()> {
    let mut currently_visiting = std::collections::HashSet::new();
    let mut completely_visited = std::collections::HashSet::new();

    // Check each node as a potential starting point to handle disconnected components
    for &node_id in children_map.keys() {
        if !completely_visited.contains(&node_id) {
            detect_cycle_from_node(
                node_id,
                children_map,
                &mut currently_visiting,
                &mut completely_visited,
            )?;
        }
    }

    Ok(())
}

/// Recursively detects cycles starting from a specific node using DFS.
///
/// # Arguments
/// * `node_id` - The current node being processed
/// * `children_map` - Map from parent node ID to vector of child node IDs
/// * `currently_visiting` - Set of nodes currently in the DFS path (gray nodes)
/// * `completely_visited` - Set of nodes that have been completely processed (black nodes)
///
/// # Returns
/// * `Ok(())` if no cycle is detected from this node
/// * `Err(anyhow::Error)` if a cycle is found involving this node
fn detect_cycle_from_node(
    node_id: NodeId,
    children_map: &ChildrenMap,
    currently_visiting: &mut std::collections::HashSet<NodeId>,
    completely_visited: &mut std::collections::HashSet<NodeId>,
) -> AnyhowResult<()> {
    // If we encounter a node we're currently visiting, we found a cycle
    if currently_visiting.contains(&node_id) {
        return Err(anyhow::anyhow!(
            "Circular reference detected: node {} is part of a cycle in the execution plan graph",
            node_id
        ));
    }

    // If we already completely processed this node, skip it
    if completely_visited.contains(&node_id) {
        return Ok(());
    }

    // Mark this node as currently being visited (gray)
    currently_visiting.insert(node_id);

    // Recursively visit all children
    if let Some(child_ids) = children_map.get(&node_id) {
        for &child_id in child_ids {
            detect_cycle_from_node(
                child_id,
                children_map,
                currently_visiting,
                completely_visited,
            )?;
        }
    }

    // Mark this node as completely processed (black) and remove from currently visiting
    currently_visiting.remove(&node_id);
    completely_visited.insert(node_id);

    Ok(())
}

/// Helper function to compact parquet scan column statistics into a more succinct format
fn compact_parquet_scan_stats(parquet_scan: &mut serde_json::Value) {
    // Remove parquetOptions to save tokens
    if let Some(parquet_scan_obj) = parquet_scan.as_object_mut() {
        parquet_scan_obj.remove("parquetOptions");
    }

    if let Some(base_conf) = parquet_scan.get_mut("baseConf") {
        // Handle schema compaction at baseConf -> schema level
        if let Some(schema) = base_conf.get_mut("schema") {
            compact_schema(schema);
        }

        // Handle column stats at baseConf -> statistics -> columnStats level
        if let Some(statistics) = base_conf.get_mut("statistics") {
            compact_statistics(statistics);
        }

        // Handle column stats at file level: baseConf -> fileGroups -> files -> statistics -> columnStats
        // More aggressive approach: remove entire file groups with duplicate file paths
        if let Some(file_groups) = base_conf.get_mut("fileGroups") {
            if let Some(file_groups_array) = file_groups.as_array_mut() {
                let mut seen_file_paths = std::collections::HashSet::new();
                let mut kept_file_groups = Vec::new();

                for file_group in file_groups_array.clone() {
                    if let Some(files) = file_group.get("files") {
                        if let Some(files_array) = files.as_array() {
                            let mut should_keep_group = true;

                            // Check if any file in this group has a path we've seen before
                            for file in files_array {
                                let file_path = file.get("path").and_then(|p| p.as_str());
                                if let Some(path) = file_path {
                                    if seen_file_paths.contains(path) {
                                        should_keep_group = false;
                                        break;
                                    }
                                }
                            }

                            if should_keep_group {
                                // This is a new file group, process it and track the file paths
                                let mut processed_file_group = file_group.clone();

                                if let Some(files) = processed_file_group.get_mut("files") {
                                    if let Some(files_array) = files.as_array_mut() {
                                        for file in files_array {
                                            let file_path =
                                                file.get("path").and_then(|p| p.as_str());

                                            if let Some(path) = file_path {
                                                seen_file_paths.insert(path.to_string());

                                                // Remove the range field to make it semantically correct
                                                if let Some(file_obj) = file.as_object_mut() {
                                                    file_obj.remove("range");
                                                }
                                            }

                                            // Compact statistics for first occurrence
                                            if let Some(statistics) = file.get_mut("statistics") {
                                                compact_statistics(statistics);
                                            }
                                        }
                                    }
                                }

                                kept_file_groups.push(processed_file_group);
                            }
                            // If should_keep_group is false, we simply don't add this group to kept_file_groups
                        }
                    }
                }

                // Replace the original file groups array with the filtered one
                *file_groups_array = kept_file_groups;
            }
        }
    }
}

/// Helper function to compact schema into a more succinct format
fn compact_schema(schema: &mut serde_json::Value) {
    if let Some(columns) = schema.get_mut("columns") {
        if let Some(columns_array) = columns.as_array() {
            let mut compacted_columns = Vec::new();

            for column in columns_array {
                if let (Some(name), Some(nullable)) = (column.get("name"), column.get("nullable")) {
                    let arrow_type = extract_arrow_type(column.get("arrowType"));

                    compacted_columns.push(serde_json::json!([name, arrow_type, nullable]));
                }
            }

            // Replace columns with compacted format
            *columns = serde_json::json!(compacted_columns);
        }
    }
}

/// Extract a concise representation of the arrow type
fn extract_arrow_type(arrow_type: Option<&serde_json::Value>) -> String {
    if let Some(arrow_type_obj) = arrow_type {
        if let Some(obj) = arrow_type_obj.as_object() {
            if let Some((type_name, type_config)) = obj.iter().next() {
                match type_name.as_str() {
                    "DECIMAL" => {
                        if let Some(precision) = type_config.get("precision") {
                            if let Some(scale) = type_config.get("scale") {
                                return format!("DECIMAL({precision},{scale})");
                            }
                        }
                        return "DECIMAL".to_string();
                    }
                    "LIST" | "LARGE_LIST" | "FIXED_SIZE_LIST" => {
                        // For complex types, might need more detailed handling
                        return type_name.to_string();
                    }
                    _ => {
                        return type_name.to_string();
                    }
                }
            }
        }
    }
    "UNKNOWN".to_string()
}

/// Helper function to compact statistics including column stats and row counts
fn compact_statistics(statistics: &mut serde_json::Value) {
    // Remove column stats entirely
    if let Some(statistics_obj) = statistics.as_object_mut() {
        statistics_obj.remove("columnStats");

        // Extract and compact numRows
        if let Some(num_rows) = statistics_obj.remove("numRows") {
            if let Some(val) =
                extract_typed_value(num_rows.get("val").unwrap_or(&serde_json::Value::Null))
                    .as_u64()
            {
                statistics_obj.insert("n_rows".to_string(), serde_json::json!(val));
            }
        }

        // Extract and compact totalByteSize
        if let Some(total_byte_size) = statistics_obj.remove("totalByteSize") {
            if let Some(val) = extract_typed_value(
                total_byte_size
                    .get("val")
                    .unwrap_or(&serde_json::Value::Null),
            )
            .as_u64()
            {
                statistics_obj.insert("total_bytes".to_string(), serde_json::json!(val));
            }
        }
    }
}

/// Extract the actual value from typed value wrappers (int64Value, utf8ViewValue, etc.)
/// This function also handles cases where numeric values are stored as strings
fn extract_typed_value(val: &serde_json::Value) -> serde_json::Value {
    if let Some(obj) = val.as_object() {
        // Check for common typed value fields and extract their content
        // For numeric types, also handle string representations
        if let Some(int64_val) = obj.get("int64Value") {
            return parse_numeric_value(int64_val);
        }
        if let Some(uint64_val) = obj.get("uint64Value") {
            return parse_numeric_value(uint64_val);
        }
        if let Some(int32_val) = obj.get("int32Value") {
            return parse_numeric_value(int32_val);
        }
        if let Some(uint32_val) = obj.get("uint32Value") {
            return parse_numeric_value(uint32_val);
        }
        if let Some(float32_val) = obj.get("float32Value") {
            return parse_numeric_value(float32_val);
        }
        if let Some(float64_val) = obj.get("float64Value") {
            return parse_numeric_value(float64_val);
        }
        if let Some(utf8_val) = obj.get("utf8Value") {
            return utf8_val.clone();
        }
        if let Some(utf8_view_val) = obj.get("utf8ViewValue") {
            return utf8_view_val.clone();
        }
        if let Some(binary_val) = obj.get("binaryValue") {
            return binary_val.clone();
        }
        if let Some(bool_val) = obj.get("boolValue") {
            return bool_val.clone();
        }
        if let Some(date32_val) = obj.get("date32Value") {
            return parse_numeric_value(date32_val);
        }
        if let Some(date64_val) = obj.get("date64Value") {
            return parse_numeric_value(date64_val);
        }
        if let Some(timestamp_val) = obj.get("timestampValue") {
            return parse_numeric_value(timestamp_val);
        }
        if let Some(decimal128_val) = obj.get("decimal128Value") {
            return decimal128_val.clone();
        }
        if let Some(decimal256_val) = obj.get("decimal256Value") {
            return decimal256_val.clone();
        }
        if let Some(time32_val) = obj.get("time32Value") {
            return parse_numeric_value(time32_val);
        }
        if let Some(time64_val) = obj.get("time64Value") {
            return parse_numeric_value(time64_val);
        }
        if let Some(duration_val) = obj.get("durationValue") {
            return parse_numeric_value(duration_val);
        }
        if let Some(interval_val) = obj.get("intervalValue") {
            return interval_val.clone();
        }
    }

    // If no known typed wrapper found, return the original value
    val.clone()
}

/// Helper function to parse numeric values that might be stored as strings
fn parse_numeric_value(val: &serde_json::Value) -> serde_json::Value {
    match val {
        // If it's already a number, return as-is
        serde_json::Value::Number(_) => val.clone(),
        // If it's a string, try to parse it as a number
        serde_json::Value::String(s) => {
            // Try to parse as u64 first (most common case)
            if let Ok(num) = s.parse::<u64>() {
                return serde_json::json!(num);
            }
            // Try to parse as i64
            if let Ok(num) = s.parse::<i64>() {
                return serde_json::json!(num);
            }
            // Try to parse as f64
            if let Ok(num) = s.parse::<f64>() {
                return serde_json::json!(num);
            }
            // If parsing fails, return the original string
            val.clone()
        }
        // For any other type, return as-is
        _ => val.clone(),
    }
}

/// Information about a node in the execution plan
#[derive(Debug)]
struct NodeInfo {
    id: usize,
    raw_node: serde_json::Value,
    compacted_node: serde_json::Value,
    child_ids: Vec<usize>,
    is_parquet_scan: bool,
}

/// Convert an execution plan to succinct JSON format
fn execution_plan_to_succinct_json(plan: Arc<dyn ExecutionPlan>) -> AnyhowResult<String> {
    let extension_codec = DefaultPhysicalExtensionCodec {};
    let mut node_infos = Vec::new();
    let mut id_counter = 0;
    let mut visited_plans = Vec::new();

    // First pass: collect all nodes and their relationships
    plan.apply(|plan| {
        let id = id_counter;
        id_counter += 1;
        visited_plans.push((id, Arc::clone(plan)));

        // Serialize the node
        let protobuf = datafusion_proto::protobuf::PhysicalPlanNode::try_from_physical_plan(
            Arc::clone(plan),
            &extension_codec,
        )?;

        let serialized = serde_json::to_value(protobuf).map_err(|e| {
            datafusion::common::DataFusionError::External(
                format!("Failed to serialize protobuf to JSON: {e}").into(),
            )
        })?;

        // Check if this is a parquet scan
        let is_parquet_scan = serialized
            .as_object()
            .map(|obj| obj.contains_key("parquetScan"))
            .unwrap_or(false);

        // Create raw node with empty child placeholders
        let raw_node = create_node_with_empty_children(&serialized);

        // Create compacted node for parquet scans
        let mut compacted_node = raw_node.clone();
        if is_parquet_scan {
            if let Some(parquet_scan_data) = compacted_node.get_mut("parquetScan") {
                compact_parquet_scan_stats(parquet_scan_data);
            }
        }

        node_infos.push(NodeInfo {
            id,
            raw_node,
            compacted_node,
            child_ids: Vec::new(), // Will be filled in second pass
            is_parquet_scan,
        });

        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
    })
    .context("Error during plan traversal")?;

    // Second pass: build child relationships
    for (node_info, (_, plan)) in node_infos.iter_mut().zip(&visited_plans) {
        node_info.child_ids = plan
            .children()
            .iter()
            .map(|child| {
                visited_plans
                    .iter()
                    .find(|(_, visited_plan)| Arc::ptr_eq(child, visited_plan))
                    .map(|(child_id, _)| *child_id)
                    .unwrap() // Safe: all nodes were visited
            })
            .collect();
    }

    // Third pass: generate final output structures
    let mut structure = serde_json::Map::new();
    let mut succinct_table_info = Vec::new();
    let mut full_table_info = Vec::new();

    for node_info in &node_infos {
        if node_info.is_parquet_scan {
            // Add to table collections
            let node_json = serde_json::json!({
                "id": node_info.id,
                "node": node_info.raw_node
            });
            full_table_info.push(node_json);

            let compacted_json = serde_json::json!({
                "id": node_info.id,
                "node": node_info.compacted_node
            });
            succinct_table_info.push(compacted_json);
        } else {
            // Add to structure with child IDs populated as key-value pairs
            let mut modified_node = node_info.raw_node.clone();
            populate_child_ids(&mut modified_node, &node_info.child_ids);

            // Use node ID as key and node data as value directly
            structure.insert(node_info.id.to_string(), modified_node);
        }
    }

    let result = serde_json::json!({
        "structure": serde_json::Value::Object(structure),
        "succinct_table_info": succinct_table_info,
        "full_table_info": full_table_info,
        "v": "3"
    });

    serde_json::to_string(&result).context("Failed to serialize result")
}

/// Create a node with empty child placeholders
fn create_node_with_empty_children(serialized: &serde_json::Value) -> serde_json::Value {
    let mut node = serialized.clone();

    if let Some(obj) = node.as_object_mut() {
        if let Some((_, plan_node)) = obj.iter_mut().next() {
            if let Some(plan_obj) = plan_node.as_object_mut() {
                // Clear child references - they'll be populated later for structure
                if plan_obj.contains_key("input") {
                    plan_obj.insert("input".to_string(), serde_json::json!({}));
                }
                if plan_obj.contains_key("left") {
                    plan_obj.insert("left".to_string(), serde_json::json!({}));
                }
                if plan_obj.contains_key("right") {
                    plan_obj.insert("right".to_string(), serde_json::json!({}));
                }
                if plan_obj.contains_key("inputs") {
                    plan_obj.insert("inputs".to_string(), serde_json::json!([]));
                }
            }
        }
    }

    node
}

/// Populate child IDs in a node's input/left/right/inputs fields
fn populate_child_ids(node: &mut serde_json::Value, child_ids: &[usize]) {
    if let Some(obj) = node.as_object_mut() {
        if let Some((_, plan_node)) = obj.iter_mut().next() {
            if let Some(plan_obj) = plan_node.as_object_mut() {
                match child_ids.len() {
                    0 => {} // No children - leave empty
                    1 => {
                        if plan_obj.contains_key("input") {
                            plan_obj.insert("input".to_string(), serde_json::json!(child_ids[0]));
                        }
                    }
                    2 => {
                        if plan_obj.contains_key("left") {
                            plan_obj.insert("left".to_string(), serde_json::json!(child_ids[0]));
                        }
                        if plan_obj.contains_key("right") {
                            plan_obj.insert("right".to_string(), serde_json::json!(child_ids[1]));
                        }
                    }
                    _ => {
                        if plan_obj.contains_key("inputs") {
                            plan_obj.insert("inputs".to_string(), serde_json::json!(child_ids));
                        }
                    }
                }
            }
        }
    }
}

/// Parsed node data from JSON
#[derive(Debug)]
struct ParsedNode {
    id: usize,
    node_data: serde_json::Value,
    child_ids: Vec<usize>,
    is_parquet_scan: bool,
}

/// Convert succinct JSON format back to an execution plan
fn execution_plan_from_succinct_json(
    ctx: &PySessionContext,
    json: String,
) -> AnyhowResult<Arc<dyn ExecutionPlan>> {
    let summary: serde_json::Value = serde_json::from_str(&json).context("Invalid JSON")?;

    let structure = summary["structure"]
        .as_object()
        .context("Missing 'structure' object in JSON")?;
    let full_table_info = summary["full_table_info"]
        .as_array()
        .context("Missing 'full_table_info' array in JSON")?;

    // Parse all nodes into a unified structure
    let mut all_nodes = Vec::new();

    // Parse regular nodes from structure
    for (id_str, node_data) in structure {
        let id = id_str
            .parse::<usize>()
            .context("Invalid node ID in structure")?;
        let child_ids = extract_child_ids(node_data);

        all_nodes.push(ParsedNode {
            id,
            node_data: node_data.clone(),
            child_ids,
            is_parquet_scan: false,
        });
    }

    // Parse parquet nodes from full_table_info
    for node in full_table_info {
        let id = node["id"]
            .as_u64()
            .context("Missing 'id' in parquet node")? as usize;
        let node_data = node["node"].clone();

        all_nodes.push(ParsedNode {
            id,
            node_data,
            child_ids: Vec::new(), // Parquet scans are leaf nodes
            is_parquet_scan: true,
        });
    }

    // Build lookup maps
    let node_map: HashMap<usize, &ParsedNode> =
        all_nodes.iter().map(|node| (node.id, node)).collect();
    let children_map: HashMap<usize, Vec<usize>> = all_nodes
        .iter()
        .map(|node| (node.id, node.child_ids.clone()))
        .collect();

    // Validate graph structure
    validate_execution_plan_graph(&children_map)?;

    // Find root node
    let all_children: std::collections::HashSet<usize> =
        children_map.values().flatten().copied().collect();
    let root_id = all_nodes
        .iter()
        .find(|node| !all_children.contains(&node.id))
        .context("Could not find root node")?
        .id;

    // Build execution plan tree
    build_execution_plan_tree(root_id, &node_map, ctx)
}

/// Extract child IDs from a node's input/left/right/inputs fields
fn extract_child_ids(node_data: &serde_json::Value) -> Vec<usize> {
    let mut child_ids = Vec::new();

    if let Some(obj) = node_data.as_object() {
        if let Some((_, plan_node)) = obj.iter().next() {
            if let Some(plan_obj) = plan_node.as_object() {
                // Single input
                if let Some(input_id) = plan_obj.get("input").and_then(|v| v.as_u64()) {
                    child_ids.push(input_id as usize);
                }
                // Binary inputs
                if let Some(left_id) = plan_obj.get("left").and_then(|v| v.as_u64()) {
                    child_ids.push(left_id as usize);
                }
                if let Some(right_id) = plan_obj.get("right").and_then(|v| v.as_u64()) {
                    child_ids.push(right_id as usize);
                }
                // Multiple inputs
                if let Some(inputs_array) = plan_obj.get("inputs").and_then(|v| v.as_array()) {
                    for input_val in inputs_array {
                        if let Some(input_id) = input_val.as_u64() {
                            child_ids.push(input_id as usize);
                        }
                    }
                }
            }
        }
    }

    child_ids
}

/// Helper function to recursively build execution plan tree from parsed nodes
fn build_execution_plan_tree(
    id: usize,
    node_map: &HashMap<usize, &ParsedNode>,
    ctx: &PySessionContext,
) -> AnyhowResult<Arc<dyn ExecutionPlan>> {
    let node = node_map
        .get(&id)
        .with_context(|| format!("Node with id {id} not found"))?;

    // For parquet nodes, deserialize directly (they have no children)
    if node.is_parquet_scan {
        let node_json =
            serde_json::to_string(&node.node_data).context("Failed to serialize parquet node")?;
        let plan = datafusion_proto::bytes::physical_plan_from_json(&node_json, &ctx.ctx)
            .context("Failed to deserialize parquet plan from JSON")?;
        return Ok(plan);
    }

    // For regular nodes, recursively build children first
    let mut node_data = node.node_data.clone();

    if !node.child_ids.is_empty() {
        let mut children = Vec::new();
        for &child_id in &node.child_ids {
            children.push(build_execution_plan_tree(child_id, node_map, ctx)?);
        }

        // Restore the children in the node data by converting them to protobuf format
        if let Some(obj) = node_data.as_object_mut() {
            if let Some((_, plan_node)) = obj.iter_mut().next() {
                if let Some(plan_obj) = plan_node.as_object_mut() {
                    match children.len() {
                        1 => {
                            // Single child - restore "input"
                            let child_protobuf = datafusion_proto::protobuf::PhysicalPlanNode::try_from_physical_plan(
                                children[0].clone(),
                                &DefaultPhysicalExtensionCodec {},
                            ).context("Failed to convert child to protobuf")?;
                            let child_json = serde_json::to_value(child_protobuf)
                                .context("Failed to serialize child to JSON")?;
                            plan_obj.insert("input".to_string(), child_json);
                        }
                        2 => {
                            // Binary children - restore "left" and "right"
                            let left_protobuf = datafusion_proto::protobuf::PhysicalPlanNode::try_from_physical_plan(
                                children[0].clone(),
                                &DefaultPhysicalExtensionCodec {},
                            ).context("Failed to convert left child to protobuf")?;
                            let left_json = serde_json::to_value(left_protobuf)
                                .context("Failed to serialize left child to JSON")?;
                            plan_obj.insert("left".to_string(), left_json);

                            let right_protobuf = datafusion_proto::protobuf::PhysicalPlanNode::try_from_physical_plan(
                                children[1].clone(),
                                &DefaultPhysicalExtensionCodec {},
                            ).context("Failed to convert right child to protobuf")?;
                            let right_json = serde_json::to_value(right_protobuf)
                                .context("Failed to serialize right child to JSON")?;
                            plan_obj.insert("right".to_string(), right_json);
                        }
                        _ => {
                            // Multiple children - restore "inputs" array
                            if plan_obj.contains_key("inputs") {
                                let inputs_json: anyhow::Result<Vec<serde_json::Value>> = children
                                    .into_iter()
                                    .map(|child_plan| {
                                        let child_proto = datafusion_proto::protobuf::PhysicalPlanNode::try_from_physical_plan(
                                            child_plan,
                                            &DefaultPhysicalExtensionCodec {},
                                        )
                                        .context("Failed to convert child to protobuf")?;
                                        let child_json = serde_json::to_value(child_proto)
                                            .context("Failed to serialize child to JSON")?;
                                        Ok(child_json)
                                    })
                                    .collect();
                                plan_obj.insert(
                                    "inputs".to_string(),
                                    serde_json::Value::Array(inputs_json?),
                                );
                            } else {
                                return Err(anyhow::anyhow!(
                                    "Unsupported number of children: {}. Expected 1, 2, or an 'inputs' array field present.",
                                    children.len()
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    // Deserialize the complete node with children
    let node_json = serde_json::to_string(&node_data).context("Failed to serialize node")?;
    let plan = datafusion_proto::bytes::physical_plan_from_json(&node_json, &ctx.ctx)
        .map_err(|e| anyhow::anyhow!("Failed to deserialize plan from JSON: {e}"))?;
    Ok(plan)
}

#[pyclass(name = "ExecutionPlan", module = "datafusion", subclass)]
#[derive(Debug, Clone)]
pub struct PyExecutionPlan {
    pub plan: Arc<dyn ExecutionPlan>,
}

impl PyExecutionPlan {
    /// creates a new PyPhysicalPlan
    pub fn new(plan: Arc<dyn ExecutionPlan>) -> Self {
        Self { plan }
    }
}

#[pymethods]
impl PyExecutionPlan {
    /// Get the inputs to this plan
    pub fn children(&self) -> Vec<PyExecutionPlan> {
        self.plan
            .children()
            .iter()
            .map(|&p| p.to_owned().into())
            .collect()
    }

    pub fn display(&self) -> String {
        let d = displayable(self.plan.as_ref());
        format!("{}", d.one_line())
    }

    pub fn display_indent(&self) -> String {
        let d = displayable(self.plan.as_ref());
        format!("{}", d.indent(false))
    }

    pub fn display_with_metrics(&self) -> String {
        let d = DisplayableExecutionPlan::with_metrics(self.plan.as_ref());
        format!("{}", d.indent(true))
    }

    pub fn to_proto<'py>(&'py self, py: Python<'py>) -> PyDataFusionResult<Bound<'py, PyBytes>> {
        let codec = DefaultPhysicalExtensionCodec {};
        let proto = datafusion_proto::protobuf::PhysicalPlanNode::try_from_physical_plan(
            self.plan.clone(),
            &codec,
        )?;

        let bytes = proto.encode_to_vec();
        Ok(PyBytes::new(py, &bytes))
    }

    #[staticmethod]
    pub fn from_proto(
        ctx: PySessionContext,
        proto_msg: Bound<'_, PyBytes>,
    ) -> PyDataFusionResult<Self> {
        let bytes: &[u8] = proto_msg.extract()?;
        let proto_plan =
            datafusion_proto::protobuf::PhysicalPlanNode::decode(bytes).map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "Unable to decode logical node from serialized bytes: {e}"
                ))
            })?;

        let codec = DefaultPhysicalExtensionCodec {};
        let plan = proto_plan.try_into_physical_plan(&ctx.ctx, &ctx.ctx.runtime_env(), &codec)?;
        Ok(Self::new(plan))
    }

    pub fn to_json(&self) -> PyDataFusionResult<String> {
        let json = datafusion_proto::bytes::physical_plan_to_json(self.plan.clone())?;
        Ok(json)
    }

    pub fn to_succinct_json(&self) -> PyDataFusionResult<String> {
        execution_plan_to_succinct_json(self.plan.clone()).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to convert to succinct JSON: {e}")).into()
        })
    }

    #[staticmethod]
    pub fn from_succinct_json(ctx: PySessionContext, json: String) -> PyDataFusionResult<Self> {
        // Catch any unwinding panic to avoid aborting the Python interpreter
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            execution_plan_from_succinct_json(&ctx, json)
        }));
        match result {
            Ok(plan_res) => {
                let plan = plan_res.map_err::<PyDataFusionError, _>(|e| {
                    PyRuntimeError::new_err(format!("Failed to convert from succinct JSON: {e}"))
                        .into()
                })?;
                Ok(Self::new(plan))
            }
            Err(panic_payload) => {
                // Attempt to extract panic information
                let panic_msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic".to_string()
                };
                Err(PyRuntimeError::new_err(format!(
                    "Rust panic while converting from succinct JSON: {panic_msg}"
                ))
                .into())
            }
        }
    }

    #[staticmethod]
    pub fn from_json(ctx: PySessionContext, json: String) -> PyDataFusionResult<Self> {
        let plan = datafusion_proto::bytes::physical_plan_from_json(&json, &ctx.ctx)?;
        Ok(Self::new(plan))
    }

    fn __repr__(&self) -> String {
        self.display_indent()
    }

    #[getter]
    pub fn partition_count(&self) -> usize {
        self.plan.output_partitioning().partition_count()
    }
}

impl From<PyExecutionPlan> for Arc<dyn ExecutionPlan> {
    fn from(plan: PyExecutionPlan) -> Arc<dyn ExecutionPlan> {
        plan.plan.clone()
    }
}

impl From<Arc<dyn ExecutionPlan>> for PyExecutionPlan {
    fn from(plan: Arc<dyn ExecutionPlan>) -> PyExecutionPlan {
        PyExecutionPlan { plan: plan.clone() }
    }
}
