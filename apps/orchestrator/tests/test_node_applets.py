"""
Unit tests for node applet on_message() methods (T-072).

Covers: TransformNodeApplet, IfElseNodeApplet, MergeNodeApplet,
ForEachNodeApplet, HTTPRequestNodeApplet, CodeNodeApplet,
MemoryNodeApplet, SQLiteFTSMemoryStoreBackend, MemoryStoreFactory.
"""
import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from apps.orchestrator.main import (
    AppletMessage,
    TransformNodeApplet,
    IfElseNodeApplet,
    MergeNodeApplet,
    ForEachNodeApplet,
    HTTPRequestNodeApplet,
    CodeNodeApplet,
    MemoryNodeApplet,
    SQLiteFTSMemoryStoreBackend,
    MemoryStoreFactory,
    TRANSFORM_NODE_TYPE,
    IF_ELSE_NODE_TYPE,
    MERGE_NODE_TYPE,
    FOR_EACH_NODE_TYPE,
    HTTP_REQUEST_NODE_TYPE,
    CODE_NODE_TYPE,
    MEMORY_NODE_TYPE,
    _as_text,
    _as_serialized_text,
    _normalize_memory_tags,
    _fts_terms,
    _parse_json_or_default,
)


def _msg(content=None, context=None, metadata=None):
    return AppletMessage(
        content=content if content is not None else {},
        context=context if context is not None else {},
        metadata=metadata if metadata is not None else {},
    )


# ============================================================
# TransformNodeApplet Tests
# ============================================================

class TestTransformNodeApplet:
    @pytest.fixture
    def applet(self):
        return TransformNodeApplet()

    @pytest.mark.asyncio
    async def test_template_operation_default(self, applet):
        msg = _msg(
            content="hello world",
            metadata={"node_data": {"operation": "template", "template": "result: {{source}}"}},
        )
        result = await applet.on_message(msg)
        assert result.content["ok"] is True
        assert result.content["operation"] == "template"
        assert result.content["output"] == "result: hello world"
        assert result.metadata["status"] == "success"

    @pytest.mark.asyncio
    async def test_json_path_operation(self, applet):
        msg = _msg(
            content={"users": [{"name": "Alice"}]},
            metadata={"node_data": {"operation": "json_path", "source": "{{content}}", "json_path": "$.users[0].name"}},
        )
        result = await applet.on_message(msg)
        assert result.content["ok"] is True
        assert result.content["output"] == "Alice"

    @pytest.mark.asyncio
    async def test_json_path_not_found(self, applet):
        msg = _msg(
            content={"foo": 1},
            metadata={"node_data": {"operation": "json_path", "json_path": "$.nonexistent"}},
        )
        result = await applet.on_message(msg)
        assert result.content["ok"] is False
        assert "not found" in result.content["error"]

    @pytest.mark.asyncio
    async def test_regex_replace_operation(self, applet):
        msg = _msg(
            content="Hello World 123",
            metadata={"node_data": {
                "operation": "regex_replace",
                "source": "{{content}}",
                "regex_pattern": r"\d+",
                "regex_replacement": "NUM",
            }},
        )
        result = await applet.on_message(msg)
        assert result.content["ok"] is True
        assert result.content["output"] == "Hello World NUM"

    @pytest.mark.asyncio
    async def test_regex_replace_with_flags(self, applet):
        msg = _msg(
            content="Hello hello HELLO",
            metadata={"node_data": {
                "operation": "regex_replace",
                "source": "{{content}}",
                "regex_pattern": "hello",
                "regex_replacement": "HI",
                "regex_flags": "i",
                "regex_count": 1,
            }},
        )
        result = await applet.on_message(msg)
        assert result.content["ok"] is True
        assert result.content["output"].startswith("HI")

    @pytest.mark.asyncio
    async def test_regex_replace_empty_pattern_error(self, applet):
        msg = _msg(
            content="test",
            metadata={"node_data": {
                "operation": "regex_replace",
                "source": "{{content}}",
                "regex_pattern": "",
            }},
        )
        result = await applet.on_message(msg)
        assert result.content["ok"] is False
        assert "regex_pattern is required" in result.content["error"]

    @pytest.mark.asyncio
    async def test_split_join_basic(self, applet):
        msg = _msg(
            content="a,b,c",
            metadata={"node_data": {
                "operation": "split_join",
                "source": "{{content}}",
                "split_delimiter": ",",
                "join_delimiter": " | ",
            }},
        )
        result = await applet.on_message(msg)
        assert result.content["ok"] is True
        assert result.content["output"] == "a | b | c"

    @pytest.mark.asyncio
    async def test_split_join_return_list(self, applet):
        msg = _msg(
            content="x,y,z",
            metadata={"node_data": {
                "operation": "split_join",
                "source": "{{content}}",
                "split_delimiter": ",",
                "return_list": True,
            }},
        )
        result = await applet.on_message(msg)
        assert result.content["ok"] is True
        assert result.content["output"] == ["x", "y", "z"]

    @pytest.mark.asyncio
    async def test_split_join_strip_and_drop_empty(self, applet):
        msg = _msg(
            content=" a , , b , ",
            metadata={"node_data": {
                "operation": "split_join",
                "source": "{{content}}",
                "split_delimiter": ",",
                "strip_items": True,
                "drop_empty": True,
                "return_list": True,
            }},
        )
        result = await applet.on_message(msg)
        assert result.content["output"] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_split_join_with_index(self, applet):
        msg = _msg(
            content="first,second,third",
            metadata={"node_data": {
                "operation": "split_join",
                "source": "{{content}}",
                "split_delimiter": ",",
                "split_index": 1,
            }},
        )
        result = await applet.on_message(msg)
        assert result.content["output"] == "second"

    @pytest.mark.asyncio
    async def test_split_join_index_out_of_range(self, applet):
        msg = _msg(
            content="a,b",
            metadata={"node_data": {
                "operation": "split_join",
                "source": "{{content}}",
                "split_delimiter": ",",
                "split_index": 99,
            }},
        )
        result = await applet.on_message(msg)
        assert result.content["ok"] is False
        assert "out of range" in result.content["error"]

    @pytest.mark.asyncio
    async def test_split_join_empty_delimiter_splits_chars(self, applet):
        msg = _msg(
            content="abc",
            metadata={"node_data": {
                "operation": "split_join",
                "source": "{{content}}",
                "split_delimiter": "",
                "return_list": True,
            }},
        )
        result = await applet.on_message(msg)
        assert result.content["output"] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_split_join_list_input(self, applet):
        msg = _msg(
            content=["first", "second"],
            metadata={"node_data": {
                "operation": "split_join",
                "source": "{{content}}",
                "join_delimiter": "-",
            }},
        )
        result = await applet.on_message(msg)
        assert result.content["ok"] is True
        assert result.content["output"] == "first-second"

    @pytest.mark.asyncio
    async def test_unsupported_operation(self, applet):
        """'bogus' is rejected by validation, so it triggers config error path."""
        msg = _msg(
            content="test",
            metadata={"node_data": {"operation": "bogus"}},
        )
        result = await applet.on_message(msg)
        assert "error" in result.content
        assert "Invalid transform configuration" in result.content["error"]

    @pytest.mark.asyncio
    async def test_invalid_config(self, applet):
        msg = _msg(
            content="test",
            metadata={"node_data": {"operation": 12345}},
        )
        result = await applet.on_message(msg)
        assert "error" in result.content
        assert "Invalid transform configuration" in result.content["error"]

    @pytest.mark.asyncio
    async def test_config_from_context_legacy(self, applet):
        msg = _msg(
            content="test",
            context={"transform": {"operation": "template", "template": "legacy: {{source}}"}},
        )
        result = await applet.on_message(msg)
        assert result.content["ok"] is True
        assert result.content["output"] == "legacy: test"

    @pytest.mark.asyncio
    async def test_context_gets_last_transform_response(self, applet):
        msg = _msg(
            content="data",
            metadata={"node_data": {"operation": "template", "template": "{{source}}"}},
        )
        result = await applet.on_message(msg)
        assert "last_transform_response" in result.context

    @pytest.mark.asyncio
    async def test_split_join_maxsplit(self, applet):
        msg = _msg(
            content="a,b,c,d",
            metadata={"node_data": {
                "operation": "split_join",
                "source": "{{content}}",
                "split_delimiter": ",",
                "split_maxsplit": 2,
                "return_list": True,
            }},
        )
        result = await applet.on_message(msg)
        assert result.content["output"] == ["a", "b", "c,d"]


# ============================================================
# IfElseNodeApplet Tests
# ============================================================

class TestIfElseNodeApplet:
    @pytest.fixture
    def applet(self):
        return IfElseNodeApplet()

    @pytest.mark.asyncio
    async def test_equals_true(self, applet):
        msg = _msg(
            content="hello",
            metadata={"node_data": {"operation": "equals", "source": "{{content}}", "value": "hello"}},
        )
        result = await applet.on_message(msg)
        assert result.content["ok"] is True
        assert result.content["result"] is True
        assert result.content["branch"] == "true"

    @pytest.mark.asyncio
    async def test_equals_false(self, applet):
        msg = _msg(
            content="hello",
            metadata={"node_data": {"operation": "equals", "source": "{{content}}", "value": "goodbye"}},
        )
        result = await applet.on_message(msg)
        assert result.content["result"] is False
        assert result.content["branch"] == "false"

    @pytest.mark.asyncio
    async def test_equals_case_insensitive(self, applet):
        msg = _msg(
            content="Hello",
            metadata={"node_data": {"operation": "equals", "source": "{{content}}", "value": "hello", "case_sensitive": False}},
        )
        result = await applet.on_message(msg)
        assert result.content["result"] is True

    @pytest.mark.asyncio
    async def test_contains_string(self, applet):
        msg = _msg(
            content="hello world",
            metadata={"node_data": {"operation": "contains", "source": "{{content}}", "value": "world"}},
        )
        result = await applet.on_message(msg)
        assert result.content["result"] is True

    @pytest.mark.asyncio
    async def test_contains_case_insensitive(self, applet):
        msg = _msg(
            content="Hello World",
            metadata={"node_data": {"operation": "contains", "source": "{{content}}", "value": "WORLD", "case_sensitive": False}},
        )
        result = await applet.on_message(msg)
        assert result.content["result"] is True

    @pytest.mark.asyncio
    async def test_contains_none_expected(self, applet):
        msg = _msg(
            content="hello",
            metadata={"node_data": {"operation": "contains", "source": "{{content}}"}},
        )
        result = await applet.on_message(msg)
        assert result.content["result"] is False

    @pytest.mark.asyncio
    async def test_contains_dict_key(self, applet):
        msg = _msg(
            content={"name": "Alice", "age": 30},
            metadata={"node_data": {"operation": "contains", "source": "{{content}}", "value": "name"}},
        )
        result = await applet.on_message(msg)
        assert result.content["result"] is True

    @pytest.mark.asyncio
    async def test_contains_dict_value(self, applet):
        msg = _msg(
            content={"name": "Alice"},
            metadata={"node_data": {"operation": "contains", "source": "{{content}}", "value": "Alice"}},
        )
        result = await applet.on_message(msg)
        assert result.content["result"] is True

    @pytest.mark.asyncio
    async def test_contains_list(self, applet):
        msg = _msg(
            content=[1, 2, 3],
            metadata={"node_data": {"operation": "contains", "source": "{{content}}", "value": 2}},
        )
        result = await applet.on_message(msg)
        assert result.content["result"] is True

    @pytest.mark.asyncio
    async def test_regex_match(self, applet):
        msg = _msg(
            content="Order #12345",
            metadata={"node_data": {"operation": "regex", "source": "{{content}}", "regex_pattern": r"#\d+"}},
        )
        result = await applet.on_message(msg)
        assert result.content["result"] is True

    @pytest.mark.asyncio
    async def test_regex_no_match(self, applet):
        msg = _msg(
            content="No numbers here",
            metadata={"node_data": {"operation": "regex", "source": "{{content}}", "regex_pattern": r"\d+"}},
        )
        result = await applet.on_message(msg)
        assert result.content["result"] is False

    @pytest.mark.asyncio
    async def test_regex_flags(self, applet):
        msg = _msg(
            content="Hello\nworld",
            metadata={"node_data": {"operation": "regex", "source": "{{content}}", "regex_pattern": "hello.*world", "regex_flags": "is"}},
        )
        result = await applet.on_message(msg)
        assert result.content["result"] is True

    @pytest.mark.asyncio
    async def test_regex_from_value_fallback(self, applet):
        msg = _msg(
            content="test123",
            metadata={"node_data": {"operation": "regex", "source": "{{content}}", "value": r"\d+"}},
        )
        result = await applet.on_message(msg)
        assert result.content["result"] is True

    @pytest.mark.asyncio
    async def test_regex_empty_pattern_error(self, applet):
        msg = _msg(
            content="test",
            metadata={"node_data": {"operation": "regex", "source": "{{content}}"}},
        )
        result = await applet.on_message(msg)
        assert result.content["ok"] is False
        assert "regex_pattern is required" in result.content["error"]

    @pytest.mark.asyncio
    async def test_json_path_exists(self, applet):
        msg = _msg(
            content={"data": {"status": "active"}},
            metadata={"node_data": {"operation": "json_path", "source": "{{content}}", "json_path": "$.data.status"}},
        )
        result = await applet.on_message(msg)
        assert result.content["result"] is True

    @pytest.mark.asyncio
    async def test_json_path_with_expected_value(self, applet):
        msg = _msg(
            content={"data": {"status": "active"}},
            metadata={"node_data": {
                "operation": "json_path",
                "source": "{{content}}",
                "json_path": "$.data.status",
                "value": "active",
            }},
        )
        result = await applet.on_message(msg)
        assert result.content["result"] is True

    @pytest.mark.asyncio
    async def test_json_path_not_found(self, applet):
        msg = _msg(
            content={"foo": 1},
            metadata={"node_data": {"operation": "json_path", "source": "{{content}}", "json_path": "$.missing"}},
        )
        result = await applet.on_message(msg)
        assert result.content["result"] is False

    @pytest.mark.asyncio
    async def test_negate(self, applet):
        msg = _msg(
            content="hello",
            metadata={"node_data": {"operation": "equals", "source": "{{content}}", "value": "hello", "negate": True}},
        )
        result = await applet.on_message(msg)
        assert result.content["result"] is False
        assert result.content["branch"] == "false"

    @pytest.mark.asyncio
    async def test_unsupported_operation(self, applet):
        """'bogus' is rejected by validation, so hits config error path."""
        msg = _msg(
            content="test",
            metadata={"node_data": {"operation": "bogus"}},
        )
        result = await applet.on_message(msg)
        assert result.content["ok"] is False
        assert "Invalid if/else configuration" in result.content["error"]

    @pytest.mark.asyncio
    async def test_invalid_config(self, applet):
        msg = _msg(metadata={"node_data": {"operation": 12345}})
        result = await applet.on_message(msg)
        assert result.content["ok"] is False
        assert "Invalid if/else configuration" in result.content["error"]

    @pytest.mark.asyncio
    async def test_context_gets_last_response(self, applet):
        msg = _msg(
            content="test",
            metadata={"node_data": {"operation": "equals", "source": "{{content}}", "value": "test"}},
        )
        result = await applet.on_message(msg)
        assert "last_if_else_response" in result.context

    @pytest.mark.asyncio
    async def test_config_from_context_keys(self, applet):
        msg = _msg(
            content="test",
            context={"if_else_config": {"operation": "equals", "source": "{{content}}", "value": "test"}},
        )
        result = await applet.on_message(msg)
        assert result.content["ok"] is True

    @pytest.mark.asyncio
    async def test_config_from_legacy_context_key(self, applet):
        msg = _msg(
            content="test",
            context={"condition_config": {"operation": "equals", "source": "{{content}}", "value": "test"}},
        )
        result = await applet.on_message(msg)
        assert result.content["ok"] is True


# ============================================================
# MergeNodeApplet Tests
# ============================================================

class TestMergeNodeApplet:
    @pytest.fixture
    def applet(self):
        return MergeNodeApplet()

    @pytest.mark.asyncio
    async def test_array_strategy_default(self, applet):
        msg = _msg(
            content={"inputs": ["a", "b", "c"]},
            metadata={"node_data": {}},
        )
        result = await applet.on_message(msg)
        assert result.content["ok"] is True
        assert result.content["strategy"] == "array"
        assert result.content["output"] == ["a", "b", "c"]
        assert result.content["count"] == 3

    @pytest.mark.asyncio
    async def test_concatenate_strategy(self, applet):
        msg = _msg(
            content={"inputs": ["hello", "world"]},
            metadata={"node_data": {"strategy": "concatenate", "delimiter": " "}},
        )
        result = await applet.on_message(msg)
        assert result.content["output"] == "hello world"

    @pytest.mark.asyncio
    async def test_first_wins_strategy(self, applet):
        msg = _msg(
            content={"inputs": ["first", "second", "third"]},
            metadata={"node_data": {"strategy": "first_wins"}},
        )
        result = await applet.on_message(msg)
        assert result.content["output"] == "first"

    @pytest.mark.asyncio
    async def test_first_wins_empty_list_content(self, applet):
        """With content=[], _normalize_inputs returns [], so first_wins returns None."""
        msg = _msg(content=[], metadata={"node_data": {"strategy": "first_wins"}})
        result = await applet.on_message(msg)
        assert result.content["output"] is None

    @pytest.mark.asyncio
    async def test_normalize_inputs_dict_with_input_key(self, applet):
        msg = _msg(content={"input": "single"}, metadata={"node_data": {}})
        result = await applet.on_message(msg)
        assert result.content["output"] == ["single"]

    @pytest.mark.asyncio
    async def test_normalize_inputs_dict_no_special_keys(self, applet):
        msg = _msg(content={"foo": "bar"}, metadata={"node_data": {}})
        result = await applet.on_message(msg)
        assert result.content["output"] == [{"foo": "bar"}]

    @pytest.mark.asyncio
    async def test_normalize_inputs_list_directly(self, applet):
        msg = _msg(content=[1, 2, 3], metadata={"node_data": {}})
        result = await applet.on_message(msg)
        assert result.content["output"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_normalize_inputs_scalar(self, applet):
        msg = _msg(content="scalar_value", metadata={"node_data": {}})
        result = await applet.on_message(msg)
        assert result.content["output"] == ["scalar_value"]

    @pytest.mark.asyncio
    async def test_normalize_inputs_empty_list(self, applet):
        msg = _msg(content=[], metadata={"node_data": {}})
        result = await applet.on_message(msg)
        assert result.content["output"] == []

    @pytest.mark.asyncio
    async def test_invalid_config(self, applet):
        msg = _msg(metadata={"node_data": {"strategy": 12345}})
        result = await applet.on_message(msg)
        assert result.content["ok"] is False
        assert "Invalid merge configuration" in result.content["error"]

    @pytest.mark.asyncio
    async def test_context_gets_last_merge_response(self, applet):
        msg = _msg(content=[1], metadata={"node_data": {}})
        result = await applet.on_message(msg)
        assert "last_merge_response" in result.context

    @pytest.mark.asyncio
    async def test_normalize_inputs_dict_with_non_list_inputs(self, applet):
        msg = _msg(content={"inputs": "not_a_list"}, metadata={"node_data": {}})
        result = await applet.on_message(msg)
        assert result.content["output"] == ["not_a_list"]


# ============================================================
# ForEachNodeApplet Tests
# ============================================================

class TestForEachNodeApplet:
    @pytest.fixture
    def applet(self):
        return ForEachNodeApplet()

    @pytest.mark.asyncio
    async def test_sequential_iteration(self, applet):
        msg = _msg(
            content=[10, 20, 30],
            metadata={"node_data": {"array_source": "{{input}}"}},
        )
        result = await applet.on_message(msg)
        assert result.content["ok"] is True
        assert result.content["iterated"] == 3
        assert len(result.content["results"]) == 3
        for i, r in enumerate(result.content["results"]):
            assert r["index"] == i
            assert r["output"] == [10, 20, 30][i]

    @pytest.mark.asyncio
    async def test_parallel_iteration(self, applet):
        msg = _msg(
            content=[1, 2, 3],
            metadata={"node_data": {"array_source": "{{input}}", "parallel": True}},
        )
        result = await applet.on_message(msg)
        assert result.content["ok"] is True
        assert result.content["parallel"] is True
        assert result.content["iterated"] == 3

    @pytest.mark.asyncio
    async def test_max_iterations_truncation(self, applet):
        msg = _msg(
            content=list(range(100)),
            metadata={"node_data": {"array_source": "{{input}}", "max_iterations": 5}},
        )
        result = await applet.on_message(msg)
        assert result.content["ok"] is True
        assert result.content["iterated"] == 5
        assert result.content["truncated"] is True
        assert result.content["total_items"] == 100

    @pytest.mark.asyncio
    async def test_non_iterable_source_error(self, applet):
        msg = _msg(
            content={"not_an_array": True},
            metadata={"node_data": {"array_source": "{{input}}"}},
        )
        result = await applet.on_message(msg)
        assert result.content["ok"] is False
        assert "did not resolve to an iterable" in result.content["error"]

    @pytest.mark.asyncio
    async def test_json_string_array(self, applet):
        msg = _msg(
            content='[1, 2, 3]',
            metadata={"node_data": {"array_source": "{{input}}"}},
        )
        result = await applet.on_message(msg)
        assert result.content["ok"] is True
        assert result.content["iterated"] == 3

    @pytest.mark.asyncio
    async def test_non_json_string_error(self, applet):
        msg = _msg(
            content="not a list",
            metadata={"node_data": {"array_source": "{{input}}"}},
        )
        result = await applet.on_message(msg)
        assert result.content["ok"] is False

    @pytest.mark.asyncio
    async def test_invalid_config(self, applet):
        msg = _msg(metadata={"node_data": {"max_iterations": "not_a_number"}})
        result = await applet.on_message(msg)
        assert "error" in result.content
        assert "Invalid for-each configuration" in result.content["error"]

    @pytest.mark.asyncio
    async def test_context_gets_for_each_results(self, applet):
        msg = _msg(
            content=[1],
            metadata={"node_data": {"array_source": "{{input}}"}},
        )
        result = await applet.on_message(msg)
        assert "for_each_results" in result.context
        assert "last_for_each_response" in result.context

    @pytest.mark.asyncio
    async def test_coerce_to_list_tuple(self, applet):
        assert ForEachNodeApplet._coerce_to_list((1, 2, 3)) == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_coerce_to_list_not_coercible(self, applet):
        assert ForEachNodeApplet._coerce_to_list(42) is None

    @pytest.mark.asyncio
    async def test_get_downstream_nodes(self, applet):
        msg = _msg(metadata={"node_data": {"sub_nodes": [{"type": "transform", "id": "t1"}]}})
        nodes = applet._get_downstream_nodes(msg)
        assert len(nodes) == 1
        assert nodes[0]["type"] == "transform"

    @pytest.mark.asyncio
    async def test_get_downstream_nodes_empty(self, applet):
        msg = _msg(metadata={})
        nodes = applet._get_downstream_nodes(msg)
        assert nodes == []

    @pytest.mark.asyncio
    async def test_build_iteration_message(self, applet):
        msg = _msg(content="parent", context={"run_id": "r1"}, metadata={"node_id": "fe1"})
        iter_msg = ForEachNodeApplet._build_iteration_message("item_val", 2, msg, "fe1", "r1")
        assert iter_msg.content == "item_val"
        assert iter_msg.context["for_each_item"] == "item_val"
        assert iter_msg.context["for_each_index"] == 2
        assert iter_msg.metadata["for_each_index"] == 2


# ============================================================
# HTTPRequestNodeApplet Tests
# ============================================================

class TestHTTPRequestNodeApplet:
    @pytest.fixture
    def applet(self):
        return HTTPRequestNodeApplet()

    @pytest.mark.asyncio
    async def test_invalid_config(self, applet):
        msg = _msg(metadata={"node_data": {"url": None, "method": 12345}})
        result = await applet.on_message(msg)
        assert "error" in result.content

    @pytest.mark.asyncio
    async def test_empty_url_error(self, applet):
        msg = _msg(metadata={"node_data": {"url": "{{nonexistent}}"}})
        result = await applet.on_message(msg)
        assert "error" in result.content

    @pytest.mark.asyncio
    async def test_successful_get_request(self, applet):
        import httpx
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.url = "https://api.example.com/data"
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"result": "ok"}
        mock_response.text = '{"result": "ok"}'

        with patch("apps.orchestrator.main.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            msg = _msg(metadata={"node_data": {"url": "https://api.example.com/data", "method": "GET"}})
            result = await applet.on_message(msg)

        assert result.content["ok"] is True
        assert result.content["status_code"] == 200
        assert result.content["data"] == {"result": "ok"}

    @pytest.mark.asyncio
    async def test_http_error(self, applet):
        import httpx
        with patch("apps.orchestrator.main.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            msg = _msg(metadata={"node_data": {"url": "https://fail.example.com", "method": "GET"}})
            result = await applet.on_message(msg)

        assert "error" in result.content
        assert result.metadata["status"] == "error"

    @pytest.mark.asyncio
    async def test_normalize_headers(self, applet):
        assert applet._normalize_headers({"X-Key": "val", "empty": None, "": "skip"}) == {"X-Key": "val"}
        assert applet._normalize_headers("not_dict") == {}

    @pytest.mark.asyncio
    async def test_normalize_headers_complex_values(self, applet):
        result = applet._normalize_headers({"X-Data": {"nested": True}, "X-List": [1, 2]})
        assert json.loads(result["X-Data"]) == {"nested": True}
        assert json.loads(result["X-List"]) == [1, 2]

    @pytest.mark.asyncio
    async def test_normalize_query_params(self, applet):
        assert applet._normalize_query_params({"q": "test", "": "skip", "none": None}) == {"q": "test"}
        assert applet._normalize_query_params("not_dict") == {}

    @pytest.mark.asyncio
    async def test_default_body_template_get(self, applet):
        msg = _msg(content="data")
        assert applet._default_body_template(msg, "GET") is None

    @pytest.mark.asyncio
    async def test_default_body_template_post(self, applet):
        msg = _msg(content={"key": "val"})
        assert applet._default_body_template(msg, "POST") == {"key": "val"}

    @pytest.mark.asyncio
    async def test_default_body_template_none_content(self, applet):
        msg = AppletMessage(content=None, context={}, metadata={})
        assert applet._default_body_template(msg, "POST") is None

    @pytest.mark.asyncio
    async def test_apply_body_payload_json(self, applet):
        kwargs = {"headers": {}}
        applet._apply_body_payload("json", {"key": "val"}, kwargs)
        assert kwargs["json"] == {"key": "val"}

    @pytest.mark.asyncio
    async def test_apply_body_payload_auto_dict(self, applet):
        kwargs = {"headers": {}}
        applet._apply_body_payload("auto", {"key": "val"}, kwargs)
        assert kwargs["json"] == {"key": "val"}

    @pytest.mark.asyncio
    async def test_apply_body_payload_form(self, applet):
        kwargs = {"headers": {}}
        applet._apply_body_payload("form", {"name": "Alice", "scores": [1, 2]}, kwargs)
        assert kwargs["data"]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_apply_body_payload_form_scalar(self, applet):
        kwargs = {"headers": {}}
        applet._apply_body_payload("form", "raw_text", kwargs)
        assert kwargs["data"]["value"] == "raw_text"

    @pytest.mark.asyncio
    async def test_apply_body_payload_none_type(self, applet):
        kwargs = {"headers": {}}
        applet._apply_body_payload("none", "anything", kwargs)
        assert "json" not in kwargs
        assert "data" not in kwargs
        assert "content" not in kwargs

    @pytest.mark.asyncio
    async def test_apply_body_payload_raw_dict(self, applet):
        kwargs = {"headers": {}}
        applet._apply_body_payload("raw", {"foo": 1}, kwargs)
        assert "content" in kwargs

    @pytest.mark.asyncio
    async def test_apply_body_payload_raw_string(self, applet):
        kwargs = {"headers": {}}
        applet._apply_body_payload("raw", "text_body", kwargs)
        assert kwargs["content"] == "text_body"

    @pytest.mark.asyncio
    async def test_parse_response_json(self, applet):
        import httpx
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.headers = {"content-type": "application/json; charset=utf-8"}
        mock_resp.json.return_value = {"parsed": True}
        assert applet._parse_response_data(mock_resp) == {"parsed": True}

    @pytest.mark.asyncio
    async def test_parse_response_text_json_fallback(self, applet):
        import httpx
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.headers = {"content-type": "text/plain"}
        mock_resp.text = '{"fallback": true}'
        assert applet._parse_response_data(mock_resp) == {"fallback": True}

    @pytest.mark.asyncio
    async def test_parse_response_plain_text(self, applet):
        import httpx
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.headers = {"content-type": "text/plain"}
        mock_resp.text = "just a string"
        assert applet._parse_response_data(mock_resp) == "just a string"

    @pytest.mark.asyncio
    async def test_parse_response_empty(self, applet):
        import httpx
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.headers = {"content-type": "text/plain"}
        mock_resp.text = ""
        assert applet._parse_response_data(mock_resp) == ""

    @pytest.mark.asyncio
    async def test_config_from_legacy_context(self, applet):
        msg = _msg(
            context={"http_config": {"url": "https://example.com", "method": "POST"}},
        )
        config = applet._resolve_config(msg)
        assert config.url == "https://example.com"
        assert config.method == "POST"

    @pytest.mark.asyncio
    async def test_include_response_headers(self, applet):
        import httpx
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.url = "https://api.example.com"
        mock_response.headers = {"content-type": "application/json", "x-custom": "value"}
        mock_response.json.return_value = {}
        mock_response.text = "{}"

        with patch("apps.orchestrator.main.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            msg = _msg(metadata={"node_data": {
                "url": "https://api.example.com",
                "method": "GET",
                "include_response_headers": True,
            }})
            result = await applet.on_message(msg)

        assert "headers" in result.content


# ============================================================
# CodeNodeApplet Tests
# ============================================================

class TestCodeNodeApplet:
    @pytest.fixture
    def applet(self):
        return CodeNodeApplet()

    @pytest.mark.asyncio
    async def test_invalid_config(self, applet):
        msg = _msg(metadata={"node_data": {"language": 12345}})
        result = await applet.on_message(msg)
        assert "error" in result.content
        assert "Invalid code node configuration" in result.content["error"]

    @pytest.mark.asyncio
    async def test_no_code_provided(self, applet):
        msg = _msg(
            content={"no_code_key": True},
            metadata={"node_data": {"code": ""}},
        )
        result = await applet.on_message(msg)
        assert "error" in result.content
        assert result.content["error"] == "No code provided"

    @pytest.mark.asyncio
    async def test_code_from_content_fallback(self, applet):
        msg = _msg(
            content={"code": "print('hello')"},
            metadata={"node_data": {"code": "", "language": "python"}},
        )
        with patch.object(applet, "_execute_sandboxed_code", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"ok": True, "result": "hello", "timed_out": False, "exit_code": 0}
            result = await applet.on_message(msg)
        assert result.content["ok"] is True
        mock_exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_successful_execution(self, applet):
        msg = _msg(
            content={},
            metadata={"node_data": {"code": "print('hello')", "language": "python"}},
        )
        with patch.object(applet, "_execute_sandboxed_code", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"ok": True, "result": "hello", "timed_out": False, "exit_code": 0}
            result = await applet.on_message(msg)
        assert result.content["ok"] is True
        assert result.content["language"] == "python"
        assert "duration_ms" in result.content
        assert result.metadata["status"] == "success"

    @pytest.mark.asyncio
    async def test_failed_execution(self, applet):
        msg = _msg(
            content={},
            metadata={"node_data": {"code": "raise Exception('fail')", "language": "python"}},
        )
        with patch.object(applet, "_execute_sandboxed_code", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"ok": False, "result": None, "timed_out": False, "exit_code": 1, "error": {"message": "fail"}}
            result = await applet.on_message(msg)
        assert result.content["ok"] is False
        assert result.metadata["status"] == "error"

    @pytest.mark.asyncio
    async def test_resolve_config_defaults(self, applet):
        msg = _msg(metadata={"node_data": {"code": "x=1"}})
        config = applet._resolve_config(msg)
        assert config.language == "python"
        assert config.timeout_seconds == 5.0
        assert config.memory_limit_mb == 256

    @pytest.mark.asyncio
    async def test_resolve_config_javascript(self, applet):
        msg = _msg(metadata={"node_data": {"code": "console.log(1)", "language": "javascript"}})
        config = applet._resolve_config(msg)
        assert config.language == "javascript"


# ============================================================
# MemoryNodeApplet Tests
# ============================================================

class TestMemoryNodeApplet:
    @pytest.fixture
    def applet(self):
        return MemoryNodeApplet()

    @pytest.fixture
    def mock_store(self):
        store = MagicMock()
        store.backend_name = "sqlite_fts"
        store.upsert = MagicMock()
        store.get = MagicMock(return_value=None)
        store.search = MagicMock(return_value=[])
        store.delete = MagicMock(return_value=False)
        store.clear = MagicMock(return_value=0)
        return store

    @pytest.mark.asyncio
    async def test_store_operation(self, applet, mock_store):
        with patch("apps.orchestrator.main.MemoryStoreFactory.get_store", return_value=mock_store):
            msg = _msg(
                content={"data": "test_value"},
                metadata={"node_data": {"operation": "store"}},
            )
            result = await applet.on_message(msg)
        assert result.content["status"] == "stored"
        assert result.content["backend"] == "sqlite_fts"
        mock_store.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_by_key_found(self, applet, mock_store):
        mock_store.get.return_value = {
            "key": "k1",
            "data": {"value": "found_data"},
            "metadata": {"created_at": 123.0},
            "score": 1.0,
        }
        with patch("apps.orchestrator.main.MemoryStoreFactory.get_store", return_value=mock_store):
            msg = _msg(
                content={"key": "k1"},
                metadata={"node_data": {"operation": "retrieve"}},
            )
            result = await applet.on_message(msg)
        assert result.content == {"value": "found_data"}
        assert result.context["memory_retrieved"] is True

    @pytest.mark.asyncio
    async def test_retrieve_by_key_with_metadata(self, applet, mock_store):
        mock_store.get.return_value = {
            "key": "k1",
            "data": "found",
            "metadata": {"tag": "x"},
            "score": 1.0,
        }
        with patch("apps.orchestrator.main.MemoryStoreFactory.get_store", return_value=mock_store):
            msg = _msg(
                content={"key": "k1"},
                metadata={"node_data": {"operation": "retrieve", "include_metadata": True}},
            )
            result = await applet.on_message(msg)
        assert result.content["key"] == "k1"
        assert result.content["data"] == "found"
        assert "metadata" in result.content

    @pytest.mark.asyncio
    async def test_retrieve_search_results(self, applet, mock_store):
        mock_store.search.return_value = [
            {"key": "k1", "data": "d1", "metadata": {}, "score": 0.9},
            {"key": "k2", "data": "d2", "metadata": {}, "score": 0.8},
        ]
        with patch("apps.orchestrator.main.MemoryStoreFactory.get_store", return_value=mock_store):
            msg = _msg(
                content="search query",
                metadata={"node_data": {"operation": "retrieve"}},
            )
            result = await applet.on_message(msg)
        assert result.content["status"] == "retrieved"
        assert "k1" in result.content["memories"]

    @pytest.mark.asyncio
    async def test_retrieve_search_with_metadata(self, applet, mock_store):
        mock_store.search.return_value = [
            {"key": "k1", "data": "d1", "metadata": {}, "score": 0.9},
        ]
        with patch("apps.orchestrator.main.MemoryStoreFactory.get_store", return_value=mock_store):
            msg = _msg(
                content="search",
                metadata={"node_data": {"operation": "retrieve", "include_metadata": True}},
            )
            result = await applet.on_message(msg)
        assert "results" in result.content
        assert result.content["count"] == 1

    @pytest.mark.asyncio
    async def test_retrieve_not_found(self, applet, mock_store):
        with patch("apps.orchestrator.main.MemoryStoreFactory.get_store", return_value=mock_store):
            msg = _msg(
                content="no match",
                metadata={"node_data": {"operation": "retrieve"}},
            )
            result = await applet.on_message(msg)
        assert result.content["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_delete_no_key(self, applet, mock_store):
        with patch("apps.orchestrator.main.MemoryStoreFactory.get_store", return_value=mock_store):
            msg = _msg(
                content={},
                metadata={"node_data": {"operation": "delete"}},
            )
            result = await applet.on_message(msg)
        assert result.content["status"] == "not_found"
        assert result.content["key"] is None

    @pytest.mark.asyncio
    async def test_delete_with_key(self, applet, mock_store):
        mock_store.delete.return_value = True
        with patch("apps.orchestrator.main.MemoryStoreFactory.get_store", return_value=mock_store):
            msg = _msg(
                content={"key": "k1"},
                metadata={"node_data": {"operation": "delete"}},
            )
            result = await applet.on_message(msg)
        assert result.content["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_delete_not_found(self, applet, mock_store):
        mock_store.delete.return_value = False
        with patch("apps.orchestrator.main.MemoryStoreFactory.get_store", return_value=mock_store):
            msg = _msg(
                content={"key": "k_missing"},
                metadata={"node_data": {"operation": "delete"}},
            )
            result = await applet.on_message(msg)
        assert result.content["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_clear_operation(self, applet, mock_store):
        mock_store.clear.return_value = 5
        with patch("apps.orchestrator.main.MemoryStoreFactory.get_store", return_value=mock_store):
            msg = _msg(
                content={},
                metadata={"node_data": {"operation": "clear"}},
            )
            result = await applet.on_message(msg)
        assert result.content["status"] == "cleared"
        assert result.content["count"] == 5

    @pytest.mark.asyncio
    async def test_unsupported_operation(self, applet, mock_store):
        with patch("apps.orchestrator.main.MemoryStoreFactory.get_store", return_value=mock_store):
            with patch.object(applet, "_resolve_operation", return_value="unknown_op"):
                msg = _msg(
                    content={},
                    metadata={"node_data": {"operation": "store"}},
                )
                result = await applet.on_message(msg)
        assert "Unsupported memory operation" in result.content["error"]

    @pytest.mark.asyncio
    async def test_invalid_config(self, applet):
        msg = _msg(metadata={"node_data": {"operation": 12345, "backend": 12345}})
        result = await applet.on_message(msg)
        assert "error" in result.content

    @pytest.mark.asyncio
    async def test_operation_from_content_override(self, applet, mock_store):
        mock_store.clear.return_value = 0
        with patch("apps.orchestrator.main.MemoryStoreFactory.get_store", return_value=mock_store):
            msg = _msg(
                content={"operation": "clear"},
                metadata={"node_data": {"operation": "store"}},
            )
            result = await applet.on_message(msg)
        assert result.content["status"] == "cleared"

    @pytest.mark.asyncio
    async def test_namespace_from_context(self, applet, mock_store):
        mock_store.clear.return_value = 0
        with patch("apps.orchestrator.main.MemoryStoreFactory.get_store", return_value=mock_store):
            msg = _msg(
                content={},
                context={"memory_namespace": "custom_ns"},
                metadata={"node_data": {"operation": "clear"}},
            )
            result = await applet.on_message(msg)
        assert result.content["namespace"] == "custom_ns"

    @pytest.mark.asyncio
    async def test_namespace_from_content(self, applet, mock_store):
        mock_store.clear.return_value = 0
        with patch("apps.orchestrator.main.MemoryStoreFactory.get_store", return_value=mock_store):
            msg = _msg(
                content={"namespace": "from_content"},
                metadata={"node_data": {"operation": "clear"}},
            )
            result = await applet.on_message(msg)
        assert result.content["namespace"] == "from_content"

    @pytest.mark.asyncio
    async def test_store_key_from_context(self, applet, mock_store):
        with patch("apps.orchestrator.main.MemoryStoreFactory.get_store", return_value=mock_store):
            msg = _msg(
                content={"data": "value"},
                context={"memory_key": "ctx_key"},
                metadata={"node_data": {"operation": "store"}},
            )
            result = await applet.on_message(msg)
        assert result.content["key"] == "ctx_key"

    @pytest.mark.asyncio
    async def test_tags_merge(self, applet, mock_store):
        with patch("apps.orchestrator.main.MemoryStoreFactory.get_store", return_value=mock_store):
            msg = _msg(
                content={"data": "value", "tags": ["from_content"]},
                context={"memory_tags": ["from_context"]},
                metadata={"node_data": {"operation": "store", "tags": ["from_config"]}},
            )
            result = await applet.on_message(msg)
        # The store was called, verify tags were merged
        call_args = mock_store.upsert.call_args
        metadata_passed = call_args[0][4]  # 5th positional arg = metadata
        assert "from_config" in metadata_passed["tags"]
        assert "from_context" in metadata_passed["tags"]
        assert "from_content" in metadata_passed["tags"]

    @pytest.mark.asyncio
    async def test_extract_store_payload_data_key(self, applet):
        msg = _msg(content={"data": "the_payload", "operation": "store"})
        assert applet._extract_store_payload(msg) == "the_payload"

    @pytest.mark.asyncio
    async def test_extract_store_payload_filters_keys(self, applet):
        msg = _msg(content={"operation": "store", "key": "k1", "custom": "kept"})
        result = applet._extract_store_payload(msg)
        assert result == {"custom": "kept"}

    @pytest.mark.asyncio
    async def test_extract_store_payload_non_dict(self, applet):
        msg = _msg(content="raw_string")
        assert applet._extract_store_payload(msg) == {"value": "raw_string"}

    @pytest.mark.asyncio
    async def test_resolve_query_from_string_content(self, applet):
        from apps.orchestrator.models import MemoryNodeConfigModel
        config = MemoryNodeConfigModel(operation="retrieve")
        msg = _msg(content="my search query")
        assert applet._resolve_query(msg, config) == "my search query"

    @pytest.mark.asyncio
    async def test_resolve_query_from_dict_content(self, applet):
        from apps.orchestrator.models import MemoryNodeConfigModel
        config = MemoryNodeConfigModel(operation="retrieve")
        msg = _msg(content={"query": "dict query"})
        assert applet._resolve_query(msg, config) == "dict query"

    @pytest.mark.asyncio
    async def test_operation_error_handling(self, applet, mock_store):
        mock_store.clear.side_effect = RuntimeError("DB error")
        with patch("apps.orchestrator.main.MemoryStoreFactory.get_store", return_value=mock_store):
            msg = _msg(
                content={},
                metadata={"node_data": {"operation": "clear"}},
            )
            result = await applet.on_message(msg)
        assert "Memory operation failed" in result.content["error"]
        assert result.metadata["status"] == "error"


# ============================================================
# SQLiteFTSMemoryStoreBackend Tests
# ============================================================

class TestSQLiteFTSBackend:
    @pytest.fixture
    def store(self, tmp_path):
        db_path = str(tmp_path / "test_memory.sqlite3")
        return SQLiteFTSMemoryStoreBackend(db_path)

    def test_upsert_and_get(self, store):
        store.upsert("k1", "ns1", "hello world", {"val": 1}, {"timestamp": time.time(), "tags": ["t1"]})
        result = store.get("k1", "ns1")
        assert result is not None
        assert result["key"] == "k1"
        assert result["data"] == {"val": 1}

    def test_get_without_namespace(self, store):
        store.upsert("k1", "ns1", "hello", "payload", {"timestamp": time.time(), "tags": []})
        result = store.get("k1")
        assert result is not None
        assert result["key"] == "k1"

    def test_get_not_found(self, store):
        assert store.get("nonexistent", "ns1") is None

    def test_upsert_update(self, store):
        store.upsert("k1", "ns1", "v1", "p1", {"timestamp": 1.0, "tags": []})
        store.upsert("k1", "ns1", "v2", "p2", {"timestamp": 2.0, "tags": []})
        result = store.get("k1", "ns1")
        assert result["data"] == "p2"

    def test_search_fts(self, store):
        store.upsert("k1", "ns1", "hello world", "p1", {"timestamp": 1.0, "tags": ["greet"]})
        store.upsert("k2", "ns1", "goodbye world", "p2", {"timestamp": 2.0, "tags": ["farewell"]})
        results = store.search("ns1", "hello", [], 10)
        assert len(results) >= 1
        keys = [r["key"] for r in results]
        assert "k1" in keys

    def test_search_with_tags(self, store):
        store.upsert("k1", "ns1", "hello", "p1", {"timestamp": 1.0, "tags": ["alpha"]})
        store.upsert("k2", "ns1", "world", "p2", {"timestamp": 2.0, "tags": ["beta"]})
        results = store.search("ns1", "", ["alpha"], 10)
        keys = [r["key"] for r in results]
        assert "k1" in keys

    def test_search_empty_query_and_tags(self, store):
        store.upsert("k1", "ns1", "data", "p1", {"timestamp": 1.0, "tags": []})
        results = store.search("ns1", "", [], 10)
        assert len(results) >= 1

    def test_delete_existing(self, store):
        store.upsert("k1", "ns1", "hello", "p1", {"timestamp": 1.0, "tags": []})
        assert store.delete("k1", "ns1") is True
        assert store.get("k1", "ns1") is None

    def test_delete_nonexistent(self, store):
        assert store.delete("nope", "ns1") is False

    def test_delete_without_namespace(self, store):
        store.upsert("k1", "ns1", "hello", "p1", {"timestamp": 1.0, "tags": []})
        assert store.delete("k1") is True
        assert store.get("k1") is None

    def test_delete_without_namespace_not_found(self, store):
        assert store.delete("nope") is False

    def test_clear(self, store):
        store.upsert("k1", "ns1", "hello", "p1", {"timestamp": 1.0, "tags": []})
        store.upsert("k2", "ns1", "world", "p2", {"timestamp": 2.0, "tags": []})
        store.upsert("k3", "ns2", "other", "p3", {"timestamp": 3.0, "tags": []})
        count = store.clear("ns1")
        assert count == 2
        assert store.get("k1", "ns1") is None
        assert store.get("k3", "ns2") is not None

    def test_schema_initialized_once(self, store):
        assert store._initialized is True
        store._ensure_schema()
        assert store._initialized is True


# ============================================================
# MemoryStoreFactory Tests
# ============================================================

class TestMemoryStoreFactory:
    def setup_method(self):
        MemoryStoreFactory._stores.clear()

    def test_get_store_sqlite_default(self, tmp_path):
        from apps.orchestrator.models import MemoryNodeConfigModel
        config = MemoryNodeConfigModel(
            operation="store",
            backend="sqlite_fts",
            persist_path=str(tmp_path / "test.db"),
        )
        store = MemoryStoreFactory.get_store(config)
        assert store.backend_name == "sqlite_fts"

    def test_get_store_caching(self, tmp_path):
        from apps.orchestrator.models import MemoryNodeConfigModel
        config = MemoryNodeConfigModel(
            operation="store",
            backend="sqlite_fts",
            persist_path=str(tmp_path / "cached.db"),
        )
        store1 = MemoryStoreFactory.get_store(config)
        store2 = MemoryStoreFactory.get_store(config)
        assert store1 is store2

    def test_get_store_persist_path_directory(self, tmp_path):
        from apps.orchestrator.models import MemoryNodeConfigModel
        config = MemoryNodeConfigModel(
            operation="store",
            backend="sqlite_fts",
            persist_path=str(tmp_path / "subdir"),
        )
        store = MemoryStoreFactory.get_store(config)
        assert store.backend_name == "sqlite_fts"

    def test_get_store_chroma_fallback(self, tmp_path):
        from apps.orchestrator.models import MemoryNodeConfigModel
        config = MemoryNodeConfigModel(
            operation="store",
            backend="chroma",
            persist_path=str(tmp_path / "chroma_test"),
        )
        # Chroma may or may not be installed; either way should not crash
        store = MemoryStoreFactory.get_store(config)
        assert store.backend_name in ("chroma", "sqlite_fts")


# ============================================================
# Helper function tests (expanding coverage)
# ============================================================

class TestHelperFunctions:
    def test_as_text_string(self):
        assert _as_text("hello") == "hello"

    def test_as_text_number(self):
        assert _as_text(42) == "42"

    def test_as_text_dict(self):
        result = _as_text({"key": "val"})
        assert '"key"' in result

    def test_as_text_list(self):
        result = _as_text([1, 2])
        assert "[1, 2]" in result

    def test_as_text_none(self):
        assert _as_text(None) == "None"

    def test_as_serialized_text_string(self):
        assert _as_serialized_text("hello") == "hello"

    def test_as_serialized_text_dict(self):
        result = _as_serialized_text({"key": "val"})
        parsed = json.loads(result)
        assert parsed == {"key": "val"}

    def test_normalize_memory_tags_list(self):
        assert _normalize_memory_tags(["hello", "  world  ", "", "  "]) == ["hello", "world"]

    def test_normalize_memory_tags_string(self):
        assert _normalize_memory_tags("single_tag") == ["single_tag"]

    def test_normalize_memory_tags_none(self):
        assert _normalize_memory_tags(None) == []

    def test_normalize_memory_tags_non_iterable(self):
        assert _normalize_memory_tags(42) == []

    def test_fts_terms_basic(self):
        terms = _fts_terms("hello world test")
        assert "hello" in terms
        assert "world" in terms

    def test_fts_terms_empty(self):
        assert _fts_terms("") == []

    def test_parse_json_or_default_valid(self):
        assert _parse_json_or_default('{"key": 1}', {}) == {"key": 1}

    def test_parse_json_or_default_invalid(self):
        assert _parse_json_or_default("not json", "default") == "default"

    def test_parse_json_or_default_none(self):
        assert _parse_json_or_default(None, "fallback") == "fallback"
