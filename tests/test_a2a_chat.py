"""Tests for A2A chat pure functions."""

import json

from a2a.types import AgentCard
from a2a.utils.constants import TransportProtocol
from google.protobuf import json_format

from agent_engine_cli.a2a_chat import (
    _SYNTHETIC_AGENT_CARD,
    SlashCommand,
    UserMessage,
    _rebind_missing_a2a_methods,
    build_message_kwargs,
    extract_response_text,
    parse_context,
    parse_input,
)


class TestParseInput:
    def test_slash_command_no_args(self):
        result = parse_input("/help")
        assert isinstance(result, SlashCommand)
        assert result.name == "/help"
        assert result.args == ""

    def test_slash_command_with_args(self):
        result = parse_input("/get-task abc-123")
        assert isinstance(result, SlashCommand)
        assert result.name == "/get-task"
        assert result.args == "abc-123"

    def test_slash_command_with_multi_word_args(self):
        result = parse_input("/context key1=val1, key2=val2")
        assert isinstance(result, SlashCommand)
        assert result.name == "/context"
        assert result.args == "key1=val1, key2=val2"

    def test_user_message(self):
        result = parse_input("hello world")
        assert isinstance(result, UserMessage)
        assert result.text == "hello world"

    def test_user_message_with_leading_space(self):
        result = parse_input("  hello world  ")
        assert isinstance(result, UserMessage)
        assert result.text == "hello world"

    def test_slash_command_with_leading_space(self):
        result = parse_input("  /help  ")
        assert isinstance(result, SlashCommand)
        assert result.name == "/help"
        assert result.args == ""


class TestParseContext:
    def test_single_pair(self):
        assert parse_context("key=value") == {"key": "value"}

    def test_multiple_pairs(self):
        result = parse_context("key1=val1, key2=val2")
        assert result == {"key1": "val1", "key2": "val2"}

    def test_whitespace_handling(self):
        result = parse_context("  key1 = val1 ,  key2 = val2  ")
        assert result == {"key1": "val1", "key2": "val2"}

    def test_empty_string(self):
        assert parse_context("") == {}

    def test_no_equals(self):
        """Entries without '=' are skipped."""
        assert parse_context("justtext") == {}

    def test_value_with_equals(self):
        """Only first '=' splits key/value."""
        result = parse_context("url=http://foo=bar")
        assert result == {"url": "http://foo=bar"}


class TestBuildMessageKwargs:
    def test_basic_message(self):
        result = build_message_kwargs("hello", None, {})
        assert result["role"] == "user"
        assert result["parts"] == [{"kind": "text", "text": "hello"}]
        assert "messageId" in result
        assert "contextId" not in result
        assert "metadata" not in result

    def test_with_context_id(self):
        result = build_message_kwargs("hello", "ctx-123", {})
        assert result["contextId"] == "ctx-123"

    def test_with_context(self):
        result = build_message_kwargs("hello", None, {"env": "prod"})
        assert result["metadata"] == {"env": "prod"}

    def test_with_context_id_and_context(self):
        result = build_message_kwargs("hello", "ctx-1", {"k": "v"})
        assert result["contextId"] == "ctx-1"
        assert result["metadata"] == {"k": "v"}

    def test_unique_message_ids(self):
        r1 = build_message_kwargs("a", None, {})
        r2 = build_message_kwargs("b", None, {})
        assert r1["messageId"] != r2["messageId"]

    def test_context_is_copied(self):
        """Ensure the original context dict is not mutated."""
        ctx = {"key": "val"}
        result = build_message_kwargs("hello", None, ctx)
        result["metadata"]["new"] = "added"
        assert "new" not in ctx


class TestExtractResponseText:
    def test_dict_style_artifacts(self):
        """Test extraction from dict-style response."""
        result = {
            "artifacts": [
                {"parts": [{"text": "Hello from agent"}]}
            ]
        }
        assert extract_response_text(result) == "Hello from agent"

    def test_dict_multiple_artifacts(self):
        result = {
            "artifacts": [
                {"parts": [{"text": "Part 1"}]},
                {"parts": [{"text": "Part 2"}]},
            ]
        }
        assert extract_response_text(result) == "Part 1\nPart 2"

    def test_object_style_with_root(self):
        """Test extraction from object with .root.text pattern."""
        class Root:
            text = "Answer text"
        class Part:
            root = Root()
        class Artifact:
            parts = [Part()]
        class Result:
            artifacts = [Artifact()]

        assert extract_response_text(Result()) == "Answer text"

    def test_object_style_direct_text(self):
        """Test extraction from object with .text pattern."""
        class Part:
            root = None
            text = "Direct text"
        class Artifact:
            parts = [Part()]
        class Result:
            artifacts = [Artifact()]

        assert extract_response_text(Result()) == "Direct text"

    def test_no_artifacts(self):
        assert extract_response_text({}) is None
        assert extract_response_text({"artifacts": []}) is None

    def test_empty_parts(self):
        result = {"artifacts": [{"parts": []}]}
        assert extract_response_text(result) is None

    def test_no_text_in_parts(self):
        result = {"artifacts": [{"parts": [{"kind": "image"}]}]}
        assert extract_response_text(result) is None


class TestRebindMissingA2AMethods:
    def test_synthetic_card_satisfies_sdk(self):
        """The SDK wrapper requires an HTTP+JSON supported interface on the card."""
        card = AgentCard()
        json_format.ParseDict(
            json.loads(_SYNTHETIC_AGENT_CARD), card, ignore_unknown_fields=True
        )
        assert card.supported_interfaces
        assert card.supported_interfaces[0].protocol_binding == TransportProtocol.HTTP_JSON

    def test_rebinds_method_missing_agent_card(self):
        def original(self):
            pass

        class RemoteAgent:
            on_message_send = original

            def operation_schemas(self):
                return [
                    {"name": "on_message_send", "api_mode": "a2a_extension"},
                    {"name": "query", "api_mode": ""},
                ]

        agent = RemoteAgent()
        _rebind_missing_a2a_methods(agent)
        assert agent.on_message_send.__func__ is not original
        assert agent.on_message_send.__name__ == "on_message_send"

    def test_keeps_method_with_agent_card(self):
        def original(self):
            pass

        class RemoteAgent:
            on_message_send = original

            def operation_schemas(self):
                return [
                    {
                        "name": "on_message_send",
                        "api_mode": "a2a_extension",
                        "a2a_agent_card": "{}",
                    }
                ]

        agent = RemoteAgent()
        _rebind_missing_a2a_methods(agent)
        assert agent.on_message_send.__func__ is original
