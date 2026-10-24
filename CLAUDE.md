# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agent Engine CLI (`ae`) is a command-line interface for managing Google Cloud Vertex AI Agent Engine deployments. Built with Python 3.11+ using Typer for CLI framework and Rich for terminal output. Wraps the `google-cloud-aiplatform` / `agentplatform` SDK.

## Development Commands

```bash
# Install dependencies and project in development mode
uv sync

# Run the CLI
uv run ae --help

# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_main.py

# Run a specific test
uv run pytest tests/test_main.py::TestListCommand::test_list_with_agents
```

## Architecture

### Module Layout (`src/agent_engine_cli/`)

- **`main.py`** — Typer app instance with all CLI commands. Top-level commands: `version`, `list`, `get`, `create`, `delete`, `chat`, `a2a-chat`. Subcommand groups: `sessions list`, `sandboxes list`, `memories list`. Every command accepts `--project/-p`, `--location/-l`, `--base-url`, and `--api-version`.
- **`client.py`** — `AgentEngineClient` wraps `agentplatform.Client` (`runtimes`, `sessions`, `sandboxes`, `memory_banks.memories`) and exposes typed methods (`list_agents`, `get_agent`, `get_runtime`, `create_agent`, `delete_agent`, `list_sessions`, `list_sandboxes`, `list_memories`). Agent IDs are resolved to full resource names (`projects/.../reasoningEngines/...`) via `_resolve_resource_name`.
- **`dependencies.py`** — `get_client()` factory function that serves as the DI seam. Tests patch `agent_engine_cli.main.get_client` to inject `FakeAgentEngineClient`.
- **`config.py`** — `resolve_project()` resolves `--project` or falls back to Application Default Credentials (ADC). Raises `ConfigurationError` on failure.
- **`chat.py`** — `run_chat()` async function for interactive streaming chat with an agent; fetches the agent via `AgentEngineClient.get_runtime`. Monkey-patches `google.genai._api_client` for debug logging.

### Testing Patterns

- Tests use `typer.testing.CliRunner` to invoke CLI commands.
- `tests/fakes.py` provides `FakeAgentEngineClient` — an in-memory fake that mirrors the real client interface. Tests populate its `_agents`, `_sessions`, `_sandboxes`, and `_memories` dicts directly.
- Commands are tested by patching `agent_engine_cli.main.get_client` to return a `FakeAgentEngineClient`. The `chat` command patches `agent_engine_cli.main.run_chat` instead.
- `test_client.py` mocks `agentplatform` at the module level via `sys.modules` patching.
- The `CliRunner` is configured with `env={"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"}` to get stable output for assertions.

### Key Design Decisions

- `agentplatform` is imported lazily inside `AgentEngineClient.__init__` and `chat.py` to keep CLI startup fast.
- `AgentEngineClient` returns the SDK's pydantic types (`ReasoningEngine`, `Session`, `Memory`, `SandboxEnvironment` from `agentplatform._genai.types.common`), unwrapped from the `Runtime` wrapper; fakes and tests use the same types, so `main.py` accesses fields directly.
- `a2a_chat._rebind_missing_a2a_methods` works around an SDK bug: A2A methods whose operation schema lacks `a2a_agent_card` crash on `json.loads(None)`, so they are re-wrapped with a synthetic v1.0 agent card.
