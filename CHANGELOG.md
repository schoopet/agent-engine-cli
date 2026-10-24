# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-24

### Added

- `ae create` accepts `--image-uri` to deploy a prebuilt Artifact Registry container and `--agent-framework` to declare the agent framework (`google-adk`, `langchain`, `langgraph`, `ag2`, `llama-index`, `custom`).

### Changed

- Migrated from the deprecated `vertexai.Client` to `agentplatform.Client`. Requires `google-cloud-aiplatform>=2.6.0`.
- Upgraded all dependencies, including `google-cloud-aiplatform` 2.6, `protobuf` 7, `a2a-sdk` 1.2, and `typer` 0.27.
- `ae delete` now goes through the SDK client, so it honors `--base-url`.
- CRUD commands only catch API and validation errors; unexpected errors are no longer swallowed.
- Debug mode masks auth tokens fully as `Bearer ***`.

### Fixed

- `ae a2a-chat` works with agents whose operation schema lacks an agent card (the fallback card now uses the A2A v1.0 format).
- `ae sandboxes list` no longer crashes on import (added `google-cloud-iam` dependency).
- `ae version` reports the installed package version instead of a stale hard-coded one.

### Performance

- Deferred chat imports for faster CLI startup.

## [0.1.8] - 2026-03-07

### Changed

- Project URLs now point to the `schoopet` GitHub organization.

## [0.1.7] - 2026-02-24

### Added

- New `ae a2a-chat` command for interacting with agents over the A2A protocol.

### Changed

- Upgraded dependencies to latest versions.

### Performance

- Session listing streams results through iterators instead of materializing the full list.

## [0.1.6] - 2026-02-11

### Changed

- **Refactored CLI Structure**: Shared parameters (`--location`, `--project`, `--base-url`, `--api-version`) are now global and can be placed before or after subcommands.
- **Simplified Implementation**: Centralized client initialization and unified resource ID parsing, significantly reducing boilerplate.

### Internal

- Improved test suite to verify global parameter handling.
- Fixed `RuntimeWarning` in tests related to unawaited `AsyncMock` coroutines.

## [0.1.5] - 2026-02-11

### Added

- New `--base-url` and `--api-version` options to `ae` for Vertex AI endpoint overrides.

### Performance

- Prevented uncontrolled monkey patching in the chat client.

### Documentation

- Expanded `CLAUDE.md` with detailed architecture and testing patterns.
- Updated installation instructions to specify Python 3.11+.

### Internal

- Refactored CRUD operations to use dependency injection and Fakes for improved testing.
- Cleaned up test fakes and strengthened assertions.

## [0.1.4] - 2026-02-10

### Security

- Escape all dynamic values in Rich `console.print()` calls to prevent markup injection from agent responses and API errors.
- Add credential exposure warning when `--debug` mode is enabled.
- Remove unused `requests` dependency to reduce attack surface.
- Validate agent IDs to reject empty, whitespace-only, and control-character inputs.
- Narrow exception handling in ADC config resolution to avoid swallowing unexpected errors.

### Changed

- Minimum Python version bumped from 3.10 to 3.11.
- Deduplicated resource name construction into a shared `_resolve_resource_name()` helper.

## [0.1.3] - 2026-01-25

### Added

- `ae get` now includes class method parameters with types and descriptions in the output.

### Changed

- Updated author information in `pyproject.toml`.
- Version bumped to 0.1.3 to resolve tagging conflicts with 0.1.2.

## [0.1.1] - 2026-01-19

### Added

- `ae sessions list` - List active sessions for an agent
- `ae sandboxes list` - List active sandboxes
- `ae memories list` - List agent memories
- Support for automatic project detection from Application Default Credentials (ADC)

### Changed

- Improved PyPI package metadata

## [0.1.0] - 2025-01-19

### Added

- Initial release of Agent Engine CLI (`ae`)
- `ae list` - List all agents in a Google Cloud project
- `ae get` - Get details for a specific agent
- `ae create` - Create a new agent with configurable identity type
- `ae delete` - Delete an agent with optional force flag
- `ae chat` - Interactive chat session with an agent (streaming support)
- `ae version` - Display CLI version
- Rich terminal output with tables and panels
- Support for Google Cloud authentication via ADC
