"""
Interactive chat client for Agent Engine.
"""
import asyncio
import json
import logging
import os
import warnings
from typing import Any

from rich.console import Console
from rich.markup import escape

console = Console()


def _format_tool_args(args: dict | None) -> str:
    """Format tool arguments as a compact string for display."""
    if not args:
        return ""

    formatted = []
    for key, value in args.items():
        if isinstance(value, str):
            value_str = value
            if len(value_str) > 50:
                value_str = value_str[:47] + "..."
            formatted.append(f'{key}="{value_str}"')
        elif isinstance(value, (dict, list)):
            value_str = json.dumps(value)
            if len(value_str) > 50:
                value_str = value_str[:47] + "..."
            formatted.append(f"{key}={value_str}")
        else:
            formatted.append(f"{key}={value}")
    return ", ".join(formatted)


def _setup_debug_logging() -> None:
    """Enable verbose HTTP debugging with request/response bodies."""
    os.environ["HTTPX_LOG_LEVEL"] = "trace"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s [%(name)s] %(message)s",
    )

    # Enable HTTP debugging with request/response bodies
    logging.getLogger("httpx").setLevel(logging.DEBUG)
    logging.getLogger("httpx._client").setLevel(logging.DEBUG)
    logging.getLogger("httpcore").setLevel(logging.DEBUG)
    logging.getLogger("httpcore.http11").setLevel(logging.DEBUG)
    logging.getLogger("google.auth").setLevel(logging.DEBUG)
    logging.getLogger("google.api_core").setLevel(logging.DEBUG)


def _install_api_logging_hooks(debug: bool) -> None:
    """Install monkey patches for API request/response logging."""
    from google.genai import _api_client

    # Prevent repeated monkey patching
    if getattr(
        _api_client.BaseApiClient.async_request, "_is_logged_async_request", False
    ):
        return

    # Monkey patch async_request for non-streaming requests (like create_session)
    _original_async_request = _api_client.BaseApiClient.async_request

    async def _logged_async_request(
        self, http_method, path, request_dict, http_options=None
    ):
        if debug:
            console.print(f"\n{'='*60}")
            console.print(
                f"API REQUEST (non-streaming): {http_method.upper()} {path}"
            )
            console.print(f"Request dict: {request_dict}")
            console.print(f"{'='*60}\n")

        result = await _original_async_request(
            self, http_method, path, request_dict, http_options
        )

        if debug:
            console.print(f"\n{'='*60}")
            console.print(f"API RESPONSE: {result}")
            console.print(f"{'='*60}\n")

        return result

    _logged_async_request._is_logged_async_request = True
    _api_client.BaseApiClient.async_request = _logged_async_request

    # Monkey patch async_request_streamed for streaming requests
    _original_async_request_streamed = (
        _api_client.BaseApiClient.async_request_streamed
    )

    async def _logged_async_request_streamed(
        self, http_method, path, request_dict, http_options=None
    ):
        if debug:
            console.print(f"\n{'='*60}")
            console.print(f"API REQUEST (streaming): {http_method.upper()} {path}")
            console.print(f"Request dict: {request_dict}")
            console.print(f"{'='*60}\n")

        result = await _original_async_request_streamed(
            self, http_method, path, request_dict, http_options
        )
        return result

    _api_client.BaseApiClient.async_request_streamed = _logged_async_request_streamed


async def run_chat(
    project: str,
    location: str,
    agent_id: str,
    user_id: str = "cli-user",
    debug: bool = False,
    base_url: str | None = None,
    api_version: str | None = None,
) -> None:
    """
    Run an interactive chat session with an Agent Engine instance.

    Args:
        project: Google Cloud project ID.
        location: Google Cloud region (e.g., "us-central1").
        agent_id: Agent ID or full resource name.
        user_id: User ID for the chat session.
        debug: Enable verbose HTTP debug logging.
        base_url: Optional override for the Vertex AI base URL.
        api_version: Optional API version override.
    """
    # Suppress vertexai experimental warnings
    try:
        from vertexai._genai.constants import ExperimentalWarning

        warnings.filterwarnings("ignore", category=ExperimentalWarning)
    except ImportError:
        pass

    if debug:
        console.print(
            "[yellow]Warning: Debug mode logs HTTP requests/responses which "
            "may include authentication tokens and credentials.[/yellow]"
        )
        _setup_debug_logging()
        _install_api_logging_hooks(debug=True)

    import vertexai

    # Get agent instance
    http_options: dict[str, Any] = {"timeout": 10_000}
    if api_version:
        http_options["api_version"] = api_version
    if base_url:
        http_options["base_url"] = base_url
    client = vertexai.Client(project=project, location=location, http_options=http_options)
    resource_name = (
        f"projects/{project}/locations/{location}/reasoningEngines/{agent_id}"
    )
    adk_app = await asyncio.to_thread(client.agent_engines.get, name=resource_name)

    # Create session
    session = await adk_app.async_create_session(user_id=user_id)
    session_id = session["id"]

    console.print(f"[green]Ready.[/green] User: {escape(user_id)}, Session: {escape(session_id)}\n")
    console.print("Type your message and press Enter. Type 'quit' or 'exit' to end.\n")

    # Main loop
    while True:
        user_input = await asyncio.to_thread(input, "You: ")

        if user_input.lower() in ["quit", "exit"]:
            break

        if not user_input.strip():
            continue

        # Query the agent with streaming
        full_response_text = []
        tools_used = []

        async for event in adk_app.async_stream_query(
            user_id=user_id,
            session_id=session_id,
            message=user_input,
        ):
            # Print raw event in debug mode
            if debug:
                console.print(f"  [{type(event).__name__}] {event}")

            # Normalize event to access parts for extraction
            parts = []
            if isinstance(event, dict):
                content = event.get("content", {})
                if isinstance(content, dict):
                    parts = content.get("parts", [])
            elif hasattr(event, "content") and event.content:
                if hasattr(event.content, "parts"):
                    parts = event.content.parts

            for part in parts:
                # Extract and display Tool Usage in real-time
                tool_name = None
                tool_args = None
                if isinstance(part, dict):
                    if "function_call" in part:
                        fc = part["function_call"]
                        tool_name = fc.get("name")
                        tool_args = fc.get("args")
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_name = fc.name
                    tool_args = fc.args

                if tool_name and tool_name not in tools_used:
                    tools_used.append(tool_name)
                    # Print tool call immediately
                    args_str = _format_tool_args(tool_args)
                    console.print(f"[dim]\\[{escape(tool_name)}({escape(args_str)})][/dim]")

                # Extract Text
                text = None
                if isinstance(part, dict):
                    text = part.get("text")
                elif hasattr(part, "text"):
                    text = part.text

                if text:
                    full_response_text.append(text)

        # Print extracted information on new lines
        if tools_used:
            console.print(
                f"[dim]({len(tools_used)} tool{'s' if len(tools_used) != 1 else ''} used)[/dim]"
            )

        if full_response_text:
            console.print(f"\n[cyan]Agent:[/cyan] {escape(''.join(full_response_text))}")

        print()  # Final newline
