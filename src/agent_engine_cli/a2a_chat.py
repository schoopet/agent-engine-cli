"""
Interactive A2A (Agent-to-Agent) chat client for Agent Engine.
"""

import asyncio
import json
import traceback
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

console = Console()

HELP_TEXT = """\
Slash commands:
  /get-agent-card          Fetch and display the agent card
  /new-task                Start a new task (clear current task ID)
  /get-task <id>           Fetch a task by ID (defaults to current task)
  /delete-task <id>        Cancel a task by ID (defaults to current task)
  /context [key=val, ...]  Show or set persistent metadata context
  /help                    Show this help message

Type a message to send it to the agent. Type 'quit' or 'exit' to end."""


@dataclass
class SlashCommand:
    name: str
    args: str


@dataclass
class UserMessage:
    text: str


@dataclass
class A2AChatState:
    current_task_id: str | None = None
    current_context_id: str | None = None
    context: dict[str, str] = field(default_factory=dict)


def parse_input(raw: str) -> SlashCommand | UserMessage:
    """Parse raw input into a SlashCommand or UserMessage."""
    stripped = raw.strip()
    if stripped.startswith("/"):
        parts = stripped.split(None, 1)
        name = parts[0]
        args = parts[1] if len(parts) > 1 else ""
        return SlashCommand(name=name, args=args)
    return UserMessage(text=stripped)


def parse_context(args: str) -> dict[str, str]:
    """Parse 'key1=val1, key2=val2' into a dict."""
    result: dict[str, str] = {}
    for pair in args.split(","):
        pair = pair.strip()
        if "=" in pair:
            key, value = pair.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def build_message_kwargs(
    text: str,
    context_id: str | None,
    context: dict[str, str],
) -> dict:
    """Build kwargs dict for on_message_send."""
    message: dict = {
        "messageId": str(uuid.uuid4()),
        "role": "user",
        "parts": [{"kind": "text", "text": text}],
    }
    if context_id is not None:
        message["contextId"] = context_id
    if context:
        message["metadata"] = dict(context)
    return message


def extract_response_text(result: object) -> str | None:
    """Extract text from a task result's artifacts.

    Handles both object-style (result.artifacts[].parts[].root.text)
    and dict-style (result['artifacts'][]['parts'][]['text']) access.
    """
    # Get artifacts list
    artifacts = getattr(result, "artifacts", None)
    if artifacts is None and isinstance(result, dict):
        artifacts = result.get("artifacts")
    if not artifacts:
        return None

    texts: list[str] = []
    for artifact in artifacts:
        parts = getattr(artifact, "parts", None)
        if parts is None and isinstance(artifact, dict):
            parts = artifact.get("parts")
        if not parts:
            continue
        for part in parts:
            text = None
            # Try part.root.text (SDK object pattern)
            root = getattr(part, "root", None)
            if root is not None:
                text = getattr(root, "text", None)
            # Try part.text (simple object)
            if text is None:
                text = getattr(part, "text", None)
            # Try dict access
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if text:
                texts.append(text)
    return "\n".join(texts) if texts else None


async def run_a2a_chat(
    project: str,
    location: str,
    agent_id: str,
    debug: bool = False,
    base_url: str | None = None,
    api_version: str | None = None,
) -> None:
    """
    Run an interactive A2A chat session with an Agent Engine instance.

    Args:
        project: Google Cloud project ID.
        location: Google Cloud region.
        agent_id: Agent ID or full resource name.
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
        from agent_engine_cli.chat import _install_api_logging_hooks, _setup_debug_logging

        console.print(
            "[yellow]Warning: Debug mode logs HTTP requests/responses which "
            "may include authentication tokens and credentials.[/yellow]"
        )
        _setup_debug_logging()
        _install_api_logging_hooks(debug=True)

    import vertexai

    # Get remote agent
    http_options: dict[str, Any] = {"timeout": 10_000}
    if api_version:
        http_options["api_version"] = api_version
    if base_url:
        http_options["base_url"] = base_url
    client = vertexai.Client(
        project=project, location=location, http_options=http_options
    )
    resource_name = (
        f"projects/{project}/locations/{location}/reasoningEngines/{agent_id}"
    )
    remote_agent = await asyncio.to_thread(
        client.agent_engines.get, name=resource_name
    )

    if debug:
        console.print(f"\n[yellow]Remote agent type:[/yellow] {type(remote_agent)}")
        console.print(f"[yellow]Remote agent repr:[/yellow] {remote_agent!r}")
        a2a_methods = [
            "handle_authenticated_agent_card",
            "on_message_send",
            "on_get_task",
            "on_cancel_task",
        ]
        for method_name in a2a_methods:
            attr = getattr(remote_agent, method_name, "<MISSING>")
            console.print(
                f"[yellow]  {method_name}:[/yellow] {type(attr).__name__} = {attr!r}"
            )
        # Show all public attributes/methods
        public_attrs = [a for a in dir(remote_agent) if not a.startswith("_")]
        console.print(f"[yellow]Public attributes:[/yellow] {public_attrs}\n")

    state = A2AChatState()

    console.print("[green]Ready.[/green] A2A agent connected.\n")
    console.print(
        "Commands:\n"
        "  /get-agent-card          Fetch the agent card\n"
        "  /new-task                Start a new task\n"
        "  /get-task [id]           Fetch a task\n"
        "  /delete-task [id]        Cancel a task\n"
        "  /context [key=val, ...]  Show or set context\n"
        "  /help                    Show full help\n"
    )
    console.print("Type a message or command. 'quit' or 'exit' to end.\n")

    while True:
        try:
            user_input = await asyncio.to_thread(input, "You: ")
        except EOFError:
            break
        except KeyboardInterrupt:
            break

        if user_input.lower() in ("quit", "exit"):
            break

        if not user_input.strip():
            continue

        parsed = parse_input(user_input)

        if isinstance(parsed, SlashCommand):
            await _handle_command(parsed, state, remote_agent, debug)
        else:
            await _handle_message(parsed, state, remote_agent, debug)


def _debug_agent_object(remote_agent: object) -> None:
    """Print diagnostic info about the remote agent object."""
    console.print(f"\n[yellow]--- Agent object debug ---[/yellow]")
    console.print(f"[yellow]Type:[/yellow] {type(remote_agent)}")
    for name in ("on_message_send", "handle_authenticated_agent_card",
                  "on_get_task", "on_cancel_task", "query", "operation"):
        attr = getattr(remote_agent, name, "<MISSING>")
        console.print(f"[yellow]  .{name}:[/yellow] {type(attr).__name__} = {attr!r}")
    public = [a for a in dir(remote_agent) if not a.startswith("_")]
    console.print(f"[yellow]Public attrs:[/yellow] {public}")
    console.print(f"[yellow]--- End debug ---[/yellow]\n")


async def _handle_command(
    cmd: SlashCommand, state: A2AChatState, remote_agent: object, debug: bool
) -> None:
    """Dispatch a slash command."""
    if cmd.name == "/help":
        console.print(HELP_TEXT)

    elif cmd.name == "/get-agent-card":
        try:
            result = await remote_agent.handle_authenticated_agent_card()
            console.print(
                Panel(
                    json.dumps(result, indent=2, default=str),
                    title="Agent Card",
                )
            )
        except (TypeError, AttributeError) as e:
            console.print(f"[red]Error fetching agent card: {escape(str(e))}[/red]")
            console.print(f"[yellow]{traceback.format_exc()}[/yellow]")
            _debug_agent_object(remote_agent)
        except Exception as e:
            console.print(f"[red]Error fetching agent card: {escape(str(e))}[/red]")

    elif cmd.name == "/new-task":
        state.current_task_id = None
        state.current_context_id = None
        console.print("Context and task cleared. Next message will start a new conversation.")

    elif cmd.name == "/get-task":
        task_id = cmd.args.strip() or state.current_task_id
        if not task_id:
            console.print("[yellow]No task ID specified and no current task.[/yellow]")
            return
        try:
            result = await remote_agent.on_get_task(id=task_id)
            if debug:
                console.print(
                    Panel(
                        json.dumps(result, indent=2, default=str),
                        title=f"Task {escape(task_id)} (raw)",
                    )
                )
            text = extract_response_text(result)
            if text:
                console.print(f"\n[cyan]Agent:[/cyan] {escape(text)}")
            else:
                console.print(
                    Panel(
                        json.dumps(result, indent=2, default=str),
                        title=f"Task {escape(task_id)}",
                    )
                )
        except Exception as e:
            console.print(f"[red]Error fetching task: {escape(str(e))}[/red]")

    elif cmd.name == "/delete-task":
        task_id = cmd.args.strip() or state.current_task_id
        if not task_id:
            console.print("[yellow]No task ID specified and no current task.[/yellow]")
            return
        try:
            result = await remote_agent.on_cancel_task(id=task_id)
            console.print(f"Task {escape(task_id)} cancelled: {result}")
            if task_id == state.current_task_id:
                state.current_task_id = None
        except Exception as e:
            console.print(f"[red]Error cancelling task: {escape(str(e))}[/red]")

    elif cmd.name == "/context":
        if not cmd.args.strip():
            if state.context:
                formatted = ", ".join(f"{k}={v}" for k, v in state.context.items())
                console.print(f"Context: {escape(formatted)}")
            else:
                console.print("Context is empty.")
        else:
            state.context = parse_context(cmd.args)
            formatted = ", ".join(f"{k}={v}" for k, v in state.context.items())
            console.print(f"Context set: {escape(formatted)}")

    else:
        console.print(f"[yellow]Unknown command: {escape(cmd.name)}[/yellow]")
        console.print("Type /help for available commands.")


async def _handle_message(
    msg: UserMessage, state: A2AChatState, remote_agent: object, debug: bool
) -> None:
    """Send a user message to the A2A agent."""
    kwargs = build_message_kwargs(msg.text, state.current_context_id, state.context)

    if debug:
        console.print(
            Panel(
                json.dumps(kwargs, indent=2),
                title="Outgoing Message",
            )
        )

    # Send the message — returns an async iterable of streaming chunks
    try:
        response = await remote_agent.on_message_send(**kwargs)
    except (TypeError, AttributeError) as e:
        console.print(f"[red]Error sending message: {escape(str(e))}[/red]")
        console.print(f"[yellow]{traceback.format_exc()}[/yellow]")
        _debug_agent_object(remote_agent)
        return
    except Exception as e:
        console.print(f"[red]Error sending message: {escape(str(e))}[/red]")
        return

    # Extract task object from response (may be async iterable or plain list)
    task_object = None
    try:
        chunks = []
        if hasattr(response, "__aiter__"):
            async for chunk in response:
                chunks.append(chunk)
        elif isinstance(response, (list, tuple)):
            chunks = response
        else:
            chunks = [response]

        for chunk in chunks:
            if debug:
                console.print(f"  [dim]\\[chunk] {chunk}[/dim]")
            if isinstance(chunk, tuple) and len(chunk) > 0 and hasattr(chunk[0], "id"):
                task_object = chunk[0]
            elif hasattr(chunk, "id"):
                task_object = chunk
    except Exception as e:
        console.print(f"[red]Error reading response: {escape(str(e))}[/red]")
        return

    if task_object is None:
        console.print("[yellow]No task object received in response.[/yellow]")
        return

    state.current_task_id = task_object.id
    context_id = getattr(task_object, "context_id", None)
    if context_id:
        state.current_context_id = context_id

    # The task object from on_message_send already contains artifacts with the
    # agent's response — extract text directly without a separate on_get_task call.
    text = extract_response_text(task_object)
    if text:
        console.print(f"\n[cyan]Agent:[/cyan] {escape(text)}")
    else:
        console.print(
            Panel(
                json.dumps(task_object, indent=2, default=str),
                title="Response",
            )
        )
    console.print(f"[dim](task: {escape(task_object.id)})[/dim]\n")
