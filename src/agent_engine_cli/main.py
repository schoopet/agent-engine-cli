import asyncio
import json
import logging
from collections.abc import MutableMapping
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal

import typer
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from agent_engine_cli import __version__
from agent_engine_cli.client import AgentEngineClient
from agent_engine_cli.config import resolve_project
from agent_engine_cli.console import console
from agent_engine_cli.dependencies import get_client

logger = logging.getLogger(__name__)


@dataclass
class State:
    location: str | None = None
    project: str | None = None
    base_url: str | None = None
    api_version: str | None = None


state = State()


def _parse_resource_name(agent_id: str) -> tuple[str, str] | None:
    """Extract (project, location) from a full resource name, or None if not a full name.

    Expects: projects/{project}/locations/{location}/reasoningEngines/{id}
    """
    parts = agent_id.split("/")
    if (
        len(parts) == 6
        and parts[0] == "projects"
        and parts[2] == "locations"
        and parts[4] == "reasoningEngines"
    ):
        return parts[1], parts[3]
    return None


def _resolve_config(agent_id: str | None = None) -> tuple[str, str]:
    """Validate location and resolve project, returning (project, location).

    If agent_id is a full resource name, project and location are extracted from it
    as fallbacks when --project / --location flags were not provided.
    """
    parsed = _parse_resource_name(agent_id) if agent_id else None
    fallback_project = parsed[0] if parsed else None
    fallback_location = parsed[1] if parsed else None

    location = state.location or fallback_location
    if not location:
        console.print(
            "[red]Error: Location is required. Use --location or set it globally.[/red]"
        )
        raise typer.Exit(code=1)

    try:
        project = resolve_project(state.project or fallback_project)
    except Exception as e:
        console.print(f"[red]Error: {escape(str(e))}[/red]")
        raise typer.Exit(code=1)

    return project, location


def get_ready_client(agent_id: str | None = None) -> AgentEngineClient:
    """Helper to resolve project and return an initialized client."""
    project, location = _resolve_config(agent_id)
    return get_client(
        project=project,
        location=location,
        base_url=state.base_url,
        api_version=state.api_version,
    )


def get_id(resource: object) -> str:
    """Extract ID from a resource name string or object."""
    name = (
        resource
        if isinstance(resource, str)
        else (getattr(resource, "name", None) or getattr(resource, "resource_name", ""))
    )
    return name.split("/")[-1] if name else ""


def format_timestamp(value: object) -> str:
    """Format a timestamp value for display, returning '' if unavailable."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    # Handle proto Timestamp objects that have a strftime-like interface
    strftime_fn = getattr(value, "strftime", None)
    if strftime_fn is not None:
        return strftime_fn("%Y-%m-%d %H:%M")
    return ""


def _get_agent_spec(agent: object) -> object | None:
    """Extract the spec from an agent, checking api_resource first."""
    api_resource = getattr(agent, "api_resource", None)
    if api_resource is not None:
        spec = getattr(api_resource, "spec", None)
        if spec:
            return spec
    spec = getattr(agent, "spec", None)
    if spec:
        return spec
    return None


def _format_class_methods(spec: object) -> tuple[str, str]:
    """Extract formatted class methods and agent card from a spec.

    Returns:
        A tuple of (class_methods_str, agent_card_str) with "N/A" as defaults.
    """
    raw_methods = getattr(spec, "class_methods", []) or []
    method_names: list[str] = []
    agent_card = "N/A"

    for m in raw_methods:
        try:
            m_name = (
                getattr(m, "name", None)
                or getattr(m, "method", None)
                or (m.get("name") if hasattr(m, "get") else None)
                or (m.get("method") if hasattr(m, "get") else None)
            )
            if not m_name:
                continue

            # Extract parameters and their types
            m_params = getattr(m, "parameters", None) or (
                m.get("parameters") if hasattr(m, "get") else None
            )

            if m_params and isinstance(m_params, dict):
                properties = m_params.get("properties", {})
                required = m_params.get("required", [])
                p_list = []
                for p, p_info in properties.items():
                    p_type = ""
                    if isinstance(p_info, dict):
                        p_type = p_info.get("type", "")
                        if not p_type and "anyOf" in p_info:
                            types = [
                                t.get("type", "any")
                                for t in p_info["anyOf"]
                                if isinstance(t, dict)
                            ]
                            p_type = "|".join(types)

                    p_str = f"{p}: {p_type}" if p_type else p
                    if p in required:
                        p_list.append(f"{p_str}*")
                    else:
                        p_list.append(p_str)
                method_names.append(f"{m_name}({', '.join(p_list)})")
            else:
                method_names.append(str(m_name))

            # Extract and add description
            m_desc = getattr(m, "description", None) or (
                m.get("description") if hasattr(m, "get") else None
            )
            if m_desc:
                m_desc_clean = m_desc.strip().split("\n")[0]
                method_names.append(f"    {m_desc_clean}")

            if agent_card == "N/A":
                m_metadata = getattr(m, "metadata", None) or (
                    m.get("metadata") if hasattr(m, "get") else None
                )
                if m_metadata:
                    card = getattr(m_metadata, "get", lambda k, d: None)(
                        "a2a_agent_card", None
                    ) or getattr(m_metadata, "get", lambda k, d: None)(
                        "agent_card", None
                    )
                    if card:
                        agent_card = card
        except (TypeError, KeyError, AttributeError) as e:
            logger.debug("Skipping malformed class method entry: %s", e)

    class_methods = ("\n  " + "\n  ".join(method_names)) if method_names else "N/A"
    return class_methods, agent_card


app = typer.Typer(
    help="Agent Engine CLI - Manage your agents with ease.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main(
    location: Annotated[
        str | None,
        typer.Option(
            "--location", "-l", help="Google Cloud region", envvar="CLOUD_SDK_LOCATION"
        ),
    ] = None,
    project: Annotated[
        str | None,
        typer.Option(
            "--project",
            "-p",
            help="Google Cloud project ID (defaults to ADC project)",
            envvar="CLOUD_SDK_PROJECT",
        ),
    ] = None,
    base_url: Annotated[
        str | None, typer.Option("--base-url", help="Override the Vertex AI base URL")
    ] = None,
    api_version: Annotated[
        str | None, typer.Option("--api-version", help="Override the API version")
    ] = None,
):
    """
    Agent Engine CLI - Manage your agents with ease.
    """
    state.location = location
    state.project = project
    state.base_url = base_url
    state.api_version = api_version


@app.command()
def version():
    """Show the CLI version."""
    print(f"Agent Engine CLI v{__version__}")


@app.command("list")
def list_agents() -> None:
    """List all agents in the project."""
    client = get_ready_client()
    try:
        agents = list(client.list_agents())

        if not agents:
            console.print("No agents found.")
            return

        table = Table(title="Agents")
        table.add_column("Name", style="cyan")
        table.add_column("Display Name", style="green")
        table.add_column("Created")
        table.add_column("Updated")
        table.add_column("Identity", overflow="fold")

        for agent in agents:
            name = get_id(agent)
            display_name = getattr(agent, "display_name", "") or ""
            create_time = format_timestamp(getattr(agent, "create_time", None))
            update_time = format_timestamp(getattr(agent, "update_time", None))

            effective_identity = "N/A"
            spec = getattr(agent, "spec", None)
            if spec:
                effective_identity = getattr(spec, "effective_identity", "N/A")

            table.add_row(
                escape(name),
                escape(display_name),
                create_time,
                update_time,
                escape(effective_identity),
            )

        console.print(table)
    except Exception as e:
        console.print(f"[red]Error listing agents: {escape(str(e))}[/red]")
        raise typer.Exit(code=1)


@app.command("get")
def get_agent(
    agent_id: Annotated[str, typer.Argument(help="Agent ID or full resource name")],
    full: Annotated[
        bool, typer.Option("--full", "-f", help="Show full JSON output")
    ] = False,
) -> None:
    """Get details for a specific agent."""
    client = get_ready_client(agent_id)
    try:
        agent = client.get_agent(agent_id)

        if full:
            agent_dict = {
                "resource_name": getattr(agent, "name", None)
                or getattr(agent, "resource_name", ""),
                "display_name": getattr(agent, "display_name", None),
                "description": getattr(agent, "description", None),
                "create_time": str(getattr(agent, "create_time", None)),
                "update_time": str(getattr(agent, "update_time", None)),
            }
            spec = _get_agent_spec(agent)
            if spec:
                agent_dict["spec"] = str(spec)
            console.print(json.dumps(agent_dict, indent=2, default=str))
        else:
            name = get_id(agent)
            display_name = getattr(agent, "display_name", "") or "N/A"
            description = getattr(agent, "description", "") or "N/A"
            create_time = format_timestamp(getattr(agent, "create_time", None)) or "N/A"
            update_time = format_timestamp(getattr(agent, "update_time", None)) or "N/A"

            effective_identity = "N/A"
            agent_framework_str = "N/A"
            class_methods = "N/A"
            agent_card = "N/A"

            spec = _get_agent_spec(agent)
            if spec:
                effective_identity = getattr(spec, "effective_identity", "N/A")
                agent_framework_str = getattr(spec, "agent_framework", "N/A")
                class_methods, agent_card = _format_class_methods(spec)

            content = (
                f"[bold]Name:[/bold] {escape(name)}\n"
                f"[bold]Display Name:[/bold] {escape(display_name)}\n"
                f"[bold]Description:[/bold] {escape(description)}\n"
                f"[bold]Created:[/bold] {create_time}\n"
                f"[bold]Updated:[/bold] {update_time}\n"
                f"[bold]Effective Identity:[/bold] {escape(effective_identity)}\n"
                f"[bold]Agent Framework:[/bold] {escape(str(agent_framework_str))}\n"
                f"[bold]Class Methods:[/bold] {escape(class_methods)}\n"
                f"[bold]Agent Card:[/bold] {escape(str(agent_card))}"
            )
            console.print(Panel(content, title="Agent Details"))
    except Exception as e:
        console.print(f"[red]Error getting agent: {escape(str(e))}[/red]")
        raise typer.Exit(code=1)


@app.command("create")
def create_agent(
    display_name: Annotated[str, typer.Argument(help="Display name for the agent")],
    identity: Annotated[
        Literal["agent_identity", "service_account"],
        typer.Option("--identity", "-i", help="Identity type for the agent"),
    ] = "agent_identity",
    service_account: Annotated[
        str | None,
        typer.Option(
            "--service-account",
            "-s",
            help="Service account email (only used with --identity service_account)",
        ),
    ] = None,
    image_uri: Annotated[
        str | None,
        typer.Option(
            "--image-uri",
            help="Artifact Registry image URI to deploy as the agent container (e.g. us-central1-docker.pkg.dev/my-project/my-repo/my-image:tag)",
        ),
    ] = None,
    agent_framework: Annotated[
        str | None,
        typer.Option(
            "--agent-framework",
            help="OSS framework used to build the agent. One of: google-adk, langchain, langgraph, ag2, llama-index, custom",
        ),
    ] = None,
) -> None:
    """Create a new agent (without deploying code)."""
    client = get_ready_client()
    try:
        console.print(f"Creating agent '{escape(display_name)}'...")

        agent = client.create_agent(
            display_name=display_name,
            identity_type=identity,
            service_account=service_account,
            image_uri=image_uri,
            agent_framework=agent_framework,
        )

        name = get_id(agent)
        resource_name = getattr(agent, "name", None) or getattr(
            agent, "resource_name", ""
        )
        console.print("[green]Agent created successfully![/green]")
        console.print(f"Name: {name}")
        console.print(f"Resource: {resource_name}")
    except Exception as e:
        console.print(f"[red]Error creating agent: {escape(str(e))}[/red]")
        raise typer.Exit(code=1)


@app.command("delete")
def delete_agent(
    agent_id: Annotated[str, typer.Argument(help="Agent ID or full resource name")],
    force: Annotated[
        bool,
        typer.Option(
            "--force", "-f", help="Force deletion of agents with sessions/memory"
        ),
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Skip confirmation prompt")
    ] = False,
) -> None:
    """Delete an agent."""
    if not yes:
        confirm = typer.confirm(f"Are you sure you want to delete agent '{agent_id}'?")
        if not confirm:
            console.print("Aborted.")
            raise typer.Exit()

    client = get_ready_client(agent_id)
    try:
        client.delete_agent(agent_id, force=force)
        console.print(f"[red]Agent '{escape(agent_id)}' deleted.[/red]")
    except Exception as e:
        console.print(f"[red]Error deleting agent: {escape(str(e))}[/red]")
        raise typer.Exit(code=1)


# Create sessions subcommand group
sessions_app = typer.Typer(help="Manage agent sessions.")
app.add_typer(sessions_app, name="sessions")


@sessions_app.command("list")
def list_sessions(
    agent_id: Annotated[str, typer.Argument(help="Agent ID or full resource name")],
) -> None:
    """List all sessions for an agent."""
    client = get_ready_client(agent_id)
    try:
        sessions = client.list_sessions(agent_id)

        table = Table(title="Sessions")
        table.add_column("Session ID", style="cyan")
        table.add_column("Display Name", style="green")
        table.add_column("User ID")
        table.add_column("Created")
        table.add_column("Expires")

        has_sessions = False
        for session in sessions:
            has_sessions = True
            session_id = get_id(session)
            display_name = getattr(session, "display_name", "") or ""
            user_id = getattr(session, "user_id", "") or ""
            create_time = format_timestamp(getattr(session, "create_time", None))
            expire_time = format_timestamp(getattr(session, "expire_time", None))

            table.add_row(
                escape(session_id),
                escape(display_name),
                escape(user_id),
                create_time,
                expire_time,
            )

        if not has_sessions:
            console.print("No sessions found.")
            return

        console.print(table)
    except Exception as e:
        console.print(f"[red]Error listing sessions: {escape(str(e))}[/red]")
        raise typer.Exit(code=1)


# Create sandboxes subcommand group
sandboxes_app = typer.Typer(help="Manage agent sandboxes.")
app.add_typer(sandboxes_app, name="sandboxes")


@sandboxes_app.command("list")
def list_sandboxes(
    agent_id: Annotated[str, typer.Argument(help="Agent ID or full resource name")],
) -> None:
    """List all sandboxes for an agent."""
    client = get_ready_client(agent_id)
    try:
        sandboxes = list(client.list_sandboxes(agent_id))

        if not sandboxes:
            console.print("No sandboxes found.")
            return

        table = Table(title="Sandboxes")
        table.add_column("Sandbox ID", style="cyan")
        table.add_column("Display Name", style="green")
        table.add_column("State")
        table.add_column("Created")
        table.add_column("Expires")

        for sandbox in sandboxes:
            sandbox_id = get_id(sandbox)
            display_name = getattr(sandbox, "display_name", "") or ""

            # Format state (remove STATE_ prefix if present)
            sandbox_state_raw = getattr(sandbox, "state", None)
            if sandbox_state_raw:
                sandbox_state = (
                    str(sandbox_state_raw.value).replace("STATE_", "")
                    if hasattr(sandbox_state_raw, "value")
                    else str(sandbox_state_raw)
                )
            else:
                sandbox_state = ""

            create_time = format_timestamp(getattr(sandbox, "create_time", None))
            expire_time = format_timestamp(getattr(sandbox, "expire_time", None))

            table.add_row(
                escape(sandbox_id),
                escape(display_name),
                sandbox_state,
                create_time,
                expire_time,
            )

        console.print(table)
    except Exception as e:
        console.print(f"[red]Error listing sandboxes: {escape(str(e))}[/red]")
        raise typer.Exit(code=1)


# Create memories subcommand group
memories_app = typer.Typer(help="Manage agent memories.")
app.add_typer(memories_app, name="memories")


@memories_app.command("list")
def list_memories(
    agent_id: Annotated[str, typer.Argument(help="Agent ID or full resource name")],
) -> None:
    """List all memories for an agent."""
    client = get_ready_client(agent_id)
    try:
        memories = list(client.list_memories(agent_id))

        table = Table(title="Memories")
        table.add_column("Memory ID", style="cyan")
        table.add_column("Display Name", style="green")
        table.add_column("Scope")
        table.add_column("Fact", max_width=40, overflow="ellipsis")
        table.add_column("Created")
        table.add_column("Expires")

        has_items = False
        for memory in memories:
            has_items = True
            memory_id = get_id(memory)
            display_name = getattr(memory, "display_name", "") or ""
            fact = getattr(memory, "fact", "") or ""

            # Format scope dict as key=value pairs
            scope_raw = getattr(memory, "scope", None)
            if scope_raw and isinstance(scope_raw, (dict, MutableMapping)):
                scope = ", ".join(f"{k}={v}" for k, v in scope_raw.items())
            else:
                scope = ""

            create_time = format_timestamp(getattr(memory, "create_time", None))
            expire_time = format_timestamp(getattr(memory, "expire_time", None))

            table.add_row(
                escape(memory_id),
                escape(display_name),
                escape(scope),
                escape(fact),
                create_time,
                expire_time,
            )

        if not has_items:
            console.print("No memories found.")
            return

        console.print(table)
    except Exception as e:
        console.print(f"[red]Error listing memories: {escape(str(e))}[/red]")
        raise typer.Exit(code=1)


@app.command("chat")
def chat(
    agent_id: Annotated[str, typer.Argument(help="Agent ID or full resource name")],
    user: Annotated[
        str, typer.Option("--user", "-u", help="User ID for the chat session")
    ] = "cli-user",
    debug: Annotated[
        bool, typer.Option("--debug", "-d", help="Enable verbose HTTP debug logging")
    ] = False,
) -> None:
    """Start an interactive chat session with an agent."""
    project, location = _resolve_config(agent_id)

    try:
        from agent_engine_cli.chat import run_chat

        asyncio.run(
            run_chat(
                project=project,
                location=location,
                agent_id=agent_id,
                user_id=user,
                debug=debug,
                base_url=state.base_url,
                api_version=state.api_version,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Chat session ended.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error in chat session: {escape(str(e))}[/red]")
        raise typer.Exit(code=1)


@app.command("a2a-chat")
def a2a_chat(
    agent_id: Annotated[str, typer.Argument(help="Agent ID or full resource name")],
    debug: Annotated[
        bool, typer.Option("--debug", "-d", help="Enable verbose HTTP debug logging")
    ] = False,
) -> None:
    """Start an interactive A2A chat session with an agent."""
    project, location = _resolve_config(agent_id)

    try:
        from agent_engine_cli.a2a_chat import run_a2a_chat

        asyncio.run(
            run_a2a_chat(
                project=project,
                location=location,
                agent_id=agent_id,
                debug=debug,
                base_url=state.base_url,
                api_version=state.api_version,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Chat session ended.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error in A2A chat session: {escape(str(e))}[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
