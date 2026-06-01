"""Client wrapper for Vertex AI Agent Engine API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from vertexai._genai.client import Client as _VertexClient
    from vertexai._genai.types.common import (
        Memory,
        ReasoningEngine,
        SandboxEnvironment,
        Session,
    )


def resolve_resource_name(project: str, location: str, agent_id: str) -> str:
    """Resolve an agent ID to a full resource name.

    If agent_id already contains a '/', it is returned as-is (assumed to be
    a full resource name).  Otherwise it is expanded to
    ``projects/{project}/locations/{location}/reasoningEngines/{agent_id}``.
    """
    if "/" in agent_id:
        return agent_id
    return f"projects/{project}/locations/{location}/reasoningEngines/{agent_id}"


class AgentEngineClient:
    """Client for interacting with Vertex AI Agent Engine."""

    def __init__(
        self,
        project: str,
        location: str,
        *,
        base_url: str | None = None,
        api_version: str | None = None,
    ):
        """Initialize the client with project and location.

        Args:
            project: Google Cloud project ID
            location: Google Cloud region
            base_url: Optional override for the Vertex AI base URL
            api_version: Optional API version override
        """
        self.project = project
        self.location = location
        # Import vertexai locally to improve CLI startup time
        import vertexai

        http_options: dict[str, str] = {}
        if api_version:
            http_options["api_version"] = api_version
        if base_url:
            http_options["base_url"] = base_url

        # vertexai.Client is lazy-exported via __getattr__; use getattr
        # so the type checker doesn't flag the dynamic attribute lookup.
        client_cls: type[_VertexClient] = getattr(vertexai, "Client")
        self._client = client_cls(
            project=project,
            location=location,
            http_options=http_options or None,  # type: ignore[arg-type]
        )

    def _resolve_resource_name(self, agent_id: str) -> str:
        """Resolve an agent ID or full resource name to a full resource name.

        Args:
            agent_id: The agent resource ID or full resource name.

        Returns:
            The full resource name.

        Raises:
            ValueError: If agent_id is empty or contains invalid characters.
        """
        if not agent_id or not agent_id.strip():
            raise ValueError("agent_id must not be empty")
        if any(c.isspace() or ord(c) < 32 for c in agent_id):
            raise ValueError(f"agent_id contains invalid characters: {agent_id!r}")
        if "/" not in agent_id:
            return (
                f"projects/{self.project}/locations/{self.location}/"
                f"reasoningEngines/{agent_id}"
            )
        return agent_id

    def list_agents(self) -> Iterator[ReasoningEngine]:
        """List all agents in the project.

        Returns:
            Iterator of ReasoningEngine instances
        """
        return (agent.api_resource for agent in self._client.agent_engines.list())

    def get_agent(self, agent_id: str) -> ReasoningEngine:
        """Get details for a specific agent.

        Args:
            agent_id: The agent resource ID or full resource name

        Returns:
            ReasoningEngine with agent details
        """
        resource_name = self._resolve_resource_name(agent_id)
        agent = self._client.agent_engines.get(name=resource_name)
        return agent.api_resource

    def create_agent(
        self,
        display_name: str,
        identity_type: str,
        service_account: str | None = None,
        image_uri: str | None = None,
        agent_framework: str | None = None,
    ) -> ReasoningEngine:
        """Create a new agent without deploying code.

        Args:
            display_name: Human-readable name for the agent
            identity_type: Identity type ('agent_identity' or 'service_account')
            service_account: Service account email (only used with service_account identity)
            image_uri: Artifact Registry image URI to deploy as the agent container
            agent_framework: OSS framework used to build the agent

        Returns:
            The created agent's ReasoningEngine
        """
        from vertexai import types

        config: dict[str, Any] = {
            "display_name": display_name,
        }

        if identity_type == "agent_identity":
            config["identity_type"] = types.IdentityType.AGENT_IDENTITY
        elif identity_type == "service_account":
            config["identity_type"] = types.IdentityType.SERVICE_ACCOUNT
            if service_account:
                config["service_account"] = service_account

        if image_uri:
            config["container_spec"] = {"image_uri": image_uri}

        if agent_framework:
            config["agent_framework"] = agent_framework

        result = self._client.agent_engines.create(config=config)  # type: ignore[arg-type]
        return result.api_resource

    def delete_agent(self, agent_id: str, force: bool = False) -> None:
        """Delete an agent.

        Args:
            agent_id: The agent resource ID or full resource name
            force: Force deletion even if agent has associated resources
        """
        resource_name = self._resolve_resource_name(agent_id)
        self._client.agent_engines.delete(name=resource_name, force=force)

    def list_sessions(self, agent_id: str) -> Iterator[Session]:
        """List all sessions for an agent.

        Args:
            agent_id: The agent resource ID or full resource name

        Returns:
            Iterator of Session instances
        """
        resource_name = self._resolve_resource_name(agent_id)

        return self._client.agent_engines.list_sessions(name=resource_name)

    def list_sandboxes(self, agent_id: str) -> Iterator[SandboxEnvironment]:
        """List all sandboxes for an agent.

        Args:
            agent_id: The agent resource ID or full resource name

        Returns:
            Iterator of SandboxEnvironment instances
        """
        resource_name = self._resolve_resource_name(agent_id)

        return self._client.agent_engines.sandboxes.list(name=resource_name)

    def list_memories(self, agent_id: str) -> Iterator[Memory]:
        """List all memories for an agent.

        Args:
            agent_id: The agent resource ID or full resource name

        Returns:
            Iterator of Memory instances
        """
        resource_name = self._resolve_resource_name(agent_id)

        return self._client.agent_engines.memories.list(name=resource_name)
