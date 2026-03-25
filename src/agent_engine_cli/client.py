"""Client wrapper for Vertex AI Agent Engine API."""

from datetime import datetime
from typing import Any, Iterator, Protocol, runtime_checkable

@runtime_checkable
class AgentSpec(Protocol):
    """Agent specification with identity information."""

    effective_identity: str


@runtime_checkable
class AgentResource(Protocol):
    """Agent resource from Vertex AI API."""

    name: str
    display_name: str
    create_time: datetime
    update_time: datetime
    spec: AgentSpec | None


class AgentEngineClient:
    """Client for interacting with Vertex AI Agent Engine."""

    def __init__(self, project: str, location: str, *, base_url: str | None = None, api_version: str | None = None):
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

        http_options: dict[str, Any] = {}
        if api_version:
            http_options["api_version"] = api_version
        if base_url:
            http_options["base_url"] = base_url

        self._client = vertexai.Client(
            project=project,
            location=location,
            http_options=http_options or None,
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
            raise ValueError(
                f"agent_id contains invalid characters: {agent_id!r}"
            )
        if "/" not in agent_id:
            return (
                f"projects/{self.project}/locations/{self.location}/"
                f"reasoningEngines/{agent_id}"
            )
        return agent_id

    def list_agents(self) -> Iterator[AgentResource]:
        """List all agents in the project.

        Returns:
            Iterator of AgentEngine api_resource instances (v1beta1)
        """
        return (agent.api_resource for agent in self._client.agent_engines.list())

    def get_agent(self, agent_id: str) -> AgentResource:
        """Get details for a specific agent.

        Args:
            agent_id: The agent resource ID or full resource name

        Returns:
            AgentEngine instance with agent details
        """
        resource_name = self._resolve_resource_name(agent_id)

        agent = self._client.agent_engines.get(name=resource_name)
        return getattr(agent, "api_resource", agent)

    def create_agent(
        self,
        display_name: str,
        identity_type: str,
        service_account: str | None = None,
    ) -> AgentResource:
        """Create a new agent without deploying code.

        Args:
            display_name: Human-readable name for the agent
            identity_type: Identity type ('agent_identity' or 'service_account')
            service_account: Service account email (only used with service_account identity)

        Returns:
            The created agent's api_resource
        """
        from vertexai import types

        config = {
            "display_name": display_name,
        }

        if identity_type == "agent_identity":
            config["identity_type"] = types.IdentityType.AGENT_IDENTITY
        elif identity_type == "service_account":
            config["identity_type"] = types.IdentityType.SERVICE_ACCOUNT
            if service_account:
                config["service_account"] = service_account

        result = self._client.agent_engines.create(config=config)
        return result.api_resource

    def delete_agent(self, agent_id: str, force: bool = False) -> None:
        """Delete an agent.

        Args:
            agent_id: The agent resource ID or full resource name
            force: Force deletion even if agent has associated resources
        """
        resource_name = self._resolve_resource_name(agent_id)
        self._client.agent_engines.delete(name=resource_name, force=force)

    def list_sessions(self, agent_id: str) -> Iterator:
        """List all sessions for an agent.

        Args:
            agent_id: The agent resource ID or full resource name

        Returns:
            Iterator of session objects
        """
        resource_name = self._resolve_resource_name(agent_id)

        return self._client.agent_engines.list_sessions(name=resource_name)

    def list_sandboxes(self, agent_id: str) -> Iterator[Any]:
        """List all sandboxes for an agent.

        Args:
            agent_id: The agent resource ID or full resource name

        Returns:
            Iterator of sandbox objects
        """
        resource_name = self._resolve_resource_name(agent_id)

        return self._client.agent_engines.sandboxes.list(name=resource_name)

    def list_memories(self, agent_id: str) -> Iterator[Any]:
        """List all memories for an agent.

        Args:
            agent_id: The agent resource ID or full resource name

        Returns:
            Iterator of memory objects
        """
        resource_name = self._resolve_resource_name(agent_id)

        return self._client.agent_engines.memories.list(name=resource_name)
