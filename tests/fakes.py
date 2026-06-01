"""Fake implementations for testing."""

from datetime import datetime
from typing import Any, Iterator
from dataclasses import dataclass
from enum import Enum

from google.cloud.aiplatform_v1beta1.types import ReasoningEngine, ReasoningEngineSpec, Session, Memory


class SandboxState(str, Enum):
    STATE_UNSPECIFIED = "STATE_UNSPECIFIED"
    STATE_RUNNING = "STATE_RUNNING"
    STATE_STOPPED = "STATE_STOPPED"


@dataclass
class Sandbox:
    """Fake Sandbox since the real one is not easily importable."""
    name: str
    display_name: str
    state: Any
    create_time: datetime
    expire_time: datetime


@dataclass
class CreateAgentCall:
    """Records the arguments passed to create_agent."""
    display_name: str
    identity_type: str
    service_account: str | None
    image_uri: str | None = None
    agent_framework: str | None = None


class FakeAgentEngineClient:
    """Fake client for Agent Engine."""

    def __init__(self, project: str, location: str):
        self.project = project
        self.location = location
        self._agents: dict[str, ReasoningEngine] = {}
        self._sessions: dict[str, list[Session]] = {}
        self._sandboxes: dict[str, list[Sandbox]] = {}
        self._memories: dict[str, list[Memory]] = {}
        self.create_agent_calls: list[CreateAgentCall] = []

    def _get_full_name(self, resource_type: str, resource_id: str) -> str:
        return f"projects/{self.project}/locations/{self.location}/{resource_type}/{resource_id}"

    def list_agents(self) -> Iterator[ReasoningEngine]:
        return iter(self._agents.values())

    def get_agent(self, agent_id: str) -> ReasoningEngine:
        if "/" in agent_id:
            name = agent_id
        else:
            name = self._get_full_name("reasoningEngines", agent_id)

        if name in self._agents:
            return self._agents[name]

        for agent_name, agent in self._agents.items():
            if agent_name.endswith(f"/{agent_id}"):
                return agent

        raise Exception(f"Agent {agent_id} not found")

    def create_agent(
        self,
        display_name: str,
        identity_type: str,
        service_account: str | None = None,
        image_uri: str | None = None,
        agent_framework: str | None = None,
    ) -> ReasoningEngine:
        self.create_agent_calls.append(
            CreateAgentCall(display_name=display_name, identity_type=identity_type, service_account=service_account, image_uri=image_uri, agent_framework=agent_framework)
        )

        agent_id = f"agent-{len(self._agents) + 1}"
        name = self._get_full_name("reasoningEngines", agent_id)

        spec = ReasoningEngineSpec(agent_framework="langchain")
        agent = ReasoningEngine(
            name=name,
            display_name=display_name,
            spec=spec,
            create_time=datetime.now(),
            update_time=datetime.now(),
        )

        self._agents[name] = agent
        return agent

    def delete_agent(self, agent_id: str, force: bool = False) -> None:
        if "/" in agent_id:
            name = agent_id
        else:
            name = self._get_full_name("reasoningEngines", agent_id)

        if name not in self._agents:
            for agent_name in list(self._agents.keys()):
                if agent_name.endswith(f"/{agent_id}"):
                    name = agent_name
                    break

        if name in self._agents:
            if not force:
                if self._sessions.get(name) or self._memories.get(name) or self._sandboxes.get(name):
                    raise Exception("Agent has resources, use force to delete")

            del self._agents[name]
            self._sessions.pop(name, None)
            self._sandboxes.pop(name, None)
            self._memories.pop(name, None)
        else:
            raise Exception(f"Agent {agent_id} not found")

    def list_sessions(self, agent_id: str) -> Iterator[Session]:
        agent = self.get_agent(agent_id)
        return iter(self._sessions.get(agent.name, []))

    def list_sandboxes(self, agent_id: str) -> Iterator[Sandbox]:
        agent = self.get_agent(agent_id)
        return iter(self._sandboxes.get(agent.name, []))

    def list_memories(self, agent_id: str) -> Iterator[Memory]:
        agent = self.get_agent(agent_id)
        return iter(self._memories.get(agent.name, []))
