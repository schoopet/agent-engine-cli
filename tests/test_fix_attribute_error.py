"""Regression test: spec.class_methods=None must not crash."""

from datetime import datetime
from unittest.mock import patch

from typer.testing import CliRunner
from vertexai._genai.types.common import ReasoningEngine, ReasoningEngineSpec

from agent_engine_cli.main import app

runner = CliRunner(env={"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"})


@patch("agent_engine_cli.main.get_client")
def test_get_agent_with_none_class_methods(mock_get_client):
    """Test get command when spec.class_methods is None (regression test)."""
    from tests.fakes import FakeAgentEngineClient

    fake_client = FakeAgentEngineClient(project="test-project", location="us-central1")
    mock_get_client.return_value = fake_client

    agent_name = "projects/test-project/locations/us-central1/reasoningEngines/agent1"
    fake_client._agents[agent_name] = ReasoningEngine(
        name=agent_name,
        display_name="Test Agent",
        description="A test agent",
        spec=ReasoningEngineSpec(
            effective_identity="test-identity",
            agent_framework="langchain",
            class_methods=None,
        ),
        create_time=datetime(2024, 1, 1),
        update_time=datetime(2024, 1, 2),
    )

    result = runner.invoke(
        app, ["--project", "test-project", "--location", "us-central1", "get", "agent1"]
    )

    assert result.exit_code == 0
    assert "Test Agent" in result.stdout
    assert "Class Methods: N/A" in result.stdout
