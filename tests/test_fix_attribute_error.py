from unittest.mock import MagicMock, patch
from typer.testing import CliRunner
from agent_engine_cli.main import app

runner = CliRunner(env={"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"})


@patch("agent_engine_cli.main.get_client")
def test_get_agent_with_none_class_methods(mock_get_client):
    """Test get command when spec.class_methods is None (regression test)."""

    # Mock spec with class_methods=None
    mock_spec = MagicMock()
    mock_spec.effective_identity = "test-identity"
    mock_spec.agent_framework = "langchain"
    mock_spec.class_methods = None  # This was causing 'NoneType' object is not iterable

    mock_agent = MagicMock()
    mock_agent.name = "projects/test/locations/us-central1/reasoningEngines/agent1"
    mock_agent.display_name = "Test Agent"
    mock_agent.description = "A test agent"
    mock_agent.create_time = "2024-01-01T00:00:00Z"
    mock_agent.update_time = "2024-01-02T00:00:00Z"
    mock_agent.api_resource.spec = mock_spec
    mock_agent.spec = mock_spec

    mock_client = MagicMock()
    mock_client.get_agent.return_value = mock_agent
    mock_get_client.return_value = mock_client

    result = runner.invoke(
        app, ["--project", "test-project", "--location", "us-central1", "get", "agent1"]
    )

    assert result.exit_code == 0
    assert "Test Agent" in result.stdout
    assert "Class Methods: N/A" in result.stdout
