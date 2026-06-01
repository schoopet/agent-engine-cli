"""Tests for CLI commands."""

from datetime import datetime
from unittest.mock import patch

from typer.testing import CliRunner
from vertexai._genai.types.common import (
    Memory,
    ReasoningEngine,
    ReasoningEngineSpec,
    SandboxEnvironment,
    SandboxState,
    Session,
)

from agent_engine_cli import __version__
from agent_engine_cli.config import ConfigurationError
from agent_engine_cli.main import (
    _format_class_methods,
    _parse_resource_name,
    app,
    format_timestamp,
    get_id,
)
from tests.fakes import CreateAgentCall, FakeAgentEngineClient

runner = CliRunner(env={"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"})


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert f"Agent Engine CLI v{__version__}" in result.stdout


def test_global_options_in_help():
    """Test that global options appear in the main help."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "--project" in result.stdout
    assert "--location" in result.stdout
    assert "--base-url" in result.stdout
    assert "--api-version" in result.stdout


class TestListCommand:
    @patch("agent_engine_cli.main.get_client")
    def test_list_no_agents(self, mock_get_client):
        """Test list command with no agents."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        # Global options must come before the command in some Typer configurations,
        # but here we can just ensure they are handled correctly.
        result = runner.invoke(
            app, ["--project", "test-project", "--location", "us-central1", "list"]
        )
        assert result.exit_code == 0
        assert "No agents found" in result.stdout

    @patch("agent_engine_cli.main.get_client")
    def test_list_with_agents(self, mock_get_client):
        """Test list command with agents."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        agent = fake_client.create_agent(
            display_name="Test Agent", identity_type="agent_identity"
        )
        agent.create_time = datetime(2024, 1, 1, 12, 30, 0)
        agent.update_time = datetime(2024, 1, 2, 14, 45, 0)

        result = runner.invoke(
            app, ["--project", "test-project", "--location", "us-central1", "list"]
        )
        assert result.exit_code == 0
        assert "agent-1" in result.stdout
        assert "Test Agent" in result.stdout
        assert "2024-01-01" in result.stdout


class TestGetCommand:
    def test_get_help(self):
        """Test get command help."""
        result = runner.invoke(app, ["get", "--help"])
        assert result.exit_code == 0
        assert "--full" in result.stdout

    @patch("agent_engine_cli.main.get_client")
    def test_get_agent(self, mock_get_client):
        """Test get command."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        agent = fake_client.create_agent(
            display_name="Test Agent", identity_type="agent_identity"
        )
        agent.description = "A test agent"

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "get",
                agent.name.split("/")[-1],
            ],
        )
        assert result.exit_code == 0
        assert "Test Agent" in result.stdout
        assert "A test agent" in result.stdout
        assert "Agent Framework: langchain" in result.stdout

    @patch("agent_engine_cli.main.get_client")
    def test_get_agent_full_output(self, mock_get_client):
        """Test get command with --full flag."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        agent = fake_client.create_agent(
            display_name="Test Agent", identity_type="agent_identity"
        )
        agent.description = "A test agent"

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "get",
                agent.name.split("/")[-1],
                "--full",
            ],
        )
        assert result.exit_code == 0
        assert "resource_name" in result.stdout


class TestCreateCommand:
    def test_create_help(self):
        """Test create command help."""
        result = runner.invoke(app, ["create", "--help"])
        assert result.exit_code == 0
        assert "--identity" in result.stdout

    @patch("agent_engine_cli.main.get_client")
    def test_create_agent(self, mock_get_client):
        """Test create command with default agent_identity type."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "create",
                "My Agent",
            ],
        )
        assert result.exit_code == 0
        assert "Agent created successfully" in result.stdout
        assert "agent-1" in result.stdout

        assert len(fake_client.create_agent_calls) == 1
        assert fake_client.create_agent_calls[0] == CreateAgentCall(
            display_name="My Agent",
            identity_type="agent_identity",
            service_account=None,
        )

    @patch("agent_engine_cli.main.get_client")
    def test_create_agent_with_service_account_identity(self, mock_get_client):
        """Test create command with service_account identity type."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "create",
                "My Agent",
                "--identity",
                "service_account",
            ],
        )
        assert result.exit_code == 0

        assert len(fake_client.create_agent_calls) == 1
        assert fake_client.create_agent_calls[0] == CreateAgentCall(
            display_name="My Agent",
            identity_type="service_account",
            service_account=None,
        )

    @patch("agent_engine_cli.main.get_client")
    def test_create_agent_with_custom_service_account(self, mock_get_client):
        """Test create command with a specific service account."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "create",
                "My Agent",
                "--identity",
                "service_account",
                "--service-account",
                "my-sa@proj.iam.gserviceaccount.com",
            ],
        )
        assert result.exit_code == 0

        assert len(fake_client.create_agent_calls) == 1
        assert fake_client.create_agent_calls[0] == CreateAgentCall(
            display_name="My Agent",
            identity_type="service_account",
            service_account="my-sa@proj.iam.gserviceaccount.com",
        )

    @patch("agent_engine_cli.main.get_client")
    def test_create_agent_with_image_uri(self, mock_get_client):
        """Test create command with a container image URI."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "create",
                "My Agent",
                "--image-uri",
                "us-central1-docker.pkg.dev/my-project/my-repo/my-image:latest",
            ],
        )
        assert result.exit_code == 0

        assert len(fake_client.create_agent_calls) == 1
        assert fake_client.create_agent_calls[0] == CreateAgentCall(
            display_name="My Agent",
            identity_type="agent_identity",
            service_account=None,
            image_uri="us-central1-docker.pkg.dev/my-project/my-repo/my-image:latest",
        )

    @patch("agent_engine_cli.main.get_client")
    def test_create_agent_with_framework(self, mock_get_client):
        """Test create command with an agent framework."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "create",
                "My Agent",
                "--agent-framework",
                "google-adk",
            ],
        )
        assert result.exit_code == 0

        assert len(fake_client.create_agent_calls) == 1
        assert fake_client.create_agent_calls[0] == CreateAgentCall(
            display_name="My Agent",
            identity_type="agent_identity",
            service_account=None,
            agent_framework="google-adk",
        )

    @patch("agent_engine_cli.main.get_client")
    def test_create_agent_with_image_uri_and_framework(self, mock_get_client):
        """Test create command combining --image-uri and --agent-framework."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "create",
                "My Agent",
                "--image-uri",
                "us-central1-docker.pkg.dev/my-project/my-repo/my-image:latest",
                "--agent-framework",
                "langchain",
            ],
        )
        assert result.exit_code == 0

        assert len(fake_client.create_agent_calls) == 1
        assert fake_client.create_agent_calls[0] == CreateAgentCall(
            display_name="My Agent",
            identity_type="agent_identity",
            service_account=None,
            image_uri="us-central1-docker.pkg.dev/my-project/my-repo/my-image:latest",
            agent_framework="langchain",
        )


class TestDeleteCommand:
    def test_delete_help(self):
        """Test delete command help."""
        result = runner.invoke(app, ["delete", "--help"])
        assert result.exit_code == 0
        assert "--force" in result.stdout
        assert "--yes" in result.stdout

    @patch("agent_engine_cli.main.get_client")
    def test_delete_agent_with_confirmation(self, mock_get_client):
        """Test delete command with confirmation prompt."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        agent_name = (
            "projects/test-project/locations/us-central1/reasoningEngines/agent123"
        )
        fake_client._agents[agent_name] = ReasoningEngine(name=agent_name)

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "delete",
                "agent123",
            ],
            input="y\n",
        )
        assert result.exit_code == 0
        assert "deleted" in result.stdout
        assert agent_name not in fake_client._agents

    @patch("agent_engine_cli.main.get_client")
    def test_delete_agent_abort(self, mock_get_client):
        """Test delete command when user aborts."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        agent_name = (
            "projects/test-project/locations/us-central1/reasoningEngines/agent123"
        )
        fake_client._agents[agent_name] = ReasoningEngine(name=agent_name)

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "delete",
                "agent123",
            ],
            input="n\n",
        )
        assert result.exit_code == 0
        assert "Aborted" in result.stdout
        assert agent_name in fake_client._agents

    @patch("agent_engine_cli.main.get_client")
    def test_delete_agent_with_yes_flag(self, mock_get_client):
        """Test delete command with --yes flag to skip confirmation."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        agent_name = (
            "projects/test-project/locations/us-central1/reasoningEngines/agent123"
        )
        fake_client._agents[agent_name] = ReasoningEngine(name=agent_name)

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "delete",
                "agent123",
                "--yes",
            ],
        )
        assert result.exit_code == 0
        assert "deleted" in result.stdout
        assert agent_name not in fake_client._agents

    @patch("agent_engine_cli.main.get_client")
    def test_delete_agent_with_force(self, mock_get_client):
        """Test delete command with --force flag."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        agent_name = (
            "projects/test-project/locations/us-central1/reasoningEngines/agent123"
        )
        fake_client._agents[agent_name] = ReasoningEngine(name=agent_name)
        fake_client._sessions[agent_name] = [Session(name=f"{agent_name}/sessions/s1")]

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "delete",
                "agent123",
                "--yes",
                "--force",
            ],
        )
        assert result.exit_code == 0
        assert agent_name not in fake_client._agents

    @patch("agent_engine_cli.main.get_client")
    def test_delete_agent_error(self, mock_get_client):
        """Test delete command when an error occurs (e.g. not found)."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "delete",
                "agent123",
                "--yes",
            ],
        )
        assert result.exit_code == 1
        assert "Error deleting agent" in result.stdout


class TestChatCommand:
    def test_chat_help(self):
        """Test chat command help."""
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "--user" in result.stdout
        assert "--debug" in result.stdout

    @patch("agent_engine_cli.chat.run_chat")
    def test_chat_invokes_run_chat(self, mock_run_chat):
        """Test chat command invokes run_chat with correct arguments."""
        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "chat",
                "agent123",
            ],
        )
        assert result.exit_code == 0
        mock_run_chat.assert_called_once_with(
            project="test-project",
            location="us-central1",
            agent_id="agent123",
            user_id="cli-user",
            debug=False,
            base_url=None,
            api_version=None,
        )

    @patch("agent_engine_cli.chat.run_chat")
    def test_chat_with_user_and_debug(self, mock_run_chat):
        """Test chat command with custom user and debug flag."""
        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "chat",
                "agent123",
                "--user",
                "my-user",
                "--debug",
            ],
        )
        assert result.exit_code == 0
        mock_run_chat.assert_called_once_with(
            project="test-project",
            location="us-central1",
            agent_id="agent123",
            user_id="my-user",
            debug=True,
            base_url=None,
            api_version=None,
        )

    @patch("agent_engine_cli.chat.run_chat")
    def test_chat_error_handling(self, mock_run_chat):
        """Test chat command handles errors gracefully."""
        mock_run_chat.side_effect = Exception("Connection failed")

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "chat",
                "agent123",
            ],
        )
        assert result.exit_code == 1
        assert "Error in chat session" in result.stdout


class TestADCFallback:
    """Tests for ADC (Application Default Credentials) project fallback."""

    @patch("agent_engine_cli.main.get_client")
    @patch("agent_engine_cli.main.resolve_project")
    def test_list_uses_adc_project(self, mock_resolve, mock_get_client):
        """Test list command uses ADC project when --project not provided."""
        mock_resolve.return_value = "adc-project"
        fake_client = FakeAgentEngineClient(
            project="adc-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        result = runner.invoke(app, ["--location", "us-central1", "list"])
        assert result.exit_code == 0
        mock_resolve.assert_called_once_with(None)
        mock_get_client.assert_called_once_with(
            project="adc-project",
            location="us-central1",
            base_url=None,
            api_version=None,
        )

    @patch("agent_engine_cli.main.resolve_project")
    def test_list_error_when_no_project(self, mock_resolve):
        """Test list command shows error when no project available."""
        mock_resolve.side_effect = ConfigurationError("No project specified")

        result = runner.invoke(app, ["--location", "us-central1", "list"])
        assert result.exit_code == 1
        assert "Error: No project specified" in result.stdout

    @patch("agent_engine_cli.main.get_client")
    @patch("agent_engine_cli.main.resolve_project")
    def test_get_uses_adc_project(self, mock_resolve, mock_get_client):
        """Test get command uses ADC project when --project not provided."""
        mock_resolve.return_value = "adc-project"
        fake_client = FakeAgentEngineClient(
            project="adc-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        agent_name = (
            "projects/adc-project/locations/us-central1/reasoningEngines/agent1"
        )
        fake_client._agents[agent_name] = ReasoningEngine(name=agent_name)

        result = runner.invoke(app, ["--location", "us-central1", "get", "agent1"])
        assert result.exit_code == 0
        mock_resolve.assert_called_once_with(None)

    @patch("agent_engine_cli.main.get_client")
    @patch("agent_engine_cli.main.resolve_project")
    def test_create_uses_adc_project(self, mock_resolve, mock_get_client):
        """Test create command uses ADC project when --project not provided."""
        mock_resolve.return_value = "adc-project"
        fake_client = FakeAgentEngineClient(
            project="adc-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        result = runner.invoke(
            app, ["--location", "us-central1", "create", "Test Agent"]
        )
        assert result.exit_code == 0
        mock_resolve.assert_called_once_with(None)

    @patch("agent_engine_cli.main.get_client")
    @patch("agent_engine_cli.main.resolve_project")
    def test_delete_uses_adc_project(self, mock_resolve, mock_get_client):
        """Test delete command uses ADC project when --project not provided."""
        mock_resolve.return_value = "adc-project"
        fake_client = FakeAgentEngineClient(
            project="adc-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        agent_name = (
            "projects/adc-project/locations/us-central1/reasoningEngines/agent1"
        )
        fake_client._agents[agent_name] = ReasoningEngine(name=agent_name)

        result = runner.invoke(
            app, ["--location", "us-central1", "delete", "agent1", "--yes"]
        )
        assert result.exit_code == 0
        mock_resolve.assert_called_once_with(None)

    @patch("agent_engine_cli.chat.run_chat")
    @patch("agent_engine_cli.main.resolve_project")
    def test_chat_uses_adc_project(self, mock_resolve, mock_run_chat):
        """Test chat command uses ADC project when --project not provided."""
        mock_resolve.return_value = "adc-project"

        result = runner.invoke(app, ["--location", "us-central1", "chat", "agent1"])
        assert result.exit_code == 0
        mock_resolve.assert_called_once_with(None)
        mock_run_chat.assert_called_once_with(
            project="adc-project",
            location="us-central1",
            agent_id="agent1",
            user_id="cli-user",
            debug=False,
            base_url=None,
            api_version=None,
        )

    @patch("agent_engine_cli.main.get_client")
    @patch("agent_engine_cli.main.resolve_project")
    def test_explicit_project_still_works(self, mock_resolve, mock_get_client):
        """Test that explicit --project still works and is passed to resolve_project."""
        mock_resolve.return_value = "explicit-project"
        fake_client = FakeAgentEngineClient(
            project="explicit-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        result = runner.invoke(
            app, ["--project", "explicit-project", "--location", "us-central1", "list"]
        )
        assert result.exit_code == 0
        mock_resolve.assert_called_once_with("explicit-project")


class TestSessionsListCommand:
    def test_sessions_list_help(self):
        """Test sessions list command help."""
        result = runner.invoke(app, ["sessions", "list", "--help"])
        assert result.exit_code == 0
        assert "AGENT_ID" in result.stdout

    @patch("agent_engine_cli.main.get_client")
    def test_sessions_list_no_sessions(self, mock_get_client):
        """Test sessions list with no sessions."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        agent = fake_client.create_agent(
            display_name="Test Agent", identity_type="agent_identity"
        )

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "sessions",
                "list",
                agent.name,
            ],
        )
        assert result.exit_code == 0
        assert "No sessions found" in result.stdout

    @patch("agent_engine_cli.main.get_client")
    def test_sessions_list_with_sessions(self, mock_get_client):
        """Test sessions list with sessions."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        agent_name = (
            "projects/test-project/locations/us-central1/reasoningEngines/agent1"
        )
        fake_client._agents[agent_name] = ReasoningEngine(name=agent_name)

        session = Session(
            name=f"{agent_name}/sessions/session123",
            user_id="user-456",
            create_time=datetime(2024, 1, 15, 10, 30, 0),
            expire_time=datetime(2024, 1, 16, 10, 30, 0),
        )
        fake_client._sessions[agent_name] = [session]

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "sessions",
                "list",
                "agent1",
            ],
        )
        assert result.exit_code == 0
        assert "session123" in result.stdout
        assert "user-456" in result.stdout
        assert "2024-01-15" in result.stdout

    @patch("agent_engine_cli.main.get_client")
    def test_sessions_list_error(self, mock_get_client):
        """Test sessions list when an error occurs."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "sessions",
                "list",
                "agent123",
            ],
        )
        assert result.exit_code == 1
        assert "Error listing sessions" in result.stdout

    @patch("agent_engine_cli.main.get_client")
    @patch("agent_engine_cli.main.resolve_project")
    def test_sessions_list_uses_adc_project(self, mock_resolve, mock_get_client):
        """Test sessions list uses ADC project when --project not provided."""
        mock_resolve.return_value = "adc-project"
        fake_client = FakeAgentEngineClient(
            project="adc-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        agent_name = (
            "projects/adc-project/locations/us-central1/reasoningEngines/agent1"
        )
        fake_client._agents[agent_name] = ReasoningEngine(name=agent_name)

        result = runner.invoke(
            app, ["--location", "us-central1", "sessions", "list", "agent1"]
        )
        assert result.exit_code == 0
        mock_resolve.assert_called_once_with(None)
        mock_get_client.assert_called_once_with(
            project="adc-project",
            location="us-central1",
            base_url=None,
            api_version=None,
        )


class TestSandboxesListCommand:
    def test_sandboxes_list_help(self):
        """Test sandboxes list command help."""
        result = runner.invoke(app, ["sandboxes", "list", "--help"])
        assert result.exit_code == 0
        assert "AGENT_ID" in result.stdout

    @patch("agent_engine_cli.main.get_client")
    def test_sandboxes_list_no_sandboxes(self, mock_get_client):
        """Test sandboxes list with no sandboxes."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        agent = fake_client.create_agent(
            display_name="Test Agent", identity_type="agent_identity"
        )

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "sandboxes",
                "list",
                agent.name,
            ],
        )
        assert result.exit_code == 0
        assert "No sandboxes found" in result.stdout

    @patch("agent_engine_cli.main.get_client")
    def test_sandboxes_list_with_sandboxes(self, mock_get_client):
        """Test sandboxes list with sandboxes."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        agent_name = (
            "projects/test-project/locations/us-central1/reasoningEngines/agent1"
        )
        fake_client._agents[agent_name] = ReasoningEngine(name=agent_name)

        sandbox = SandboxEnvironment(
            name=f"{agent_name}/sandboxes/sandbox123",
            display_name="my_sandbox",
            state=SandboxState.STATE_RUNNING,
            create_time=datetime(2024, 2, 20, 14, 30, 0),
            expire_time=datetime(2024, 2, 21, 14, 30, 0),
        )
        fake_client._sandboxes[agent_name] = [sandbox]

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "sandboxes",
                "list",
                "agent1",
            ],
        )
        assert result.exit_code == 0
        assert "sandbox123" in result.stdout
        assert "my_sandbox" in result.stdout
        assert "RUNNING" in result.stdout
        assert "2024-02-20" in result.stdout

    @patch("agent_engine_cli.main.get_client")
    def test_sandboxes_list_error(self, mock_get_client):
        """Test sandboxes list when an error occurs."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "sandboxes",
                "list",
                "agent123",
            ],
        )
        assert result.exit_code == 1
        assert "Error listing sandboxes" in result.stdout

    @patch("agent_engine_cli.main.get_client")
    @patch("agent_engine_cli.main.resolve_project")
    def test_sandboxes_list_uses_adc_project(self, mock_resolve, mock_get_client):
        """Test sandboxes list uses ADC project when --project not provided."""
        mock_resolve.return_value = "adc-project"
        fake_client = FakeAgentEngineClient(
            project="adc-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        agent_name = (
            "projects/adc-project/locations/us-central1/reasoningEngines/agent1"
        )
        fake_client._agents[agent_name] = ReasoningEngine(name=agent_name)

        result = runner.invoke(
            app, ["--location", "us-central1", "sandboxes", "list", "agent1"]
        )
        assert result.exit_code == 0
        mock_resolve.assert_called_once_with(None)
        mock_get_client.assert_called_once_with(
            project="adc-project",
            location="us-central1",
            base_url=None,
            api_version=None,
        )


class TestMemoriesListCommand:
    def test_memories_list_help(self):
        """Test memories list command help."""
        result = runner.invoke(app, ["memories", "list", "--help"])
        assert result.exit_code == 0
        assert "AGENT_ID" in result.stdout

    @patch("agent_engine_cli.main.get_client")
    def test_memories_list_no_memories(self, mock_get_client):
        """Test memories list with no memories."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        agent = fake_client.create_agent(
            display_name="Test Agent", identity_type="agent_identity"
        )

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "memories",
                "list",
                agent.name,
            ],
        )
        assert result.exit_code == 0
        assert "No memories found" in result.stdout

    @patch("agent_engine_cli.main.get_client")
    def test_memories_list_with_memories(self, mock_get_client):
        """Test memories list with memories."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        agent_name = (
            "projects/test-project/locations/us-central1/reasoningEngines/agent1"
        )
        fake_client._agents[agent_name] = ReasoningEngine(name=agent_name)

        memory = Memory(
            name=f"{agent_name}/memories/memory123",
            display_name="user_preference",
            scope={"user_id": "user-123"},
            fact="User prefers dark mode",
            create_time=datetime(2024, 3, 10, 9, 15, 0),
            expire_time=datetime(2024, 4, 10, 9, 15, 0),
        )
        fake_client._memories[agent_name] = [memory]

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "memories",
                "list",
                "agent1",
            ],
        )
        assert result.exit_code == 0
        assert "memory123" in result.stdout
        assert "user_id=" in result.stdout
        assert "dark mode" in result.stdout
        assert "2024-03-10" in result.stdout

    @patch("agent_engine_cli.main.get_client")
    def test_memories_list_error(self, mock_get_client):
        """Test memories list when an error occurs (e.g. agent not found)."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "memories",
                "list",
                "agent123",
            ],
        )
        assert result.exit_code == 1
        assert "Error listing memories" in result.stdout

    @patch("agent_engine_cli.main.get_client")
    @patch("agent_engine_cli.main.resolve_project")
    def test_memories_list_uses_adc_project(self, mock_resolve, mock_get_client):
        """Test memories list uses ADC project when --project not provided."""
        mock_resolve.return_value = "adc-project"
        fake_client = FakeAgentEngineClient(
            project="adc-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        agent_name = (
            "projects/adc-project/locations/us-central1/reasoningEngines/agent1"
        )
        fake_client._agents[agent_name] = ReasoningEngine(name=agent_name)

        result = runner.invoke(
            app, ["--location", "us-central1", "memories", "list", "agent1"]
        )
        assert result.exit_code == 0
        mock_resolve.assert_called_once_with(None)
        mock_get_client.assert_called_once_with(
            project="adc-project",
            location="us-central1",
            base_url=None,
            api_version=None,
        )


class TestA2AChatCommand:
    def test_a2a_chat_help(self):
        """Test a2a-chat command help."""
        result = runner.invoke(app, ["a2a-chat", "--help"])
        assert result.exit_code == 0
        assert "--debug" in result.stdout
        assert "AGENT_ID" in result.stdout

    @patch("agent_engine_cli.a2a_chat.run_a2a_chat")
    def test_a2a_chat_invokes_run_a2a_chat(self, mock_run_a2a_chat):
        """Test a2a-chat command invokes run_a2a_chat with correct arguments."""
        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "a2a-chat",
                "agent123",
            ],
        )
        assert result.exit_code == 0
        mock_run_a2a_chat.assert_called_once_with(
            project="test-project",
            location="us-central1",
            agent_id="agent123",
            debug=False,
            base_url=None,
            api_version=None,
        )

    @patch("agent_engine_cli.a2a_chat.run_a2a_chat")
    def test_a2a_chat_with_debug(self, mock_run_a2a_chat):
        """Test a2a-chat command with debug flag."""
        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "a2a-chat",
                "agent123",
                "--debug",
            ],
        )
        assert result.exit_code == 0
        mock_run_a2a_chat.assert_called_once_with(
            project="test-project",
            location="us-central1",
            agent_id="agent123",
            debug=True,
            base_url=None,
            api_version=None,
        )

    @patch("agent_engine_cli.a2a_chat.run_a2a_chat")
    def test_a2a_chat_error_handling(self, mock_run_a2a_chat):
        """Test a2a-chat command handles errors gracefully."""
        mock_run_a2a_chat.side_effect = Exception("Connection failed")

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "a2a-chat",
                "agent123",
            ],
        )
        assert result.exit_code == 1
        assert "Error in A2A chat session" in result.stdout

    @patch("agent_engine_cli.a2a_chat.run_a2a_chat")
    @patch("agent_engine_cli.main.resolve_project")
    def test_a2a_chat_uses_adc_project(self, mock_resolve, mock_run_a2a_chat):
        """Test a2a-chat command uses ADC project when --project not provided."""
        mock_resolve.return_value = "adc-project"

        result = runner.invoke(app, ["--location", "us-central1", "a2a-chat", "agent1"])
        assert result.exit_code == 0
        mock_resolve.assert_called_once_with(None)
        mock_run_a2a_chat.assert_called_once_with(
            project="adc-project",
            location="us-central1",
            agent_id="agent1",
            debug=False,
            base_url=None,
            api_version=None,
        )

    @patch("agent_engine_cli.a2a_chat.run_a2a_chat")
    def test_a2a_chat_with_custom_endpoint_options(self, mock_run_a2a_chat):
        """Test a2a-chat command passes custom base_url and api_version."""
        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "--base-url",
                "https://staging.example.com",
                "--api-version",
                "v1",
                "a2a-chat",
                "agent123",
            ],
        )
        assert result.exit_code == 0
        mock_run_a2a_chat.assert_called_once_with(
            project="test-project",
            location="us-central1",
            agent_id="agent123",
            debug=False,
            base_url="https://staging.example.com",
            api_version="v1",
        )


class TestEndpointOverrideOptions:
    """Tests for --base-url and --api-version options."""

    @patch("agent_engine_cli.main.get_client")
    def test_list_with_custom_endpoint_options(self, mock_get_client):
        """Test list command passes custom base_url and api_version."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "--base-url",
                "https://custom.example.com",
                "--api-version",
                "v1",
                "list",
            ],
        )
        assert result.exit_code == 0
        mock_get_client.assert_called_once_with(
            project="test-project",
            location="us-central1",
            base_url="https://custom.example.com",
            api_version="v1",
        )

    @patch("agent_engine_cli.chat.run_chat")
    def test_chat_with_custom_endpoint_options(self, mock_run_chat):
        """Test chat command passes custom base_url and api_version to run_chat."""
        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "--base-url",
                "https://staging.example.com",
                "--api-version",
                "v1",
                "chat",
                "agent123",
            ],
        )
        assert result.exit_code == 0
        mock_run_chat.assert_called_once_with(
            project="test-project",
            location="us-central1",
            agent_id="agent123",
            user_id="cli-user",
            debug=False,
            base_url="https://staging.example.com",
            api_version="v1",
        )


class TestHelpers:
    """Tests for pure helper functions."""

    def test_get_id_full_resource_name(self):
        assert get_id("projects/p/locations/l/reasoningEngines/abc") == "abc"

    def test_get_id_short_id(self):
        assert get_id("abc") == "abc"

    def test_get_id_none(self):
        assert get_id(None) == ""

    def test_get_id_empty(self):
        assert get_id("") == ""

    def test_format_timestamp_datetime(self):
        assert format_timestamp(datetime(2024, 3, 15, 9, 30)) == "2024-03-15 09:30"

    def test_format_timestamp_none(self):
        assert format_timestamp(None) == ""

    def test_parse_resource_name_valid(self):
        result = _parse_resource_name(
            "projects/my-proj/locations/us-central1/reasoningEngines/agent1"
        )
        assert result == ("my-proj", "us-central1")

    def test_parse_resource_name_short_id(self):
        assert _parse_resource_name("agent1") is None

    def test_parse_resource_name_wrong_prefix(self):
        assert _parse_resource_name("foo/bar/baz/qux/quux/corge") is None

    def test_parse_resource_name_wrong_segment_count(self):
        assert _parse_resource_name("projects/p/locations/l") is None

    def test_format_class_methods_none(self):
        methods, card = _format_class_methods(None)
        assert methods == "N/A"
        assert card == "N/A"

    def test_format_class_methods_empty(self):
        methods, card = _format_class_methods([])
        assert methods == "N/A"
        assert card == "N/A"

    def test_format_class_methods_simple(self):
        methods, card = _format_class_methods([{"name": "query"}])
        assert "query" in methods
        assert card == "N/A"

    def test_format_class_methods_with_params(self):
        methods, card = _format_class_methods(
            [
                {
                    "name": "search",
                    "parameters": {
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                }
            ]
        )
        assert "search(query: string*, limit: integer)" in methods

    def test_format_class_methods_with_any_of(self):
        methods, _ = _format_class_methods(
            [
                {
                    "name": "fetch",
                    "parameters": {
                        "properties": {
                            "url": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        },
                        "required": [],
                    },
                }
            ]
        )
        assert "string|null" in methods

    def test_format_class_methods_with_description(self):
        methods, _ = _format_class_methods(
            [
                {"name": "greet", "description": "Say hello\nMore details here"},
            ]
        )
        assert "greet" in methods
        assert "Say hello" in methods
        assert "More details" not in methods  # Only first line

    def test_format_class_methods_extracts_agent_card(self):
        _, card = _format_class_methods(
            [
                {
                    "name": "query",
                    "metadata": {"a2a_agent_card": '{"name": "MyAgent"}'},
                }
            ]
        )
        assert card == '{"name": "MyAgent"}'

    def test_format_class_methods_skips_malformed(self):
        """Malformed entries are skipped without crashing."""
        methods, _ = _format_class_methods(
            [
                {"name": "good"},
                {"not_name": "bad"},  # no name or method key
                {"name": "also_good"},
            ]
        )
        assert "good" in methods
        assert "also_good" in methods
        assert "bad" not in methods


class TestDeleteWithResources:
    """Test delete behavior when agent has associated resources."""

    @patch("agent_engine_cli.main.get_client")
    def test_delete_agent_with_sessions_no_force_fails(self, mock_get_client):
        """Delete without --force fails when agent has sessions."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        agent_name = (
            "projects/test-project/locations/us-central1/reasoningEngines/agent123"
        )
        fake_client._agents[agent_name] = ReasoningEngine(name=agent_name)
        fake_client._sessions[agent_name] = [Session(name=f"{agent_name}/sessions/s1")]

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "delete",
                "agent123",
                "--yes",
            ],
        )
        assert result.exit_code == 1
        assert "Error deleting agent" in result.stdout
        assert agent_name in fake_client._agents  # Not deleted

    @patch("agent_engine_cli.main.get_client")
    def test_delete_agent_with_memories_no_force_fails(self, mock_get_client):
        """Delete without --force fails when agent has memories."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        agent_name = (
            "projects/test-project/locations/us-central1/reasoningEngines/agent123"
        )
        fake_client._agents[agent_name] = ReasoningEngine(name=agent_name)
        fake_client._memories[agent_name] = [
            Memory(name=f"{agent_name}/memories/m1", fact="test")
        ]

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "delete",
                "agent123",
                "--yes",
            ],
        )
        assert result.exit_code == 1
        assert agent_name in fake_client._agents  # Not deleted


class TestGetAgentRegressions:
    """Regression tests consolidated from standalone test files."""

    @patch("agent_engine_cli.main.get_client")
    def test_get_agent_with_none_class_methods(self, mock_get_client):
        """spec.class_methods=None must not crash (regression)."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        agent_name = (
            "projects/test-project/locations/us-central1/reasoningEngines/agent1"
        )
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
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "get",
                "agent1",
            ],
        )
        assert result.exit_code == 0
        assert "Test Agent" in result.stdout
        assert "Class Methods: N/A" in result.stdout

    @patch("agent_engine_cli.main.get_client")
    def test_get_agent_shows_effective_identity(self, mock_get_client):
        """Get command shows effective identity."""
        fake_client = FakeAgentEngineClient(
            project="test-project", location="us-central1"
        )
        mock_get_client.return_value = fake_client

        agent_name = (
            "projects/test-project/locations/us-central1/reasoningEngines/agent1"
        )
        fake_client._agents[agent_name] = ReasoningEngine(
            name=agent_name,
            display_name="Test Agent",
            description="A test agent",
            spec=ReasoningEngineSpec(
                effective_identity="service-account@test.iam.gserviceaccount.com",
            ),
            create_time=datetime(2024, 1, 1),
            update_time=datetime(2024, 1, 2),
        )

        result = runner.invoke(
            app,
            [
                "--project",
                "test-project",
                "--location",
                "us-central1",
                "get",
                "agent1",
            ],
        )
        assert result.exit_code == 0
        assert "Effective Identity" in result.stdout
        assert "service-account@test.iam.gserviceaccount.com" in result.stdout
