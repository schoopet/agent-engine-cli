
# We need to import the function to test
from agent_engine_cli.chat import _install_api_logging_hooks

def test_install_api_logging_hooks_idempotency():
    """Test that _install_api_logging_hooks does not double-patch."""

    # We need to mock google.genai._api_client because it might not be installed in all environments
    # or we want to isolate the test.
    # However, since the code imports it directly, we need to mock it in sys.modules
    # OR we can rely on the fact that we are in an environment where it is installed.
    # Given the previous exploration, it is installed.

    from google.genai import _api_client

    # Save original state to restore later
    original_async_request = _api_client.BaseApiClient.async_request
    original_async_request_streamed = _api_client.BaseApiClient.async_request_streamed

    # Remove any existing patch markers if present (unlikely in fresh process but good for safety)
    if hasattr(original_async_request, "_is_logged_async_request"):
        delattr(original_async_request, "_is_logged_async_request")

    try:
        # First patch
        _install_api_logging_hooks(debug=True)
        patched_once = _api_client.BaseApiClient.async_request

        assert patched_once != original_async_request
        assert getattr(patched_once, "_is_logged_async_request", False)

        # Second patch
        _install_api_logging_hooks(debug=True)
        patched_twice = _api_client.BaseApiClient.async_request

        # Should be the same object
        assert patched_twice == patched_once

    finally:
        # Restore original state
        _api_client.BaseApiClient.async_request = original_async_request
        _api_client.BaseApiClient.async_request_streamed = original_async_request_streamed
