# tests/core/cloud/conftest.py
from __future__ import annotations

import os

import perceval as pcvl
import perceval.providers.scaleway as scw
import pytest
from perceval.runtime import RemoteConfig


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add Scaleway test option (--run-cloud-tests is defined in parent conftest)."""
    try:
        parser.addoption(
            "--run-scaleway-tests",
            action="store_true",
            default=False,
            help="Run tests that require Scaleway credentials (SCW_PROJECT_ID, SCW_SECRET_KEY)",
        )
    except ValueError:
        pass  # Already defined elsewhere


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """By default, only skip tests that truly require a cloud token or Scaleway credentials.

    - If --run-cloud-tests is NOT passed: skip tests that use the
      'remote_processor' fixture (i.e. tests that require an actual
      remote platform/token). Other tests under tests/core/cloud that do
      not require the token will still run.
    - If --run-cloud-tests IS passed: do not skip at collection time;
      tests depending on 'remote_processor' will be skipped later by the
      fixture itself when no token is configured.
    - If --run-scaleway-tests is NOT passed: skip tests that use the
      'scaleway_session' fixture.
    """
    run_cloud = config.getoption("--run-cloud-tests", default=False)
    run_scaleway = config.getoption("--run-scaleway-tests", default=False)

    cloud_skip = pytest.mark.skip(
        reason="requires Quandela Cloud token; run with --run-cloud-tests and configure RemoteConfig"
    )
    scaleway_skip = pytest.mark.skip(
        reason="requires Scaleway credentials; run with --run-scaleway-tests"
    )

    for item in items:
        fixturenames = getattr(item, "fixturenames", ())

        # Skip cloud tests unless --run-cloud-tests
        if not run_cloud and "remote_processor" in fixturenames:
            item.add_marker(cloud_skip)

        # Skip scaleway tests unless --run-scaleway-tests
        if not run_scaleway and "scaleway_session" in fixturenames:
            item.add_marker(scaleway_skip)


# ---------------------------------------------------------------------------
# Quandela Cloud fixtures
# ---------------------------------------------------------------------------


def _has_configured_token() -> bool:
    """
    True only if a token has already been configured (e.g. via
    RemoteConfig.set_token(...) in user env or prior setup).
    We intentionally do NOT read environment variables here.
    """
    rc = RemoteConfig()
    token = (rc.get_token() or "").strip()
    return bool(token)


@pytest.fixture(scope="session")
def has_cloud_token() -> bool:
    """Whether a Quandela Cloud token is already configured."""
    return _has_configured_token()


@pytest.fixture
def remote_processor(has_cloud_token: bool):
    """
    Provide a RemoteProcessor if a token is configured; otherwise skip.
    We do not configure the token here—this is on the user environment.
    """
    if not has_cloud_token:
        pytest.skip(
            "Quandela Cloud token not configured: please call RemoteConfig.set_token(...) "
            "or set up your environment before running cloud tests."
        )
    return pcvl.RemoteProcessor("sim:slos")


# ---------------------------------------------------------------------------
# Scaleway fixtures
# ---------------------------------------------------------------------------


def _has_scaleway_credentials() -> bool:
    """Check if Scaleway credentials are available in environment."""
    return bool(
        os.environ.get("SCW_PROJECT_ID") and os.environ.get("SCW_SECRET_KEY")
    )


@pytest.fixture(scope="session")
def has_scaleway_credentials() -> bool:
    """Whether Scaleway credentials are configured."""
    return _has_scaleway_credentials()


@pytest.fixture(scope="module")
def scaleway_credentials(has_scaleway_credentials: bool):
    """Provide Scaleway credentials or skip if not available."""
    if not has_scaleway_credentials:
        pytest.skip(
            "Scaleway credentials not configured: set SCW_PROJECT_ID and SCW_SECRET_KEY"
        )
    return {
        "project_id": os.environ["SCW_PROJECT_ID"],
        "token": os.environ["SCW_SECRET_KEY"],
    }


@pytest.fixture(scope="module")
def scaleway_session(scaleway_credentials):
    """
    Provide a Scaleway Session for testing.

    Uses EMU-ASCELLA-6PQ platform with reasonable timeouts for testing.
    The session is shared across all tests in a module to avoid
    repeatedly creating/destroying sessions.
    """
    with scw.Session(
        "EMU-ASCELLA-6PQ",
        project_id=scaleway_credentials["project_id"],
        token=scaleway_credentials["token"],
        deduplication_id="merlin-test-session",
        max_idle_duration_s=300,
        max_duration_s=600,
    ) as session:
        # Workaround for MerlinProcessor compatibility
        session.default_rpc_handler = session._rpc_handler
        yield session
