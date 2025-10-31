# tests/core/cloud/conftest.py
from __future__ import annotations

import perceval as pcvl
import pytest
from perceval.runtime import RemoteConfig
from pathlib import Path


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-cloud-tests"):
        return
    skip_marker = pytest.mark.skip(
        reason="use --run-cloud-tests to enable Quandela Cloud integration tests"
    )
    for item in items:
        try:
            item_path = Path(item.fspath).resolve()
        except AttributeError:
            continue
        if _is_under_cloud_dir(item_path):
            item.add_marker(skip_marker)


def _is_under_cloud_dir(path: Path) -> bool:
    cloud_root = Path(__file__).parent.resolve()
    try:
        path.relative_to(cloud_root)
        return True
    except ValueError:
        return False


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
