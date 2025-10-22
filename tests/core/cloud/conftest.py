# Shared setup for cloud tests.
#
# - Prefer Perceval's RemoteConfig token (cache or PCVL_CLOUD_TOKEN env).
# - Also accept QUANDELA_TOKEN as a convenience for local dev/CI.
# - Sets RemoteConfig token once (session scope).
# - Exposes `has_cloud_token` and `remote_processor` fixtures.
#
# Any test that needs the cloud can depend on `remote_processor`; it will
# auto-skip if no token is present.

from __future__ import annotations

import os

import perceval as pcvl
import pytest
from perceval.runtime import RemoteConfig


def _resolved_token() -> str | None:
    """Resolve a usable Quandela token from RemoteConfig or env."""
    # 1) Ask RemoteConfig (uses cache or PCVL_CLOUD_TOKEN)
    rc = RemoteConfig()
    token = (rc.get_token() or "").strip()
    if token:
        return token

    # 2) Fallback for projects/scripts that set QUANDELA_TOKEN
    token = os.environ.get("QUANDELA_TOKEN", "").strip()
    return token or None


@pytest.fixture(scope="session", autouse=True)
def _configure_quandela_token() -> None:
    """Configure Perceval's RemoteConfig with the token (if available)."""
    token = _resolved_token()
    if token:
        RemoteConfig.set_token(token)


@pytest.fixture(scope="session")
def has_cloud_token() -> bool:
    """True if a token is available for cloud tests."""
    return bool(_resolved_token())


@pytest.fixture
def remote_processor(has_cloud_token: bool):
    """Provide a RemoteProcessor if a token is available; otherwise skip."""
    if not has_cloud_token:
        pytest.skip("No token configured; skipping cloud-dependent test.")
    # Default to the SLOS simulator used throughout tests.
    return pcvl.RemoteProcessor("sim:slos")
