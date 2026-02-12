"""Verify MerlinProcessor raises ValueError when no token can be extracted."""
from unittest.mock import MagicMock, patch

import pytest
from perceval.runtime import RemoteProcessor

from merlin.core.merlin_processor import MerlinProcessor


def test_raises_on_missing_token():
    """MerlinProcessor must raise ValueError at init if the token cannot be found."""
    # Build a fake RemoteProcessor that passes isinstance checks
    mock_rp = MagicMock(spec=RemoteProcessor)
    mock_rp.name = "sim:slos"
    mock_rp.available_commands = ["probs"]
    mock_rp.proxies = None

    # Patch _extract_rp_token to return None — simulates no token anywhere
    with patch.object(MerlinProcessor, "_extract_rp_token", return_value=None):
        with pytest.raises(ValueError, match="Could not extract auth token"):
            MerlinProcessor(remote_processor=mock_rp)