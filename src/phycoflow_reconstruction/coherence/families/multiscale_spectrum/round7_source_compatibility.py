"""Narrow identity bridge for the telemetry-only Round 7 Scale change."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

# Exact bytes at the checked-out pre-Round-7 HEAD, obtained with git show.
PRIOR = {
    'family.py': '0cc4ea52a87651bccce901020e8f88c12a1eba23be781fdd9c9330215417d9bb',
    'objectives.py': 'abda92e70044351a7241761d24c248d5a741210a5a1e942a67c5bff65315f1ae',
}
# Exact new telemetry-only implementation bytes. Any subsequent implementation
# change invalidates this bridge until it is independently reviewed again.
CURRENT = {
    'family.py': 'e13de39c44dad5db165fa1496d1c3704a1dfd6448e38f2e2b482ed413d4dd570',
    'objectives.py': '3f82e464fed9228aa2f1280fe0ec3632d3a2470c63a67d40dd01662c570a539a',
}


def accepts_prior_scale_identity(saved: Any, current: dict[str, Any]) -> bool:
    if not isinstance(saved, dict) or not isinstance(current.get('sources'), dict):
        return False
    if any(current['sources'].get(name) != value for name, value in CURRENT.items()):
        return False
    expected = deepcopy(current)
    expected['sources'].update(PRIOR)
    return saved == expected
