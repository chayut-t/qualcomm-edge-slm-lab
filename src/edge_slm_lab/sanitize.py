"""Shared secret-scan helper for files written to results/.

Every record this project commits must pass check_text. The check is a
substring scan for credential-like content: tokens, signed URLs, and
any URL at all. Records store names, sizes, hashes, and metrics only.
"""

from __future__ import annotations

FORBIDDEN_SUBSTRINGS = (
    "http://",
    "https://",
    "X-Amz",
    "hf_",
    "api_token",
    "api_key",
    "Bearer ",
    "AKIA",
    "qai-hub.env",
)


def check_text(text: str) -> list[str]:
    """Return the forbidden substrings found in text. Empty list = clean."""
    return [s for s in FORBIDDEN_SUBSTRINGS if s in text]
