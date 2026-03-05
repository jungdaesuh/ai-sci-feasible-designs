"""OAuth PKCE flow and token management for ChatGPT subscription.

M1 stub: signatures defined, bodies raise NotImplementedError.
Full implementation in M7.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CodexCredentials:
    """Stored OAuth credentials for the Codex Responses API."""

    access_token: str
    refresh_token: str
    expires_at: int
    account_id: str


def load_codex_credentials() -> CodexCredentials:
    """Load credentials from ~/.harness/auth.json."""
    raise NotImplementedError("Full auth in M7")


def refresh_if_expired(creds: CodexCredentials) -> CodexCredentials:
    """Refresh the access token if expired."""
    raise NotImplementedError("Full auth in M7")
