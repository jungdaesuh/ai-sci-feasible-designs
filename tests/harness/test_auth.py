"""Tests for harness.auth — M1 stub acceptance."""

from __future__ import annotations

import pytest

from harness.auth import (
    CodexCredentials,
    load_codex_credentials,
    refresh_if_expired,
)


def test_load_credentials_stub_raises():
    with pytest.raises(NotImplementedError, match="Full auth in M7"):
        load_codex_credentials()


def test_refresh_stub_raises():
    creds = CodexCredentials(
        access_token="fake",
        refresh_token="fake",
        expires_at=0,
        account_id="fake",
    )
    with pytest.raises(NotImplementedError, match="Full auth in M7"):
        refresh_if_expired(creds)
