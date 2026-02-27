from __future__ import annotations

import io
import json
import time
from email.message import Message
from urllib.error import HTTPError

import pytest

from ai_scientist.auth_profile import (
    list_auth_profiles,
    record_profile_failure,
    record_profile_success,
    resolve_auth_candidates,
    upsert_auth_profile,
)


@pytest.fixture(autouse=True)
def _auth_store_isolation(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv(
        "AI_SCIENTIST_AUTH_PROFILE_STORE_PATH", str(tmp_path / "auth_profiles.json")
    )
    monkeypatch.setenv(
        "AI_SCIENTIST_AUTH_SECRET_STORE_PATH", str(tmp_path / "auth_secrets.json")
    )
    monkeypatch.setenv("AI_SCIENTIST_AUTH_MANAGED_PROVIDERS", "codex_native")


def test_resolve_uses_profile_secret_before_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upsert_auth_profile(
        profile_id="codex-main",
        provider="codex_native",
        mode="api_key",
        account_label="main",
        priority=20,
        api_key="profile-token",
    )
    monkeypatch.setenv("CODEX_NATIVE_BEARER_TOKEN", "env-token")
    candidates = resolve_auth_candidates("codex_native", "CODEX_NATIVE_BEARER_TOKEN")
    assert candidates[0].authorization_header == "Bearer profile-token"
    assert candidates[0].profile_id == "codex-main"
    assert candidates[1].authorization_header == "Bearer env-token"


def test_resolve_refreshes_expired_oauth(monkeypatch: pytest.MonkeyPatch) -> None:
    upsert_auth_profile(
        profile_id="codex-oauth",
        provider="codex_native",
        mode="oauth",
        account_label="oauth",
        priority=10,
        access_token="stale-token",
        refresh_token="refresh-token",
        token_endpoint="https://auth.example/token",
        expires_at=time.time() - 20.0,
    )

    class _FakeResponse:
        status = 200

        def read(self) -> bytes:
            payload = {"access_token": "fresh-token", "expires_in": 3600}
            return json.dumps(payload).encode("utf-8")

        def getcode(self) -> int:
            return self.status

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *_: object) -> bool:
            return False

    def _fake_urlopen(*_: object, **__: object) -> _FakeResponse:
        return _FakeResponse()

    monkeypatch.setattr("ai_scientist.auth_profile.urlopen", _fake_urlopen)
    candidates = resolve_auth_candidates("codex_native", "CODEX_NATIVE_BEARER_TOKEN")
    assert candidates[0].authorization_header == "Bearer fresh-token"
    profiles = list_auth_profiles(provider="codex_native")
    assert profiles[0].last_refresh is not None


def test_profile_failure_enters_cooldown_then_success_resets() -> None:
    upsert_auth_profile(
        profile_id="codex-main",
        provider="codex_native",
        mode="api_key",
        account_label="main",
        api_key="profile-token",
    )
    record_profile_failure("codex-main")
    after_failure = list_auth_profiles(provider="codex_native")[0]
    assert after_failure.consecutive_failures == 1
    assert after_failure.cooldown_until > time.time()

    record_profile_success("codex-main")
    after_success = list_auth_profiles(provider="codex_native")[0]
    assert after_success.consecutive_failures == 0
    assert after_success.cooldown_until == 0.0


def test_refresh_failure_falls_back_to_secondary_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upsert_auth_profile(
        profile_id="codex-oauth",
        provider="codex_native",
        mode="oauth",
        account_label="oauth",
        priority=20,
        access_token="stale-token",
        refresh_token="refresh-token",
        token_endpoint="https://auth.example/token",
        expires_at=time.time() - 20.0,
    )
    upsert_auth_profile(
        profile_id="codex-backup",
        provider="codex_native",
        mode="api_key",
        account_label="backup",
        priority=10,
        api_key="backup-token",
    )

    def _fake_urlopen(*_: object, **__: object) -> _FakeResponse:
        headers = Message()
        raise HTTPError(
            url="https://auth.example/token",
            code=401,
            msg="Unauthorized",
            hdrs=headers,
            fp=io.BytesIO(b'{"error":"bad refresh"}'),
        )

    class _FakeResponse:
        def read(self) -> bytes:
            return b"{}"

    monkeypatch.setattr("ai_scientist.auth_profile.urlopen", _fake_urlopen)
    candidates = resolve_auth_candidates("codex_native", "CODEX_NATIVE_BEARER_TOKEN")
    assert candidates[0].authorization_header == "Bearer backup-token"
    profiles = {profile.profile_id: profile for profile in list_auth_profiles()}
    assert profiles["codex-oauth"].consecutive_failures == 1


def test_refresh_parse_failure_falls_back_to_secondary_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upsert_auth_profile(
        profile_id="codex-oauth",
        provider="codex_native",
        mode="oauth",
        account_label="oauth",
        priority=20,
        access_token="stale-token",
        refresh_token="refresh-token",
        token_endpoint="https://auth.example/token",
        expires_at=time.time() - 20.0,
    )
    upsert_auth_profile(
        profile_id="codex-backup",
        provider="codex_native",
        mode="api_key",
        account_label="backup",
        priority=10,
        api_key="backup-token",
    )

    class _FakeResponse:
        status = 200

        def read(self) -> bytes:
            return b"{not-json"

        def getcode(self) -> int:
            return self.status

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *_: object) -> bool:
            return False

    def _fake_urlopen(*_: object, **__: object) -> _FakeResponse:
        return _FakeResponse()

    monkeypatch.setattr("ai_scientist.auth_profile.urlopen", _fake_urlopen)
    candidates = resolve_auth_candidates("codex_native", "CODEX_NATIVE_BEARER_TOKEN")
    assert candidates[0].authorization_header == "Bearer backup-token"
    profiles = {profile.profile_id: profile for profile in list_auth_profiles()}
    assert profiles["codex-oauth"].consecutive_failures == 1
