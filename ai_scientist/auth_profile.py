"""Auth profile store, OAuth refresh, and runtime credential selection."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

_PROFILE_STORE_ENV = "AI_SCIENTIST_AUTH_PROFILE_STORE_PATH"
_SECRET_STORE_ENV = "AI_SCIENTIST_AUTH_SECRET_STORE_PATH"
_PROFILE_ENV = "AI_SCIENTIST_AUTH_PROFILE"
_MANAGED_PROVIDERS_ENV = "AI_SCIENTIST_AUTH_MANAGED_PROVIDERS"
_FAILOVER_COOLDOWN_ENV = "AI_SCIENTIST_AUTH_FAILOVER_COOLDOWN_SECONDS"
_REFRESH_SKEW_ENV = "AI_SCIENTIST_AUTH_REFRESH_SKEW_SECONDS"

_DEFAULT_PROFILE_STORE = Path.home() / ".ai_scientist" / "auth_profiles.json"
_DEFAULT_SECRET_STORE = Path.home() / ".ai_scientist" / "auth_secrets.json"
_DEFAULT_FAILOVER_COOLDOWN_SECONDS = 120
_DEFAULT_REFRESH_SKEW_SECONDS = 60


@dataclass(frozen=True)
class OAuthSecret:
    access_token: str
    refresh_token: str | None
    token_endpoint: str | None
    client_id: str | None
    client_secret_env: str | None
    scope: str | None
    expires_at: float | None


@dataclass(frozen=True)
class ProfileSecret:
    api_key: str | None
    oauth: OAuthSecret | None


@dataclass(frozen=True)
class AuthProfile:
    profile_id: str
    provider: str
    mode: str
    account_label: str
    secret_id: str
    priority: int
    last_refresh: float | None
    last_used: float | None
    cooldown_until: float
    consecutive_failures: int


@dataclass(frozen=True)
class AuthCandidate:
    authorization_header: str
    profile_id: str | None
    source: str


@dataclass(frozen=True)
class _AuthState:
    profiles: tuple[AuthProfile, ...]
    secrets: Mapping[str, ProfileSecret]


def _profile_store_path() -> Path:
    raw = os.getenv(_PROFILE_STORE_ENV, "")
    return Path(raw) if raw else _DEFAULT_PROFILE_STORE


def _secret_store_path() -> Path:
    raw = os.getenv(_SECRET_STORE_ENV, "")
    return Path(raw) if raw else _DEFAULT_SECRET_STORE


def _managed_providers() -> tuple[str, ...]:
    raw = os.getenv(_MANAGED_PROVIDERS_ENV, "codex_native")
    values = [item.strip().lower() for item in raw.split(",")]
    return tuple(item for item in values if item)


def _should_manage_provider(provider_name: str) -> bool:
    return provider_name.lower() in _managed_providers()


def _env_token(auth_env: str) -> str | None:
    if not auth_env:
        return None
    token = os.getenv(auth_env)
    return token if token else None


def _env_candidate(auth_env: str) -> AuthCandidate | None:
    token = _env_token(auth_env)
    if token is None:
        return None
    return AuthCandidate(
        authorization_header=f"Bearer {token}",
        profile_id=None,
        source="env",
    )


def _placeholder_candidate(provider_name: str, auth_env: str) -> AuthCandidate:
    placeholder = auth_env or provider_name.upper()
    return AuthCandidate(
        authorization_header=f"Bearer LOCAL-{placeholder}",
        profile_id=None,
        source="placeholder",
    )


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON in {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _oauth_from_dict(data: Mapping[str, Any] | None) -> OAuthSecret | None:
    if not data:
        return None
    expires_raw = data.get("expires_at")
    expires_at = float(expires_raw) if expires_raw not in (None, "") else None
    return OAuthSecret(
        access_token=str(data.get("access_token", "")),
        refresh_token=(
            str(data.get("refresh_token"))
            if data.get("refresh_token") not in (None, "")
            else None
        ),
        token_endpoint=(
            str(data.get("token_endpoint"))
            if data.get("token_endpoint") not in (None, "")
            else None
        ),
        client_id=(
            str(data.get("client_id"))
            if data.get("client_id") not in (None, "")
            else None
        ),
        client_secret_env=(
            str(data.get("client_secret_env"))
            if data.get("client_secret_env") not in (None, "")
            else None
        ),
        scope=str(data.get("scope")) if data.get("scope") not in (None, "") else None,
        expires_at=expires_at,
    )


def _oauth_to_dict(secret: OAuthSecret) -> dict[str, Any]:
    return {
        "access_token": secret.access_token,
        "refresh_token": secret.refresh_token,
        "token_endpoint": secret.token_endpoint,
        "client_id": secret.client_id,
        "client_secret_env": secret.client_secret_env,
        "scope": secret.scope,
        "expires_at": secret.expires_at,
    }


def _profile_secret_from_dict(data: Mapping[str, Any] | None) -> ProfileSecret:
    values = data or {}
    api_key = values.get("api_key")
    oauth_payload = values.get("oauth")
    oauth_data = oauth_payload if isinstance(oauth_payload, dict) else None
    return ProfileSecret(
        api_key=str(api_key) if api_key not in (None, "") else None,
        oauth=_oauth_from_dict(oauth_data),
    )


def _profile_secret_to_dict(secret: ProfileSecret) -> dict[str, Any]:
    return {
        "api_key": secret.api_key,
        "oauth": _oauth_to_dict(secret.oauth) if secret.oauth else None,
    }


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _profile_from_dict(data: Mapping[str, Any]) -> AuthProfile:
    return AuthProfile(
        profile_id=str(data.get("id", "")),
        provider=str(data.get("provider", "")),
        mode=str(data.get("mode", "api_key")),
        account_label=str(data.get("account_label", "")),
        secret_id=str(data.get("secret_id", data.get("id", ""))),
        priority=int(data.get("priority", 0)),
        last_refresh=_optional_float(data.get("last_refresh")),
        last_used=_optional_float(data.get("last_used")),
        cooldown_until=float(data.get("cooldown_until", 0.0)),
        consecutive_failures=int(data.get("consecutive_failures", 0)),
    )


def _profile_to_dict(profile: AuthProfile) -> dict[str, Any]:
    return {
        "id": profile.profile_id,
        "provider": profile.provider,
        "mode": profile.mode,
        "account_label": profile.account_label,
        "secret_id": profile.secret_id,
        "priority": profile.priority,
        "last_refresh": profile.last_refresh,
        "last_used": profile.last_used,
        "cooldown_until": profile.cooldown_until,
        "consecutive_failures": profile.consecutive_failures,
    }


def _load_state() -> _AuthState:
    profile_payload = _read_json(_profile_store_path())
    secret_payload = _read_json(_secret_store_path())
    raw_profiles = profile_payload.get("profiles") if profile_payload else []
    if not isinstance(raw_profiles, list):
        raise ValueError("auth profile store must contain list field 'profiles'")
    profiles = tuple(_profile_from_dict(item) for item in raw_profiles)
    raw_secrets = secret_payload.get("secrets") if secret_payload else {}
    if not isinstance(raw_secrets, dict):
        raise ValueError("auth secret store must contain object field 'secrets'")
    secrets = {
        str(secret_id): _profile_secret_from_dict(payload)
        for secret_id, payload in raw_secrets.items()
        if isinstance(payload, dict)
    }
    return _AuthState(profiles=profiles, secrets=secrets)


def _save_state(state: _AuthState) -> None:
    profile_payload = {
        "version": 1,
        "profiles": [_profile_to_dict(profile) for profile in state.profiles],
    }
    secret_payload = {
        "version": 1,
        "secrets": {
            secret_id: _profile_secret_to_dict(secret)
            for secret_id, secret in state.secrets.items()
        },
    }
    _write_json(_profile_store_path(), profile_payload)
    _write_json(_secret_store_path(), secret_payload)


def list_auth_profiles(*, provider: str | None = None) -> tuple[AuthProfile, ...]:
    profiles = _load_state().profiles
    if provider is None:
        return profiles
    target = provider.lower()
    return tuple(item for item in profiles if item.provider.lower() == target)


def upsert_auth_profile(
    *,
    profile_id: str,
    provider: str,
    mode: str,
    account_label: str,
    priority: int = 0,
    api_key: str | None = None,
    access_token: str | None = None,
    refresh_token: str | None = None,
    token_endpoint: str | None = None,
    client_id: str | None = None,
    client_secret_env: str | None = None,
    scope: str | None = None,
    expires_at: float | None = None,
) -> AuthProfile:
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"api_key", "oauth"}:
        raise ValueError("mode must be one of: api_key, oauth")
    if normalized_mode == "api_key" and not api_key:
        raise ValueError("api_key mode requires api_key")
    if normalized_mode == "oauth" and not access_token:
        raise ValueError("oauth mode requires access_token")

    state = _load_state()
    profile_map = {item.profile_id: item for item in state.profiles}
    existing = profile_map.get(profile_id)
    secret_id = existing.secret_id if existing else profile_id
    profile = AuthProfile(
        profile_id=profile_id,
        provider=provider,
        mode=normalized_mode,
        account_label=account_label,
        secret_id=secret_id,
        priority=priority,
        last_refresh=existing.last_refresh if existing else None,
        last_used=existing.last_used if existing else None,
        cooldown_until=existing.cooldown_until if existing else 0.0,
        consecutive_failures=existing.consecutive_failures if existing else 0,
    )

    secret = ProfileSecret(
        api_key=api_key if normalized_mode == "api_key" else None,
        oauth=(
            OAuthSecret(
                access_token=access_token or "",
                refresh_token=refresh_token,
                token_endpoint=token_endpoint,
                client_id=client_id,
                client_secret_env=client_secret_env,
                scope=scope,
                expires_at=expires_at,
            )
            if normalized_mode == "oauth"
            else None
        ),
    )

    profile_map[profile_id] = profile
    secret_map = dict(state.secrets)
    secret_map[secret_id] = secret
    existing_ids = {item.profile_id for item in state.profiles}
    updated_profiles = tuple(
        profile_map[stored.profile_id] for stored in state.profiles
    )
    if profile_id not in existing_ids:
        updated_profiles = updated_profiles + (profile,)
    _save_state(_AuthState(profiles=updated_profiles, secrets=secret_map))
    return profile


def _selected_profile_id(provider_name: str) -> str | None:
    provider_key = provider_name.upper().replace("-", "_")
    provider_value = os.getenv(f"{_PROFILE_ENV}_{provider_key}")
    global_value = os.getenv(_PROFILE_ENV)
    return provider_value or global_value or None


def _refresh_safeguard_seconds() -> int:
    raw = os.getenv(_REFRESH_SKEW_ENV, "")
    if not raw:
        return _DEFAULT_REFRESH_SKEW_SECONDS
    return max(0, int(raw))


def _oauth_needs_refresh(secret: OAuthSecret) -> bool:
    if secret.expires_at is None:
        return False
    return secret.expires_at <= time.time() + float(_refresh_safeguard_seconds())


def _refresh_oauth_secret(secret: OAuthSecret, *, timeout: float) -> OAuthSecret:
    if not secret.refresh_token or not secret.token_endpoint:
        raise RuntimeError("OAuth profile is missing refresh_token or token_endpoint")
    payload: dict[str, str] = {
        "grant_type": "refresh_token",
        "refresh_token": secret.refresh_token,
    }
    if secret.client_id:
        payload["client_id"] = secret.client_id
    if secret.scope:
        payload["scope"] = secret.scope
    if secret.client_secret_env:
        client_secret = os.getenv(secret.client_secret_env)
        if client_secret:
            payload["client_secret"] = client_secret

    encoded = urlencode(payload).encode("utf-8")
    request = Request(
        secret.token_endpoint,
        data=encoded,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        details = exc.read().decode("utf-8", "replace") if exc.fp else ""
        raise RuntimeError(
            f"OAuth refresh failed with HTTP {exc.code}: {details}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"OAuth refresh failed: {exc}") from exc

    try:
        parsed = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError("OAuth refresh response is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("OAuth refresh response must be a JSON object")
    new_access_token = str(parsed.get("access_token", ""))
    if not new_access_token:
        raise RuntimeError("OAuth refresh response is missing access_token")
    expires_at = secret.expires_at
    expires_in_raw = parsed.get("expires_in")
    if expires_in_raw not in (None, ""):
        try:
            expires_at = time.time() + float(expires_in_raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("OAuth refresh response has invalid expires_in") from exc
    new_refresh = parsed.get("refresh_token")
    refreshed_token = (
        str(new_refresh) if new_refresh not in (None, "") else secret.refresh_token
    )
    return replace(
        secret,
        access_token=new_access_token,
        refresh_token=refreshed_token,
        expires_at=expires_at,
    )


def refresh_auth_profile(profile_id: str, *, timeout: float = 20.0) -> bool:
    state = _load_state()
    profile_map = {item.profile_id: item for item in state.profiles}
    profile = profile_map.get(profile_id)
    if profile is None:
        return False
    secret = state.secrets.get(profile.secret_id)
    if secret is None or secret.oauth is None:
        return False
    refreshed = _refresh_oauth_secret(secret.oauth, timeout=timeout)
    secret_map = dict(state.secrets)
    secret_map[profile.secret_id] = replace(secret, oauth=refreshed)
    profile_map[profile_id] = replace(profile, last_refresh=time.time())
    ordered_profiles = tuple(profile_map[item.profile_id] for item in state.profiles)
    _save_state(_AuthState(profiles=ordered_profiles, secrets=secret_map))
    return True


def _ordered_profiles(
    profiles: tuple[AuthProfile, ...],
    provider_name: str,
    selected_profile_id: str | None,
) -> list[AuthProfile]:
    provider_profiles = [
        item for item in profiles if item.provider.lower() == provider_name.lower()
    ]
    if not selected_profile_id:
        return sorted(
            provider_profiles, key=lambda item: (-item.priority, item.profile_id)
        )
    preferred = [
        item for item in provider_profiles if item.profile_id == selected_profile_id
    ]
    non_preferred = [
        item for item in provider_profiles if item.profile_id != selected_profile_id
    ]
    return preferred + sorted(
        non_preferred, key=lambda item: (-item.priority, item.profile_id)
    )


def _profile_token(profile: AuthProfile, secret: ProfileSecret) -> str:
    if profile.mode == "api_key" and secret.api_key:
        return secret.api_key
    if profile.mode == "oauth" and secret.oauth and secret.oauth.access_token:
        return secret.oauth.access_token
    return ""


def _candidates_from_profiles(
    provider_name: str,
    auth_env: str,
) -> tuple[AuthCandidate, ...]:
    state = _load_state()
    selected_profile_id = _selected_profile_id(provider_name)
    ordered = _ordered_profiles(state.profiles, provider_name, selected_profile_id)

    candidates: list[AuthCandidate] = []
    for profile in ordered:
        if profile.cooldown_until > time.time():
            continue
        secret = state.secrets.get(profile.secret_id)
        if secret is None:
            continue
        if (
            profile.mode == "oauth"
            and secret.oauth
            and _oauth_needs_refresh(secret.oauth)
        ):
            try:
                if refresh_auth_profile(profile.profile_id):
                    state = _load_state()
                    secret = state.secrets.get(profile.secret_id)
            except RuntimeError:
                record_profile_failure(profile.profile_id)
                continue
        if secret is None:
            continue
        token = _profile_token(profile, secret)
        if token:
            candidates.append(
                AuthCandidate(
                    authorization_header=f"Bearer {token}",
                    profile_id=profile.profile_id,
                    source="profile",
                )
            )

    env_candidate = _env_candidate(auth_env)
    if env_candidate:
        if env_candidate.authorization_header not in {
            item.authorization_header for item in candidates
        }:
            candidates.append(env_candidate)

    if candidates:
        return tuple(candidates)
    return (_placeholder_candidate(provider_name, auth_env),)


def resolve_auth_candidates(
    provider_name: str,
    auth_env: str,
) -> tuple[AuthCandidate, ...]:
    if not _should_manage_provider(provider_name):
        env_candidate = _env_candidate(auth_env)
        if env_candidate:
            return (env_candidate,)
        return (_placeholder_candidate(provider_name, auth_env),)
    return _candidates_from_profiles(provider_name, auth_env)


def resolve_runtime_auth_header(provider_name: str, auth_env: str) -> str:
    return resolve_auth_candidates(provider_name, auth_env)[0].authorization_header


def _failover_cooldown_seconds() -> int:
    raw = os.getenv(_FAILOVER_COOLDOWN_ENV, "")
    if not raw:
        return _DEFAULT_FAILOVER_COOLDOWN_SECONDS
    return max(1, int(raw))


def _rewrite_profiles(
    profiles: tuple[AuthProfile, ...],
    profile_id: str,
    mutate: Callable[[AuthProfile], AuthProfile],
) -> tuple[AuthProfile, ...]:
    updated: list[AuthProfile] = []
    for profile in profiles:
        if profile.profile_id == profile_id:
            updated.append(mutate(profile))
        else:
            updated.append(profile)
    return tuple(updated)


def record_profile_success(profile_id: str) -> None:
    state = _load_state()
    updated_profiles = _rewrite_profiles(
        state.profiles,
        profile_id,
        lambda profile: replace(
            profile,
            last_used=time.time(),
            consecutive_failures=0,
            cooldown_until=0.0,
        ),
    )
    _save_state(_AuthState(profiles=updated_profiles, secrets=state.secrets))


def record_profile_failure(profile_id: str) -> None:
    state = _load_state()

    def _mutate(profile: AuthProfile) -> AuthProfile:
        failures = profile.consecutive_failures + 1
        cooldown = float(_failover_cooldown_seconds()) * float(
            2 ** min(4, failures - 1)
        )
        return replace(
            profile,
            consecutive_failures=failures,
            cooldown_until=time.time() + cooldown,
        )

    updated_profiles = _rewrite_profiles(state.profiles, profile_id, _mutate)
    _save_state(_AuthState(profiles=updated_profiles, secrets=state.secrets))


__all__ = [
    "AuthCandidate",
    "AuthProfile",
    "list_auth_profiles",
    "record_profile_failure",
    "record_profile_success",
    "refresh_auth_profile",
    "resolve_auth_candidates",
    "resolve_runtime_auth_header",
    "upsert_auth_profile",
]
