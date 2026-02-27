#!/usr/bin/env python3
"""Manage codex-native auth profiles for AI Scientist."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage auth profiles.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List auth profiles")
    list_parser.add_argument("--provider", default=None, help="Filter by provider")

    upsert_parser = subparsers.add_parser("upsert", help="Create or update a profile")
    upsert_parser.add_argument("--profile-id", required=True)
    upsert_parser.add_argument("--provider", default="codex_native")
    upsert_parser.add_argument("--mode", choices=("api_key", "oauth"), required=True)
    upsert_parser.add_argument("--account-label", required=True)
    upsert_parser.add_argument("--priority", type=int, default=0)
    upsert_parser.add_argument("--api-key", default=None)
    upsert_parser.add_argument("--access-token", default=None)
    upsert_parser.add_argument("--refresh-token", default=None)
    upsert_parser.add_argument("--token-endpoint", default=None)
    upsert_parser.add_argument("--client-id", default=None)
    upsert_parser.add_argument("--client-secret-env", default=None)
    upsert_parser.add_argument("--scope", default=None)
    upsert_parser.add_argument("--expires-at", type=float, default=None)

    refresh_parser = subparsers.add_parser("refresh", help="Refresh OAuth profile")
    refresh_parser.add_argument("--profile-id", required=True)
    refresh_parser.add_argument("--timeout", type=float, default=20.0)
    return parser


def main() -> None:
    from ai_scientist.auth_profile import (
        list_auth_profiles,
        refresh_auth_profile,
        upsert_auth_profile,
    )

    args = _parser().parse_args()
    if args.command == "list":
        profiles = list_auth_profiles(provider=args.provider)
        payload = [
            {
                "id": profile.profile_id,
                "provider": profile.provider,
                "mode": profile.mode,
                "account_label": profile.account_label,
                "priority": profile.priority,
                "last_refresh": profile.last_refresh,
                "last_used": profile.last_used,
                "cooldown_until": profile.cooldown_until,
                "consecutive_failures": profile.consecutive_failures,
            }
            for profile in profiles
        ]
        print(json.dumps(payload, indent=2))
        return

    if args.command == "upsert":
        profile = upsert_auth_profile(
            profile_id=args.profile_id,
            provider=args.provider,
            mode=args.mode,
            account_label=args.account_label,
            priority=args.priority,
            api_key=args.api_key,
            access_token=args.access_token,
            refresh_token=args.refresh_token,
            token_endpoint=args.token_endpoint,
            client_id=args.client_id,
            client_secret_env=args.client_secret_env,
            scope=args.scope,
            expires_at=args.expires_at,
        )
        print(
            json.dumps(
                {
                    "id": profile.profile_id,
                    "provider": profile.provider,
                    "mode": profile.mode,
                    "account_label": profile.account_label,
                },
                indent=2,
            )
        )
        return

    if args.command == "refresh":
        refreshed = refresh_auth_profile(args.profile_id, timeout=args.timeout)
        print(json.dumps({"profile_id": args.profile_id, "refreshed": refreshed}))
        return

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
