"""Stellar operations — delegates to Web API.

The Prime Agent never holds Stellar private keys. All on-chain operations
(balance queries, token transfers, dividends) go through the Talos Web
server, which uses Stellar SDK server-side or reads from Horizon.

Failures are classified with :mod:`talos_agent.payments.stellar_retry` so that
callers and operators can decide deterministically whether a transaction is
safe to retry.  The returned classification is privacy-safe: no exception
text, seeds, payment proofs, or response bodies are ever surfaced.
"""

from __future__ import annotations

import os
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx
from rich.console import Console

from talos_agent.adapters.snapshots import StellarHealthSnapshot
from talos_agent.http import request_with_retry
from talos_agent.payments.stellar_retry import (
    attach_stellar_failure,
    classify_stellar_failure,
    classify_stellar_result,
)

_HORIZON_URL = os.getenv("STELLAR_HORIZON_URL", "https://horizon-testnet.stellar.org")
_SEVEN_DECIMALS = Decimal("0.0000001")

console = Console()


def to_stellar_amount(value: Decimal | float | int | str) -> Decimal:
    """Parse a numeric or string value to Decimal with exact precision.

    Float inputs are converted via str(value) to avoid binary float artifacts
    where possible, while Decimal and string inputs retain exact precision.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, str)):
        return Decimal(str(value).strip())
    # float input
    return Decimal(str(value))


def format_stellar_amount(amount: Decimal | float | int | str) -> str:
    """Serialize an on-chain Stellar amount as a string with exactly 7 decimal places."""
    d = to_stellar_amount(amount)
    return f"{d.quantize(_SEVEN_DECIMALS):f}"


class StellarKit:
    """Proxy for Stellar operations via Talos Web API.

    Read operations use Stellar Horizon (public, no key needed).
    Write operations are forwarded to Web, which handles signing.
    """

    def __init__(self, api_client: Any):
        self._api = api_client
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        console.print("[green]Stellar proxy ready (via Web API + Horizon).[/green]")

    @property
    def available(self) -> bool:
        return self._initialized

    def health_snapshot(self) -> StellarHealthSnapshot:
        """Return in-process readiness snapshot for health probes (side-effect free)."""
        return StellarHealthSnapshot(
            has_api=self._api is not None,
            initialized=bool(self._initialized),
        )

    async def get_balance(self, account_id: str = "") -> dict[str, Any]:
        """Query XLM balance via Horizon (public API)."""
        try:
            talos = await self._api.get_talos(self._api._talos_id)
            acct = account_id or (talos.get("stellarAccountId", "") if talos else "")
            if not acct:
                return {"error": "No Stellar account configured"}
            # Horizon is public — no auth needed
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await request_with_retry(
                    lambda: client.get(f"{_HORIZON_URL}/accounts/{acct}"),
                    provider="talos_web_api",
                )
                if r.status_code == 200:
                    data = r.json()
                    balance = data.get("balances", [])
                    xlm_balance = next((b for b in balance if b.get("asset_type") == "native"), None)
                    if xlm_balance:
                        dec_val = Decimal(str(xlm_balance["balance"]))
                        return {
                            "balance_xlm": format_stellar_amount(dec_val),
                            "balance_xlm_decimal": dec_val,
                            "account": acct,
                        }
                    zero_dec = Decimal("0")
                    return {
                        "balance_xlm": format_stellar_amount(zero_dec),
                        "balance_xlm_decimal": zero_dec,
                        "account": acct,
                    }
            return attach_stellar_failure(
                None,
                classify_stellar_failure(status_code=r.status_code),
            )
        except Exception as exc:
            return attach_stellar_failure(
                None,
                classify_stellar_failure(exc=exc),
            )

    async def get_token_balance(self, account_id: str, token_id: str) -> dict[str, Any]:
        """Query Stellar asset balance via Horizon."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await request_with_retry(
                    lambda: client.get(f"{_HORIZON_URL}/accounts/{account_id}"),
                    provider="talos_web_api",
                )
                if r.status_code == 200:
                    data = r.json()
                    balances = data.get("balances", [])
                    token_balance = next((b for b in balances if b.get("asset_code") == token_id), None)
                    dec_val = Decimal(str(token_balance["balance"])) if token_balance else Decimal("0")
                    return {
                        "balance": format_stellar_amount(dec_val),
                        "balance_decimal": dec_val,
                        "token_id": token_id,
                        "account": account_id,
                    }
            return attach_stellar_failure(
                None,
                classify_stellar_failure(status_code=r.status_code),
            )
        except Exception as exc:
            return attach_stellar_failure(
                None,
                classify_stellar_failure(exc=exc),
            )

    async def transfer_xlm(self, to_account: str, amount: Decimal | float | str) -> dict[str, Any]:
        """Request XLM transfer via Web API (Web handles signing).

        On failure the returned dict carries ``error`` plus the bounded
        classification fields (``failure_class``, ``failure_code``,
        ``retryable``, ``indeterminate``, ``terminal``).  On success it keeps
        the historical ``{"status": "submitted", ...}`` shape.
        """
        try:
            dec_amount = to_stellar_amount(amount)
            formatted_amount = format_stellar_amount(dec_amount)
            result = await self._api.request_transfer(
                to_account=to_account, amount=formatted_amount, currency="XLM"
            )
        except Exception as exc:
            return attach_stellar_failure(None, classify_stellar_failure(exc=exc))

        failure = classify_stellar_result(result)
        if failure is not None:
            return attach_stellar_failure(result, failure)

        if result:
            return {"status": "submitted", "to": to_account, "amount": formatted_amount}

        # Empty response with no error payload — surface an unclassified failure.
        return attach_stellar_failure(None, classify_stellar_failure())
