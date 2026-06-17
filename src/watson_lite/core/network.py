"""Shared HTTP helpers with retry logic for external API and SPARQL requests."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

USER_AGENT = "WatsonLite/1.0 (research project; clavijodario@gmail.com)"
WIKI_HEADERS: dict[str, str] = {"User-Agent": USER_AGENT}
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
DEFAULT_TIMEOUT_SECONDS: int = 10
DEFAULT_MAX_ATTEMPTS: int = 3
DEFAULT_BACKOFF_SECONDS: float = 1.0


def retry_delay_seconds(response: Any, attempt: int) -> float:  # noqa: ANN401
    """Compute the wait time before the next retry attempt."""
    retry_after: str | None = None
    headers = getattr(response, "headers", None)
    if headers is not None and hasattr(headers, "get"):
        retry_after_value = headers.get("Retry-After")
        if retry_after_value is not None:
            retry_after = str(retry_after_value)
    if retry_after:
        try:
            return float(max(float(retry_after), 0.0))
        except ValueError:
            logger.debug("Ignoring invalid Retry-After header: %s", retry_after)
    return float(DEFAULT_BACKOFF_SECONDS * (2**attempt))


def request_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    context: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> dict[str, Any] | None:
    """GET *url* with retries on transient failures and return parsed JSON.

    Returns ``None`` when the request fails after all attempts.
    """
    effective_headers = headers or WIKI_HEADERS
    for attempt in range(max_attempts):
        try:
            response = requests.get(
                url,
                params=params,
                headers=effective_headers,
                timeout=timeout,
            )
        except Exception as err:
            if attempt == max_attempts - 1:
                logger.warning("%s request failed: %s", context, err)
                return None
            wait = DEFAULT_BACKOFF_SECONDS * (2**attempt)
            logger.warning(
                "%s request failed, retrying in %.1fs: %s", context, wait, err
            )
            time.sleep(wait)
            continue

        status = int(getattr(response, "status_code", 200))
        if status in RETRYABLE_STATUS_CODES:
            if attempt == max_attempts - 1:
                logger.warning("%s request failed: HTTP %s", context, status)
                return None
            wait = retry_delay_seconds(response, attempt)
            logger.warning(
                "%s transient failure/rate limit: HTTP %s; retrying in %.1fs",
                context,
                status,
                wait,
            )
            time.sleep(wait)
            continue

        if status >= 400:
            logger.warning("%s request failed: HTTP %s", context, status)
            return None

        try:
            payload = response.json()
        except Exception as err:
            logger.warning("%s response parse failed: %s", context, err)
            return None

        if not isinstance(payload, dict):
            logger.warning("%s response was not a JSON object", context)
            return None
        return payload
    return None
