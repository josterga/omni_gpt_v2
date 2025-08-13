# fathom_api.py
import os
import time
from typing import Any, Dict, Generator, List, Optional, Union

import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
FATHOM_API_KEY = os.getenv("FATHOM_API_KEY")
FATHOM_BASE_URL = "https://api.fathom.ai/external/v1"

if not FATHOM_API_KEY:
    raise RuntimeError("FATHOM_API_KEY not set in environment or .env file.")

class FathomAPIError(Exception):
    """Raised for any non-success response from the Fathom API."""
    pass

def _prepare_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert list values into Fathom's array param format (param[]=val1&param[]=val2)."""
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, list):
            out.update({f"{k}[]": v})
        else:
            out[k] = v
    return out

def list_meetings(
    params: Optional[Dict[str, Any]] = None,
    auto_paginate: bool = True,
    max_retries: int = 3,
    timeout: float = 15.0
) -> Generator[Union[Dict[str, Any], Dict[str, str]], None, None]:
    """
    Yield meetings from Fathom API /meetings endpoint.

    If max retries are reached due to rate limits or server errors,
    yields a single error dict instead of raising.
    """
    headers = {"X-Api-Key": FATHOM_API_KEY}
    client = httpx.Client(timeout=timeout, headers=headers)

    cursor = None
    base_params = params or {}

    while True:
        query = _prepare_params(base_params.copy())
        if cursor:
            query["cursor"] = cursor

        retries = 0
        while True:
            resp = client.get(f"{FATHOM_BASE_URL}/meetings", params=query)
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                if retries >= max_retries:
                    yield {
                        "error": f"Max retries reached: {resp.status_code}",
                        "status_code": resp.status_code,
                        "params": query
                    }
                    client.close()
                    return
                wait = min(2 ** retries, 8)
                print(f"[WARN] Retrying in {wait}s after {resp.status_code}...")
                time.sleep(wait)
                retries += 1
                continue
            if not resp.is_success:
                yield {
                    "error": f"Request failed: {resp.status_code}",
                    "status_code": resp.status_code,
                    "params": query
                }
                client.close()
                return
            break

        data = resp.json()
        for item in data.get("items", []):
            yield item

        cursor = data.get("next_cursor")
        if not (auto_paginate and cursor):
            break

    client.close()