from __future__ import annotations

import argparse
import threading
import time

import uvicorn

from src.api import app


def run_api_smoke_check(timeout_s: float = 12.0) -> int:
    """
    Start the API server, verify startup, then shut it down.
    This keeps automations finite while still validating API bootability.
    """
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    deadline = time.monotonic() + max(timeout_s, 1.0)
    while time.monotonic() < deadline:
        if getattr(server, "started", False):
            print("[OK] API startup check passed on 127.0.0.1:8000")
            server.should_exit = True
            t.join(timeout=5)
            return 0
        if not t.is_alive():
            print("[ERROR] API process exited before startup check could pass")
            return 1
        time.sleep(0.1)

    print(f"[ERROR] API startup check timed out after {timeout_s:.1f}s")
    server.should_exit = True
    t.join(timeout=5)
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run API server continuously (blocking).",
    )
    parser.add_argument(
        "--smoke-timeout",
        type=float,
        default=12.0,
        help="Seconds to wait for startup when running finite smoke-check mode.",
    )
    args = parser.parse_args()

    if args.serve:
        uvicorn.run(app, host="127.0.0.1", port=8000)
    else:
        raise SystemExit(run_api_smoke_check(timeout_s=args.smoke_timeout))
