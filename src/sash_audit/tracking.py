"""Optional, privacy-preserving Weights & Biases experiment tracking."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


class _NoOpTracker:
    enabled = False

    def log(self, values: Mapping[str, Any]) -> None:
        return None

    def finish(self, *, summary: Mapping[str, Any] | None = None) -> None:
        return None


class _WandbTracker:
    enabled = True

    def __init__(self, run: Any) -> None:
        self._run = run

    def log(self, values: Mapping[str, Any]) -> None:
        self._run.log(dict(values))

    def finish(self, *, summary: Mapping[str, Any] | None = None) -> None:
        if summary:
            self._run.summary.update(dict(summary))
        self._run.finish()


def create_tracker(
    *,
    model_key: str,
    model_id: str,
    model_revision: str,
    config: Mapping[str, Any] | None = None,
) -> _NoOpTracker | _WandbTracker:
    """Create a W&B run only when explicitly enabled and authenticated.

    Raw questions, images, answers, and prediction rows are intentionally never
    sent to W&B. The tracker is therefore safe to leave in the inference path
    when the API key is absent or tracking is disabled.
    """

    api_key = os.getenv("WANDB_API_KEY", "").strip()
    disabled = os.getenv("SASH_DISABLE_WANDB", "").lower() in {"1", "true", "yes"}
    if not api_key or disabled:
        return _NoOpTracker()
    try:
        import wandb
    except ImportError:
        return _NoOpTracker()

    run_config = {
        "model_key": model_key,
        "model_id": model_id,
        "model_revision": model_revision,
        "tracking_scope": "aggregate_metrics_and_runtime_only",
    }
    if config:
        run_config.update(config)
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "sash-vlm-replication"),
        entity=os.getenv("WANDB_ENTITY") or None,
        job_type="inference",
        name=f"{model_key}-replication",
        config=run_config,
        reinit="finish_previous",
    )
    return _WandbTracker(run)


__all__ = ["create_tracker"]
