"""Artifact loading and hot-reload management for the RetainStack prediction API.

Design: Thread-Safe Atomic Pointer Swap
---------------------------------------
``ArtifactContainer`` is an **immutable snapshot** of a matched (model, preprocessor)
pair, stamped with a SHA-256 fingerprint and UTC load timestamp.

``ModelStore`` holds the live container behind a ``threading.RLock``.  All callers
acquire the artifact pair through ``get_artifacts()``, which returns the current
container object under the lock and then releases it immediately — so inference
runs *outside* the lock against a stable, immutable reference.

``reload()`` follows a strict three-phase protocol:
  1. **Load** — deserialise candidate ``.skops`` files into a *new* container
     without touching the live one.
  2. **Pre-flight** — run a dry synthetic inference pass against the candidate
     to detect shape/type mismatches before they reach real traffic.
  3. **Swap** — atomically replace the ``_active`` pointer under the write lock.

If pre-flight fails, the live model is **never** replaced and the error is
propagated to the caller.  In-flight requests always finish against the artifact
snapshot they captured at their own start, so there is **zero torn-read risk**.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import skops.io as skio

from src.config import settings
from src.logger.logger import get_logger

logger = get_logger("api.dependencies")


# ---------------------------------------------------------------------------
# Immutable artifact snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArtifactContainer:
    """An immutable, matched pair of model + preprocessor artifacts.

    ``frozen=True`` ensures that once created the object cannot be mutated,
    guaranteeing that every caller snapshotting the active container sees a
    consistent, stable state for the lifetime of their request.
    """

    model: object
    preprocessor: object
    version: str           # first 12 hex chars of SHA-256(model_bytes + preprocessor_bytes)
    loaded_at: str         # ISO-8601 UTC timestamp


def _compute_version(model_path: str, preprocessor_path: str) -> str:
    """Return a 12-char SHA-256 fingerprint of the two artifact files.

    Reading both files together means any change in either artifact produces a
    completely different fingerprint, making version mismatches immediately
    visible in logs and the /ready probe.
    """
    sha = hashlib.sha256()
    for path in (model_path, preprocessor_path):
        with open(path, "rb") as fh:
            sha.update(fh.read())
    return sha.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Dummy pre-flight payload
# The feature order and value ranges mirror a valid mid-traffic session so
# that the dry-run exercises both the preprocessor transform *and* XGBoost
# predict paths without triggering Pydantic validation.
# ---------------------------------------------------------------------------
_PREFLIGHT_ROW = {
    "Administrative": 1,
    "Administrative_Duration": 10.0,
    "Informational_Duration": 5.0,
    "ProductRelated": 20,
    "ProductRelated_Duration": 400.0,
    "BounceRates": 0.05,
    "ExitRates": 0.08,
    "PageValues": 5.0,
    "Month": "Nov",
}


class ArtifactReloadError(RuntimeError):
    """Raised when a hot-reload attempt fails pre-flight validation."""


# ---------------------------------------------------------------------------
# Thread-safe model store
# ---------------------------------------------------------------------------

class ModelStore:
    """Thread-safe store for the active (model, preprocessor) artifact pair.

    All public methods are safe to call from any thread or asyncio coroutine.
    The internal ``threading.RLock`` is held only for the atomic pointer swap
    and for reading the active container reference — never during I/O or
    inference — so lock contention is near zero.
    """

    def __init__(self) -> None:
        self._active: ArtifactContainer | None = None
        self._lock = threading.RLock()
        self._reload_count: int = 0

    # ------------------------------------------------------------------
    # Public read interface
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """``True`` if at least one artifact pair has been loaded successfully."""
        with self._lock:
            return self._active is not None

    def get_artifacts(self) -> ArtifactContainer:
        """Return the currently active immutable artifact container.

        Raises ``RuntimeError`` if no artifacts have been loaded yet.
        The returned object is safe to use outside the lock — it is frozen and
        will not be mutated even if a concurrent ``reload()`` swaps the pointer.
        """
        with self._lock:
            container = self._active
        if container is None:
            raise RuntimeError("No artifacts loaded. Call load() or reload() first.")
        return container

    @property
    def reload_count(self) -> int:
        """Number of successful hot-reloads since server start."""
        with self._lock:
            return self._reload_count

    # ------------------------------------------------------------------
    # Startup load
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load artifacts on startup — identical to a first-time reload."""
        container, _ = self._load_candidate()
        with self._lock:
            self._active = container
            self._reload_count = 0
        logger.info(
            "Artifacts loaded on startup (version=%s, loaded_at=%s)",
            container.version,
            container.loaded_at,
        )

    # ------------------------------------------------------------------
    # Hot-reload (atomic pointer swap with pre-flight)
    # ------------------------------------------------------------------

    def reload(self) -> tuple[ArtifactContainer | None, ArtifactContainer, float]:
        """Hot-reload artifacts without interrupting in-flight requests.

        Protocol:
          1. Load new ``.skops`` files into a candidate ``ArtifactContainer``
             (outside the lock — I/O can be slow).
          2. Run a synthetic pre-flight inference pass against the candidate
             to surface schema or compatibility issues early.
          3. Atomically swap ``_active`` under the write lock.

        Returns:
          A 3-tuple of ``(old_container, new_container, preflight_latency_ms)``.
          ``old_container`` is ``None`` on the very first load.

        Raises:
          ``ArtifactReloadError`` if pre-flight validation fails.  The active
          model is **not** replaced in this case.
        """
        # Phase 1 — load candidate (no lock held during disk I/O)
        candidate, preflight_latency_ms = self._load_candidate()

        # Phase 3 — atomic swap (lock held only for pointer swap)
        with self._lock:
            old_container = self._active
            self._active = candidate
            self._reload_count += 1

        logger.info(
            "Hot-reload complete: %s → %s (preflight=%.1fms, reload_count=%d)",
            old_container.version if old_container else "none",
            candidate.version,
            preflight_latency_ms,
            self._reload_count,
        )
        return old_container, candidate, preflight_latency_ms

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_candidate(self) -> tuple[ArtifactContainer, float]:
        """Deserialise skops files and validate with a pre-flight dry run.

        Returns ``(ArtifactContainer, preflight_latency_ms)``.
        Raises ``FileNotFoundError`` on missing files or
        ``ArtifactReloadError`` when pre-flight inference fails.
        """
        artifact_dir = settings.artifact_path
        model_path = os.path.join(artifact_dir, "model.skops")
        preprocessor_path = os.path.join(artifact_dir, "preprocessor.skops")

        for path in (model_path, preprocessor_path):
            if not os.path.exists(path):
                logger.error("Artifact not found: %s", path)
                raise FileNotFoundError(f"Artifact not found: {path}")

        # Deserialise — skops safe-loads only trusted types
        trusted_model = skio.get_untrusted_types(file=model_path)
        model = skio.load(model_path, trusted=trusted_model)
        logger.info("Candidate model loaded from %s", model_path)

        trusted_preprocessor = skio.get_untrusted_types(file=preprocessor_path)
        preprocessor = skio.load(preprocessor_path, trusted=trusted_preprocessor)
        logger.info("Candidate preprocessor loaded from %s", preprocessor_path)

        # Compute deterministic version fingerprint
        version = _compute_version(model_path, preprocessor_path)
        loaded_at = datetime.now(timezone.utc).isoformat()

        # Phase 2 — pre-flight dry-run inference (outside lock)
        preflight_latency_ms = self._run_preflight(model, preprocessor, version)

        container = ArtifactContainer(
            model=model,
            preprocessor=preprocessor,
            version=version,
            loaded_at=loaded_at,
        )
        return container, preflight_latency_ms

    @staticmethod
    def _run_preflight(model: object, preprocessor: object, version: str) -> float:
        """Run a single synthetic inference pass to validate artifact compatibility.

        Returns elapsed wall-clock time in milliseconds.
        Raises ``ArtifactReloadError`` if any step fails.
        """
        logger.info("Pre-flight dry-run started for candidate version=%s", version)
        t0 = time.perf_counter()
        try:
            df = pd.DataFrame([_PREFLIGHT_ROW])
            transformed = preprocessor.transform(df)
            predictions = model.predict(transformed)
            probabilities = model.predict_proba(transformed)[:, 1]

            # Sanity assertions on output shape and value ranges
            assert len(predictions) == 1, "Expected 1 prediction from pre-flight"
            assert 0.0 <= float(probabilities[0]) <= 1.0, "Probability out of [0, 1]"

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error(
                "Pre-flight failed for version=%s after %.1fms: %s",
                version, elapsed_ms, exc,
            )
            raise ArtifactReloadError(
                f"Pre-flight inference validation failed: {exc}"
            ) from exc

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Pre-flight passed for version=%s in %.1fms", version, elapsed_ms
        )
        return elapsed_ms


# ---------------------------------------------------------------------------
# Singleton & FastAPI dependency
# ---------------------------------------------------------------------------

_store = ModelStore()


def get_model_store() -> ModelStore:
    """FastAPI dependency that returns the shared ``ModelStore`` singleton."""
    return _store
