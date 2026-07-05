"""Dynamic (server-side) request batcher for the RetainStack prediction API.

Individual ``POST /api/v1/predict`` requests are enqueued as ``PendingRequest``
objects, each carrying an ``asyncio.Future``.  A single background worker task
drains the queue into a vectorised inference pass whenever one of two conditions
fires first:

* The queue reaches ``BATCH_MAX_SIZE`` items, **or**
* ``BATCH_TIMEOUT_MS`` milliseconds have elapsed since the first item joined
  the current batch.

Callers simply ``await batcher.submit(row)`` and block until the batch they
were grouped into has been processed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pandas as pd

from src.config import settings
from src.logger.logger import get_logger
from src.api.schemas.response import PredictionResult

logger = get_logger("api.batcher")


def _confidence_label(probability: float) -> str:
    """Map a purchase probability to a human-readable confidence bucket."""
    if probability >= 0.75:
        return "high"
    if probability >= 0.40:
        return "medium"
    return "low"


@dataclass
class PendingRequest:
    """One session waiting to be batched.

    ``row`` is the ordered feature dict ready for ``pd.DataFrame``.
    ``future`` is resolved with a ``PredictionResult`` when the batch runs,
    or set with an exception if inference fails.
    """

    row: dict
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())


class DynamicBatcher:
    """Accumulates individual prediction requests and processes them in batches.

    Usage::

        batcher = DynamicBatcher()
        batcher.start(store, feature_order)   # called once at startup
        result = await batcher.submit(row)    # called per request
        await batcher.stop()                  # called once at shutdown
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[PendingRequest] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._store = None
        self._feature_order: list[str] = []

    def start(self, store, feature_order: list[str]) -> None:
        """Launch the background worker. Must be called after the model is loaded."""
        self._store = store
        self._feature_order = feature_order
        self._worker_task = asyncio.create_task(self._worker(), name="batcher-worker")
        logger.info(
            "DynamicBatcher started (max_size=%d, timeout=%dms)",
            settings.BATCH_MAX_SIZE,
            settings.BATCH_TIMEOUT_MS,
        )

    async def stop(self) -> None:
        """Cancel the background worker and await its completion."""
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("DynamicBatcher stopped.")

    async def submit(self, row: dict) -> PredictionResult:
        """Enqueue one session and await its prediction result.

        Suspends the calling coroutine until the batch this request is
        grouped into has been processed by the background worker.
        """
        loop = asyncio.get_running_loop()
        pending = PendingRequest(row=row, future=loop.create_future())
        await self._queue.put(pending)
        return await pending.future

    async def _worker(self) -> None:
        """Drain loop: waits for the first item then collects more until
        BATCH_MAX_SIZE or BATCH_TIMEOUT_MS, then flushes."""
        timeout_s = settings.BATCH_TIMEOUT_MS / 1000.0

        while True:
            try:
                first = await self._queue.get()
            except asyncio.CancelledError:
                break

            pending: list[PendingRequest] = [first]

            deadline = asyncio.get_event_loop().time() + timeout_s
            while len(pending) < settings.BATCH_MAX_SIZE:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    pending.append(item)
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    await self._run_batch(pending)
                    return

            await self._run_batch(pending)

    async def _run_batch(self, pending: list[PendingRequest]) -> None:
        """Run a single vectorised inference pass and resolve all futures."""
        logger.info("Flushing batch of %d request(s)", len(pending))
        df = pd.DataFrame([p.row for p in pending])

        try:
            transformed = self._store.preprocessor.transform(df)
            raw_predictions = self._store.model.predict(transformed).tolist()
            raw_probabilities = self._store.model.predict_proba(transformed)[:, 1].tolist()
        except Exception as exc:
            logger.error("Batch inference failed: %s", exc)
            for p in pending:
                if not p.future.done():
                    p.future.set_exception(exc)
            return

        for p, pred, prob in zip(pending, raw_predictions, raw_probabilities):
            result = PredictionResult(
                prediction=int(pred),
                purchase_probability=round(float(prob), 4),
                confidence=_confidence_label(float(prob)),
            )
            if not p.future.done():
                p.future.set_result(result)

        logger.debug(
            "Batch resolved: preds=%s probs=%s",
            raw_predictions,
            [round(p, 4) for p in raw_probabilities],
        )


_batcher = DynamicBatcher()


def get_batcher() -> DynamicBatcher:
    """FastAPI dependency that returns the shared DynamicBatcher."""
    return _batcher

