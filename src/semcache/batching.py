"""Micro-batching for the embedder.

A CPU transformer embeds 32 short prompts in one forward pass much faster than
32 separate passes, and separate passes from many request threads fight over
the same cores. ``EmbedBatcher`` queues concurrent requests, waits at most
``max_wait_ms`` for company, and runs one ``embed`` call per batch in a worker
thread. Under no load a request waits only the ``max_wait_ms`` window.
"""

from __future__ import annotations

import asyncio
from typing import Callable, List, Optional, Sequence, Tuple

import anyio
import numpy as np

EmbedFn = Callable[[Sequence[str]], np.ndarray]


class EmbedBatcher:
    def __init__(self, embed: EmbedFn, max_batch: int = 32, max_wait_ms: float = 2.0) -> None:
        if max_batch <= 0:
            raise ValueError("max_batch must be positive")
        self._embed = embed
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000.0
        self._queue: Optional[asyncio.Queue[Tuple[str, asyncio.Future[np.ndarray]]]] = None
        self._worker: Optional[asyncio.Task[None]] = None
        self.batches = 0
        self.items = 0

    def _ensure(self) -> asyncio.Queue[Tuple[str, asyncio.Future[np.ndarray]]]:
        if self._queue is None or self._worker is None or self._worker.done():
            self._queue = asyncio.Queue()
            self._worker = asyncio.get_running_loop().create_task(self._run(self._queue))
        return self._queue

    async def embed(self, text: str) -> np.ndarray:
        fut: asyncio.Future[np.ndarray] = asyncio.get_running_loop().create_future()
        await self._ensure().put((text, fut))
        return await fut

    async def _run(self, queue: asyncio.Queue[Tuple[str, asyncio.Future[np.ndarray]]]) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch: List[Tuple[str, asyncio.Future[np.ndarray]]] = [await queue.get()]
            deadline = loop.time() + self.max_wait
            while len(batch) < self.max_batch:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(queue.get(), remaining))
                except asyncio.TimeoutError:
                    break
            texts = [t for t, _ in batch]
            try:
                vecs = await anyio.to_thread.run_sync(self._embed, texts)
            except Exception as exc:
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)
                continue
            self.batches += 1
            self.items += len(batch)
            for i, (_, fut) in enumerate(batch):
                if not fut.done():
                    fut.set_result(vecs[i])

    async def aclose(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):
                pass
            self._worker = None
            self._queue = None
