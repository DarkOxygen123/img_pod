import asyncio
from typing import Any, Callable, Optional

from shared.logging_config import get_logger

logger = get_logger(__name__)


class BoundedQueue:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._queue: asyncio.Queue[Any] = asyncio.Queue()

    @property
    def size(self) -> int:
        return self._queue.qsize()

    def full(self) -> bool:
        return self.size >= self.capacity

    async def put(self, item: Any) -> None:
        if self.full():
            raise asyncio.QueueFull()
        await self._queue.put(item)

    async def get(self) -> Any:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()


class SingleWorker:
    def __init__(self, queue: BoundedQueue, handler: Callable[[Any], asyncio.Future]):
        self.queue = queue
        self.handler = handler
        self._running = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lock:
            if self._running:
                return
            self._running = True
            asyncio.create_task(self._run())

    async def _run(self) -> None:
        while self._running:
            item = await self.queue.get()
            try:
                await self.handler(item)
            except Exception:  # pragma: no cover - logging safeguard
                logger.exception("worker_handler_failed", extra_fields={"item": str(item)})
            finally:
                self.queue.task_done()

    async def stop(self) -> None:
        async with self._lock:
            self._running = False
