"""Tracked asyncio background tasks — canceled on shutdown to prevent blocking uvicorn reload."""
import asyncio
from typing import Set


class BackgroundTaskRegistry:
    def __init__(self):
        self._tasks: Set[asyncio.Task] = set()

    def create_task(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def cancel_all(self, timeout: float = 5.0) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=timeout)
            for task in pending:
                self._tasks.discard(task)

    @property
    def active_count(self) -> int:
        return len(self._tasks)


task_registry = BackgroundTaskRegistry()
