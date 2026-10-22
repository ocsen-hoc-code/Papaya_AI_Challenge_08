import asyncio
import threading

import pytest
from PIL import Image

from medextract.backends.base import GenerationResult, ThreadedBackend, VLMError
from medextract.backends.fake import FakeBackend


def _img():
    return Image.new("RGB", (8, 8), "white")


async def test_fake_backend_returns_scripted_responses_in_order():
    backend = FakeBackend(["first", "second"])
    assert (await backend.generate([_img()], "p1")).text == "first"
    assert (await backend.generate([_img()], "p2")).text == "second"
    assert (await backend.generate([_img()], "p3")).text == "second"  # last one repeats
    assert [c.prompt for c in backend.calls] == ["p1", "p2", "p3"]


async def test_fake_backend_callable_and_exceptions():
    backend = FakeBackend(lambda images, prompt: "A" if "classify" in prompt else "B")
    assert (await backend.generate([], "please classify")).text == "A"
    assert (await backend.generate([], "extract")).text == "B"

    failing = FakeBackend([VLMError("boom")])
    with pytest.raises(VLMError):
        await failing.generate([], "x")


async def test_fake_backend_as_async_context_manager():
    async with FakeBackend(["ok"]) as backend:
        assert (await backend.generate([], "x")).text == "ok"
    assert backend.describe() == {"backend": "fake", "model_id": "fake-model"}


class _SlowThreaded(ThreadedBackend):
    """Records which threads run model code and how many calls overlap."""

    name = "test"

    def __init__(self):
        super().__init__()
        self.load_count = 0
        self.threads: set[str] = set()
        self.active = 0
        self.max_active = 0
        self._guard = threading.Lock()

    def _load_sync(self):
        self.load_count += 1

    def _generate_sync(self, images, prompt, system, max_tokens, temperature):
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.threads.add(threading.current_thread().name)
        threading.Event().wait(0.02)
        with self._guard:
            self.active -= 1
        return GenerationResult(text=prompt)


async def test_threaded_backend_serializes_calls_on_one_thread_and_loads_once():
    backend = _SlowThreaded()
    results = await asyncio.gather(*(backend.generate([_img()], f"p{i}") for i in range(5)))
    assert [r.text for r in results] == [f"p{i}" for i in range(5)]
    assert backend.load_count == 1
    assert backend.max_active == 1
    assert len(backend.threads) == 1
    assert all(r.elapsed_s > 0 for r in results)
    await backend.close()


async def test_threaded_backend_keeps_event_loop_responsive():
    backend = _SlowThreaded()
    ticks = 0

    async def ticker():
        nonlocal ticks
        for _ in range(5):
            await asyncio.sleep(0.005)
            ticks += 1

    await asyncio.gather(backend.generate([], "x"), ticker())
    assert ticks == 5
    await backend.close()
