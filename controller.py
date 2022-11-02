from typing import Any

import numpy as np

from clock import VirtualClock
from invoker import CacheInvoker, Invoker
from simulated_concept import Invocation


class Controller:
    def __init__(self):
        self._invoker_stat = []
        self._queue: list[Invocation] = []
        self.__free_memory = None
        self._job_numbers = None
        self._cores = None

    def _check_memory(self):
        if not self.has_invocation():
            return None
        invocation = self._queue[0]
        mask = self.__free_memory >= invocation.app.memory
        if not np.any(mask):
            return None
        return mask

    def has_invocation(self) -> bool:
        return bool(self._queue)

    def queue_invocation(self, invocation: Invocation):
        self._queue.append(invocation)

    def collect_invoker_stat(self, invokers: list[Invoker]) -> None:
        self._invoker_stat = [i.get_performance_stat() for i in invokers]
        self.__free_memory = np.asarray(
            [stat["free_memory"] for stat in self._invoker_stat]
        )
        self._job_numbers = np.asarray(
            [stat["job_number"] for stat in self._invoker_stat]
        )
        self._cores = np.asarray([stat["cores"] for stat in self._invoker_stat])

    def route_invocation(self, invokers: list[Invoker], clock: VirtualClock) -> bool:
        self.collect_invoker_stat(invokers=invokers)
        mask = self._check_memory()
        if mask is None:
            return False
        invocation = self._queue.pop(0)
        index, cache_idx, cache_level = self.decide_invoker(
            mask=mask, invokers=invokers, invocation=invocation
        )
        match cache_level:
            case 0:
                invocation.set_exec_time(invocation.fun.exec_time)
            case 1:
                invocation.set_exec_time(
                    invocation.fun.exec_time + invocation.fun.app_init_time
                )
            case 2:
                invocation.set_exec_time(
                    invocation.fun.exec_time
                    + invocation.fun.app_init_time
                    + invocation.fun.container_init_time
                )
            case 3:
                invocation.set_exec_time(invocation.fun.total_cost)
        self.__free_memory[index] -= invocation.app.memory
        invokers[index].add_new_job(
            invocation=invocation, clock=clock, cache_idx=cache_idx
        )
        self._job_numbers[index] += 1
        return True

    def decide_invoker(
        self, mask, invokers: list[Invoker], invocation: Invocation
    ) -> tuple[int, None | int]:
        raise NotImplementedError()


class LeastLoadController(Controller):
    def decide_invoker(
        self, mask, invokers: list[Invoker], invocation: Invocation
    ) -> tuple[int, None | int]:
        loads = (self._job_numbers / self._cores)[mask]
        idx = np.argmin(loads)
        return np.arange(mask.shape[0])[mask][idx], None, 3


class CacheAwareController(Controller):
    def decide_invoker(
        self, mask, invokers: list[Invoker], invocation: Invocation
    ) -> tuple[int, Any]:
        highest_cache_level = 3
        invoker_idx = None
        final_cache_idx = None
        load = None
        loads = self._job_numbers / self._cores
        has_cache = False
        for idx, stat in enumerate(self._invoker_stat):
            if not mask[idx]:
                continue
            cache = stat["cache"]
            if not cache and loads[idx] == 0:
                invoker_idx = idx
                final_cache_idx = None
                break
            if cache:
                has_cache = True
            cache_idx, cache_level = CacheInvoker.get_cache(
                cache=cache, invocation=invocation
            )
            if cache_level < highest_cache_level:
                highest_cache_level = cache_level
                invoker_idx = idx
                final_cache_idx = cache_idx
                load = loads[idx]
            elif cache_level == highest_cache_level and (
                load is None or loads[idx] < load
            ):
                invoker_idx = idx
                final_cache_idx = cache_idx
                load = loads[idx]
        if highest_cache_level == 3:
            assert not has_cache
        assert invoker_idx is not None
        return invoker_idx, final_cache_idx, highest_cache_level


def get_controller(name: str) -> Controller:
    match name:
        case "leastload":
            return LeastLoadController()
        case "cacheaware":
            return CacheAwareController()
    raise NotImplementedError()
