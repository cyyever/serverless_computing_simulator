import heapq
from typing import Callable

from simulated_concept import Container


class CachePolicy:
    def add_to_cache(self, cache: list[Container], container: Container):
        raise NotImplementedError()

    def evict(
        self, cache: list[Container], stop_criteria: Callable, new_container=None
    ) -> list[Container]:
        raise NotImplementedError()


class LRUCachePolicy(CachePolicy):
    def add_to_cache(self, cache: list[Container], container: Container):
        cache.append(container)

    def evict(
        self, cache: list[Container], stop_criteria: Callable, new_container=None
    ) -> list[Container]:
        assert cache
        containers = [(container.reuse_time, container) for container in cache]
        released_memory = 0
        heapq.heapify(containers)
        while containers and not stop_criteria(released_memory):
            container = heapq.heappop(containers)
            released_memory += container[1].memory
        return [container[1] for container in containers]


class GDSFCachePolicy(CachePolicy):
    clock = 0

    def add_to_cache(self, cache: list[Container], container: Container):
        container.set_data("GDSF_clock", GDSFCachePolicy.clock)
        cache.append(container)

    def evict(
        self, cache: list[Container], stop_criteria: Callable, new_container=None
    ) -> list[Container]:
        assert cache
        containers = [
            (
                container.get_data("GDSF_clock")
                + container.use_count
                * container.invocation.fun.container_init_time
                / container.memory,
                container,
            )
            for container in cache
        ]
        released_memory = 0
        max_clock = 0
        heapq.heapify(containers)
        removed_containers = []
        while containers and not stop_criteria(released_memory):
            container = heapq.heappop(containers)
            released_memory += container[1].memory
            max_clock = max(max_clock, container[0])
            if container[1] != new_container:
                removed_containers.append(container[1])
            else:
                return [container[1] for container in containers] + removed_containers
        assert max_clock >= GDSFCachePolicy.clock
        GDSFCachePolicy.clock = max_clock
        return [container[1] for container in containers]


def get_cache_policy(name: str) -> CachePolicy:
    match name:
        case "LRU":
            return LRUCachePolicy()
        case "GDSF":
            return GDSFCachePolicy()
    raise NotImplementedError()
