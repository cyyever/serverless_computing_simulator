import copy
from datetime import timedelta

from cache_policy import get_cache_policy
from clock import VirtualClock
from config import global_config
from job_scheduler import Scheduler, get_scheduler
from simulated_concept import Container, Invocation


class Invoker:
    time_slice = timedelta(milliseconds=10)
    __next_id: int = 0

    def __init__(self, memory, cores):
        self.id = f"invoker_{Invoker.__next_id}"
        Invoker.__next_id += 1
        self._total_memory = memory
        self._free_memory = memory
        self.__cores = cores
        self.__slowdown = []
        self._scheduler: Scheduler = get_scheduler(
            name=global_config["scheduler_type"], cores=cores
        )
        self.__clock = VirtualClock()

    def has_job(self) -> bool:
        return self._scheduler.has_job()

    @property
    def slowdown(self) -> list:
        return self.__slowdown

    @property
    def load(self) -> float:
        return self._scheduler.job_number() / self.__cores

    def add_new_job(self, invocation: Invocation, clock: VirtualClock, cache_idx=None):
        self._scheduler.add_job(Container(invocation=invocation, clock=clock))
        self._free_memory -= invocation.app.memory

    def get_performance_stat(self) -> dict:
        return {
            "free_memory": self._free_memory,
            "cores": self.__cores,
            "job_number": self._scheduler.job_number(),
        }

    def sync_local_clock(self, global_clock: VirtualClock):
        assert self.__clock.time_point <= global_clock.time_point
        self.__clock = copy.deepcopy(global_clock)

    def run(self, time_duration: timedelta):
        assert self._scheduler is not None
        number_of_time_slice = int(time_duration / self.time_slice)
        for _ in range(number_of_time_slice):
            if not self._scheduler.has_job():
                return
            finished_containers = self._scheduler(
                time_slice=self.time_slice,
                clock=self.__clock,
            )
            if finished_containers:
                self._process_finished_container(finished_containers)
                if not self.has_job:
                    assert self._total_memory == self._free_memory

    def _process_finished_container(self, finished_containers):
        for container in finished_containers:
            self._free_memory += container.memory
            self.__slowdown.append(container.invocation.slowdown)


class CacheInvoker(Invoker):
    def __init__(self, memory, cores):
        super().__init__(memory=memory, cores=cores)
        self.__cache: list[Container] = []
        self.__cache_policy = get_cache_policy(name=global_config["cache_policy"])

    @property
    def free_memory_without_cache(self):
        return self._free_memory - sum(
            (invocation.app.memory for invocation in self.__cache), start=0
        )

    def get_performance_stat(self) -> dict:
        return super().get_performance_stat() | {
            "cache": self.__cache,
        }

    def add_new_job(self, invocation: Invocation, clock: VirtualClock, cache_idx=None):
        if cache_idx is not None:
            container = self.__cache.pop(cache_idx)
            container.load_invocation(invocation, clock=clock)
            self._scheduler.add_job(container=container)
            self._free_memory -= invocation.app.memory
        else:
            self._scheduler.add_job(Container(invocation=invocation, clock=clock))
            self._free_memory -= invocation.app.memory
            free_memory_without_cache = self.free_memory_without_cache
            if free_memory_without_cache < 0:
                print("perform evict")
                self.__cache_policy.evict(
                    self.__cache,
                    lambda released_memory: free_memory_without_cache + released_memory
                    >= 0,
                )
                assert self.free_memory_without_cache >= 0

    @classmethod
    def get_cache(
        cls, cache: list[Invocation], invocation: Invocation
    ) -> tuple[int, int]:
        min_memory = None
        cache_idx = None
        cache_level: int = 3
        for idx, cached_container in enumerate(cache):
            if cached_container.fun_id == invocation.fun.id:
                cache_level = 0
                cache_idx = idx
                break
            if cache_level == 1:
                continue
            if invocation.app.id == cached_container.app_id:
                cache_level = 1
                cache_idx = idx
                continue

            if invocation.memory <= cached_container.memory and (
                min_memory is None or cached_container.memory < min_memory
            ):
                cache_idx = idx
                cache_level = 2
                min_memory = cached_container.memory
        if cache_level == 3 and cache:
            print("cache miss", cache)
            raise RuntimeError("fdfds")
        return cache_idx, cache_level

    def _process_finished_container(self, finished_containers):
        super()._process_finished_container(finished_containers)
        assert finished_containers
        for container in finished_containers:
            self.__cache_policy.add_to_cache(cache=self.__cache, container=container)
