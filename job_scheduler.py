import heapq
from datetime import timedelta

from scipy.stats import bernoulli

from clock import VirtualClock
from simulated_concept import Container


class Scheduler:
    def __init__(self, cores):
        self._jobs: list[Container] = []
        self._cores = cores

    def __call__(
        self, cores: int, time_slice: timedelta, clock: VirtualClock
    ) -> list[Container]:
        raise NotImplementedError()

    def has_job(self) -> bool:
        return self.job_number() != 0

    def job_number(self) -> int:
        return len(self._jobs)

    def add_job(self, container: Container) -> None:
        self._jobs.append(container)

    def _run_batch(
        self, batch: list[Container], time_slice: timedelta, clock: VirtualClock
    ) -> timedelta:
        assert batch
        min_remain_time = min(container.invocation.remain_time for container in batch)
        min_remain_time = min(min_remain_time, time_slice)
        for container in batch:
            container.invocation.run(time_slice=min_remain_time, clock=clock)
        clock.advance(min_remain_time)
        time_slice -= min_remain_time
        return time_slice


class FIFOScheduler(Scheduler):
    def __call__(
        self,
        time_slice: timedelta,
        clock: VirtualClock,
    ) -> list[Container]:
        completed_jobs: list[Container] = []
        while self._jobs and time_slice:
            batch = self._jobs[: self._cores]
            time_slice = self._run_batch(
                batch=batch, time_slice=time_slice, clock=clock
            )
            for i in reversed(range(len(batch))):
                if self._jobs[i].invocation.complete:
                    completed_jobs.append(self._jobs[i])
                    self._jobs.pop(i)
        return completed_jobs


class RRScheduler(Scheduler):
    def __call__(
        self,
        time_slice: timedelta,
        clock: VirtualClock,
    ) -> list[Container]:
        completed_jobs: list[Container] = []
        while self._jobs and time_slice:
            batch = self._jobs[: self._cores]
            time_slice = self._run_batch(
                batch=batch, time_slice=time_slice, clock=clock
            )
            remain_jobs = self._jobs[self._cores:]
            for container in batch:
                if container.invocation.complete:
                    completed_jobs.append(container)
                else:
                    remain_jobs.append(container)
            self._jobs = remain_jobs

        return completed_jobs


class LASScheduler(Scheduler):
    def add_job(self, container: Container) -> None:
        heapq.heappush(
            self._jobs,
            (container.invocation.used_time, container),
        )

    def __call__(
        self,
        time_slice: timedelta,
        clock: VirtualClock,
    ) -> list[Container]:
        completed_jobs: list[Container] = []
        while self._jobs and time_slice:
            batch = []
            for _ in range(self._cores):
                if not self._jobs:
                    break
                batch.append(heapq.heappop(self._jobs)[1])
            time_slice = self._run_batch(
                batch=batch, time_slice=time_slice, clock=clock
            )
            for container in batch:
                if container.invocation.complete:
                    completed_jobs.append(container)
                else:
                    self.add_job(container)
        return completed_jobs


class SRTFScheduler(Scheduler):
    def add_job(self, container: Container) -> None:
        heapq.heappush(
            self._jobs,
            (container.invocation.remain_time, container),
        )

    def __call__(
        self,
        time_slice: timedelta,
        clock: VirtualClock,
    ) -> tuple[list[Container], list[Container]]:
        completed_jobs: list[Container] = []
        while self._jobs and time_slice:
            batch = []
            for _ in range(self._cores):
                if not self._jobs:
                    break
                batch.append(heapq.heappop(self._jobs)[1])
            time_slice = self._run_batch(
                batch=batch, time_slice=time_slice, clock=clock
            )
            for container in batch:
                if container.invocation.complete:
                    completed_jobs.append(container)
                else:
                    self.add_job(container)
        return completed_jobs


class LotterySRTFScheduler(Scheduler):
    known_job_IDs: set[str] = set()
    max_prob = 9 / 10

    def __init__(self, cores):
        super().__init__(cores=cores)
        self.__SRTF = None
        self.__LAS = None
        self._known_jobs = {}
        self._unknown_jobs = {}
        self._unknown_fun_ids = {}

    def add_job(self, container: Container) -> None:
        is_known_job = False
        if container.fun_id in LotterySRTFScheduler.known_job_IDs:
            is_known_job = True
            self._known_jobs[container.id] = container
        else:
            self._unknown_jobs[container.id] = container
            if container.fun_id not in self._unknown_fun_ids:
                self._unknown_fun_ids[container.fun_id] = []
            self._unknown_fun_ids[container.fun_id].append(container)
        if self.__LAS is not None:
            self.__LAS.add_job(container)
        else:
            if is_known_job and self.__SRTF is not None:
                self.__SRTF.add_job(container)

    def job_number(self) -> int:
        return len(self._known_jobs) + len(self._unknown_jobs)

    def __call__(
        self,
        time_slice: timedelta,
        clock: VirtualClock,
    ) -> tuple[list[Container], list[Container]]:
        already_known_jobs = LotterySRTFScheduler.known_job_IDs.intersection(
            set(self._unknown_fun_ids.keys())
        )

        if already_known_jobs:
            self.__LAS = None
            for k in already_known_jobs:
                for container in self._unknown_fun_ids[k]:
                    self._unknown_jobs.pop(container.id)
                    self.add_job(container=container)
                self._unknown_fun_ids.pop(k)

        use_SRTF_completed_with_LAS = False
        if self.job_number() <= self._cores:
            use_SRTF = False
        elif not self._known_jobs:
            use_SRTF = False
        else:
            p = min(self.max_prob, len(self._known_jobs) / self.job_number())
            use_SRTF = bernoulli.rvs(p=p, size=1).tolist()[0]
            if use_SRTF and len(self._known_jobs) < self._cores:
                use_SRTF = False
                use_SRTF_completed_with_LAS = True
        if use_SRTF:
            assert self._known_jobs
            self.__LAS = None
            if self.__SRTF is None:
                self.__SRTF = SRTFScheduler(cores=self._cores)
                for job in self._known_jobs.values():
                    self.__SRTF.add_job(job)
            completed_jobs = self.__SRTF(
                time_slice=time_slice,
                clock=clock,
            )
            for container in completed_jobs:
                self._known_jobs.pop(container.id)
            return completed_jobs
        # print("use LAS")
        self.__SRTF = None
        self.__LAS = LASScheduler(cores=self._cores)
        for job in self._known_jobs.values():
            self.__LAS.add_job(job)
        if use_SRTF_completed_with_LAS:
            for job in self._unknown_jobs.values():
                if self.__LAS.job_number() < self._cores:
                    self.__LAS.add_job(job)
                else:
                    break
        else:
            for job in self._unknown_jobs.values():
                self.__LAS.add_job(job)
        completed_jobs = self.__LAS(
            time_slice=time_slice,
            clock=clock,
        )
        for container in completed_jobs:
            self._known_jobs.pop(container.id, None)
            if container.id in self._unknown_jobs:
                self._unknown_jobs.pop(container.id)
                fun_id = container.fun_id
                LotterySRTFScheduler.known_job_IDs.add(fun_id)
                for other_container in self._unknown_fun_ids.get(fun_id, []):
                    if container.id == other_container.id:
                        continue
                    if not container.invocation.complete:
                        self.add_job(container=other_container)
                    self.__LAS = None
                self._unknown_fun_ids.pop(fun_id, None)
        return completed_jobs


def get_scheduler(name: str, cores: int) -> Scheduler:
    print(f"use {name} scheduler")
    match name:
        case "RR":
            return RRScheduler(cores=cores)
        case "FIFO":
            return FIFOScheduler(cores=cores)
        case "SRTF":
            return SRTFScheduler(cores=cores)
        case "LotterySRTF":
            return LotterySRTFScheduler(cores=cores)
        case "LAS":
            return LASScheduler(cores=cores)
    raise NotImplementedError()
