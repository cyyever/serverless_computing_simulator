import random
from datetime import timedelta

from clock import VirtualClock


class SimulatedFunction:
    __next_id: int = 0

    def __init__(self, exec_time: timedelta):
        self.id = f"fun_{SimulatedFunction.__next_id}"
        self.__exec_time: timedelta = exec_time
        # add container startup time
        self.__container_init_time = timedelta(milliseconds=random.randint(1000, 1500))
        exec_time_ms = exec_time / timedelta(milliseconds=1)
        self.__app_init_time = timedelta(
            milliseconds=random.randint(
                int(exec_time_ms * 5 / 100), int(exec_time_ms * 10 / 100)
            )
        )
        self.__fun_init_time = timedelta(
            milliseconds=random.randint(
                int(exec_time_ms * 5 / 100), int(exec_time_ms * 10 / 100)
            )
        )
        SimulatedFunction.__next_id += 1

    @property
    def exec_time(self):
        return self.__exec_time

    @property
    def container_init_time(self):
        return self.__container_init_time

    @property
    def app_init_time(self):
        return self.__app_init_time

    @property
    def fun_init_time(self):
        return self.__fun_init_time

    @property
    def total_cost(self):
        return (
            self.__exec_time
            + self.__container_init_time
            + self.__app_init_time
            + self.__fun_init_time
        )


class SimulatedApplication:
    __next_id: int = 0

    def __init__(self, memory):
        self.id = f"app_{SimulatedApplication.__next_id}"
        self.memory = memory
        self.functions: list[SimulatedFunction] = []
        SimulatedApplication.__next_id += 1

    def add_fun(self, fun: SimulatedFunction) -> None:
        self.functions.append(fun)


class Invocation:
    __next_id: int = 0

    def __init__(self, fun: SimulatedFunction, app: SimulatedApplication):
        self.id = f"invocation_{Invocation.__next_id}"
        Invocation.__next_id += 1
        self.fun: SimulatedFunction = fun
        self.app: SimulatedApplication = app
        self.invoke_time: None | timedelta = None
        self.finish_time: None | timedelta = None
        self.__used_time: timedelta = timedelta()
        self.__remain_time: timedelta = fun.total_cost

    def set_exec_time(self, exec_time: int):
        self.__remain_time = exec_time

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self):
        return f"{self.fun}_{self.__remain_time}"

    @property
    def used_time(self) -> timedelta:
        return self.__used_time

    @property
    def remain_time(self) -> timedelta:
        return self.__remain_time

    @property
    def memory(self):
        return self.app.memory

    @property
    def complete(self) -> bool:
        return self.finish_time is not None

    @property
    def slowdown(self) -> float:
        assert self.complete
        return (self.finish_time - self.invoke_time) / self.fun.exec_time

    def run(self, time_slice: timedelta, clock: VirtualClock) -> None:
        assert not self.complete
        assert self.invoke_time is not None

        if self.__remain_time <= time_slice:
            self.finish_time = clock.time_point + self.__remain_time
            self.__used_time += self.__remain_time
            self.__remain_time -= self.__remain_time
            return
        self.__remain_time -= time_slice
        self.__used_time += time_slice


class Container:
    __next_id: int = 0

    def __init__(self, invocation: Invocation, clock: VirtualClock):
        self.id = f"container_{Container.__next_id}"
        Container.__next_id += 1
        self.__use_count = 1
        self.__reuse_time = clock.time_point
        self.invocation = invocation
        self.data = {}

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):
        return self.id < other.id

    def load_invocation(self, invocation: Invocation, clock: VirtualClock):
        self.invocation = invocation
        self.__use_count += 1
        self.__reuse_time = clock.time_point

    def set_data(self, key, value):
        self.data[key] = value

    def get_data(self, key):
        return self.data[key]

    @property
    def reuse_time(self):
        return self.__reuse_time

    @property
    def use_count(self):
        return self.__use_count

    @property
    def fun_id(self):
        return self.invocation.fun.id

    @property
    def app_id(self):
        return self.invocation.app.id

    @property
    def memory(self):
        return self.invocation.app.memory
