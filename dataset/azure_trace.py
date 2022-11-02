import os
import pickle
import sys

import numpy
import numpy.typing as npt
import pandas
from config import global_config

_application_trace_list = None


class FunctionTrace:
    def __init__(self, name):
        self.name = name
        self.__invocation_numbers = {}
        # self.__percentile_avg_execution_times: list | npt.ArrayLike = []
        self.__avg_execution_times: list[int] = []
        self.__trigger_count: dict = {}

    @property
    def trigger_count(self) -> dict:
        return self.__trigger_count

    @property
    def avg_execution_times(self) -> list[int]:
        return self.__avg_execution_times

    # @property
    # def avg_execution_time_pct75(self) -> int | None:
    #     if isinstance(self.__percentile_avg_execution_times, list):
    #         self.__percentile_avg_execution_times = numpy.asarray(
    #             self.__percentile_avg_execution_times
    #         )
    #     return self.__percentile_avg_execution_times[:, -3]

    def add_execution_time(
        self, avg_exec_time: int, percentile_avg_exec_times: npt.ArrayLike
    ) -> None:
        if avg_exec_time > 0 and numpy.sum(percentile_avg_exec_times) > 0:
            self.__avg_execution_times.append(avg_exec_time)
            # self.__percentile_avg_execution_times.append(percentile_avg_exec_times)

    @property
    def invocation_numbers(self):
        return self.__invocation_numbers

    def add_invocation_number(
        self, day: int, trigger: str, invocation_number: npt.ArrayLike
    ) -> None:
        if numpy.sum(invocation_number) <= 0:
            return
        if trigger not in self.__trigger_count:
            self.__trigger_count[trigger] = 0
        self.__trigger_count[trigger] += numpy.sum(invocation_number)
        if day not in self.__invocation_numbers:
            self.__invocation_numbers[day] = {}

        day_invocation_numbers = self.__invocation_numbers[day]
        if trigger not in day_invocation_numbers:
            day_invocation_numbers[trigger] = invocation_number
        else:
            day_invocation_numbers[trigger] += invocation_number

    def is_complete(self) -> bool:
        return (
            self.invocation_numbers
            and self.avg_execution_times
            # and self.__percentile_avg_execution_times
        )


class ApplicationTrace:
    def __init__(self, name):
        self.name = name
        self.functions: dict[str, FunctionTrace] = {}
        self.avg_memory = []
        # self.avg_memory_pct75 = []

    def add_function_trace(self, fun: FunctionTrace) -> None:
        assert fun.name not in self.functions
        self.functions[fun.name] = fun

    def get_function_trace(self, fun_name: str) -> FunctionTrace | None:
        return self.functions.get(fun_name, None)

    def drop_incomplete_fun_traces(self) -> None:
        self.functions = {k: v for k, v in self.functions.items() if v.is_complete()}

    def is_complete(self) -> bool:
        return self.functions and self.avg_memory


def __load_fun_invocation(trace_dir: str, application_traces: dict) -> None:
    for day in range(1, 15):
        file = os.path.join(
            trace_dir, f"invocations_per_function_md.anon.d{day:02}.csv"
        )
        df = pandas.read_csv(file)
        for _, row in df.iterrows():
            app_name = row["HashApp"]
            if app_name not in application_traces:
                application_traces[app_name] = ApplicationTrace(name=app_name)
            app = application_traces[app_name]
            fun_name = row["HashFunction"]
            fun = app.get_function_trace(fun_name=fun_name)
            if fun is None:
                fun = FunctionTrace(name=fun_name)
                app.add_function_trace(fun)
            fun.add_invocation_number(day, row["Trigger"], row[4:].to_numpy())


def __load_fun_duration(trace_dir: str, application_traces: dict) -> None:
    for day in range(1, 15):
        file = os.path.join(
            trace_dir, f"function_durations_percentiles.anon.d{day:02}.csv"
        )
        df = pandas.read_csv(file)
        for _, row in df.iterrows():
            app_name = row["HashApp"]
            if app_name not in application_traces:
                application_traces[app_name] = ApplicationTrace(app_name)
            app = application_traces[app_name]
            fun_name = row["HashFunction"]
            fun = app.get_function_trace(fun_name=fun_name)
            if fun is None:
                fun = FunctionTrace(name=fun_name)
                app.add_function_trace(fun)
            fun.add_execution_time(row["Average"], row[7:].to_numpy())


def __load_memory(trace_dir: str, application_traces: dict) -> None:
    for day in range(1, 12):
        file = os.path.join(trace_dir, f"app_memory_percentiles.anon.d{day:02}.csv")
        df = pandas.read_csv(file)
        for _, row in df.iterrows():
            app_name = row["HashApp"]
            if app_name not in application_traces:
                application_traces[app_name] = ApplicationTrace(app_name)
            app = application_traces[app_name]
            app.avg_memory.append(row["AverageAllocatedMb"])


cache_dir = os.path.join(os.path.dirname(__file__), ".cache")


def load_azure_trace() -> list[ApplicationTrace]:
    trace_dir = global_config["azure_trace_dir"]
    global _application_trace_list
    if _application_trace_list is not None:
        return _application_trace_list
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "azure_trace.pk")
    if os.path.isfile(cache_file):
        print(f"load cache from {cache_file}")
        with open(cache_file, "rb") as f:
            sys.path.insert(0, os.path.dirname(__file__))
            _application_trace_list = pickle.load(f)
            return _application_trace_list
    application_traces: dict[str, ApplicationTrace] = {}
    __load_fun_invocation(trace_dir=trace_dir, application_traces=application_traces)
    __load_fun_duration(trace_dir=trace_dir, application_traces=application_traces)
    __load_memory(trace_dir=trace_dir, application_traces=application_traces)
    _application_trace_list = list(application_traces.values())
    with open(cache_file, "wb") as f:
        pickle.dump(_application_trace_list, f)
    return _application_trace_list
