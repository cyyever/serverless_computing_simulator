import os
import random
import sys
from datetime import timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import global_config
from simulated_concept import (Invocation, SimulatedApplication,
                               SimulatedFunction)

from azure_distribution import (fit_fun_execution_time_distribution,
                                fit_fun_invocation_distribution,
                                fit_memory_distribution)


class AzureWorkload:
    def __init__(self):
        self.__registered_applications: list[
            SimulatedApplication
        ] = self.__sample_azure_application()
        assert self.__registered_applications
        print(
            "generate",
            len(self.__registered_applications),
            "applications",
        )
        print(
            "generate",
            sum(len(app.functions) for app in self.__registered_applications),
            "functions",
        )
        self.__invocation_poly = fit_fun_invocation_distribution(
            trigger="http", weekday=True
        )[1]

    def generate_invocations(self, cur_minute: int) -> list[Invocation]:
        application_invocation_limit = global_config["application_invocation_limit"]

        function_invocations = {}
        for application in self.__registered_applications:
            for function in application.functions:
                invocation_count = int(self.__invocation_poly(cur_minute))
                if invocation_count <= 0:
                    raise RuntimeError(cur_minute, self.__invocation_poly(cur_minute))
                function_invocations[function] = invocation_count

        total_invocation = sum(function_invocations.values())
        invocations: list[Invocation] = []
        for fun, invocation_number in function_invocations.items():
            invocation_number = int(
                round(
                    invocation_number * application_invocation_limit / total_invocation,
                    0,
                )
            )
            assert invocation_number > 0
            for _ in range(invocation_number):
                invocations.append(Invocation(fun=fun, app=application))

        assert invocations
        random.shuffle(invocations)
        return invocations

    @classmethod
    def __sample_azure_application(cls) -> list[SimulatedApplication]:
        gm = fit_memory_distribution()[-1]
        application_number = global_config["application_number"]
        X = gm.sample(n_samples=application_number)
        memory_list = X[0].reshape(-1).astype(dtype=np.int64).tolist()
        applications: list[SimulatedApplication] = []
        functions = cls.__sample_azure_function(size=5 * application_number)
        for i in range(application_number):
            applications.append(SimulatedApplication(memory=memory_list[i]))
            for _ in range(random.randint(1, 5)):
                applications[-1].add_fun(functions[0])
                functions = functions[1:]
        return applications

    @classmethod
    def __sample_azure_function(cls, size: int) -> list[SimulatedFunction]:
        exec_time_list = (
            fit_fun_execution_time_distribution(triggers={"http"}, fit_trigger="http")[
                -1
            ](size=size)
            .reshape(-1)
            .astype(dtype=np.int64)
            .clip(1, None)
            .tolist()
        )
        return [
            SimulatedFunction(exec_time=timedelta(milliseconds=exec_time))
            for exec_time in exec_time_list
        ]
