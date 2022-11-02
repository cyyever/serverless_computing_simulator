import os
from datetime import timedelta

import numpy as np
from cyy_naive_lib.reproducible_random_env import ReproducibleRandomEnv

from clock import VirtualClock
from config import global_config, load_config
from controller import CacheAwareController, Controller, get_controller
from dataset.azure_workload import AzureWorkload
from invoker import CacheInvoker, Invoker


class Simulator:
    def __init__(self):
        self.__global_clock = VirtualClock()
        self.__workload = AzureWorkload()
        node_config = global_config["invoker"]
        self.__controller: Controller = get_controller(global_config["controller_type"])
        self.__invokers: list[Invoker] = []

        if isinstance(self.__controller, CacheAwareController):
            invoker_cls = CacheInvoker
        else:
            invoker_cls = Invoker

        for _ in range(node_config["number"]):
            self.__invokers.append(
                invoker_cls(
                    memory=node_config["memory"] * 1024,
                    cores=node_config["core"],
                )
            )

    def run(self):
        time_duration = timedelta(seconds=1)
        simulation_minutes = global_config["simulation_minutes"]
        while self.__global_clock.elapsed_minutes < simulation_minutes:
            # split invocation per-second
            print("time ", self.__global_clock.elapsed_minutes)
            cur_minute = self.__global_clock.elapsed_minutes
            invocations = self.__workload.generate_invocations(
                cur_minute=int(cur_minute * simulation_minutes / (24 * 60))
            )
            batch_size = len(invocations) // 60
            for i in range(60):
                if i + 1 < 60:
                    batch = invocations[:batch_size]
                    invocations = invocations[batch_size:]
                else:
                    batch = invocations
                assert batch
                for invocation in batch:
                    invocation.invoke_time = self.__global_clock.time_point
                    self.__controller.queue_invocation(invocation)
                while self.__controller.route_invocation(
                    invokers=self.__invokers, clock=self.__global_clock
                ):
                    pass
                for invoker in self.__invokers:
                    invoker.run(time_duration=time_duration)
                self.__global_clock.advance(amount=time_duration)
                self.sync_clock()

        # deliver remaining invocations
        while self.__controller.has_invocation() or any(
            invoker.has_job() for invoker in self.__invokers
        ):
            while self.__controller.route_invocation(
                invokers=self.__invokers, clock=self.__global_clock
            ):
                pass
            for invoker in self.__invokers:
                invoker.run(time_duration=time_duration)
            self.__global_clock.advance(amount=time_duration)
            self.sync_clock()
        total_slowdown = []
        for invoker in self.__invokers:
            total_slowdown += invoker.slowdown
        print("total_slowdown size", len(total_slowdown))
        print("slowdown mean is", np.mean(total_slowdown))
        # print("slowdown std is", np.std(total_slowdown))
        print("90 quantile slowdown is", np.quantile(total_slowdown, 0.9))
        print("max slowdown is", np.max(total_slowdown))

    def sync_clock(self):
        for invoker in self.__invokers:
            invoker.sync_local_clock(global_clock=self.__global_clock)


if __name__ == "__main__":
    load_config()
    random_seed_dir = global_config.get("random_seed_dir", None)
    if random_seed_dir is None:
        random_seed_dir = os.path.join(os.path.dirname(__file__), "random_seed")
    reproducible_env = ReproducibleRandomEnv()
    if os.path.isdir(random_seed_dir):
        reproducible_env.load(seed_dir=random_seed_dir)
    with reproducible_env:
        simulator = Simulator()
        simulator.run()
        reproducible_env.save(seed_dir=random_seed_dir)
