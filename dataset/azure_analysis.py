import os
import sys

import numpy as np
import scipy

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import global_config, load_config

from azure_distribution import (fit_fun_execution_time_distribution,
                                fit_fun_invocation_distribution,
                                fit_memory_distribution)
from azure_trace import ApplicationTrace, load_azure_trace


def plot_memory_cv_distribution(
    azure_application_traces: list[ApplicationTrace],
) -> None:
    avg_memory_list = []
    for app_trace in azure_application_traces:
        if not app_trace.avg_memory:
            continue
        mean = np.mean(app_trace.avg_memory)
        std = np.std(app_trace.avg_memory)
        avg_memory_list.append({"app_id": app_trace.name, "cv": std / mean})
    df = pd.DataFrame(data=avg_memory_list, columns=["app_id", "cv"])
    ax = sns.histplot(data=df)
    ax.set(xlabel="Average allocated memory CV")
    plt.tight_layout()
    plt.show()


def plot_memory_distribution() -> None:
    df, gm = fit_memory_distribution()
    ax = sns.histplot(data=df, binrange=(0, 500), stat="density")
    ax.set(xlabel="Average allocated memory (MB)")

    stds = np.sqrt(gm.covariances_)
    x = np.arange(0, 500, 0.1)
    y = None
    for mu, std, weight in zip(gm.means_, stds, gm.weights_):
        res = weight * scipy.stats.norm.pdf(x, loc=mu, scale=std)
        if y is None:
            y = res
        else:
            y += res
    ax = sns.lineplot(x=x, y=y, ax=ax)
    ax.legend(["mixture Gaussian PDF"])

    plt.tight_layout()
    plt.savefig("memory.png")
    plt.show()


def check_fun_execution_time_cv_distribution(
    azure_application_traces, fit_trigger
) -> None:
    avg_exec_time_list = []
    for app_trace in azure_application_traces:
        for fun_trace in app_trace.functions.values():
            if len(fun_trace.trigger_count) != 1:
                continue
            trigger = next(iter(fun_trace.trigger_count.keys()))
            if trigger != fit_trigger:
                continue
            if len(fun_trace.avg_execution_times) <= 1:
                continue
            mean = np.mean(fun_trace.avg_execution_times)
            std = np.std(fun_trace.avg_execution_times)
            assert mean != 0
            avg_exec_time_list.append(
                {"fun_id": fun_trace.name, "avg_exec_time_cv": round(std / mean, 3)}
            )
    assert avg_exec_time_list
    df = pd.DataFrame(data=avg_exec_time_list, columns=["fun_id", "avg_exec_time_cv"])
    print("75th percentile is", df["avg_exec_time_cv"].quantile(0.75))
    print("95th percentile is", df["avg_exec_time_cv"].quantile(0.95))
    ax = sns.histplot(data=df, binrange=(-1, 10))
    ax.set(xlabel="Average execution time CV")
    plt.tight_layout()
    plt.savefig("fun_time_cv.png")
    plt.show()


def plot_fun_trigger_count(
    azure_application_traces: list[ApplicationTrace], triggers: set | None = None
) -> None:
    total_count = {}
    for app_trace in azure_application_traces:
        for fun_trace in app_trace.functions.values():
            for trigger in fun_trace.trigger_count:
                if trigger not in total_count:
                    total_count[trigger] = 0
                total_count[trigger] += 1
    df = pd.DataFrame(data=[total_count])
    ax = sns.barplot(data=df)
    ax.set(xlabel="Trigger")
    ax.set(ylabel="Function count")
    plt.tight_layout()
    plt.savefig("fun_trigger.png")
    plt.show()


def plot_fun_count(azure_application_traces: list[ApplicationTrace]) -> None:
    fun_count = []
    for app_trace in azure_application_traces:
        fun_count.append(
            {"app_id": app_trace.name, "fun_count": len(app_trace.functions)}
        )

    df = pd.DataFrame(data=fun_count)
    print("95th percentile is", df.quantile(0.95))
    ax = sns.histplot(data=df, binrange=(0, 40), legend=False)
    ax.set(xlabel="Function count")
    plt.tight_layout()
    plt.savefig("fun_count.png")
    plt.show()


def plot_invocation_trigger_count(
    azure_application_traces: list[ApplicationTrace], triggers: set | None = None
) -> None:
    total_count = {}
    for app_trace in azure_application_traces:
        for fun_trace in app_trace.functions.values():
            for trigger, count in fun_trace.trigger_count.items():
                if trigger not in total_count:
                    total_count[trigger] = 0
                total_count[trigger] += count
    df = pd.DataFrame(data=[total_count])
    ax = sns.barplot(data=df)
    ax.set(xlabel="Trigger")
    ax.set(ylabel="Invocation count")
    plt.tight_layout()
    plt.savefig("invocation_trigger.png")
    plt.show()

    # ax = sns.barplot(data=df, stat="percent")
    # ax.set(xlabel="Trigger")
    # plt.tight_layout()
    # plt.savefig("invocation_trigger_percent.png")
    # plt.show()


def plot_fun_execution_time_distribution(triggers: set | None = None) -> None:
    df, fit_df, pdf, _ = fit_fun_execution_time_distribution(triggers)

    hue_order = sorted(df["trigger"].unique())
    ax = sns.histplot(
        data=df,
        x="avg_exec_time",
        binrange=(0, 6000),
        hue="trigger",
        stat="density",
        hue_order=hue_order,
    )
    ax.set(xlabel="Average execution time (ms)")

    x = np.arange(0, 6000, 10)
    y = pdf(x)

    ax = sns.lineplot(x=x, y=y, ax=ax)
    ax.legend(["Log-normal PDF"] + hue_order)
    plt.tight_layout()
    plt.savefig("exec_time_distribution.png")
    plt.show()


def plot_fun_invocation_distribution(trigger: str, weekday: bool) -> None:
    df, poly = fit_fun_invocation_distribution(trigger, weekday)
    x = np.arange(0, 1440, 1)
    df["Polynomial fit"] = poly(x)

    ax = sns.lineplot(
        data=df, palette=sns.color_palette("bright", n_colors=len(df.columns))
    )
    ax.set(xlabel="Minute")
    ax.set(ylabel="Invocation number")
    plt.tight_layout()
    if weekday:
        plt.savefig(f"invocation_distribution_{trigger}_weekday.png")
    else:
        plt.savefig(f"invocation_distribution_{trigger}_weekend.png")
    plt.show()


def check_fun_execution_time_and_memory(
    azure_application_traces: list[ApplicationTrace],
) -> None:
    exec_time_and_memory = []
    for app_trace in azure_application_traces:
        memory_mean = np.mean(app_trace.avg_memory)
        for fun_trace in app_trace.functions.values():
            if len(fun_trace.avg_execution_times) <= 1:
                continue
            exec_time_mean = np.mean(fun_trace.avg_execution_times)
            exec_time_and_memory.append(
                {"fun_time": exec_time_mean, "memory": memory_mean}
            )
            break
    assert exec_time_and_memory
    df = pd.DataFrame(data=exec_time_and_memory)
    sns.lineplot(data=df, x="memory", y="fun_time")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    load_config()
    config = global_config
    # azure_application_traces: list[ApplicationTrace] = load_azure_trace()
    # print(f"we have {len(azure_application_traces)} traces")
    # fun_number = sum(len(app.functions) for app in azure_application_traces)
    # print(f"we have {fun_number} traces")
    # plot_fun_count(azure_application_traces)
    # plot_fun_trigger_count(azure_application_traces)
    # plot_invocation_trigger_count(azure_application_traces)
    # plot_memory_distribution()
    # plot_fun_execution_time_distribution(triggers={"http"})
    plot_fun_invocation_distribution("http", weekday=True)
    # plot_fun_invocation_distribution("http", weekday=False)

    # plot_memory_cv_distribution(azure_application_traces)
    # check_fun_execution_time_cv_distribution(azure_application_traces,"http")

    # complete_azure_application_traces = []
    # for trace in azure_application_traces:
    #     trace.drop_incomplete_fun_traces()
    #     if trace.is_complete():
    #         complete_azure_application_traces.append(trace)
    # assert complete_azure_application_traces
    # # check_fun_execution_time_and_memory(complete_azure_application_traces)
