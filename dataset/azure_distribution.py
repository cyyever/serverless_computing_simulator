import functools
import os
import sys

import numpy as np
import scipy
from cyy_naive_lib.storage import persistent_cache
from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd

from azure_trace import ApplicationTrace, cache_dir, load_azure_trace


@persistent_cache(path=os.path.join(cache_dir, "__fit_memory_distribution.pk"))
def fit_memory_distribution() -> tuple[pd.DataFrame, GaussianMixture]:
    azure_application_traces: list[ApplicationTrace] = load_azure_trace()
    # follow a two modal Gaussian distribution
    avg_memory_list = []
    for app_trace in azure_application_traces:
        for memory in app_trace.avg_memory:
            avg_memory_list.append({"app_id": app_trace.name, "avg_memory": memory})

    df = pd.DataFrame(data=avg_memory_list)
    print("75th percentile is", df["avg_memory"].quantile(0.75))
    print("95th percentile is", df["avg_memory"].quantile(0.95))
    X = df["avg_memory"]

    gm = GaussianMixture(n_components=3, covariance_type="spherical").fit(
        np.asarray(X).reshape(-1, 1)
    )
    # print("means", gm.means_)
    # print("covariance", gm.covariances_)
    # print("weights", gm.weights_)
    return df, gm


@persistent_cache(
    path=os.path.join(cache_dir, "__fit_fun_execution_time_distribution.pk")
)
def fit_fun_execution_time_distribution(
    triggers: set | None = None, fit_trigger="http"
) -> tuple:
    azure_application_traces: list[ApplicationTrace] = load_azure_trace()
    avg_exec_time_list = []
    for app_trace in azure_application_traces:
        for fun_trace in app_trace.functions.values():
            for exec_time in fun_trace.avg_execution_times:
                avg_exec_time_list.append(
                    {
                        "fun_id": fun_trace.name,
                        "avg_exec_time": exec_time,
                        "trigger": "all",
                    }
                )
            if triggers is not None:
                if len(fun_trace.trigger_count) != 1:
                    continue
                trigger = next(iter(fun_trace.trigger_count.keys()))
                if trigger not in triggers:
                    continue
                for exec_time in fun_trace.avg_execution_times:
                    avg_exec_time_list.append(
                        {
                            "fun_id": fun_trace.name,
                            "avg_exec_time": exec_time,
                            "trigger": trigger,
                        }
                    )
    assert avg_exec_time_list

    df: pd.DataFrame = pd.DataFrame(
        data=avg_exec_time_list,
    )
    fit_df = df[df["trigger"] == fit_trigger]
    print("75th percentile is", fit_df["avg_exec_time"].quantile(0.75))
    print("95th percentile is", fit_df["avg_exec_time"].quantile(0.95))

    X = fit_df["avg_exec_time"].to_numpy()
    s, loc, scale = scipy.stats.lognorm.fit(X)
    pdf = functools.partial(scipy.stats.lognorm.pdf, s=s, loc=loc, scale=scale)
    return (
        df,
        fit_df,
        pdf,
        functools.partial(scipy.stats.lognorm.rvs, s, loc=loc, scale=scale),
    )


@persistent_cache(path=os.path.join(cache_dir, "__fit_fun_invocation_distribution.pk"))
def fit_fun_invocation_distribution(
    trigger: str, weekday: bool
) -> tuple[pd.DataFrame, np.poly1d]:
    azure_application_traces: list[ApplicationTrace] = load_azure_trace()
    total_invocation_number = {}

    for app_trace in azure_application_traces:
        for fun_trace in app_trace.functions.values():
            if not fun_trace.invocation_numbers:
                continue
            for day, day_invocation_numbers in fun_trace.invocation_numbers.items():
                invocation_number = day_invocation_numbers.get(trigger, None)
                if invocation_number is None:
                    continue
                if day == 9:
                    # outlier
                    continue
                if not weekday:
                    if day not in (6, 7, 13, 14):
                        continue
                else:
                    if day in (6, 7, 13, 14):
                        continue
                day = f"day_{day}"
                if day not in total_invocation_number:
                    total_invocation_number[day] = invocation_number
                else:
                    total_invocation_number[day] += invocation_number

    df = pd.DataFrame(data=total_invocation_number)

    df_np: np.ndarray = df.to_numpy(dtype=np.float64)

    columns = np.hsplit(df_np, df_np.shape[1])
    new_columns = []
    for column in columns:
        column = np.column_stack([range(column.shape[0]), column]).reshape(-1, 2)
        new_columns.append(column)
    samples = np.concatenate(new_columns, axis=0)
    z = np.polyfit(samples[:, 0].reshape(-1), samples[:, 1].reshape(-1), 5)
    poly = np.poly1d(z)
    return df, poly
