from datetime import timedelta

# from cyy_naive_lib.log import get_logger


class VirtualClock:
    def __init__(self, wall_clock_ratio: None | float = None):
        self.__time_point: timedelta = timedelta()
        self.reset(wall_clock_ratio=wall_clock_ratio)

    def __repr__(self):
        return f"{self.time_point}"

    def reset(self, wall_clock_ratio: None | float) -> None:
        self.__time_point = timedelta()

    @property
    def elapsed_minutes(self) -> int:
        return int(self.time_point / timedelta(minutes=1))

    @property
    def time_point(self) -> timedelta:
        time_point: timedelta = self.__time_point
        return time_point

    def advance(self, amount: timedelta) -> None:
        self.__time_point += amount
