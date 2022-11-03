from datetime import timedelta


class VirtualClock:
    def __init__(self):
        self.__time_point: timedelta = timedelta()
        self.reset()

    def __repr__(self) -> str:
        return f"{self.time_point}"

    def reset(self) -> None:
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
