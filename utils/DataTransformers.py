import numpy as np
from scipy.signal import savgol_filter
from detecta import detect_cusum
from typing import Any


class Scaler:
    """ 1D timeseries normalization/reverse normalization class.
    """

    def __init__(self):
        self.stat_info = None

    @staticmethod
    def info(ts: np.array) -> np.array:
        """ Time series statistical information.

        Returns:
            list of stst info as well as mean, std, var, quantile 25%, 50%, 75%.
        """
        max_val = ts.max()
        min_val = ts.min()
        mean_val = ts.mean()
        std_val = ts.std()
        var_val = ts.var()
        quantile_25 = np.quantile(ts, q=0.25)
        quantile_50 = np.quantile(ts, q=0.50)
        quantile_75 = np.quantile(ts, q=0.75)
        return np.array([max_val, min_val, mean_val,
                         std_val, var_val, quantile_25,
                         quantile_50, quantile_75])

    def linear_normalization(self, ts: np.array) -> np.array:
        self.stat_info = self.info(ts)
        normalized_ts = (ts - self.stat_info[1]) / (self.stat_info[0] - self.stat_info[1])
        return normalized_ts

    def reversed_linear_normalization(self, ts: np.array) -> np.array:
        reversed_ts = ts * (self.stat_info[0] - self.stat_info[1]) + self.stat_info[2]
        return reversed_ts

    @staticmethod
    def log_normalization(ts: np.array) -> np.array:
        normalized_ts = np.log2(ts)
        return normalized_ts

    @staticmethod
    def reversed_log_normalization(transformed_ts: np.array):
        reversed_ts = 2 ** transformed_ts
        return reversed_ts

    def z_normalization(self, ts: np.array) -> np.array:
        self.stat_info = self.info(ts)
        normalized_ts = (ts - self.stat_info[2]) / self.stat_info[3]
        return normalized_ts

    def reverse_z_normalization(self, normalized_ts: np.array) -> np.array:
        reversed_ts = normalized_ts * self.stat_info[3] + self.stat_info[1]
        return reversed_ts


class Filter(Scaler):
    @staticmethod
    def savgol(x: np.array, window_length: int,
               polyorder: int = 3, mode: str = "nearest") -> np.array:
        """ Savitsky-Golay filter implementation.
        """
        return savgol_filter(x=x, window_length=window_length, polyorder=polyorder, mode=mode)

    @staticmethod
    def queue(queue_window: int = 10, time_series: list[int] = None, reversed: bool = False) -> np.array:
        """ Filter time series based on nearest value distant.

        Notes:
            By default this function expect that array contains only binary values.

        Args:
            queue_window: minimum distance between two values in your array.
            time_series: array of values.
            reversed: should we select the last anomaly as true positive and drop others based on window.

        Returns:
            filtered array where minimum distance between nearest value preserved.
        """
        if reversed:
            time_series = time_series[::-1]
        queue_list: list[int] = [0 for i in range(queue_window)]
        filtered_score: list[int] = []
        for i in range(len(time_series)):
            value = time_series[i]
            if max(queue_list) != 0:
                filtered_score.append(0)
                queue_list.pop(0)
                queue_list.append(0)
            else:
                filtered_score.append(value)
                queue_list.pop(0)
                queue_list.append(value)
        return np.array(filtered_score)

    def cumsum(self, x: np.array) -> tuple[Any, Any, Any, Any]:
        """ cumulative sum algorithm implementation to find abnormal behaviours in data.

        Notes:
            By default x expect to contain residuals value between predicted values and real.

        Args:
            x: array of values.

        Returns:
            info arrays about change points.
        """
        self.stat_info = self.info(x)
        ending, start, alarm, cumsum = detect_cusum(x=x,
                                                    threshold=(self.stat_info[2] + self.stat_info[3] * 3),
                                                    drift=self.stat_info[3],
                                                    ending=True,
                                                    show=False)
        return ending, start, alarm, cumsum
