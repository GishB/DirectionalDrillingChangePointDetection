import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
from detecta import detect_cusum
from typing import Any, List


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


class Filter:
    @staticmethod
    def savgol(x: np.array, window_length: int,
               polyorder: int = 3, mode: str = "nearest") -> np.array:
        """ Savitsky-Golay filter implementation.
        """
        return savgol_filter(x=x, window_length=window_length, polyorder=polyorder, mode=mode)

    @staticmethod
    def queue(time_series: np.array, queue_window: int = 10, reversed: bool = False) -> np.array:
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


class StatisticalFilter(Filter):
    """ Based on statistical significant changes for two subsequences filter change point labels.

    Attributes:
        queue_window: minimal distance between two change points.
        pvalue_threshold:
    """
    def __init__(self,
                 queue_window: int = 10,
                 pvalue_threshold: float = 0.05):
        self.queue_window = queue_window
        self.pvalue_threshold = pvalue_threshold

    @staticmethod
    def extract_index_cps(cps_labels: np.ndarray) -> np.ndarray:
        """ extract all index where change points detected.

        Args:
            cps_labels: array of data where all cps saved.

        Returns:
            filtered cps labels array.
        """
        return np.where(cps_labels == 1)[0]

    @staticmethod
    def split_list_of_arrays(extracted_index_cps: np.array, data: np.array) -> list[np.array]:
        """ Split data based on extracted cps index.

        Args:
            extracted_index_cps: index for cps at data.
            data: values for target 1d series.

        Returns:
            list of subsequences split by extracted cps index.
        """
        first_indx: int = 0
        list_of_subsequence = []
        for second_indx in extracted_index_cps:
            # append new subsequence
            list_of_subsequence.append(data[first_indx:second_indx])
            # update indx
            first_indx = second_indx
        return list_of_subsequence

    def check_cps_by_ttest(self, subsequences_splitten_by_cps: list[np.array]) -> list[bool]:
        """ Check that nearest subsequences are different statistical significant.

        Notes:
            1. if significant different has been found then you will get True.
            2. Expected that distribution are normal for all subsequences.

        Args:
            subsequences_splitten_by_cps: list of subseqences of different shapes.

        Returns:
            list of boolean values.
        """
        list_bool = []
        past_subsequence = subsequences_splitten_by_cps[0]
        for next_subsequence in subsequences_splitten_by_cps:
            pvalue = stats.ttest_ind(past_subsequence, next_subsequence).pvalue
            if pvalue < self.pvalue_threshold:
                list_bool.append(True)
            else:
                list_bool.append(False)
            past_subsequence = next_subsequence
        return list_bool

    def filter(self, cps_labels: np.array, data: np.array) -> np.array:
        """ Filter cps labels based on statistical idea and minimum sequence window.

        Args:
            cps_labels: list of cps labels.
            data: list of original data.

        Returns:
            filtered list of cps labels.
        """
        cps_labels = self.queue(queue_window=self.queue_window,
                                time_series=list(cps_labels))
        extract_index = self.extract_index_cps(cps_labels)
        if extract_index.shape[0] != 0:
            subsequences_splittedby_cps = self.split_list_of_arrays(extracted_index_cps=extract_index,
                                                                    data=data)
            list_bool = self.check_cps_by_ttest(subsequences_splittedby_cps)
            extract_index = [extract_index[ind] for ind, val in enumerate(list_bool) if val is True]
        zeros_cps_label = np.zeros_like(cps_labels)
        zeros_cps_label[extract_index] = 1
        return zeros_cps_label

