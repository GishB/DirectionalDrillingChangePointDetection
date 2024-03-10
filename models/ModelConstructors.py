"""
Highly used parent model classes functions for all others implemented models.
"""
import sys
import numpy as np

from utils.DataTransformers import Filter, Scaler

sys.path.append("..")


class ChangePointDetectionConstructor(Filter, Scaler):
    def __init__(self,
                 is_cps_filter_on: bool = True,
                 threshold_std_coeff: float = 3.1,
                 queue_window: int = 10,
                 sequence_window: int = 10):

        self.is_cps_filter_on = is_cps_filter_on
        self.threshold_std_coeff = threshold_std_coeff
        self.queue_window = queue_window
        self.sequence_window = sequence_window

    def get_distances(self) -> np.ndarray:
        """ Apply custom method to get residuals from time series data.

        Returns:
            array of residuals shaped as initial ts.
        """
        ...

    def predict(self) -> np.ndarray:
        """ Change Point Detection based on failure statistics.

        Notes:
            1. By default, we expect that threshold value can be found via quantile value due the fact that CPs shape are
            less time series shape.
            2. Keep in mind that queue window algorithm always saves the first anomaly as true result and
             drop others based on queue window range.

        Returns:
            array of binary change points labels.
        """
        residuals = self.get_distances()
        dp = [val for val in residuals[:self.sequence_window]]
        cps_list = [0 for ind in range(self.queue_window)]
        mean_val = np.mean(dp)
        std_val = np.std(dp) * self.threshold_std_coeff
        for val in residuals[self.sequence_window:]:
            if val > (mean_val + std_val) or val < (mean_val - std_val):
                cps_list.append(1)
            else:
                cps_list.append(0)
            dp.append(val)
            dp.pop(0)
            mean_val = np.mean(dp)
            std_val = np.std(dp) * self.threshold_std_coeff
        if self.is_cps_filter_on:
            cps_list = self.queue(time_series=cps_list, queue_window=self.queue_window, reversed=False)
        return np.array(cps_list)
