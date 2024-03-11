"""
Highly used parent model classes functions for all others implemented models.
"""
import sys
import numpy as np

from typing import Optional, Tuple

from utils.DataTransformers import Filter, Scaler
from utils.hyperparameters.WSSAlgorithms import WindowSizeSelection

sys.path.append("..")


class ChangePointDetectionConstructor(WindowSizeSelection, Filter, Scaler):
    def __init__(self,
                 fast_optimize_algorithm: str = 'dominant_fourier_frequency',
                 is_cps_filter_on: bool = True,
                 is_fast_parameter_selection: bool = True,
                 threshold_std_coeff: float = 3.1,
                 queue_window: int = 10,
                 sequence_window: int = 10,
                 lag: int = 1):

        self.fast_optimize_algorithm = fast_optimize_algorithm
        self.is_fast_parameter_selection = is_fast_parameter_selection
        self.is_cps_filter_on = is_cps_filter_on
        self.threshold_std_coeff = threshold_std_coeff
        self.queue_window = queue_window
        self.sequence_window = sequence_window
        self.lag = lag

    def fit(self,
            x_train: np.array,
            y_train: Optional[np.array]) -> Tuple[int, int, int]:
        """ Search for the best model hyperparameters.

        Notes:
            1. In default pipe model will use window size selection algorithm.
            2. You can default all parameters manually in init method.
            3. in case of fast optimal params searching you can drop y_train.

        Args:
            x_train: array of time-series values.
            y_train: array of true labels.

        Returns:
            tuple of model hyperparameters.
        """
        if self.is_fast_parameter_selection:
            if self.sequence_window is None:
                self.sequence_window = super().__init__(time_series=x_train,
                                                        wss_algorithm=self.fast_optimize_algorithm)

            if self.queue_window is None:
                if self.sequence_window // 2 <= 10:
                    queue_window = 10
                else:
                    queue_window = self.sequence_window // 2
                self.queue_window = queue_window

            if self.lag is None:
                if self.sequence_window // 4 <= 10:
                    lag = 10
                else:
                    lag = self.sequence_window // 4
                self.lag = lag
        else:
            raise NotImplementedError("Any other optimization are not implemented yet! Select is_fast_optimize = True")
        return self.sequence_window, self.lag, self.queue_window

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
