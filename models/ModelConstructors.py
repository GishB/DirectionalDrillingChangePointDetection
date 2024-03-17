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
    """ Basic class to work with ChangePoint detection models.

    Attributes:
        parameters: dict of parameters for selected model.

    """
    def __init__(self,
                 fast_optimize_algorithm: str = 'summary_statistics_subsequence',
                 is_cps_filter_on: bool = True,
                 is_fast_parameter_selection: bool = True,
                 threshold_std_coeff: float = 3.1,
                 queue_window: int = None,
                 sequence_window: int = None,
                 lag: int = None):
        """ Highly used parameters.

        Args:
            fast_optimize_algorithm: algorithm name from WSS optimization.
            is_cps_filter_on: is filter change points option based on queue window.
            is_fast_parameter_selection: is fast optimization applied.
            threshold_std_coeff: threshold param for abnormal residuals.
            queue_window: max limited window to filter cps.
            sequence_window: max length of subsequence which is used for analysis on each step.
            lag: step between two subsequences.
        """

        self.parameters = {
            "is_fast_parameter_selection": is_fast_parameter_selection,
            "fast_optimize_algorithm": fast_optimize_algorithm,
            "is_cps_filter_on": is_cps_filter_on,
            "threshold_std_coeff": threshold_std_coeff,
            "queue_window" : queue_window,
            "sequence_window": sequence_window,
            "lag": lag
        }

    def fit(self,
            x_train: np.array,
            y_train: Optional[np.array]) -> object:
        """ Search for the best model hyperparameters.

        Notes:
            1. In default pipe model will use window size selection algorithm.
            2. You can default all parameters manually in init method.
            3. in case of fast optimal params searching you can drop y_train.
            4. all parameters are saved in self. parameters

        Args:
            x_train: array of time-series values.
            y_train: array of true labels.

        Returns:
            self model object
        """
        if self.parameters.get("is_fast_parameter_selection"):
            super().__init__(time_series=x_train, wss_algorithm=self.parameters.get("fast_optimize_algorithm"))
            sequence_window = self.runner_wss()[0]
            self.parameters["sequence_window"] = sequence_window
            if self.parameters.get("queue_window") is None:
                queue_window = 10
                self.parameters["queue_window"] = queue_window
            if self.parameters.get("lag") is None:
                lag = sequence_window // 4
                self.parameters["lag"] = lag
        else:
            raise NotImplementedError("Any other optimization are not implemented yet! Select is_fast_optimize = True")
        return self

    def get_distances(self, target_array: np.array) -> np.ndarray:
        """ Apply custom method to get residuals from time series data.

        Notes:
            Of course it is expected that data timeline has no missing points.

        Args:
            target_array: 1-d time series data values.


        Returns:
            array of residuals shaped as initial ts.
        """
        if target_array.shape[0] <= 10:
            raise ArithmeticError("x_array shape is equal or lower to 10! It dose`t make sense at all.")
        ...
        return target_array

    def predict(self, target_array: np.array) -> np.ndarray:
        """ Change Point Detection based on failure statistics.

        Notes:
            1. By default, we expect that threshold value can be found via quantile value due the fact that CPs shape are
            less time series shape.
            2. Keep in mind that queue window algorithm always saves the first anomaly as true result and
             drop others based on queue window range.

        Returns:
            array of binary change points labels.
        """
        residuals = self.get_distances(target_array)
        dp = [val for val in residuals[:self.parameters.get("sequence_window")]]
        cps_list = [0 for ind in range(self.parameters.get("queue_window"))]
        mean_val = np.mean(dp)
        std_val = np.std(dp) * self.parameters.get("threshold_std_coeff")
        for val in residuals[self.parameters.get("sequence_window"):]:
            if val > (mean_val + std_val) or val < (mean_val - std_val):
                cps_list.append(1)
            else:
                cps_list.append(0)
            dp.append(val)
            dp.pop(0)
            mean_val = np.mean(dp)
            std_val = np.std(dp) * self.parameters.get("threshold_std_coeff")
        if self.parameters.get("is_cps_filter_on"):
            cps_list = self.queue(time_series=cps_list,
                                  queue_window=self.parameters.get("queue_window"),
                                  reversed=False)
        return np.array(cps_list)
