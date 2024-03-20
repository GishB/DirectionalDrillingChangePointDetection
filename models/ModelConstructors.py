"""
Highly used parent model classes functions for all others implemented models.
"""
import sys
import numpy as np

from typing import Optional

from utils.DataTransformers import Filter, Scaler
from optimization.WSSAlgorithms import WindowSizeSelection


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
                 lag: int = None,
                 is_cumsum_applied: bool = True,
                 is_z_normalization: bool = True,
                 is_squared_residual: bool = True):
        """ Highly used parameters.

        Args:
            fast_optimize_algorithm: algorithm name from WSS optimization.
            is_cps_filter_on: is filter change points option based on queue window.
            is_fast_parameter_selection: is fast optimization applied.
            threshold_std_coeff: threshold param for abnormal residuals.
            queue_window: max limited window to filter cps.
            sequence_window: max length of subsequence which is used for analysis on each step.
            lag: step between two subsequences.
            is_cumsum_applied: should we use cumsum algorithm for CPs detection.
            is_z_normalization: normalization over residual data.
        """

        self.parameters = {
            "is_fast_parameter_selection": is_fast_parameter_selection,
            "fast_optimize_algorithm": fast_optimize_algorithm,
            "is_cps_filter_on": is_cps_filter_on,
            "threshold_std_coeff": threshold_std_coeff,
            "queue_window" : queue_window,
            "sequence_window": sequence_window,
            "lag": lag,
            "is_cumsum_applied": is_cumsum_applied,
            "is_z_normalization": is_z_normalization,
            "is_squared_residual": is_squared_residual
        }

    def fit(self,
            x_train: np.array,
            y_train: Optional[np.array]):
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
        ...
