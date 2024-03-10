import numpy as np
import pandas as pd

import sys

sys.path.append("..")
from utils.hyperparameters.WSSAlgorithms import WindowSizeSelection
from utils.DataTransformers import Filter


class KalmanFilter(WindowSizeSelection, Filter):
    """ Idea is to find data deviations based on Kalman extrapolation for nearest data.

    Attributes:
        df: pandas dataframe with target data.
        target_column: column which should be checked.
        window: expected window for analysis.
        threshold_quantile_coeff: expected deviation between generated and original data.
    """

    def __init__(self, df: pd.DataFrame = None,
                 target_column: str = None,
                 kalman_window: int = None,
                 queue_window: int = None,
                 is_cps_filter_on: bool = True,
                 is_quantile_threshold: bool = False,
                 threshold_quantile_coeff: float = 0.91,
                 threshold_std_coeff: float = 3.61,
                 kalman_power_coeff: float = 0.9):
        self.df = df
        self.target_column = target_column
        self.kalman_window = kalman_window
        self.queue_window = queue_window
        self.threshold_quantile_coeff = threshold_quantile_coeff
        self.is_cps_filter_on = is_cps_filter_on
        self.threshold_std_coeff = threshold_std_coeff
        self.is_quantile_threshold = is_quantile_threshold
        self.kalman_power_coeff = kalman_power_coeff

        if kalman_window is None:
            super().__init__(time_series=df[target_column].values)
            self.kalman_window = self.runner_wss()[0]

        if queue_window is None:
            self.queue_window = kalman_window*2

        if kalman_window <= 10:
            raise NotImplementedError("This algorithm highly like will raise an error because your initial df shape "
                                      "less then 10 points")

        if target_column is None:
            raise ValueError("You must to define target column!")

    @staticmethod
    def get_array_info(data: np.array) -> tuple[float, float]:
        """ Get gaussian stats based on an array of values.

        Args:
            data: slice of time series values.

        Returns:
            gaussian stats tuple as mean and std values
        """
        gaussian_info = np.mean(data), np.std(data)
        return gaussian_info

    def gaussian_multiply(self, g1: tuple[float, float], g2: tuple[float, float]) -> tuple[float, float]:
        """ Update gaussian stats based on prev info and current status.

        Notes:
            first index is mean value
            second index is var value

        TO DO:
            Kalman Gain.

        Args:
            g1: past gaussian stats.
            g2: current gaussian stats.

        Returns:
            likelihood gaussian statistics.
        """
        mean = (g1[1] * g2[0] + g2[1] * g1[0]) / (g1[1] + g2[1])
        variance = (g1[1] * g2[1]) / (g1[1] + g2[1])
        return mean, variance

    @staticmethod
    def forecast(mean_gauss: float, std_gauss: float) -> float:
        """ forecast next values based on estimated gaussian coefficient.

        Args:
            mean_gauss: expected mean value.
            std_gauss: expected std value.

        Returns:
            normal generated value based on expected mean and std.
        """
        return np.random.normal(loc=mean_gauss, scale=std_gauss)

    @staticmethod
    def update(mult_gaussian: tuple[float, float], actual_gaussian: tuple[float, float]) -> tuple[float, float]:
        """ Update gaussian stats based on actual and kalman filter info.

        Args:
            mult_gaussian: mean and var values based on Kalman Filter.
            actual_gaussian: mean and var values based on actual time series info.

        Returns:
            expected mean and var stat.
        """
        return mult_gaussian[0] + actual_gaussian[0], mult_gaussian[1] + actual_gaussian[1]

    def get_full_forecast(self) -> np.ndarray:
        """ Generate residuals based on gaussian forecasted values.

        Notes:
            By default, expected that array shape will be more ore equal to 100 values - (up to window size).

        Returns:
            array of residuals between generated data and real values.
        """
        gaussian_forecasted_list = [val for val in self.df[self.target_column][:self.kalman_window]]
        gaussian_likelihood = self.get_array_info(self.df[self.target_column].values)
        dp = [val for val in self.df[self.target_column][:self.kalman_window]]
        for generation_epoch in range(self.kalman_window, self.df.shape[0]):
            gaussian_forecasted_list.append(self.forecast(mean_gauss=gaussian_likelihood[0],
                                                          std_gauss=np.sqrt(gaussian_likelihood[1])))
            actual_gaussian = self.get_array_info(dp)
            mult_gaussian = self.gaussian_multiply(
                g1=gaussian_likelihood,
                g2=actual_gaussian
            )
            gaussian_likelihood = self.update(mult_gaussian, actual_gaussian)
            dp.pop(0)
            dp.append(self.df[self.target_column][generation_epoch])
        return np.array(gaussian_forecasted_list)

    def get_residuals(self):
        """ Calculate residuals between filtered time series and original.

        Returns:
            array of residuals.
        """
        return self.df[self.target_column].values - self.get_full_forecast()

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
        residuals = abs(self.get_residuals())
        dp = [val for val in residuals[:self.kalman_window]]
        cps_list = [0 for ind in range(self.kalman_window)]
        mean_val = np.mean(dp)
        std_val = np.std(dp) * self.threshold_std_coeff
        for val in residuals[self.kalman_window:]:
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


if __name__ == "__main__":
    from models.ProbabilityBased import KalmanFilter
    from data.SythData import LinearSteps, SinusoidWaves

    data = SinusoidWaves(length_data=2000, cps_number=5, white_noise_level="min").get()

    model = KalmanFilter(df=data, target_column="x", window=250, threshold_quantile_coeff=0.95, is_cps_filter_on=True)
    cps_pred = model.predict()
