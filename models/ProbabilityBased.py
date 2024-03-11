import numpy as np
import pandas as pd

import sys

sys.path.append("..")
from models.ModelConstructors import ChangePointDetectionConstructor


class KalmanFilter(ChangePointDetectionConstructor):
    """ Idea is to find data deviations based on Kalman extrapolation for nearest data.

    Attributes:
        df: pandas dataframe with target data.
        target_column: column which should be checked.
        sequence_window: expected window for analysis.
        threshold_quantile_coeff: expected deviation between generated and original data.
    """

    def __init__(self, df: pd.DataFrame = None,
                 target_column: str = None,
                 sequence_window: int = None,
                 queue_window: int = None,
                 is_cps_filter_on: bool = True,
                 is_quantile_threshold: bool = False,
                 is_fast_parameter_selection: bool = True,
                 fast_optimize_algorithm: str = 'summary_statistics_subsequence',
                 threshold_quantile_coeff: float = 0.95,
                 threshold_std_coeff: float = 3.1,
                 kalman_power_coeff: float = 0.9):
        super().__init__(queue_window=queue_window, sequence_window=sequence_window)
        self.df = df
        self.target_column = target_column
        self.sequence_window = sequence_window
        self.is_fast_parameter_selection = is_fast_parameter_selection
        self.fast_optimize_algorithm = fast_optimize_algorithm
        self.queue_window = queue_window
        self.threshold_quantile_coeff = threshold_quantile_coeff
        self.is_cps_filter_on = is_cps_filter_on
        self.threshold_std_coeff = threshold_std_coeff
        self.is_quantile_threshold = is_quantile_threshold
        self.kalman_power_coeff = kalman_power_coeff

        if target_column is None:
            raise ValueError("You must to define target column!")

    @staticmethod
    def get_array_info(array_slice: np.array) -> tuple[float, float]:
        """ Get gaussian stats based on an array of values.

        Args:
            array_slice: slice of time series values.

        Returns:
            gaussian stats tuple as mean and std values
        """
        gaussian_info = np.mean(array_slice), np.std(array_slice)
        return gaussian_info

    @staticmethod
    def gaussian_multiply(g1: tuple[float, float], g2: tuple[float, float]) -> tuple[float, float]:
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
        gaussian_forecasted_list = [val for val in self.df[self.target_column][:self.sequence_window]]
        gaussian_likelihood = self.get_array_info(self.df[self.target_column].values)
        dp = [val for val in self.df[self.target_column][:self.sequence_window]]
        for generation_epoch in range(self.sequence_window, self.df.shape[0]):
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

    def get_distances(self):
        """ Calculate residuals between filtered time series and original.

        Returns:
            array of residuals.
        """
        return self.df[self.target_column].values - self.get_full_forecast()


if __name__ == "__main__":
    from models.ProbabilityBased import KalmanFilter
    from data.SythData import SinusoidWaves

    data = SinusoidWaves(length_data=2000, cps_number=5, white_noise_level="min").get()

    model = KalmanFilter(df=data, target_column="x", queue_window=250,
                         threshold_quantile_coeff=0.95, is_cps_filter_on=True)
    cps_pred = model.predict()
