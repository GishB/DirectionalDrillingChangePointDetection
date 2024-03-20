import numpy as np
from models.ModelConstructors import ChangePointDetectionConstructor


class KalmanFilter(ChangePointDetectionConstructor):
    """ Idea is to find data deviations based on Kalman extrapolation for nearest data.
    """

    def __init__(self,
                 sequence_window: int = None,
                 queue_window: int = None,
                 is_cps_filter_on: bool = True,
                 is_quantile_threshold: bool = False,
                 is_fast_parameter_selection: bool = True,
                 is_cumsum_applied: bool = True,
                 is_z_normalization: bool = True,
                 is_squared_residual: bool = True,
                 fast_optimize_algorithm: str = 'summary_statistics_subsequence',
                 threshold_quantile_coeff: float = 0.95,
                 threshold_std_coeff: float = 3.1,
                 kalman_power_coeff: float = 0.9):
        super().__init__(queue_window=queue_window,
                         sequence_window=sequence_window,
                         fast_optimize_algorithm=fast_optimize_algorithm,
                         is_cps_filter_on=is_cps_filter_on,
                         is_fast_parameter_selection=is_fast_parameter_selection,
                         threshold_std_coeff=threshold_std_coeff,
                         is_cumsum_applied=is_cumsum_applied,
                         is_z_normalization=is_z_normalization,
                         is_squared_residual=is_squared_residual
                         )
        self.parameters["is_quantile_threshold"] = is_quantile_threshold
        self.parameters["threshold_quantile_coeff"] = threshold_quantile_coeff
        self.parameters["kalman_power_coeff"] = kalman_power_coeff

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

    def get_distances(self, target_array: np.array) -> np.ndarray:
        """ Generate residuals based on gaussian forecasted values.

        Notes:
            By default, expected that array shape will be more ore equal to 100 values - (up to window size).

        Args:
            target_array: 1-d time series data values.

        Returns:
            array of residuals between generated data and real values.
        """
        super().get_distances(target_array=target_array)
        gaussian_forecasted_list = [val for val in target_array[:self.parameters.get("sequence_window")]]
        gaussian_likelihood = self.get_array_info(target_array)
        dp = [val for val in target_array[:self.parameters.get("sequence_window")]]
        for generation_epoch in range(self.parameters.get("sequence_window"), target_array.shape[0]):
            gaussian_forecasted_list.append(self.forecast(mean_gauss=gaussian_likelihood[0],
                                                          std_gauss=np.sqrt(gaussian_likelihood[1])))
            actual_gaussian = self.get_array_info(dp)
            mult_gaussian = self.gaussian_multiply(
                g1=gaussian_likelihood,
                g2=actual_gaussian
            )
            gaussian_likelihood = self.update(mult_gaussian, actual_gaussian)
            dp.pop(0)
            dp.append(target_array[generation_epoch])
        return np.array(gaussian_forecasted_list) - target_array

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
        residuals = abs(target_array - abs(self.get_distances(target_array)))
        if self.parameters.get("is_z_normalization"):
            residuals = self.z_normalization(residuals)
        if self.parameters.get("is_squared_residual"):
            residuals = residuals**2
        if self.parameters.get('is_cumsum_applied'):
            alarm_index = self.cumsum(residuals)[2]
            cps_list = np.zeros_like(residuals)
            for index in alarm_index:
                cps_list[index] = 1
        else:
            dp = [val for val in residuals[:self.parameters.get("queue_window")]]
            cps_list = [0 for ind in range(self.parameters.get("sequence_window"))]
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


if __name__ == "__main__":
    from models.ProbabilityBased import KalmanFilter
    from data.SythData import SinusoidWaves

    data = SinusoidWaves(length_data=1000, cps_number=5, white_noise_level="min").get()
    target_array = data['x'].values

    model = KalmanFilter(sequence_window=None,
                         is_fast_parameter_selection=True,
                         fast_optimize_algorithm='highest_autocorrelation',
                         queue_window=10,
                         threshold_quantile_coeff=0.95,
                         is_cps_filter_on=True).fit(x_train=list(target_array), y_train=None)

    cps_pred = model.predict(target_array)
    stop = 0
