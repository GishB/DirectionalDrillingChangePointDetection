from typing import Tuple
import numpy as np
from scipy.linalg import hankel
from models.ModelConstructors import ChangePointDetectionConstructor


class SingularSequenceTransformer(ChangePointDetectionConstructor):
    """ Idea is to find nearest change points based on abnormal subspace distance.
    """

    def __init__(self,
                 sequence_window: int = None,
                 queue_window: int = None,
                 n_components: int = 1,
                 lag: int = None,
                 is_cps_filter_on: bool = True,
                 is_quantile_threshold: bool = False,
                 is_exp_squared: bool = False,
                 is_fast_parameter_selection: bool = True,
                 is_cumsum_applied: bool = True,
                 is_z_normalization: bool = True,
                 is_squared_residual: bool = True,
                 fast_optimize_algorithm: str = 'summary_statistics_subsequence',
                 threshold_quantile_coeff: float = 0.91,
                 threshold_std_coeff: float = 3.61,
                 ):
        """

        Args:
            n_components:  PCA components number describes changes in time-data (usually we have 1,2 or 3).
            sequence_window: window which we need to analyse each step over time series.
            queue_window: min distance between two change points.
            is_cps_filter_on: should we use queue window algorithm.
            is_quantile_threshold: should we take quantile value as threshold.
            lag: distance between two nearest matrix.
            threshold_quantile_coeff: threshold coefficient for quantile.
            threshold_std_coeff: threshold coefficient based on rule of thumb for normal distribution.
        """

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
        self.parameters["n_components"] = n_components
        self.parameters["lag"] = lag
        self.parameters["threshold_quantile_coeff"] = threshold_quantile_coeff
        # should we use exponential squared function for subspace distance
        self.parameters["is_exp_squared"] = is_exp_squared

    @staticmethod
    def get_hankel_matrix(sequence: np.array) -> np.ndarray:
        """ Apply Hankel method over 1D time-series subsequence to transform it into matrix view.

        Arg:
            sequence: time-series subsequence.

        Return:
            Hankel matrix.
        """
        return hankel(c=sequence)

    @staticmethod
    def _sst_svd(x_test: np.array, x_history: np.array, n_components: int):
        """Apply singular sequence transformation algorithm with SVD.

        Args:
            x_test: matrix which represents time-series subsequence in the target time.
            x_history: matrix which represents time-series subsequence in the past.
            n_components: PCA components number describes changes in time-data (usually we have 1,2 or 3).

        Return:
            distance between compared matrices.
        """
        u_test, s_test, _ = np.linalg.svd(x_test, full_matrices=False)
        u_history, s_hist, _ = np.linalg.svd(x_history, full_matrices=False)
        s_cov = u_test[:, :n_components].T @ u_history[:, :n_components]
        u_cov, s, _ = np.linalg.svd(s_cov, full_matrices=False)
        return 1 - s[0]

    def get_current_matrix(self, ts: np.array) -> np.ndarray:
        """ Calculate historical matrix based on lag between past and future.

        Args:
            ts: target 1d sequence.

        Returns:
            array of historical matrix.
        """
        list_matrix = []
        for ind in range(ts.shape[0] - self.parameters.get("lag") - self.parameters.get("sequence_window")):
            list_matrix.append(self.get_hankel_matrix(ts[ind:ind + self.parameters.get("sequence_window")]))
        return np.array(list_matrix)

    def get_lagged_matrix(self, ts: np.array) -> np.ndarray:
        """ Calculate future matrix based on lag between past and future.

        Args:
            ts: target 1d sequence.

        Returns:
            array of future matrix.
        """
        list_matrix = []
        for ind in range(ts.shape[0] - self.parameters.get("lag") - self.parameters.get("sequence_window")):
            list_matrix.append(self.get_hankel_matrix(ts[ind + self.parameters.get("lag"):
                                                         ind + self.parameters.get("lag") +
                                                         self.parameters.get("sequence_window")]))
        return np.array(list_matrix)

    def preprocess_ts(self, ts: np.array) -> Tuple[np.ndarray, np.ndarray]:
        """ Preprocess historical and future matrix based on array.

        Args:
            ts: target 1d sequence.

        Returns:
            tuple of arrays with historical and future matrix in each time step.
        """
        present_matrix = self.get_current_matrix(ts)
        lagged_matrix = self.get_lagged_matrix(ts)
        return present_matrix, lagged_matrix

    def get_distances(self, target_array: np.array) -> np.ndarray:
        """ Calculate subspace distances.

        Notes:
            By default, this pipline based on SST SVD idea.

        Args:
            target_array: target 1d time-series.

        Returns:
            array of subspace distance score.
        """
        score_list = np.zeros_like(target_array)
        matrix_history, matrix_next = self.preprocess_ts(target_array)
        counter: int = 0
        while counter != target_array.shape[0] - self.parameters.get("lag") - self.parameters.get("sequence_window"):
            score_list[counter] = self._sst_svd(x_test=matrix_next[counter],
                                                x_history=matrix_history[counter],
                                                n_components=self.parameters.get("n_components"))
            counter += 1
        if self.parameters.get("is_exp_squared"):
            score_list = np.exp(score_list) ** 2
        return score_list

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
        residuals = abs(self.get_distances(target_array))
        if self.parameters.get("is_z_normalization"):
            residuals = self.z_normalization(residuals)
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
    from data.SythData import SinusoidWaves

    data = SinusoidWaves(length_data=2000, cps_number=4, white_noise_level="min").get()
    target_array = data['x'].values
    model = SingularSequenceTransformer(
                                        sequence_window=None,
                                        lag=None,
                                        queue_window=10,
                                        is_cps_filter_on=True,
                                        n_components=2,
                                        is_fast_parameter_selection=True,
                                        fast_optimize_algorithm='highest_autocorrelation',
                                        threshold_std_coeff=2.65).fit(x_train=list(target_array), y_train=None)
    # distances = model.get_distances(target_array=target_array)
    cps_pred = model.predict(target_array=target_array)
    stop = 0
