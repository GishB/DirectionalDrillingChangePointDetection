from typing import Tuple

import numpy as np
import pandas as pd
import sys
from scipy.linalg import hankel

sys.path.append("..")
from utils.hyperparameters.WSSAlgorithms import WindowSizeSelection
from models.ModelConstructors import ChangePointDetectionConstructor


class SingularSequenceTransformer(WindowSizeSelection, ChangePointDetectionConstructor):
    """ Idea is to find nearest change points based on abnormal subspace distance.

    Attributes:
        df: pandas dataframe with all necessary data.
        n_components:  PCA components number describes changes in time-data (usually we have 1,2 or 3).
        target_column: target colum name in the dataframe.
        sequence_window: window which we need to analyse each step over time series.
        queue_window: min distance between two change points.
        is_cps_filter_on: should we use queue window algorithm.
        is_quantile_threshold: should we take quantile value as threshold.
        lag: distance between two nearest matrix.
        threshold_quantile_coeff: threshold coefficient for quantile.
        threshold_std_coeff: threshold coefficient based on rule of thumb for normal distribution.
    """

    def __init__(self, df: pd.DataFrame = None, target_column: str = None, sequence_window: int = None,
                 queue_window: int = None, n_components: int = None, lag: int = None, is_cps_filter_on: bool = True,
                 is_quantile_threshold: bool = False, is_exp_squared: bool = False, threshold_quantile_coeff: float = 0.91,
                 threshold_std_coeff: float = 3.61):
        self.df = df
        self.target_column = target_column
        self.sequence_window = sequence_window
        self.lag = lag
        self.n_components = n_components
        self.queue_window = queue_window
        self.is_cps_filter_on = is_cps_filter_on
        self.is_quantile_threshold = is_quantile_threshold
        self.threshold_quantile_coeff = threshold_quantile_coeff
        self.threshold_std_coeff = threshold_std_coeff
        self.is_exp_squared = is_exp_squared

        if self.sequence_window is None:
            self.sequence_window = super().__init__(time_series=df[target_column].values).runner_wss()[0]

        if self.queue_window is None:
            self.queue_window = int(self.sequence_window * 1.5)

        if self.lag is None:
            self.lag = self.sequence_window // 4

        if self.df is None:
            raise AttributeError("Dataframe is None!")

        if self.target_column is None:
            raise AttributeError("Target column is None!")

        if self.df.shape[0] <= 10:
            raise NotImplementedError("Your dataframe rows are less then 10! It has`t been expected.")

        if self.lag >= self.df.shape[0] // 2:
            raise ArithmeticError("Expected lag between subsequences too high for this dataframe!")

        if self.n_components <= 0:
            raise AttributeError("Number of components can not be equal to 0 or lower. There is no logic in it.")

        if self.n_components is None:
            self.n_components = 1

        self.ts = self.df[self.target_column].values

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
        list_matrix = []
        for ind in range(ts.shape[0] - self.lag - self.sequence_window):
            list_matrix.append(self.get_hankel_matrix(ts[ind:ind + self.sequence_window]))
        return np.array(list_matrix)

    def get_lagged_matrix(self, ts: np.array) -> np.ndarray:
        list_matrix = []
        for ind in range(ts.shape[0] - self.lag - self.sequence_window):
            list_matrix.append(self.get_hankel_matrix(ts[ind + self.lag:ind + self.lag + self.sequence_window]))
        return np.array(list_matrix)

    def preprocess_ts(self) -> Tuple[np.ndarray, np.ndarray]:
        present_matrix = self.get_current_matrix(self.ts)
        lagged_matrix = self.get_lagged_matrix(self.ts)
        return present_matrix, lagged_matrix

    def get_distances(self) -> np.ndarray:
        score_list = np.zeros_like(self.ts)
        matrix_history, matrix_next = self.preprocess_ts()
        counter: int = 0
        while counter != self.ts.shape[0] - self.lag - self.sequence_window:
            score_list[counter] = self._sst_svd(x_test=matrix_next[counter],
                                                x_history=matrix_history[counter],
                                                n_components=self.n_components)
            counter += 1
        if self.is_exp_squared:
            score_list = np.exp(score_list)**2
        return score_list



if __name__ == "__main__":
    import sys

    sys.path.append("../..")

    from data.SythData import LinearSteps, SinusoidWaves

    data = SinusoidWaves(length_data=4000, cps_number=2, white_noise_level="min").get()

    model = SingularSequenceTransformer(df=data, target_column="x",
                                        sequence_window=25,
                                        lag=20,
                                        queue_window=10,
                                        is_cps_filter_on=True,
                                        n_components=1,
                                        threshold_std_coeff=3.1)

    distances = model.get_distances()

    cps_pred = model.predict()
    stop = 0


