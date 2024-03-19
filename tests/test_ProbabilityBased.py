import pytest
import sys

import numpy as np

sys.path.append("..")

from models.ProbabilityBased import KalmanFilter


class TestKalmanFilter:
    model = KalmanFilter()

    def test_get_array_info(self):
        """ Get gaussian stats based on an array of values.

        Returns:
            gaussian stats tuple as mean and std values
        """
        array_test = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        expected_mean, expected_std = np.mean(array_test), np.std(array_test)
        test_mean, test_std = self.model.get_array_info(array_test)
        assert test_std == expected_std, "test std val equal to expected"
        assert test_mean == expected_mean, "test mean val equal to expected"

    def test_gaussian_multiply(self):
        """ Update gaussian stats based on prev info and current status.

        Notes:
            first index is mean value
            second index is var value

        Returns:
            likelihood gaussian statistics.
        """
        g1 = 10, 1
        g2 = 1, 10
        test_mean, test_var = self.model.gaussian_multiply(g1, g2)
        expected_mean, expected_var = 9.18, 0.90
        assert (test_mean <= expected_mean + 2e-1) and (test_mean >= expected_mean - 2e-1), \
            "test gaussian mean value is equal to 9.18"
        assert (test_var <= expected_var + 2e-1) and (test_var >= expected_var - 2e-1), \
            "test gaussian var value is equal to 0.90"

    def test_forecast(self):
        """ forecast next values based on estimated gaussian coefficient.

        Returns:
            normal generated value based on expected mean and std.
        """
        expected_min: float = 6.48
        expected_max: float = 12.6
        test_val: float = self.model.forecast(9.18, 0.9)
        assert expected_min <= test_val, 'test val is higher 6.48'
        assert test_val <= expected_max, 'test val is less 12.6'
