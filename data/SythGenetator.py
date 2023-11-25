"""This module helps to generate sythethic data for change point tasks"""

import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Union


class SythData:
    """ Simple idea of the class is to make some change point detection data.

    Notes:
        This is a constructor class to show syth data idea for future data.

    Args:

    """

    def __init__(self,
                 is_anomaly: bool = True,
                 is_time: bool = True,
                 is_date_range: bool = True,
                 is_uniform_distr_noise: bool = True,
                 cp_count: int = 2,
                 period: int = 48,
                 len_ts: int = None,
                 coeff_anomaly: float = 1.5,
                 freq_str: str = '1H',
                 freq_int: int = 24,
                 date_start: str = '2023-07-10',
                 date_finish: str = '2023-07-20'):

        self.is_anomaly = is_anomaly
        self.is_time = is_time
        self.is_date_range = is_date_range
        self.is_uniform_distr_noise = is_uniform_distr_noise
        self.cp_count = cp_count
        self.coeff_anomaly = coeff_anomaly
        self.period = period
        self.len_ts = len_ts
        self.freq_str = freq_str
        self.freq_int = freq_int
        self.date_start = date_start
        self.date_finish = date_finish

    def generate_date_time_range(self) -> pd.date_range:
        """ Generate index for your dataframe.

        Returns:
            Range of time index based on start and end data or numer of periods with freq.
        """
        if self.is_date_range:
            data = pd.date_range(start=self.date_start, end=self.date_finish, freq=self.freq_str, inclusive='left')
        else:
            data = pd.date_range(start=self.date_start, periods=self.period, freq=self.freq_str, inclusive='left')
        return data

    def generate_ts(self) -> npt.NDArray[float]:
        """ Generate time series with all possible data according to an object.

        Returns:
            Array of normal data and possible anomalies.
        """
        pass

    def generate_anomaly_ts(self) -> npt.NDArray[float]:
        """ Make abnormal data based on anomaly params.

        Returns:
            In default return anomaly data points according to syth data object.
        """
        pass

    def create_df(self) -> pd.DataFrame:
        """ Create data which you expect to get.

        Returns
            In default you will get DataFrame of data where index
            is a time series and columns name specified to syth class.
        """
        pass


class SinusoidTraffic(SythData):
    def __init__(self):
        super().__init__()

    def generate_sinusoid_wave(self) -> npt.NDArray[float]:
        data = np.sin(np.arange(0, 2 * np.pi, 2 * np.pi / self.freq_int))
        if self.is_uniform_distr_noise:
            data = np.add(data, np.random.uniform(low=0, high=0.05, size=data.shape))
        return data

    def generate_anomaly_ts(self) -> npt.NDArray[float]:
        return self.generate_sinusoid_wave() * self.coeff_anomaly

    def generate_ts(self, index_: npt.NDArray[float] = None) -> npt.NDArray[float]:
        count_ts_generations = index_.shape[0] // self.freq_int
        counter_cp = 0
        cashe = 0
        data = []
        for val in range(count_ts_generations):
            if ((np.random.rand() > ((1 + counter_cp) / (self.cp_count + 2))) and
                    (counter_cp < self.cp_count and cashe != 1)):
                cashe = 1
                counter_cp += 1
                data.extend(self.generate_anomaly_ts())
            else:
                cashe = 0
                data.extend(self.generate_sinusoid_wave())
        return np.array(data[:index_.shape[0]]).ravel()

    def create_df(self) -> pd.DataFrame:
        index_ = self.generate_date_time_range()
        data_ = self.generate_ts(index_=index_)
        return pd.DataFrame(index=index_,
                            data=data_,
                            columns=["sinusoid_feature"])


class LinearJumps(SythData):
    def __init__(self):
        super().__init__()

    def generate_linear_random(self) -> npt.NDArray[float]:
        data = np.full(self.freq_int, 1, dtype=int)
        if self.is_uniform_distr_noise:
            data = np.add(data, np.random.uniform(low=0, high=0.05, size=data.shape))
        return data

    def generate_anomaly_ts(self) -> npt.NDArray[float]:
        return self.generate_linear_random() * self.coeff_anomaly

    def generate_ts(self, index_: npt.NDArray[float] = None) -> npt.NDArray[float]:
        count_ts_generations = index_.shape[0] // self.freq_int
        counter_cp = 0
        cashe = 0
        data = []
        for val in range(count_ts_generations):
            if ((np.random.rand() > ((1 + counter_cp) / (self.cp_count + 2))) and
                    (counter_cp < self.cp_count and cashe != 1)):
                cashe = 1
                counter_cp += 1
                data.extend(self.generate_anomaly_ts())
            else:
                cashe = 0
                data.extend(self.generate_linear_random())
        return np.array(data[:index_.shape[0]]).ravel()

    def create_df(self) -> pd.DataFrame:
        index_ = self.generate_date_time_range()
        data_ = self.generate_ts(index_=index_)
        return pd.DataFrame(index=index_,
                            data=data_,
                            columns=["linear_feature"])


class MixData(SythData):
    def __init__(self):
        super().__init__()

    def generate_ts(self, index_: npt.NDArray[float] = None) -> npt.NDArray[float]:
        sinusoid_ts = SinusoidTraffic().generate_ts(index_=index_[:index_.shape[0] // 2])
        linear_ts = LinearJumps().generate_ts(index_=index_[:index_.shape[0] // 2])
        return np.concatenate([sinusoid_ts, linear_ts])

    def create_df(self) -> pd.DataFrame:
        index_ = self.generate_date_time_range()
        data_ = self.generate_ts(index_)
        return pd.DataFrame(index=index_,
                            data=data_,
                            columns=["mixed_feature"])


if __name__ == "__main__":
    test_sinusoid = SinusoidTraffic().create_df()
    test_linear = LinearJumps().create_df()
    test_mix = MixData().create_df()
    stop = 0
