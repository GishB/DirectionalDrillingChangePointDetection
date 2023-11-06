"""this module helps to generate sythethic data traffic with an anomaly"""

import pandas as pd
import numpy as np
import numpy.typing as npt


class SinusoidTraffic:
    def __init__(self,
                 anomaly_bool: bool = True,
                 coeff_anomaly: float = 1.5):
        self.anomaly_bool = anomaly_bool
        self.coeff_anomaly = coeff_anomaly

    @staticmethod
    def generate_time() -> pd.date_range:
        return pd.date_range(start="2023-09-01", end="2023-09-29", freq="1H", inclusive='left')

    @staticmethod
    def generate_ts() -> npt.NDArray[float]:
        return np.array([np.sin(
            np.arange(start=0,
                      stop=np.pi,
                      step=np.pi / 24))
            for i in range(28)]).ravel()

    def create_df(self) -> pd.DataFrame:
        return pd.DataFrame(index=self.generate_time(),
                            data=self.generate_ts(),
                            columns=["sinusoid_values_from_an_object"])

    def generate_dataset(self) -> pd.DataFrame:
        df = self.create_df()
        if self.anomaly_bool:
            df.sinusoid_values_from_an_object.iloc[:-24] = df[:-24].sinusoid_values_from_an_object * self.coeff_anomaly
        df.sinusoid_values_from_an_object = [np.random.uniform(low=-0.115, high=0.115) + x for x in
                                             df.sinusoid_values_from_an_object]
        return df


class GeologyTimeSeries:
    def __init__(self,
                 anomaly_bool: bool = True,
                 coeff_anomaly: float = 1.5,
                 period: int = 48):
        self.anomaly_bool = anomaly_bool
        self.coeff_anomaly = coeff_anomaly
        self.period = period

    def generate_time(self) -> pd.date_range:
        return pd.date_range(start="2023-09-01", periods=self.period,
                             freq="1min", inclusive='left')

    @staticmethod
    def generate_sinusoid_sequence(length: int = 24) -> npt.NDArray[float]:
        return np.arange(start=0,
                         stop=np.pi,
                         step=np.pi / length)

    @staticmethod
    def generate_linear_step_sequence(length: int = 24):
        low = np.random.uniform(low=0, high=100)
        return np.random.randint(low=low,
                                 high=int(low + low * np.random.rand()) + 1,
                                 size=length)

    def generate_example_cp_array(self) -> npt.NDArray:
        ts_past = self.generate_sinusoid_sequence()
        ts_future = self.generate_linear_step_sequence()
        return np.array([ts_past, ts_future]).ravel()

    def generate_dataset(self) -> pd.DataFrame:
        return pd.DataFrame(index=self.generate_time(),
                            data=self.generate_example_cp_array(),
                            columns=["example_cp_between_diff_generators"])


if __name__ == "__main__":
    test_df = SinusoidTraffic(anomaly_bool=True).generate_dataset()
    test_geology = GeologyTimeSeries(anomaly_bool=True).generate_dataset()
    stop = 0
