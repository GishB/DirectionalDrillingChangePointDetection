import numpy as np
import pandas as pd
from collections import namedtuple
import sys
sys.path.append("..")

from utils.hyperparameters.WSSAlgorithms import WindowSizeSelection


class KalmanFilter:
    """ Idea is to find data deviations based on Kalman extrapolation for nearest data.

    Attributes:
        df: pandas dataframe with target data.
        target_column: column which should be checked.
    """
    def __init__(self,
                 df: pd.DataFrame = None,
                 target_column: str = None,
                 window: int = None):
        self.df = df
        self.target_column = target_column
        self.window = window

        if window is None:
            self.window = WindowSizeSelection(df[target_column]).runner_wss()[0]

    def get_initial_guess(self) -> tuple[float, float]:
        """ Get gaussian stats based on first data.

        Returns:
            gaussian stats tuple as mean and std values
        """
        return self.df[self.target_column].mean(), self.df[self.target_column].std()

    def gaussian_multiply(self, g1: tuple[float, float], g2: tuple[float, float]) -> tuple[float, float]:
        """ Update gaussian stats for current status.

        Args:
            g1: past gaussian stats.
            g2: current gaussian stats.

        Returns:
            likelihood gaussian statistics.
        """
        ...

    def predict(self, mean_gauss: float, std_gauss: float) -> float:
        """

        Args:
            mean_gauss:
            std_gauss:

        Returns:

        """
        ...


gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: '𝒩(μ={:.3f}, 𝜎²={:.3f})'.format(s[0], s[1])


def gaussian_multiply(g1, g2) -> object:
    mean = (g1.var * g2.mean + g2.var * g1.mean) / (g1.var + g2.var)
    variance = (g1.var * g2.var) / (g1.var + g2.var)
    return gaussian(mean, variance)


def update(prior, likelihood):
    posterior = gaussian_multiply(likelihood, prior)
    return posterior


def predict(pos, movement):
    return gaussian(pos.mean + movement.mean, pos.var + movement.var)


def online_detection(time_series: list = None, window=20, queue_window=10, treshold_coef=3):
    ''' 
        An online implementation of Kalman Filter Change Point Detection based on risiduals,
    and current time-series statistics for treshold updating.
    
    Note:
        queue_window - min possible distance between two predicted CPs.
        treshold_coef - if residuals above this treshold than we can detect CP. Based on np.meean + 3*np.std
        where 3 is the coef.
        window - the length of data which we save in memory.
    '''

    dp = [0] * window  # массив с предсказаниями по CPs
    queue_risiduals = [0] * window  # очередь с рядом остатков по которому отбивают трешхолд
    queue_cp = [0] * queue_window  # очередь CPs
    queue_ts = time_series[:window]  # р
    queue_gaussian = [gaussian(np.mean(queue_ts), np.var(queue_ts))] * 2

    for i in range(window, len(time_series)):  # Это цикл позволяет нам получать данные раз в 9 секунд условно.
        gaussian_likelihood = queue_gaussian[0]  # смотрим начальное состояние MEAN VAR
        gaussian_prior = gaussian(np.mean(queue_ts),
                                  np.var(queue_ts))  # смотрим обновление состояние MEAN VAR

        x = update(gaussian_prior, gaussian_likelihood)  # обновляем значение MEAN VAR
        queue_gaussian.pop(0)  # удаляем старое значение
        queue_gaussian.append(gaussian_prior)  # добавляем новое значение likelihoo

        next_prediction = np.random.normal(
            loc=x.mean)  # Генерируем значение исходя из нового знания об gaussian
        next_value = time_series[i]  # П
        risidual_value = abs(next_prediction - next_value)

        mean_risidual_prev = np.mean(queue_risiduals)
        std_risidual_prev = np.std(queue_risiduals)
        queue_risiduals.pop(0)  # удаляем старое значение risiduals
        queue_risiduals.append(risidual_value)  # новое значение risiduals

        if risidual_value >= mean_risidual_prev + std_risidual_prev * treshold_coef:  # проверка что мы нашли переход
            if max(queue_cp) != 1:  # queue filter IF
                queue_cp.pop(0)
                queue_cp.append(1)
                dp.append(1)
            else:  # queue filter else
                queue_cp.pop(0)
                queue_cp.append(0)
                dp.append(0)
        else:
            dp.append(0)
            queue_cp.pop(0)
            queue_cp.append(0)

        queue_ts.pop(0)  # удаляем из очереди старое значение VALUE
        queue_ts.append(next_value)  # добавляем в очередь новое значение VALUE
    return dp
