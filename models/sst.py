"""This is an implementation of SST model based on experiments pipline for change point detection task in time series.

Todo:
    * unittest
    * auto optimization for threshold percent param.
"""

from typing import List, Tuple
from scipy.linalg import hankel
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy import ndarray
import pandas as pd


class SingularSequenceTransformation:
    """An example of singular sequence analysis method to solve CPD problem.

    Attributes:
        df (pd.DataFrame): pandas DataFrame which contents data.
        treshold_percent (float): value which selected as specific anomaly probability to detect anomaly at synth data.
    """

    def __init__(self,
                 df: pd.DataFrame = None,
                 window_length: int = 24,
                 treshold_percent: float = 0.415,
                 is_online: bool = True,
                 is_sigmoid: bool = True):

        self.df = self.apply_normalization(df)
        self.treshold_percent = treshold_percent
        self.window_length = window_length
        self.is_online = is_online
        self.is_sigmoid = is_sigmoid

    @staticmethod
    def apply_normalization(df: pd.DataFrame) -> pd.DataFrame:
        """ Normalize each column of DataFrame.

        Args:
            df: DataFrame to normalize.

        Returns:
            Normalized dataframe.
        """
        for column in df.columns:
            scaler = MinMaxScaler()
            scaler.fit(df[column].values.reshape(-1,1))
            df[column] = scaler.transform(df[column].values.reshape(-1,1))
        return df

    def runner(self, column: str) -> pd.Series:
        score_list = []
        if self.is_online:
            for number in range(0, self.df.shape[0]-1):
                first_vector = self.df[column][number:self.window_length+number]
                second_vector = self.df[column][number+1:self.window_length+number+1]
                score_list.append(self.calculate_score(first_vector, second_vector))
        else:
            first_vector = self.df[column][0:self.window_length]
            for number in range(0, self.df.shape[0]-1):
                second_vector = self.df[column][number+1:self.window_length+number+1]
                score_list.append(self.calculate_score(first_vector, second_vector))
        return pd.Series(data=score_list+[0], name=column, index=self.df.index)

    def calculate_score(self, first_vector: np.ndarray, second_vector: np.ndarray) -> float:
        return self.get_score(np.diag(hankel(first_vector)), np.diag(hankel(second_vector)))

    def get_score(self, first_tr: np.ndarray, second_tr: np.ndarray) -> float:
        """ Calculate scores between two sets.

        Returns:
            A list of float distances between past weeks data.
        """
        a = abs(sum(first_tr ** 2))
        b = abs(sum(second_tr ** 2))
        if self.is_sigmoid:
            c = self.sigmoid_loss(a, b)
        else:
            c = a - b
        return c

    @staticmethod
    def sigmoid_loss(first_score: float, second_score: float) -> float:
        """ Calculate probability that target data has some similarities for past weeks.

        Returns:
            Pseudo probability that we have time series similar to the past data.
        """
        return 1 / (1 + np.exp(-(np.mean(first_score) - np.mean(second_score))))


