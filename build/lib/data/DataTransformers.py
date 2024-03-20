import pandas as pd
import numpy as np


class CloudDataTransformer:
    """ This class helps to transform raw data as it expected for experiments in the project.

    Attributes:
        df: dataframe which was loaded from the cloud.
        dataset_name: dataset name which was loaded via CloudData

    """

    def __init__(self, df: pd.DataFrame = None,
                 dataset_name: str = "default"):
        self.df = df
        self.dataset_name = dataset_name

        if self.df is None:
            raise ValueError("dataset has not been defined!")

        if self.dataset_name not in ["default", "229G", "231G", "237G", "xxxAA684G", "xxxAA564G"]:
            raise NameError("There is not such dataset name.")

    @staticmethod
    def add_time_column(df: pd.DataFrame) -> pd.DataFrame:
        df['time'] = np.arange(0, df.shape[0] * 1, 1).astype('datetime64[s]')
        df = df.set_index('time')
        return df

    @staticmethod
    def replace_nan_values(df: pd.DataFrame) -> pd.DataFrame:
        return df.replace(['-9999', -9999, 'missing', '#'], np.nan)

    def drop_nan_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.replace_nan_values(df)
        df = df[df['unitless'].notna()]
        df = df[df['uR/h'].notna()]
        return df

    @staticmethod
    def add_expected_change_points(df: pd.DataFrame) -> pd.DataFrame:
        cps_list = [1 if df['unitless'].iloc[i] != df['unitless'].iloc[i + 1] else 0 for i in range(df.shape[0] - 1)]
        df['CPs'] = cps_list + [0]
        return df

    @staticmethod
    def take_expected_columns(df: pd.DataFrame) -> pd.DataFrame:
        return df[["uR/h", "ohmm", "ohmm.6", "m/hr", "unitless", "CPs"]].reset_index(drop=True)

    @staticmethod
    def rename_column_special(df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns={"uR/h": "GR",
                                  "ohmm": "Resist_short",
                                  "ohmm.6": "Resist_long",
                                  "unitless": "LITHOLOGY",
                                  "m/hr": "DrillingSpeed"})

    def transform(self) -> pd.DataFrame:
        """ Transform data initial point.

        Returns:
            DataFrame as it expected to be for any future tasks.
        """
        df = self.df
        if self.dataset_name in ["xxxAA684G", "xxxAA564G"]:
            df = self.replace_nan_values(df)
            df = self.add_expected_change_points(df)
            df = self.take_expected_columns(df)
            df = self.rename_column_special(df)
        df = self.add_time_column(df)
        return df
