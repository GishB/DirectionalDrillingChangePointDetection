import pandas as pd
import numpy as np
from io import StringIO
from typing import Tuple, List, Optional
import requests


class DrillingData:
    """ This class helps to load las files for one of selected horizontal well.

    Notes:
        - If you define name which does not exist then program will raise NameError!
        - All data are restored on public yandex cloud server.
        - If you need to load just raw data then use load_raw_data method.

    Attributes:
        dataset_name: indicate which well you need to load data.
    """

    def __init__(self,
                 dataset_name: str = "default",
                 sep: str = ","):
        self.url_dict = {
            "229G": "https://storage.yandexcloud.net/cloud-files-public/229G_las_files.csv",
            "231G": "https://storage.yandexcloud.net/cloud-files-public/231G_las_files.csv",
            "237G": "https://storage.yandexcloud.net/cloud-files-public/237G_las_files.csv",
            "xxxAA564G": "https://storage.yandexcloud.net/cloud-files-public/dataframe.csv",
            "xxxAA684G": "https://storage.yandexcloud.net/cloud-files-public/dataframe.csv"
        }
        self.dataset_name = dataset_name
        self.sep = sep
        self.list_of_available_dataset_names: List[str] = ["default", "229G", "231G", "237G", "xxxAA684G", "xxxAA564G"]
        if dataset_name not in self.list_of_available_dataset_names:
            raise NameError("There is not such dataset name.")
        if dataset_name in ["xxxAA684G", "xxxAA564G"]:
            self.sep = "|"

    def load_raw_data(self, url: str) -> pd.DataFrame:
        """ Load las files as it is in pandas format.

        Warning:
            - These files are available only for education purposes and shall not be used for any other points.

        Notes:
            - value like -9999 means that data has been missing.
            - unitless column means type of layer.
            - uR/h the most important drilling data columns for analysis.

        Returns:
            pandas dataframe with all available columns and rows from chosen las file.
        """
        return pd.read_csv(StringIO(requests.get(url).content.decode('utf-8')), sep=self.sep)

    @staticmethod
    def generate_cp_based_on_rock_types(array_of_rocks_types: np.array) -> np.array:
        """ Generate change points based on different rock types at original data.

        Arg:
            array_of_rocks_types: array of rock types

        Returns:
            array of binary data which contents change points between different rock types.
        """
        dp = np.zeros_like(array_of_rocks_types)
        first_type: int = array_of_rocks_types[0]
        for indx, val in enumerate(array_of_rocks_types):
            if first_type != val:
                dp[indx] = 1
                first_type = val
        return dp

    def extract_transform(self) -> np.ndarray[np.array, np.array]:
        """ Extract target dataframe and transform data as well as X and Y.

        Notes:
            1) If are looking for CPD task data and hot start function here it is.
            2) x features based on gamma rate.
            3) y - target features based on different rock types (there are 6 of them).

        Returns:
            numpy array for features and target data.
        """
        df = self.get()
        if "xxx" not in self.dataset_name:
            x = df["GR"].values
            y = df["CPs"].values
        else:
            df.replace(to_replace=-9999, value=np.NaN, regex=True, inplace=True)
            df.dropna(thresh=15, inplace=True)
            df.reset_index(drop=True, inplace=True)
            x = df["uR/h"].values
            y = self.generate_cp_based_on_rock_types(df["unitless"].values)
        return np.array(x, y)

    def get(self) -> pd.DataFrame:
        """ Just load data from available bucket.

        Returns:
            selected dataframe.
        """
        if self.dataset_name == "default":
            raw_data = self.load_raw_data(url=self.url_dict.get("237G"))
        else:
            raw_data = self.load_raw_data(url=self.url_dict.get(self.dataset_name))
            if self.dataset_name in ["xxxAA684G", "xxxAA564G"]:
                raw_data = raw_data[raw_data[raw_data.columns[0]] == self.dataset_name]
        return raw_data.reset_index(drop=True)
