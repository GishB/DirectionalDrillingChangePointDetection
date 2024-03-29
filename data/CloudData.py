import pandas as pd
from io import StringIO
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
        if dataset_name not in ["default", "229G", "231G", "237G", "xxxAA684G", "xxxAA564G"]:
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

    def get(self) -> pd.DataFrame:
        if self.dataset_name == "default":
            raw_data = self.load_raw_data(url=self.url_dict.get("237G"))
        else:
            raw_data = self.load_raw_data(url=self.url_dict.get(self.dataset_name))
            if self.dataset_name in ["xxxAA684G", "xxxAA564G"]:
                raw_data = raw_data[raw_data[raw_data.columns[0]] == self.dataset_name]
        return raw_data.reset_index(drop=True)
