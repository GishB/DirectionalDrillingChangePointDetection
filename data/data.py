import pandas as pd
import numpy as np
from io import StringIO
import requests


class DrillingData:
    def __init__(self,
                 dataset_name: str = "default"):
        self.url_dict = {
            "229G": 'https://storage.yandexcloud.net/cloud-files-public/229G_las_files.csv',
            "231G": 'https://storage.yandexcloud.net/cloud-files-public/231G_las_files.csv',
            "237G": 'https://storage.yandexcloud.net/cloud-files-public/237G_las_files.csv'
        }
        self.dataset_name = dataset_name
        if dataset_name not in ["default", "229G", "231G", "237G"]:
            raise NameError("There is not such dataset name.")

    @staticmethod
    def load_raw_data(url: str) -> pd.DataFrame:
        return pd.read_csv(StringIO(requests.get(url).content.decode('utf-8')))

    @staticmethod
    def transform_data(data: pd.DataFrame) -> pd.DataFrame:
        data['time'] = np.arange(0, data.shape[0]*1, 1).astype('datetime64[s]')
        data = data.set_index('time')
        return data

    def get_data(self) -> pd.DataFrame:
        if self.dataset_name == "default":
            raw_data = self.load_raw_data(url=self.url_dict.get("237G"))
        else:
            raw_data = self.load_raw_data(url=self.url_dict.get(self.dataset_name))
        return self.transform_data(raw_data)

