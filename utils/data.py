import numpy as np
import pandas as pd

def df_expirement(url="https://storage.yandexcloud.net/cloud-files-public/229G_las_files.csv"):
    df = pd.read_csv(StringIO(requests.get(url).content.decode('utf-8')))
    df['time'] = np.arange(0, len(df)*1, 1).astype('datetime64[s]')
    df = df.set_index('time')
    return df

list_links = ['https://storage.yandexcloud.net/cloud-files-public/229G_las_files.csv',
             'https://storage.yandexcloud.net/cloud-files-public/231G_las_files.csv',
             'https://storage.yandexcloud.net/cloud-files-public/237G_las_files.csv']
