import sys
import os
sys.path.append(os.path.abspath(".."))
import libs_cpd

def df_expirement(url="https://storage.yandexcloud.net/cloud-files-public/229G_las_files.csv"):
    df = libs_cpd.pd.read_csv(libs_cpd.StringIO(libs_cpd.requests.get(url).content.decode('utf-8')))
    df['time'] = libs_cpd.np.arange(0, len(df)*1, 1).astype('datetime64[s]')
    df = df.set_index('time')
    return df

list_links = ['https://storage.yandexcloud.net/cloud-files-public/229G_las_files.csv',
             'https://storage.yandexcloud.net/cloud-files-public/231G_las_files.csv',
             'https://storage.yandexcloud.net/cloud-files-public/237G_las_files.csv']
