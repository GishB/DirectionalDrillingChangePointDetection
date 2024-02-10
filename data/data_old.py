import sys
import os
sys.path.append(os.path.abspath(".."))
from utils import libs_cpd


def df_expirement(name_well='xxxAA564G'):
    """
    xxxAA684G or xxxAA564G
    """ 
    url = "https://storage.yandexcloud.net/cloud-files-public/dataframe.csv"
    df = libs_cpd.pd.read_csv(libs_cpd.StringIO(libs_cpd.requests.get(url).content.decode('utf-8')), sep='|')
    
    df = df[df[df.columns[0]] == name_well] #xxxAA684G

    df.replace(['-9999', -9999,'missing','#'], libs_cpd.np.nan, inplace=True)
    df = df[df['unitless'].notna()]
    df = df[df['uR/h'].notna()]
    df = df.drop(axis=1, labels=(df.columns[0])) \
        .drop(axis=1, labels=(df.columns[1]))[['uR/h', 'unitless']].reset_index(drop=True)

    #Create new columns for CPd and time
    df['change_points'] = [1 if df['unitless'][i] !=  df['unitless'][i+1] else 0 for i in range(len(df)-1)] + [0]
    df['time'] = libs_cpd.np.arange(0, len(df) * 1, 1).astype('datetime64[s]')
    df = df.set_index('time')
    df = df.rename(columns={"uR/h": "GR", "change_points": "CPs"})
    return df
