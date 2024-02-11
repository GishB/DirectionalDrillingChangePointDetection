import sys
import os
sys.path.append(os.path.abspath(".."))
from utils import libs_cpd


def normalization_linear(x):
    return (x-min(x))/(max(x)-min(x))

def filter_Savgol(x, window_length):
    '''
        Savitzky-Golay filter from scipy
    '''
    aaccSG = libs_cpd.savgol_filter(x, window_length, 3, mode='nearest')
    return libs_cpd.np.array(aaccSG)

def cumsum(x, quantile_=0.99):
    '''
        Just CUMSUM filter
    '''
    quantile_99 = libs_cpd.np.quantile(x, quantile_)
    new_x = [g if abs(g) < quantile_99 else quantile_99*1.5 for g in x]
    ending, start, alarm, cumsum = libs_cpd.detect_cusum(new_x, libs_cpd.np.mean(new_x) + libs_cpd.np.std(new_x) * 3, libs_cpd.np.std(new_x), True, False)
    return ending, start, alarm, cumsum 

def queue(queue_window=10, time_series=None):
    queue = [0]*queue_window
    filtered_score = []
    for i in range(len(time_series)):
        value = time_series[i]
        if max(queue) != 0: #Вариант при котором CPs уже в очереди
            filtered_score.append(0)
            queue.pop(0)
            queue.append(0)
        else: # В очереди нет CPs
            filtered_score.append(value)
            queue.pop(0)
            queue.append(value)
    return filtered_score