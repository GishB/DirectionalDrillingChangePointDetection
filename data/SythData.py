import sys
import os
sys.path.append(os.path.abspath(".."))
from utils import libs_cpd


class SythDataGenerator:
    """SythDataGenerator class (SDG).

    This class helps to generate syth data for CPD experiments.

    Note:
        At SDG class we have 1 group of function. It is step_rate_function which generate steps function with some noise.

    Args:
        number_steps: numbers of CPs.
        length: size of 1D time series which you expect to get.
        mess_rate: it is called np.random.randint where mess rate is std value for Gaussian distribution.
        normalization: normalize or not output list of syth data.
        savgol_filter: filter or not output list of syth data.

    """
    def __init__(self,
                 number_steps: int = 1,
                 length: int = 100,
                 mess_rate: float = 0,
                 normalization: bool = True,
                 savgol: bool = False):
    
        self.number_steps = number_steps
        self.length = length
        self.mess_rate = mess_rate
        self.normalization = normalization
        self.savgol = savgol
        
    def step_rate_function(self, number_steps: int, length: int, mess_rate: float) -> list:
        random_value = libs_cpd.np.random.randint(-length, length)
        function = [self.gaussian_mess(x=random_value - libs_cpd.np.random.randint(-length, length), mess_rate=mess_rate, size=length // (number_steps + 1))\
                    for w in range(number_steps+1)]
        cps_list = [[0 for i in range(length//(number_steps+1)-1)]+[1] if w != number_steps\
                    else [0 for i in range(length//(number_steps+1)-1)]+[0] for w in range(number_steps+1)]
        return list(libs_cpd.chain.from_iterable(function)), list(libs_cpd.chain.from_iterable(cps_list))

    def gaussian_mess(self, x: float, mess_rate: float, size: int) -> float:
        return libs_cpd.np.random.normal(loc=x, scale=mess_rate, size=size)

    def normalization_linear(self, x: list) -> list:
        return (x-min(x))/(max(x)-min(x))
    
    def filter_Savgol(self, x: list, window_length: int) -> list:
        return libs_cpd.savgol_filter(x, window_length, 3, mode='nearest')
    
    def wss(self, x: list) -> int:
        return libs_cpd.WindowSizeSelection(time_series = x,
                                            wss_algorithm = 'summary_statistics_subsequence').get_window_size()[0]
    
    def runner(self):
        out, cps = self.step_rate_function(number_steps=self.number_steps, length=self.length, mess_rate=self.mess_rate)
        if self.normalization:
            out = self.normalization_linear(out)
        if self.savgol:
            out = self.filter_Savgol(out, self.wss(out))
        return list(out), cps