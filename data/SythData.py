import numpy as np
import pandas as pd


class SythDataConstructor:
    """ This is fundament class which should be used for any syth data generators.

    Notes:
        length_data % cpu_numbers has to be equal 0!

    Attributes:
        frequency: which freq should be used seconds, minutes, days.
        length_data: just how many points should be generated.
        cps_number: number of change points over generated data.
    """

    def __init__(self,
                 white_noise_level: str = "default",
                 frequency: str = "s",
                 length_data: int = 24 * 7 * 15 + 15,
                 cps_number: int = 15):
        self.frequency = frequency
        self.length_data = length_data
        self.cps_number = cps_number

        self.white_mean = 0
        if white_noise_level == "default":
            self.white_std = 0.5
        elif white_noise_level == "max":
            self.white_std = 1
        elif white_noise_level == "min":
            self.white_std = 0.01
        else:
            raise NameError("Not implemented white noise level!")

        if length_data % cps_number != 0:
            raise ValueError("Not equal length of data and cpu_numbers expected from syth data!")

    def generate_empty_df(self) -> pd.DataFrame:
        """ Generate dataframe with timestamps.

        Returns:
            pandas dataframe with expected frequency and length
        """
        return pd.DataFrame(index=pd.date_range(start="10/07/1999",
                                                periods=self.length_data,
                                                freq=self.frequency,
                                                normalize=True,
                                                inclusive="both",
                                                name="time"))

    def generate_white_noise(self) -> np.array:
        """ Generate random noise for your data.

        Returns:
            array of white noise based on expected length of data.
        """
        return np.random.normal(self.white_mean,
                                self.white_std,
                                size=self.length_data)

    def generate_array_of_change_points(self) -> np.array:
        """ Generate values which represent CPs over syth data.

        Returns:
            numpy array of int values where 1 is change point and 0 is default value.
        """
        cps_index = [i for i in range(self.length_data // self.cps_number,
                                      self.length_data,
                                      self.length_data // self.cps_number)]
        dp = [0 if i not in cps_index else 1 for i in range(self.length_data)]
        return np.array(dp)

    def generate_data(self) -> np.array:
        """ Generate syth data array

        Returns:
            expected syth data based on class idea.
        """
        ...

    def get(self) -> pd.DataFrame:
        """ Get syth data.

        Returns:
            pandas dataframe with syth data and time index.
        """
        ...


class LinearSteps(SythDataConstructor):
    def get_linear_array(self,
                         beta_past: float,
                         k_past: float,
                         beta_mutation_coeff: float,
                         k_mutation_coeff: float) -> tuple[np.array, float, float]:
        """ Generate random linear array based on past observation

        Notes:
            beta_mutation_coeff as well as k_mutation_coeff should be defined based on expertise. These coefficients
            help to connect nearest arrays.

        Args:
            beta_past: beta value in the past array.
            k_past: k coefficient in the past array.
            beta_mutation_coeff: treshold for beta deviation.
            k_mutation_coeff: treshold for k coeff deviation.

        Returns:
            tuple of generated data and info for this generations beta and k_coeff.
        """
        beta = np.random.uniform(beta_past, 1)
        k_coeff = np.random.uniform(k_past, 1)
        if np.random.uniform(0, 1) > beta_mutation_coeff:
            beta = np.random.uniform(-1, 1)
        if np.random.uniform(0, 1) > k_mutation_coeff:
            k_coeff = np.random.uniform(-1, 1)
        dp = [k_coeff * x + beta for x in range(0, self.length_data // self.cps_number)]
        return np.array(dp), beta, k_coeff

    def generate_data(self, initial_beta: float = -0.01,
                      initial_k: float = 0.2,
                      beta_mutation_coeff: float = 0.8,
                      k_mutation_coeff: float = 0.2) -> np.array:
        dp = []
        for steps in range(self.cps_number):
            temp_info = self.get_linear_array(initial_beta,
                                              initial_k,
                                              beta_mutation_coeff,
                                              k_mutation_coeff)
            dp.extend(temp_info[0])
            initial_beta = temp_info[1]
            initial_k = temp_info[2]
        return np.array(dp)

    def get(self):
        df = self.generate_empty_df()
        df['x'] = np.add(self.generate_data(), self.generate_white_noise())
        df['CPs'] = self.generate_array_of_change_points()
        return df


class SinusoidWaves(SythDataConstructor):
    def get_sinusoid_array(self, beta_past: float, beta_mutation_coeff: float) -> tuple[np.array, float]:
        """ Generate sinusoid waves over expected shape.

        Args:
            beta_past: beta coefficient for sinus wave.
            beta_mutation_coeff: coeff for mutation operator.

        Returns:
            array of sinusoid data
        """
        beta_past = np.random.uniform(low=beta_past, high=2)
        if np.random.uniform(low=0, high=1) > beta_mutation_coeff:
            beta_past = np.random.uniform(low=-2, high=2)
        x = np.linspace(start=0, stop= self.length_data // self.cps_number, num=self.length_data // self.cps_number)
        return np.sin(x) * beta_past, beta_past

    def generate_data(self, initial_beta: float = 0.5, beta_mutation_coeff: float = 0.5) -> np.array:
        dp = []
        for steps in range(self.cps_number):
            temp_info = self.get_sinusoid_array(initial_beta,
                                                beta_mutation_coeff)
            dp.extend(temp_info[0])
            initial_beta = temp_info[1]
        return np.array(dp)

    def get(self):
        df = self.generate_empty_df()
        df['x'] = np.add(self.generate_data(), self.generate_white_noise())
        df['CPs'] = self.generate_array_of_change_points()
        return df
