from typing import Optional, List
from utils.DataTransformers import Filter

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
        x = np.linspace(start=0, stop=self.length_data // self.cps_number, num=self.length_data // self.cps_number)
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


class RandomChangePointsGenerator(Filter):
    """ Default idea is to generate different change point sequences for 1D cases.
    """

    def __init__(self,
                 seed: Optional[int],
                 cps_number: int = 0,
                 length_data: int = 24 * 7 * 15 + 15,
                 minimum_sequence_cp: int = 10,
                 start_mutation_coeff: float = 0.5,
                 treshold_mutation_coeff: float = 0.1,
                 power_coeff: float = 2,
                 attemps_to_failure: int = 15):

        if seed is None:
            seed = np.random.randint(low=0, high=65535)

        self.seed = seed
        self.cps_number = cps_number
        self.length_data = length_data

        self.power_coeff = power_coeff
        self.start_mutation_coeff = start_mutation_coeff
        self.treshold_mutation_coeff = treshold_mutation_coeff
        self.minimum_sequence_cp = minimum_sequence_cp
        self.attemps_to_failure = attemps_to_failure

        if cps_number < 0:
            raise AttributeError(f"Change points number has to be positive! However, you set it to be {cps_number}")

        if length_data < 10:
            raise NotImplementedError("This class expect to generate array with more then 10 values in a sequence!")

        if (length_data / cps_number) < 10:
            raise NotImplementedError("Expected length of data is to small for expected cps_number! One of your "
                                      "sequences could be less then 10 points which may lead to cps model errors.")

    def _new_mutation_coeff(self, sequence_len: int) -> float:
        """ Generate random float value between 0 and 1.

        Notes:
            we use it to update mutation coefficient.

        Arg:
            sequence_len: len of sequence from the last change points.

        Returns:
            float value
        """
        if sequence_len ** self.power_coeff > self.length_data:
            out = np.random.random() - self.treshold_mutation_coeff
        else:
            out = np.random.random()
        return out

    def _is_mutation_apply(self, sequence_len: int, past_mutation_coeff: float) -> bool:
        """ Should we apply mutation factor

        Notes:
            if true then we apply mutation.

        Arg:
            sequence_len: len of sequence from the last change points.
            past_mutation_coeff: coefficient which we use for mutation logic.

        Returns:
            boolean value
        """
        out = False
        if sequence_len > self.minimum_sequence_cp:
            if np.random.random() > past_mutation_coeff:
                out = True
        return out

    def generate_change_points_with_random(self, cps_array: Optional[np.array]) -> np.array:
        """ Generate change points based on random numpy function.

        Notes:
            1. By default, this function helps to generate all change points
             in case of any failure from mutation function.
            2. You can simply use this function to generate you own change points based on random indx.
            3. Here you might see queue filter which helps to filter change point distance.

        Args:
            cps_array: array of change points or just none if you generate a new one.

        Returns:
            array of change points.
        """
        if cps_array is None:
            cps_array: np.array = np.zeros(shape=self.length_data)
        count_cps = sum(cps_array)
        attempts: int = 0
        while (count_cps < self.cps_number) or (attempts < self.attemps_to_failure):
            random_cp_index = np.random.randint(size=self.cps_number, low=5, high=self.length_data-5)
            cps_array[random_cp_index] = 1
            cps_array = self.queue(queue_window=self.minimum_sequence_cp, time_series=cps_array)
            attempts += 1
        if attempts == self.attemps_to_failure:
            raise NotImplementedError("Failure to generate random change points due unexpected behaviour! "
                                      f"Try to set other init params or increase sequence length: "
                                      f"cps_number = {self.cps_number} |"
                                      f" minimum_sequence_cp: {self.minimum_sequence_cp}")
        return cps_array

    def generate_change_points_with_mutation(self) -> np.array:
        """ Main function to generate array of change points.

        Notes:
            Baseline idea is to generate change points based on mutation coefficient.

        Returns:
            array of change points
        """
        cps_array: np.array = np.zeros(shape=self.length_data)
        cps_counter: int = 0
        counter_sequence_len: int = 0
        indx: int = 0
        past_mutation_coeff = self.start_mutation_coeff
        while indx < self.length_data - 5:
            is_cp: bool = False
            if indx >= 5:
                is_cp: bool = self._is_mutation_apply(counter_sequence_len, past_mutation_coeff)
            if is_cp:
                past_mutation_coeff = self._new_mutation_coeff(counter_sequence_len)
                cps_counter += 1
                counter_sequence_len = 0
            else:
                counter_sequence_len += 1
            indx += 1
        if cps_counter < self.cps_number:
            cps_array = self.generate_change_points_with_random(cps_array)
        return cps_array
