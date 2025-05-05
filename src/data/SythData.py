from typing import Optional, List

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


class RandomChangePointsGenerator:
    """Improved version that distributes change points more uniformly"""

    def __init__(self,
                 seed: Optional[int] = None,
                 cps_number: int = 0,
                 length_data: int = 24 * 7 * 15 + 15,
                 minimum_sequence_cp: int = 10,
                 start_mutation_coeff: float = 0.5,
                 treshold_mutation_coeff: float = 0.1,
                 power_coeff: float = 1.2,
                 attemps_to_failure: int = 15):

        if seed is None:
            seed = np.random.randint(low=0, high=65535)
        np.random.seed(seed)

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
            raise NotImplementedError("This class expects to generate array with more than 10 values in a sequence!")

        if (length_data / (cps_number + 1e-9)) < minimum_sequence_cp:
            raise NotImplementedError("Expected length of data is too small for expected cps_number! One of your "
                                      "sequences could be less than minimum_sequence_cp points which may lead to errors.")

    def _new_mutation_coeff(self, sequence_len: int) -> float:
        """Generate random float value between 0 and 1."""
        if sequence_len ** self.power_coeff > self.length_data:
            out = np.random.random() - self.treshold_mutation_coeff
        else:
            out = np.random.random()
        return out

    def _is_mutation_apply(self, sequence_len: int, past_mutation_coeff: float) -> bool:
        """Determine if we should apply mutation factor"""
        if sequence_len > self.minimum_sequence_cp:
            # Adjust probability based on position in sequence to encourage uniform distribution
            position_factor = sequence_len / self.length_data
            adjusted_prob = past_mutation_coeff * (1 + position_factor)
            return np.random.random() > adjusted_prob
        return False

    def queue(self, queue_window: int, time_series: np.array) -> np.array:
        """Ensure minimum distance between change points"""
        cp_indices = np.where(time_series == 1)[0]
        for i in range(1, len(cp_indices)):
            if cp_indices[i] - cp_indices[i - 1] < queue_window:
                # Remove the latter change point if too close
                time_series[cp_indices[i]] = 0
        return time_series

    def generate_uniform_change_points(self) -> np.array:
        """Generate change points with more uniform distribution"""
        cps_array = np.zeros(self.length_data)

        if self.cps_number == 0:
            return cps_array

        # Calculate ideal spacing between change points
        ideal_spacing = self.length_data / (self.cps_number + 1)

        # Generate initial positions with some randomness
        positions = [int(i + 1 + np.random.uniform(-0.3, 0.3)) * ideal_spacing for i in range(self.cps_number)]
        positions = np.clip(positions, self.minimum_sequence_cp, self.length_data - self.minimum_sequence_cp)

        # Convert to integers and ensure uniqueness
        positions = np.unique(np.round(positions).astype(int))

        # If we didn't get enough points, add more randomly
        while len(positions) < self.cps_number:
            new_pos = np.random.randint(self.minimum_sequence_cp,
                                        self.length_data - self.minimum_sequence_cp)
            if np.min(np.abs(positions - new_pos)) > self.minimum_sequence_cp:
                positions = np.append(positions, new_pos)

        # Ensure minimum distance
        positions.sort()
        for i in range(1, len(positions)):
            if positions[i] - positions[i - 1] < self.minimum_sequence_cp:
                positions[i] = positions[i - 1] + self.minimum_sequence_cp

        cps_array[positions] = 1
        return cps_array

    def generate_change_points_with_mutation(self) -> np.array:
        """Main function to generate array of change points with uniform distribution"""
        # First try uniform generation
        cps_array = self.generate_uniform_change_points()

        # If we didn't get enough points, fall back to random
        if sum(cps_array) < self.cps_number:
            attempts = 0
            while sum(cps_array) < self.cps_number and attempts < self.attemps_to_failure:
                pos = np.random.randint(self.minimum_sequence_cp,
                                        self.length_data - self.minimum_sequence_cp)
                # Check distance from existing points
                existing = np.where(cps_array == 1)[0]
                if len(existing) == 0 or np.min(np.abs(existing - pos)) > self.minimum_sequence_cp:
                    cps_array[pos] = 1
                attempts += 1

        # Final check and queue filtering
        cps_array = self.queue(self.minimum_sequence_cp, cps_array)

        if sum(cps_array) < self.cps_number:
            print(f"Warning: Only generated {sum(cps_array)} change points out of requested {self.cps_number}")

        return cps_array

    def generate_change_points_with_random(self, cps_array: Optional[np.array] = None) -> np.array:
        """Alternative method using random generation with uniform distribution"""
        if cps_array is None:
            cps_array = np.zeros(self.length_data)

        positions = []
        attempts = 0

        while len(positions) < self.cps_number and attempts < self.attemps_to_failure:
            # Try to place points in different segments
            segment_size = self.length_data / (self.cps_number + 1)
            new_pos = int(np.random.uniform(len(positions) * segment_size,
                                            (len(positions) + 1) * segment_size))

            new_pos = np.clip(new_pos, self.minimum_sequence_cp, self.length_data - self.minimum_sequence_cp)

            # Check minimum distance
            if len(positions) == 0 or np.min(np.abs(np.array(positions) - new_pos)) > self.minimum_sequence_cp:
                positions.append(new_pos)
            attempts += 1

        cps_array[np.array(positions)] = 1
        return self.queue(self.minimum_sequence_cp, cps_array)


class SyntheticSinusoid(RandomChangePointsGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_data(self, cps_list: List[int],
                      min_max_amplitude: float,
                      min_max_frequency: float,
                      min_max_phase: float,
                      noise_std: float) -> List[float]:
        """
        Generate sinusoidal data with change points

        Parameters:
        - cps_list: List of change point indices (0 = no change, 1 = change)
        - min_max_amplitude: Tuple of (min, max) amplitude range
        - min_max_frequency: Tuple of (min, max) frequency range
        - min_max_phase: Tuple of (min, max) phase shift range
        - noise_std: Standard deviation of Gaussian noise to add

        Returns:
        - List of sinusoidal values with change points
        """
        time_points = len(cps_list)
        t = np.arange(time_points)

        # Initialize parameters
        amplitude = np.random.uniform(min_max_amplitude[0], min_max_amplitude[1])
        frequency = np.random.uniform(min_max_frequency[0], min_max_frequency[1])
        phase = np.random.uniform(min_max_phase[0], min_max_phase[1])

        sinusoid = []

        for i in range(time_points):
            # Check for change point
            if cps_list[i] == 1:
                amplitude = np.random.uniform(min_max_amplitude[0], min_max_amplitude[1])
                frequency = np.random.uniform(min_max_frequency[0], min_max_frequency[1])
                phase = np.random.uniform(min_max_phase[0], min_max_phase[1])

            # Generate sinusoidal value
            value = amplitude * np.sin(2 * np.pi * frequency * t[i] + phase)

            # Add noise
            value += np.random.normal(0, noise_std)

            sinusoid.append(value)

        return np.array(sinusoid)

    def get(self,
            min_max_amplitude: float = (0.5, 5.0),
            min_max_frequency: float = (0.01, 0.9),
            min_max_phase: float = (-2 * np.pi, 2 * np.pi),
            noise_std: float = 0.1) -> np.array:
        """
        Generate complete sinusoidal time series with change points

        Parameters:
        - min_max_amplitude: Amplitude range (default 0.5-2.0)
        - min_max_frequency: Frequency range (default 0.01-0.1)
        - min_max_phase: Phase shift range (default 0-2π)
        - noise_std: Noise standard deviation (default 0.1)

        Returns:
        - np.array: [change_points_list, time_series_data]
        """
        random_cps_list = self.generate_change_points_with_mutation()
        sinusoid_series = self.generate_data(
            cps_list=random_cps_list,
            min_max_amplitude=min_max_amplitude,
            min_max_frequency=min_max_frequency,
            min_max_phase=min_max_phase,
            noise_std=noise_std
        )
        return np.array([random_cps_list, sinusoid_series])

class SimpleRandomTimeSeries:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get(self, min_max_val: int, min_max_std: int) -> np.array:
        raise NotImplementedError("In a progress! Here you will create random timeseries from box")