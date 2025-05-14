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

    def get(self) -> pd.DataFrame:
        """ Get syth data.

        Returns:
            pandas dataframe with syth data and time index.
        """
        ...

    def plot(self):
        ...