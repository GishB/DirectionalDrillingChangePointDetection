import pandas as pd
from typing import Optional
import sys

sys.path.append("../..")

from data.SythData import LinearSteps, SinusoidWaves


def init_model_params(option, model):
    return None


def data_loader(option: str, params: dict) -> Optional[pd.DataFrame]:
    """ Load data based on option query.

    Args:
        option: option name.
        params: dict of params to generate syth data if any.

    Returns:
        dataframe or None in case of none option.
    """
    if option == "Syth-Steps":
        data = LinearSteps(length_data=params.get("length_data"),
                           cps_number=params.get("cps_number"),
                           white_noise_level=params.get("white_noise_level")).get()
    elif option == 'Syth-Sinusoid':
        data = SinusoidWaves(length_data=params.get("length_data"),
                             cps_number=params.get("cps_number"),
                             white_noise_level=params.get("white_noise_level")).get()
    else:
        data = None
    return data
