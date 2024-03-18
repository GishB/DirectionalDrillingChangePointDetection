import pandas as pd
from typing import Optional, Dict, Any
import sys
import streamlit as st

sys.path.append("../..")

from data.SythData import LinearSteps, SinusoidWaves


@st.cache_data
def init_model_params(model_name: str) -> Dict[str: Any]:
    """ Select params for custom model from module models.

    Returns:
        dict of params for custom model.
    """
    params = {
        "is_z_normalization": st.checkbox("is_z_normalization"),
        "is_cps_filter_on": st.checkbox("is_cps_filter_on"),
        "is_squared_residual": st.checkbox("is_squared_residual"),
        "is_cumsum_applied": st.checkbox("is_cumsum_applied"),
        "queue_window": st.slider('queue_window', 10, 100, 10),
        "threshold_std_coeff": st.slider('threshold_str_coeff', 2.1, 5.1, 3.1)
    }
    if model_name == "Singular Sequence Decomposition":
        params["n_components"] = st.slider('n_components PCA', 1, 3, 2),
    elif model_name == "Kalman Filter":
        pass
    return params


@st.cache_data
def init_data_loader_params() -> Dict[str: Any]:
    """ Init params to loan syth data.

    Notes:
        length_data should be equally split based on cps_number.

    Examples:
        1. length_data = 1000, cps_number=[2, 4, 10, 20, 50]
        2. length_data = 100, cps_number=[2, 4]

    Warnings:
        cps_number should split data equally!

    Returns:
        dict of params to generate syth data.
    """
    params = {
        "length_data": st.slider('length_data', 100, 1000, 500),
        "cps_number": st.slider('cps_number', 2, 100, 500),
        "white_noise_level": st.radio(
            "What's noise level you want to select",
            ["min", "default", "max"],
            index=0,
        )
    }
    return params


@st.cache_data
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
