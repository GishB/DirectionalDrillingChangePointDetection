import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import sys
import streamlit as st
import matplotlib.pyplot as plt

sys.path.append("../..")

from data.SythData import LinearSteps, SinusoidWaves
from models.SubspaceBased import SingularSequenceTransformer
from models.ProbabilityBased import KalmanFilter


@st.cache_data
def data_info(data: pd.DataFrame):
    """ Plot pandas dataframe for table value overview.

    Args:
        data: loaded pandas dataframe.

    Returns:
        None.
    """
    st.dataframe(data, use_container_width=True)


@st.cache_data
def data_plot(data: pd.DataFrame):
    """ Plot data for target raw values and original cps.

    Args:
        data: loaded pandas dataframe.

    Returns:
        None
    """
    target_column_value_name = "x"
    original_cps_name = "CPs"

    fig = plt.figure(figsize=(20, 5))
    plt.plot(data[target_column_value_name], label='Raw syth values')
    plt.legend()
    st.pyplot(fig=fig, use_container_width=True)

    fig_2 = plt.figure(figsize=(20, 5))
    plt.plot(data[original_cps_name], label='Original Change Points values')
    plt.legend()
    st.pyplot(fig=fig_2, use_container_width=True)


def init_model_params(model_name: str) -> Dict[str, Optional[Any]]:
    """ Select params for custom model from module models.

    Returns:
        dict of params for custom model.
    """
    params = {
        "is_cps_filter_on": st.sidebar.checkbox("is_cps_filter_on"),
        "is_cumsum_applied": st.sidebar.checkbox("is_cumsum_applied"),
        "queue_window": st.sidebar.slider('queue_window', 10, 100, 10),
        "threshold_std_coeff": st.sidebar.slider('threshold_std_coeff', 2.1, 5.1, 3.1)
    }
    if model_name == "Singular Sequence Decomposition":
        params["n_components"] = st.sidebar.slider('n_components PCA', 1, 3, 2),
    elif model_name == "Kalman Filter":
        pass
    return params


def init_data_loader_params() -> Dict[str, Optional[Any]]:
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
        "length_data": st.sidebar.slider('length_data', 100, 1000, 500),
        "cps_number": st.sidebar.slider('cps_number', 2, 100, 5),
        "white_noise_level": st.sidebar.radio(
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
    with st.spinner(text="Loading data in progress..."):
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


@st.cache_data
def init_model_pipeline(name_model: str, params: dict, df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """ Model pipeline over syth data.

    Args:
        name_model: selected model from UI.
        params: selected initial params.
        df: syth data.

    Returns:
        dataframe with change points predicted and residuals.
    """
    df_new = df.copy()
    cps_preds = None
    residuals = None
    if name_model == "Kalman Filter":
        model = KalmanFilter(is_cps_filter_on=params.get("is_cps_filter_in"),
                             is_cumsum_applied=params.get("is_cumsum_applied"),
                             queue_window=params.get("queue_window"),
                             is_z_normalization=True,
                             is_squared_residual=True,
                             threshold_std_coeff=params.get("threshold_std_coeff")).fit(list(df.x), None)
        cps_preds = model.predict(df.x.values)
        residuals = model.get_distances(df.x.values)
    elif name_model == "Singular Sequence Decomposition":
        model = SingularSequenceTransformer(
            is_cps_filter_on=params.get("is_cps_filter_in"),
            is_cumsum_applied=params.get("is_cumsum_applied"),
            is_z_normalization=True,
            is_squared_residual=True,
            n_components=params.get("n_components"),
            threshold_std_coeff=params.get("threshold_std_coeff")).fit(list(df.x), None)
        cps_preds = model.predict(df.x.values)
        residuals = model.get_distances(df.x.values)
    df_new['cps_preds'] = cps_preds
    df_new['residuals'] = residuals
    return df_new


def model_summary(preds: np.array, df: pd.DataFrame) -> pd.DataFrame:
    """

    Args:
        preds:
        df:

    Returns:

    """
    ...
