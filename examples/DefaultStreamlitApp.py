import streamlit as st
import sys

sys.path.append("..")

import utils.StreamlitFunctions as useful

st.title('Change Point Detection examples.')

st.sidebar.header('UI Model pipeline')
option_model = st.sidebar.selectbox(
    'Select CPD model',
    ("Singular Sequence Decomposition", "Kalman Filter"))
model_params = useful.init_model_params(model_name=option_model)

option_data = st.sidebar.selectbox(
    'Select dataset',
    ("None", "Syth-Steps", "Syth-Sinusoid"))

df = None
if option_data != "None":
    data_loader_params = useful.init_data_loader_params()
    df = useful.data_loader(option=option_data, params=data_loader_params)
    useful.data_info(df)
    useful.data_plot(df)

option_start_model = st.sidebar.selectbox(
    'Run selected model',
    ("None", "RUN!"))

df_updated = None
if option_start_model != "None" and df is not None:
    df_updated = useful.init_model_pipeline(name_model=option_model, params=model_params, df=df)

if df_updated is not None:
    summary_df = useful.model_summary(df=df_updated)
    useful.data_info(summary_df)
    useful.plot_results(df_updated)
