import streamlit as st
import utils.StreamlitFunctions as useful

st.title('Change Point Detection app.')

st.sidebar.header('UI Model pipeline')
option_model = st.sidebar.selectbox(
    'Select CPD model',
    ("Singular Sequence Decomposition", "Kalman Filter"))
useful.init_model_params(model_name=option_model)

option_data = st.sidebar.selectbox(
    'Select dataset',
    ("Syth-Steps", "Syth-Sinusoid"))

if option_data != "None":
    data_loader_params = useful.init_data_loader_params()
    df = useful.data_loader(option=option_data, params=data_loader_params)
    useful.data_info()
    useful.data_plot()





