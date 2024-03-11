import streamlit as st
import utils.StreamlitFunctions as useful

st.title('Change Point Detection app.')

st.sidebar.header('UI Model pipline')
option_model = st.sidebar.selectbox(
    'Select CPD model',
    ('SingularSequenceTransformation', 'Kalman'))

option_data = st.sidebar.selectbox(
    'Select dataset',
    ('None', 'Custom', 'SythData', 'CloudData'))

if option_data != "None":
    useful.data_loader(option=option_data)

option_model = st.sidebar.selectbox(
    "Select hyperparameters",
    ("Default", "Custom")
)

if option_model != "Default":
    useful.init_model_params(option=option_model, model=option_model)


