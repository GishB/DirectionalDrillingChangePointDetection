import streamlit as st
import pandas as pd
import numpy as np
import datareader.data as dtest
import datareader.data_old as dtest_old
import libs_cpd
import optimization.functions as optf
import models.kalman as kalman
import create_report as crtest
from optimization.functions import cumsum, queue

st.title('MLT Project')
st.sidebar.header('Ввод данных')
model = st.sidebar.selectbox(
     'Какую модель выбрать?',
     ('SST','Kalman'))

option = st.sidebar.selectbox(
     'Какой датасет выбрать?',
     ('None', '229G', '231G', '237G', 'xxxAA564G', 'xxxAA684G'))
      
if option !='None':
    if option == 'xxxAA564G' or option == 'xxxAA684G':
        df = dtest_old.df_expirement(name_well=option)
    else:
        if option == '229G':
            option_yes = 0
        elif option == '231G':
            option_yes = 1
        else:
            option_yes = 2
        df = dtest.df_expirement(dtest.list_links[option_yes])
    st.write(df)
    
option_ts = st.sidebar.selectbox(
     'Какой столбец данных выбрать для отображения\обработки?',
     ('None', 'GR', 'Resist_short', 'DrillingSpeed','LITHOLOGY','CPs'))
if option_ts != 'None':
    st.line_chart(df[option_ts])

option_filter = st.sidebar.selectbox(
     'Какой фильтр использовать для сглаживания временного ряда?',
     ('None', 'Savgol'))

if option_filter == 'Savgol':
    window_length_savgol = libs_cpd.WindowSizeSelection(time_series = list(df[option_ts]),
                                       wss_algorithm = 'summary_statistics_subsequence').runner_wss()[0]
    norm_filter_ts = optf.normalization_linear(optf.filter_Savgol(df[option_ts], window_length_savgol))
    st.line_chart(norm_filter_ts)

option_hyperparameters = st.sidebar.selectbox(
        'Выбираем метод подбора гиперпараметров для модели '+model,
        ('None', 'dominant_fourier_frequency', 'summary_statistics_subsequence', 'highest_autocorrelation', 'multi_window_finder'))
if option_hyperparameters != 'None':
    if option_hyperparameters == 'highest_autocorrelation':
        norm_filter_ts = list(norm_filter_ts)
    window_length = 2*libs_cpd.WindowSizeSelection(time_series = norm_filter_ts,
                                       wss_algorithm = option_hyperparameters).runner_wss()[0]
    st.write('Алгоритм '+option_hyperparameters+' выбрал параметр окна равный: ', window_length)

    if model == 'SST' or model == 'SST+Kalman':
        ts_window_length = 2*int(libs_cpd.np.mean([libs_cpd.WindowSizeSelection(time_series = norm_filter_ts[i:window_length+i], wss_algorithm='summary_statistics_subsequence'\
                ).runner_wss()[0] for i in range(0, len(norm_filter_ts)-window_length, window_length)]))
        st.write('Алгоритм dominant_fourier_frequency выбрал параметр субокна равный: ', ts_window_length)
        
option_kalman = st.sidebar.selectbox(
            'Запуск модели '+model,
            ('Нет', 'Да'))
if model == 'Kalman':
    treshold_coef = st.sidebar.slider('Параметр аномальности CPs', 1, 10, 4)
    queue_window = st.sidebar.slider('Параметр окна детектирующего фильтра CPs', 1, 100, 11)
else:
    treshold_coef = st.sidebar.slider('Значимость аномалий', 1/10, 99/100, 95/100)
    queue_window = st.sidebar.slider('Параметр окна детектирующего фильтра CPs', 1, 100, 11)
    
if option_kalman != 'Нет' and model == 'Kalman':
    cps_list_kalman = kalman.online_detection(list(norm_filter_ts), window=window_length, queue_window=queue_window, treshold_coef=treshold_coef)
    df['predicted_cps'] = cps_list_kalman
    st.line_chart(df[['predicted_cps','CPs']])
elif option_kalman != 'Нет' and model == 'SST':
    model_sst = libs_cpd.SingularSpectrumTransformation(time_series=libs_cpd.np.array(norm_filter_ts), quantile_rate=0.95,
                                       trajectory_window_length=window_length,
                                       ts_window_length=ts_window_length, lag=int(ts_window_length/2), view=False)
    score = model_sst.score_offline(dynamic_mode=True)
    score = cumsum(score, quantile_=treshold_coef)
    predicted_cps = [1 if i in score[-2] else 0 for i in range(len(df))]
    queue_cps = queue(queue_window=queue_window, time_series=predicted_cps)
    df['predicted_cps'] = queue_cps
    st.line_chart(df[['predicted_cps','CPs']])

option_report = st.sidebar.selectbox(
                'Создать отчет?',
                ('Нет', 'Да'))
if option_report != 'Нет':
    tsad_average_results = crtest.tsad_average(df.predicted_cps, df.CPs)
    tsad_nab_results = crtest.tsad_nab(df.predicted_cps, df.CPs)
    tsad_nab_results.update(tsad_average_results)
    st.write(crtest.create_report(tsad_nab_results))

            
