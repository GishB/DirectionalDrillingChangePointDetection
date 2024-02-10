import sys
import os
sys.path.append(os.path.abspath("../.."))

from utils import libs_cpd, create_report as crtest
import data.data as dtest
import models.kalman as kalman
import utils.functions as optf


if len(sys.argv) > 1:
    i =  dtest.list_links[int(sys.argv[1])];
    name = str(sys.argv[2]);
else:
    i = dtest.list_links[0]; 
    name = "test";

df = dtest.df_expirement(i)

window_length_savgol = libs_cpd.WindowSizeSelection(time_series = list(df.GR),
                                                    wss_algorithm = 'summary_statistics_subsequence').get_window_size()[0]
norm_filter_gr = optf.normalization_linear(optf.filter_Savgol(df.Resist_short, window_length_savgol))
window_length = libs_cpd.WindowSizeSelection(time_series = norm_filter_gr,
                                             wss_algorithm = 'dominant_fourier_frequency', window_max=1000, window_min=50).get_window_size()[0]

cps_list_kalman = kalman.online_detection(list(df['GR']), window=window_length, queue_window=10, treshold_coef=4.3)
df['cps_kalman'] = cps_list_kalman

tsad_average_results = crtest.tsad_average(df.cps_kalman, df.CPs)
tsad_nab_results = crtest.tsad_nab(df.cps_kalman, df.CPs)
tsad_nab_results.update(tsad_average_results)

report = crtest.create_report(tsad_nab_results)


#downloand report and image with predicted labels
libs_cpd.plt.figure(figsize=(12, 3))
libs_cpd.plt.plot(list(df.CPs), label='original CPs')
libs_cpd.plt.plot(cps_list_kalman, label='predicted CPs')
libs_cpd.plt.legend(loc="center right", bbox_to_anchor=(1.18, 0.5))
libs_cpd.plt.savefig('predicted_' + name + '.jpg')

with open('./predicted_'+name+'.txt','w') as out:
    for key,val in report.items():
        out.write('{}:{}\n'.format(key,val))
