import sys
sys.path.append(r'/home/gishb/PycharmProjects/Fedot_Industrial_old/Fedot.Industrial') #alternative_cpd_methods branch - januray 23
#sys.path.append(r'/home/gishb/PycharmProjects/Fedot.Industrial') main branch


# from fedot_ind.core.operation.transformation.WindowSelection import WindowSizeSelection
from core.operation.transformation.WindowSelection import WindowSizeSelection
# from fedot_ind.core.models.detection.subspaces.sst import SingularSpectrumTransformation
from core.models.detection.subspaces.SSTdetector import SingularSpectrumTransformation

from tsad.evaluating.evaluating import evaluating #F1 score evaluation
from detecta import detect_cusum #CUMSUM MODULE

from itertools import chain # list of list to one bit list
from collections import namedtuple #Kalman Filter

import math
import pandas as pd
import numpy as np
from numpy import random as rnd
import matplotlib.pyplot as plt

from itertools import chain # list of list to one bit list
from collections import namedtuple #Kalman Filter

from scipy.signal import butter, filtfilt
from scipy.signal import savgol_filter
from optcutfreq import optcutfreq

import requests
from io import StringIO
