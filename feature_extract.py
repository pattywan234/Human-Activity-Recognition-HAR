import numpy as np
import pywt
from scipy.fft import fft
from scipy.stats import pearsonr


all_label_1 = np.load('wesad/S2/Normalize/label_selected/norm_all_label_1.npy')

x = all_label_1[:, 0]
y = all_label_1[:, 1]
z = all_label_1[:, 2]
ecg = all_label_1[:, 3]
eda = all_label_1[:, 4]
emg = all_label_1[:, 5]
resp = all_label_1[:, 6]
temp = all_label_1[:, 7]

# Time domain features
# Root-Mean-Square
def rootMeanS(data, N):
    # data can be an array or list.
    # N is number of elements in data
    root_meanS = np.sqrt(np.sum(np.square(data))/N)
    return root_meanS

# Correlation (using Scipy.stats)
def pearsCorr(data_1, data_2):
    corrcoef, pValue = pearsonr(data_1, data_2)
    return corrcoef

# Frequency domain features
# Fast Fourier Transforms
def FastFT(data):
    fast_ft = fft(data)
    return fast_ft

# Wavelet Transforms
# Discrete wavelet transform
def discreteWT(data, wt):
    # data can be an array or list.
    # wt is a wavelet family to use. (should be a string)
    # cA, cD are approximation and detail coefficients.
    (cA, cD) = pywt.dwt(data, wt)
    return cA
