import numpy as np
import pywt
from scipy.fft import fft
from scipy.stats import pearsonr, tmean, tstd
import math


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
    result_rms = []
    for c_rms in range(0, N - 1, 2):
        d1 = data[c_rms]
        d2 = data[c_rms+1]
        root_meanS = math.sqrt(math.fsum([math.pow(d1, 2), math.pow(d2, 2)]) / 2)
        result_rms.append(root_meanS)
    return result_rms

# Correlation (using Scipy.stats)
def pearsCorr(data_1, data_2):
    corrcoef = []
    mean_1 = tmean(data_1)
    mean_2 = tmean(data_2)
    std_1 = tstd(data_1)
    std_2 = tstd(data_2)
    for c in range(0, len(data_1)-1):
        corr = math.fsum([(data_1[c]-mean_1)*(data_2[c]-mean_2), (data_1[c+1]-mean_1)*(data_2[c+1]-mean_2)])/(std_1*std_2)
        # corr, pValue = pearsonr(data_1[c], data_2[c])
        corrcoef.append(corr)
    corr_last = math.fsum([(data_1[-1]-mean_1)*(data_2[-1]-mean_2), (data_1[-2]-mean_1)*(data_2[-2]-mean_2)])/(std_1*std_2)
    corrcoef.append(corr_last)
    print(corrcoef[-1])
    result_pc = []
    for c_pc in range(0, len(corrcoef)-1, 2):
        d1 = corrcoef[c_pc]
        d2 = corrcoef[c_pc+1]
        mean_pc = (d1+d2)/2
        result_pc.append(mean_pc)
    return result_pc

# Frequency domain features
# Fast Fourier Transforms
def fastFT(data):
    fast_ft = fft(data)
    result_fft = []
    for c_fft in range(0, len(fast_ft)-1, 2):
        d1 = fast_ft[c_fft]
        d2 = fast_ft[c_fft + 1]
        mean = (d1 + d2) / 2
        result_fft.append(mean)
    return result_fft

# Wavelet Transforms
# Discrete wavelet transform
def discreteWT(data, wt):
    # data can be an array or list.
    # wt is a wavelet family to use. (should be a string)
    # cA, cD are approximation and detail coefficients.
    (cA, cD) = pywt.dwt(data, wt)
    return cA

x_rms = rootMeanS(x, len(x))
xy_corr = pearsCorr(x, y)
x_fft = fastFT(x)
x_dwt = discreteWT(x, 'db1')

# print('x_rms', x_rms)
# print('xy_corr', xy_corr)
# print('x_fft', x_fft)
# print('x_dwt', x_dwt)

print('finished')
