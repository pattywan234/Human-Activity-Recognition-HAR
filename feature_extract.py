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

def average(data_1, data_2, data_3):
    # determine mean of x-, y-, z-axis of accelerometer data
    avg = []
    for c_avg in range(0, len(data_1)):
        ans = (data_1[c_avg] + data_2[c_avg] + data_3[c_avg])/3
        avg.append(ans)
    return avg

x_rms = rootMeanS(x, len(x))
y_rms = rootMeanS(y, len(y))
z_rms = rootMeanS(z, len(z))
# ecg_rms = rootMeanS(ecg, len(ecg))
# eda_rms = rootMeanS(eda, len(eda))
# emg_rms = rootMeanS(emg, len(emg))
# resp_rms = rootMeanS(resp, len(resp))
temp_rms = rootMeanS(temp, len(temp))

xyz_avg = average(x, y, z)

accecg_corr = pearsCorr(xyz_avg, ecg)
acceda_corr = pearsCorr(xyz_avg, eda)
accemg_corr = pearsCorr(xyz_avg, emg)
accresp_corr = pearsCorr(xyz_avg, resp)
acctemp_corr = pearsCorr(xyz_avg, temp)

# ecgemg_corr = pearsCorr(ecg, emg)
# ecgeda_corr = pearsCorr(ecg, eda)
# ecgresp_corr = pearsCorr(ecg, resp)
# emgeda_corr = pearsCorr(emg, eda)
# emgtemp_corr = pearsCorr(emg, temp)
# edatemp_corr = pearsCorr(eda, temp)
# edaresp_corr = pearsCorr(eda, resp)
# tempresp_corr = pearsCorr(temp, resp)

# x_fft = fastFT(x)
# y_fft = fastFT(y)
# z_fft = fastFT(z)
ecg_fft = fastFT(ecg)
eda_fft = fastFT(eda)
emg_fft = fastFT(emg)
resp_fft = fastFT(resp)
# temp_fft = fastFT(temp)


# x_dwt = discreteWT(x, 'db1')
# y_dwt = discreteWT(y, 'db1')
# z_dwt = discreteWT(z, 'db1')
ecg_dwt = discreteWT(ecg, 'db1')
eda_dwt = discreteWT(eda, 'db1')
emg_dwt = discreteWT(emg, 'db1')
resp_dwt = discreteWT(resp, 'db1')
# temp_dwt = discreteWT(temp, 'db1')
# delete last element of dwt (label_4)
# ecg_dwt = np.delete(ecg_dwt, -1)
# eda_dwt = np.delete(eda_dwt, -1)
# emg_dwt = np.delete(emg_dwt, -1)
# resp_dwt = np.delete(resp_dwt, -1)

# print('x_rms', x_rms)
# print('xy_corr', xy_corr)
# print('x_fft', x_fft)
# print('x_dwt', x_dwt)

final_data = np.stack((x_rms, y_rms, z_rms, temp_rms, accecg_corr, acceda_corr, accemg_corr, accresp_corr, acctemp_corr,
                       ecg_dwt, eda_dwt, emg_dwt, resp_dwt), axis=-1)
np.save('wesad/S2/Normalize/label_selected/label_4/all_extracted.npy', final_data)

print('finished')
