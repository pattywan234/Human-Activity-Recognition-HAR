"""
This file contains normalization step.
"""
import numpy as np
from sklearn.preprocessing import normalize as norm
from matplotlib import pyplot as plt

# range [a,b]
def norm_cal(input, min, max, a, b):
    output = ((b-a)*((input - min)/(max-min))) + a
    return output

#x = [2, 4, 5, 4, 1, 3, 6, 7, 9, 10]
# acc = np.load('wesad/S2/raw/c_acc.npy')
# ecg = np.load('wesad/S2/raw/c_ecg.npy')
# eda = np.load('wesad/S2/raw/c_eda.npy')
# emg = np.load('wesad/S2/raw/c_emg.npy')
# resp = np.load('wesad/S2/raw/c_resp.npy')
# temp = np.load('wesad/S2/raw/c_temp.npy')
all_data = np.load('wesad/S2/raw/all_label_4.npy')
# form raw data
# x = acc[:, 0]
# y = acc[:, 1]
# z = acc[:, 2]
# ecg = np.asarray(ecg).reshape(len(ecg))
# eda = np.asarray(eda).reshape(len(eda))
# emg = np.asarray(emg).reshape(len(emg))
# resp = np.asarray(resp).reshape(len(resp))
# temp = np.asarray(temp).reshape(len(temp))
# from label selected data
x = all_data[:, 0]
y = all_data[:, 1]
z = all_data[:, 2]
ecg = all_data[:, 3]
eda = all_data[:, 4]
emg = all_data[:, 5]
resp = all_data[:, 6]
temp = all_data[:, 7]

#norm_range = [a, b]
a = 0
b = 1
new_x = []
new_y = []
new_z = []
new_ecg = []
new_eda = []
new_emg = []
new_resp = []
new_temp = []
'''
x_new = []
x_min = []
x_max = []

for i in range(0, len(acc[0])):
    x_test = acc[:, i]
    min = x_test.min()
    max = x_test.max()
    x_min.append(min)
    x_max.append(max)
'''
x_min = x.min()
x_max = x.max()
y_min = y.min()
y_max = y.max()
z_min = z.min()
z_max = z.max()
ecg_min = ecg.min()
ecg_max = ecg.max()
eda_min = eda.min()
eda_max = eda.max()
emg_min = emg.min()
emg_max = emg.max()
resp_min = resp.min()
resp_max = resp.max()
temp_min = temp.min()
temp_max = temp.max()

"""
for cc in range(0, len(x_min)):
    xmin = x_min[cc]
    xmax = x_max[cc]
    for rc in range(0, len(acc)):
        va = acc[rc, cc]
        after = norm_cal(va, xmin, xmax, a, b)
        new.append(after)
"""
for rc in range(0, len(x)):
    # va_x = x[rc]
    after_x = norm_cal(x[rc], x_min, x_max, a, b)
    new_x.append(after_x)

    # va_y = y[rc]
    after_y = norm_cal(y[rc], y_min, y_max, a, b)
    new_y.append(after_y)

    # va_z = z[rc]
    after_z = norm_cal(z[rc], z_min, z_max, a, b)
    new_z.append(after_z)

    # va_ecg = ecg[rc]
    after_ecg = norm_cal(ecg[rc], ecg_min, ecg_max, a, b)
    new_ecg.append(after_ecg)

    # va_eda = eda[rc]
    after_eda = norm_cal(eda[rc], eda_min, eda_max, a, b)
    new_eda.append(after_eda)

    # va_emg = emg[rc]
    after_emg = norm_cal(emg[rc], emg_min, emg_max, a, b)
    new_emg.append(after_emg)

    # va_resp = resp[rc]
    after_resp = norm_cal(resp[rc], resp_min, resp_max, a, b)
    new_resp.append(after_resp)

    # va_temp = temp[rc]
    after_temp = norm_cal(temp[rc], temp_min, temp_max, a, b)
    new_temp.append(after_temp)

norm_all_data = np.stack((new_x, new_y, new_z, new_ecg, new_eda, new_emg, new_resp, new_temp), axis=-1)
# r_norm_all = np.asarray(norm_all_data).reshape(len(x), 8)

# r_new_y = np.asarray(new_y).reshape(len(y))
# r_new_z = np.asarray(new_z).reshape(len(z))
# r_new_ecg = np.asarray(new_ecg).reshape(len(new_ecg))
# r_new_eda = np.asarray(new_eda).reshape(len(new_eda))
# r_new_emg = np.asarray(new_emg).reshape(len(new_emg))
# r_new_resp = np.asarray(new_resp).reshape(len(new_resp))
# r_new_temp = np.asarray(new_temp).reshape(len(new_temp))

# for rc in range(0, len(ecg)):
#     va_ecg = ecg[rc]
#     after_ecg = norm_cal(va_ecg, ecg_min, ecg_max, a, b)
#     new_ecg.append(after_ecg)
#
# for rc in range(0, len(eda)):
#     va_eda = eda[rc]
#     after_eda = norm_cal(va_eda, eda_min, eda_max, a, b)
#     new_eda.append(after_eda)
#
# for rc in range(0, len(emg)):
#     va_emg = emg[rc]
#     after_emg = norm_cal(va_emg, emg_min, emg_max, a, b)
#     new_emg.append(after_emg)
#
# for rc in range(0, len(resp)):
#     va_resp = resp[rc]
#     after_resp = norm_cal(va_resp, resp_min, resp_max, a, b)
#     new_resp.append(after_resp)
#
# for rc in range(0, len(temp)):
#     va_temp = temp[rc]
#     after_temp = norm_cal(va_temp, temp_min, temp_max, a, b)
#     new_temp.append(after_temp)


# x_reshape = np.asarray(new).reshape(len(acc), 3)
# np.save('wesad/S2/Normalize/x_norm.npy', new_x)
# np.save('wesad/S2/Normalize/y_norm.npy', new_y)
# np.save('wesad/S2/Normalize/z_norm.npy', new_z)
# np.save('wesad/S2/Normalize/ecg_norm.npy', new_ecg)
# np.save('wesad/S2/Normalize/eda_norm.npy', new_eda)
# np.save('wesad/S2/Normalize/emg_norm.npy', new_emg)
# np.save('wesad/S2/Normalize/resp_norm.npy', new_resp)
# np.save('wesad/S2/Normalize/temp_norm.npy', new_temp)

np.save('wesad/S2/Normalize/label_selected/norm_all_label_4.npy', norm_all_data)

print('finished')
