"""
This file split training and test data.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Normalized data
# x = np.load('wesad/S2/Normalize/x_norm.npy')
# y = np.load('wesad/S2/Normalize/y_norm.npy')
# z = np.load('wesad/S2/Normalize/z_norm.npy')
# ecg = np.load('wesad/S2/Normalize/ecg_norm.npy')
# eda = np.load('wesad/S2/Normalize/eda_norm.npy')
# emg = np.load('wesad/S2/Normalize/emg_norm.npy')
# resp = np.load('wesad/S2/Normalize/resp_norm.npy')
# temp = np.load('wesad/S2/Normalize/temp_norm.npy')
# label = np.load('wesad/S2/label_paper.npy')

# Normalized data + label selected data
# data_1 = np.load('wesad/S2/Normalize/label_selected/label_1/all_extracted.npy')
# data_2 = np.load('wesad/S2/Normalize/label_selected/label_2/all_extracted.npy')
# data_3 = np.load('wesad/S2/Normalize/label_selected/label_3/all_extracted.npy')
# data_4 = np.load('wesad/S2/Normalize/label_selected/label_4/all_extracted.npy')
# label_1 = np.load('wesad/S2/Normalize/label_selected/label_1.npy')[0:len(data_1)]
# label_2 = np.load('wesad/S2/Normalize/label_selected/label_2.npy')[0:len(data_2)]
# label_3 = np.load('wesad/S2/Normalize/label_selected/label_3.npy')[0:len(data_3)]
# label_4 = np.load('wesad/S2/Normalize/label_selected/label_4.npy')[0:len(data_4)]
# Normalized data + label selected data + data augmentation
data_1 = np.load('wesad/S2/Normalize/label_selected/train_keras/data_augment/f_extract_1.npy')
data_2 = np.load('wesad/S2/Normalize/label_selected/train_keras/data_augment/f_extract_2.npy')
data_3 = np.load('wesad/S2/Normalize/label_selected/train_keras/data_augment/f_extract_3.npy')
data_4 = np.load('wesad/S2/Normalize/label_selected/train_keras/data_augment/f_extract_4.npy')
label_1 = np.load('wesad/S2/Normalize/label_selected/train_keras/data_augment/label_1.npy')[0:len(data_1)]
label_2 = np.load('wesad/S2/Normalize/label_selected/train_keras/data_augment/label_2.npy')[0:len(data_2)]
label_3 = np.load('wesad/S2/Normalize/label_selected/train_keras/data_augment/label_3.npy')[0:len(data_3)]
label_4 = np.load('wesad/S2/Normalize/label_selected/train_keras/data_augment/label_4.npy')[0:len(data_4)]

all_data = np.concatenate((data_1, data_2, data_3, data_4))
all_label = np.concatenate((label_1, label_2, label_3, label_4))

N_TIME_STEPS = 200
N_FEATURES = len(all_data[1])
step = 20
segments = []
labelss = []

for i in range(0, len(all_data) - N_TIME_STEPS, step):
    # xs = x[i: i + N_TIME_STEPS]
    # ys = y[i: i + N_TIME_STEPS]
    # zs = z[i: i + N_TIME_STEPS]
    # ecgs = ecg[i: i + N_TIME_STEPS]
    # edas = eda[i: i + N_TIME_STEPS]
    # emgs = emg[i: i + N_TIME_STEPS]
    # resps = resp[i: i + N_TIME_STEPS]
    # temps = temp[i: i + N_TIME_STEPS]
    # labels = label[i: i + N_TIME_STEPS][0]
    # segments.append([xs, ys, zs, ecgs, edas, emgs, resps, temps])
    # labelss.append(labels)
    d = all_data[i: i + N_TIME_STEPS]
    l = all_label[i: i + N_TIME_STEPS][0]
    segments.append(d)
    labelss.append(l)

reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
# labeld = np.asarray(pd.get_dummies(labelss), dtype=np.float32)
labelss = np.asarray(labelss)

RANDOM_SEED = 42
# random_state is the seed used by the random number generator, if number is same random result will be same
# X_train, X_test, y_train, y_test = train_test_split(reshaped_segments, labeld, test_size=0.3, random_state=RANDOM_SEED)
X_train, X_test, y_train, y_test = train_test_split(reshaped_segments, labelss, test_size=0.3, random_state=RANDOM_SEED)

# np.save('wesad/S2/Normalize/label_selected/X_train.npy', X_train)
# np.save('wesad/S2/Normalize/label_selected/X_test.npy', X_test)
# np.save('wesad/S2/Normalize/label_selected/y_train.npy', y_train)
# np.save('wesad/S2/Normalize/label_selected/y_test.npy', y_test)
# np.save('wesad/S2/Normalize/label_selected/train_keras/X_train.npy', X_train)
# np.save('wesad/S2/Normalize/label_selected/train_keras/X_test.npy', X_test)
# np.save('wesad/S2/Normalize/label_selected/train_keras/y_train.npy', y_train)
# np.save('wesad/S2/Normalize/label_selected/train_keras/y_test.npy', y_test)
np.save('wesad/S2/Normalize/label_selected/train_keras/data_augment/X_train.npy', X_train)
np.save('wesad/S2/Normalize/label_selected/train_keras/data_augment/X_test.npy', X_test)
np.save('wesad/S2/Normalize/label_selected/train_keras/data_augment/y_train.npy', y_train)
np.save('wesad/S2/Normalize/label_selected/train_keras/data_augment/y_test.npy', y_test)

# Raw data
# acc = np.load('wesad/S2/raw/c_acc.npy')
# ecg = np.load('wesad/S2/raw/c_ecg.npy')
# eda = np.load('wesad/S2/raw/c_eda.npy')
# emg = np.load('wesad/S2/raw/c_emg.npy')
# resp = np.load('wesad/S2/raw/c_resp.npy')
# temp = np.load('wesad/S2/raw/c_temp.npy')
# label = np.load('wesad/S2/label_paper.npy')
# x = acc[:, 0]
# y = acc[:, 1]
# z = acc[:, 2]
# r_ecg = np.asarray(ecg).reshape(len(ecg))
# r_eda = np.asarray(eda).reshape(len(eda))
# r_emg = np.asarray(emg).reshape(len(emg))
# r_resp = np.asarray(resp).reshape(len(resp))
# r_temp = np.asarray(temp).reshape(len(temp))
#
# print('x max', np.std(x))
# print('y std', np.std(y))
# print('z std', np.std(z))
# print('ecg std', np.std(r_ecg))
# print('emg std', np.std(r_emg))
# print('eda std', np.std(r_eda))
# print('resp std', np.std(r_resp))
# print('temp std', np.std(r_temp))
#
# N_TIME_STEPS = 200
# N_FEATURES = 8
# step = 20
# segments = []
# labelss = []
#
# for i in range(0, len(x) - N_TIME_STEPS, step):
#     xs = x[i: i + N_TIME_STEPS]
#     ys = y[i: i + N_TIME_STEPS]
#     zs = z[i: i + N_TIME_STEPS]
#     ecgs = r_ecg[i: i + N_TIME_STEPS]
#     edas = r_eda[i: i + N_TIME_STEPS]
#     emgs = r_emg[i: i + N_TIME_STEPS]
#     resps = r_resp[i: i + N_TIME_STEPS]
#     temps = r_temp[i: i + N_TIME_STEPS]
#     labels = label[i: i + N_TIME_STEPS][0]
#     segments.append([xs, ys, zs, ecgs, edas, emgs, resps, temps])
#     labelss.append(labels)
#
#
# reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
# labeld = np.asarray(pd.get_dummies(labelss), dtype=np.float32)
#
# RANDOM_SEED = 42
# #random_state is the seed used by the random number generator, if number is same random result will be same
# X_train, X_test, y_train, y_test = train_test_split(reshaped_segments, labeld, test_size=0.3, random_state=RANDOM_SEED)
#
#
# np.save('wesad/S2/Normalize/X_train.npy', X_train)
# np.save('wesad/S2/Normalize/X_test.npy', X_test)
# np.save('wesad/S2/Normalize/y_train.npy', y_train)
# np.save('wesad/S2/Normalize/y_test.npy', y_test)

print('DONE!!')

