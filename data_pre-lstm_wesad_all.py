import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


x = np.load('wesad/S2/Normalize/x_norm.npy')
y = np.load('wesad/S2/Normalize/y_norm.npy')
z = np.load('wesad/S2/Normalize/z_norm.npy')
ecg = np.load('wesad/S2/Normalize/ecg_norm.npy')
eda = np.load('wesad/S2/Normalize/eda_norm.npy')
emg = np.load('wesad/S2/Normalize/emg_norm.npy')
resp = np.load('wesad/S2/Normalize/resp_norm.npy')
temp = np.load('wesad/S2/Normalize/temp_norm.npy')
label = np.load('wesad/S2/label_paper.npy')
#label is a number (0,1,2,3,4,6,7).
N_TIME_STEPS = 200
N_FEATURES = 8
step = 20
segments = []
labelss = []
for i in range(0, len(x) - N_TIME_STEPS, step):
    xs = x[i: i + N_TIME_STEPS]
    ys = y[i: i + N_TIME_STEPS]
    zs = z[i: i + N_TIME_STEPS]
    ecgs = ecg[i: i + N_TIME_STEPS]
    edas = x[i: i + N_TIME_STEPS]
    emgs = y[i: i + N_TIME_STEPS]
    resps = z[i: i + N_TIME_STEPS]
    temps = ecg[i: i + N_TIME_STEPS]
    labels = label[i: i + N_TIME_STEPS][0]
    segments.append([xs, ys, zs, ecgs, edas, emgs, resps, temps])
    labelss.append(labels)


reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
labeld = np.asarray(pd.get_dummies(labelss), dtype=np.float32)

RANDOM_SEED = 42
#random_state is the seed used by the random number generator, if number is same random result will be same
X_train, X_test, y_train, y_test = train_test_split(reshaped_segments, labeld, test_size=0.3, random_state=RANDOM_SEED)


np.save('wesad/S2/Normalize/X_train.npy', X_train)
np.save('wesad/S2/Normalize/X_test.npy', X_test)
np.save('wesad/S2/Normalize/y_train.npy', y_train)
np.save('wesad/S2/Normalize/y_test.npy', y_test)

print("finished")

