import numpy as np
from matplotlib import pyplot as plt
#Raw data
acc = np.load('wesad/S2/raw/c_acc.npy')
ecg = np.load('wesad/S2/raw/c_ecg.npy')
eda = np.load('wesad/S2/raw/c_eda.npy')
emg = np.load('wesad/S2/raw/c_emg.npy')
resp = np.load('wesad/S2/raw/c_resp.npy')
temp = np.load('wesad/S2/raw/c_temp.npy')
label = np.load('wesad/S2/label_paper.npy')
x = acc[:, 0]
y = acc[:, 1]
z = acc[:, 2]
r_ecg = np.asarray(ecg).reshape(len(ecg))
r_eda = np.asarray(eda).reshape(len(eda))
r_emg = np.asarray(emg).reshape(len(emg))
r_resp = np.asarray(resp).reshape(len(resp))
r_temp = np.asarray(temp).reshape(len(temp))

# unique, counts = np.unique(label, return_counts=True)
# print(dict(zip(unique, counts)))

# label_0 = []
label_1 = []
label_2 = []
label_3 = []
label_4 = []
# label_6 = []
# label_7 = []
for i in range(0, len(label)):
    # if label[i] == 0:
    #     label_0.append(i)
    if label[i] == 1:
        label_1.append(i)
    if label[i] == 2:
        label_2.append(i)
    if label[i] == 3:
        label_3.append(i)
    if label[i] == 4:
        label_4.append(i)
    # if label[i] == 6:
    #     label_6.append(i)
    # if label[i] == 7:
    #     label_7.append(i)

# print('label_0:', len(label_0))
# print('label_1:', len(label_1))
# print('label_2:', len(label_2))
# print('label_3:', len(label_3))
# print('label_4:', len(label_4))
# print('label_6:', len(label_6))
# print('label_7:', len(label_7))
x_1 = []
y_1 = []
z_1 = []
ecg_1 = []
eda_1 = []
emg_1 = []
resp_1 = []
temp_1 = []
l_1 = []

x_2 = []
y_2 = []
z_2 = []
ecg_2 = []
eda_2 = []
emg_2 = []
resp_2 = []
temp_2 = []
l_2 = []

x_3 = []
y_3 = []
z_3 = []
ecg_3 = []
eda_3 = []
emg_3 = []
resp_3 = []
temp_3 = []
l_3 = []

x_4 = []
y_4 = []
z_4 = []
ecg_4 = []
eda_4 = []
emg_4 = []
resp_4 = []
temp_4 = []
l_4 = []

for k in range(0, len(label_1)):
    indexk = label_1[k]
    x_1.append(x[indexk])
    y_1.append(y[indexk])
    z_1.append(z[indexk])
    ecg_1.append(r_ecg[indexk])
    eda_1.append(r_eda[indexk])
    emg_1.append(r_emg[indexk])
    resp_1.append(r_resp[indexk])
    temp_1.append(r_temp[indexk])
    l_1.append(label[indexk])

rx_1 = np.asarray(x_1).reshape(len(label_1))
ry_1 = np.asarray(y_1).reshape(len(label_1))
rz_1 = np.asarray(z_1).reshape(len(label_1))
recg_1 = np.asarray(ecg_1).reshape(len(label_1))
reda_1 = np.asarray(eda_1).reshape(len(label_1))
remg_1 = np.asarray(emg_1).reshape(len(label_1))
rresp_1 = np.asarray(resp_1).reshape(len(label_1))
rtemp_1 = np.asarray(temp_1).reshape(len(label_1))
rl_1 = np.asarray(l_1).reshape(len(label_1))
all_label_1 = np.stack((rx_1, ry_1, rz_1, recg_1, eda_1, remg_1, rresp_1, rtemp_1), axis=-1)

for j in range(0, len(label_2)):
    indexj = label_2[j]
    x_2.append(x[indexj])
    y_2.append(y[indexj])
    z_2.append(z[indexj])
    ecg_2.append(r_ecg[indexj])
    eda_2.append(r_eda[indexj])
    emg_2.append(r_emg[indexj])
    resp_2.append(r_resp[indexj])
    temp_2.append(r_temp[indexj])
    l_2.append(label[indexj])

rx_2 = np.asarray(x_2).reshape(len(label_2), 1)
ry_2 = np.asarray(y_2).reshape(len(label_2), 1)
rz_2 = np.asarray(z_2).reshape(len(label_2), 1)
recg_2 = np.asarray(ecg_2).reshape(len(label_2), 1)
reda_2 = np.asarray(eda_2).reshape(len(label_2), 1)
remg_2 = np.asarray(emg_2).reshape(len(label_2), 1)
rresp_2 = np.asarray(resp_2).reshape(len(label_2), 1)
rtemp_2 = np.asarray(temp_2).reshape(len(label_2), 1)
rl_2 = np.asarray(l_2).reshape(len(label_2))
all_label_2 = np.stack((rx_2, ry_2, rz_2, recg_2, eda_2, remg_2, rresp_2, rtemp_2), axis=-1)

for l in range(0, len(label_3)):
    indexl = label_3[l]
    x_3.append(x[indexl])
    y_3.append(y[indexl])
    z_3.append(z[indexl])
    ecg_3.append(r_ecg[indexl])
    eda_3.append(r_eda[indexl])
    emg_3.append(r_emg[indexl])
    resp_3.append(r_resp[indexl])
    temp_3.append(r_temp[indexl])
    l_3.append(label[indexl])

rx_3 = np.asarray(x_3).reshape(len(label_3))
ry_3 = np.asarray(y_3).reshape(len(label_3))
rz_3 = np.asarray(z_3).reshape(len(label_3))
recg_3 = np.asarray(ecg_3).reshape(len(label_3))
reda_3 = np.asarray(eda_3).reshape(len(label_3))
remg_3 = np.asarray(emg_3).reshape(len(label_3))
rresp_3 = np.asarray(resp_3).reshape(len(label_3))
rtemp_3 = np.asarray(temp_3).reshape(len(label_3))
rl_3 = np.asarray(l_3).reshape(len(label_3))
all_label_3 = np.stack((rx_3, ry_3, rz_3, recg_3, eda_3, remg_3, rresp_3, rtemp_3), axis=-1)

for m in range(0, len(label_4)):
    indexm = label_4[m]
    x_4.append(x[indexm])
    y_4.append(y[indexm])
    z_4.append(z[indexm])
    ecg_4.append(r_ecg[indexm])
    eda_4.append(r_eda[indexm])
    emg_4.append(r_emg[indexm])
    resp_4.append(r_resp[indexm])
    temp_4.append(r_temp[indexm])
    l_4.append(label[indexm])

rx_4 = np.asarray(x_4).reshape(len(label_4), 1)
ry_4 = np.asarray(y_4).reshape(len(label_4), 1)
rz_4 = np.asarray(z_4).reshape(len(label_4), 1)
recg_4 = np.asarray(ecg_4).reshape(len(label_4), 1)
reda_4 = np.asarray(eda_4).reshape(len(label_4), 1)
remg_4 = np.asarray(emg_4).reshape(len(label_4), 1)
rresp_4 = np.asarray(resp_4).reshape(len(label_4), 1)
rtemp_4 = np.asarray(temp_4).reshape(len(label_4), 1)
rl_4 = np.asarray(l_4).reshape(len(label_4))
all_label_4 = np.stack((rx_4, ry_4, rz_4, recg_4, eda_4, remg_4, rresp_4, rtemp_4), axis=-1)

np.save('wesad/S2/raw/all_label_1.npy', all_label_1)
np.save('wesad/S2/raw/all_label_2.npy', all_label_2)
np.save('wesad/S2/raw/all_label_3.npy', all_label_3)
np.save('wesad/S2/raw/all_label_4.npy', all_label_4)
np.save('wesad/S2/raw/label_1.npy', rl_1)
np.save('wesad/S2/raw/label_2.npy', rl_2)
np.save('wesad/S2/raw/label_3.npy', rl_3)
np.save('wesad/S2/raw/label_4.npy', rl_4)

print('finished')
