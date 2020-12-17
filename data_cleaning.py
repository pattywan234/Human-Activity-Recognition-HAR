import numpy as np

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

all_label_1 = x_1 + y_1 + z_1 + ecg_1 + eda_1 + emg_1 + resp_1 + temp_1
r_all_label_1 = np.asarray(all_label_1).reshape(len(label_1), 8)
r_l_1 = np.asarray(l_1).reshape(len(label_1))

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

all_label_2 = x_2 + y_2 + z_2 + ecg_2 + eda_2 + emg_2 + resp_2 + temp_2
r_all_label_2 = np.asarray(all_label_2).reshape(len(label_2), 8)
r_l_2 = np.asarray(l_2).reshape(len(label_2))

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

all_label_3 = x_3 + y_3 + z_3 + ecg_3 + eda_3 + emg_3 + resp_3 + temp_3
r_all_label_3 = np.asarray(all_label_3).reshape(len(label_3), 8)
r_l_3 = np.asarray(l_3).reshape(len(label_3))

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

all_label_4 = x_4 + y_4 + z_4 + ecg_4 + eda_4 + emg_4 + resp_4 + temp_4
r_all_label_4 = np.asarray(all_label_4).reshape(len(label_4), 8)
r_l_4 = np.asarray(l_4).reshape(len(label_4))

# np.save('wesad/S2/raw/all_label_1.npy', r_all_label_1)
# np.save('wesad/S2/raw/all_label_2.npy', r_all_label_2)
# np.save('wesad/S2/raw/all_label_3.npy', r_all_label_3)
# np.save('wesad/S2/raw/all_label_4.npy', r_all_label_4)
np.save('wesad/S2/raw/label_1.npy', r_l_1)
np.save('wesad/S2/raw/label_2.npy', r_l_2)
np.save('wesad/S2/raw/label_3.npy', r_l_3)
np.save('wesad/S2/raw/label_4.npy', r_l_4)

print('finished')
