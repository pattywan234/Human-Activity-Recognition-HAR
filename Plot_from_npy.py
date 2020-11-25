import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#data = np.load('wesad/S2/data_no_acc.npy')  ##
#data = pd.read_csv('wesad/S2/test/HR.csv').to_numpy()

# one_sec = []

#for i in range(0, 5):
# all = np.load('wesad/S2/raw/c_acc.npy')
# x = np.load('wesad/S2/Normalize/x_norm.npy')
# y = np.load('wesad/S2/Normalize/y_norm.npy')
# z = np.load('wesad/S2/Normalize/z_norm.npy')
# ecg = np.load('wesad/S2/Normalize/ecg_norm.npy')
# eda = np.load('wesad/S2/Normalize/eda_norm.npy')
# emg = np.load('wesad/S2/Normalize/emg_norm.npy')
# resp = np.load('wesad/S2/Normalize/resp_norm.npy')
# temp = np.load('wesad/S2/Normalize/temp_norm.npy')

# plt.figure()
# plt.subplot(511)
#plt.plot(ecg[:1500])
#plt.xlabel('ECG')
#plt.subplot(512)
#plt.plot(eda)
#plt.xlabel('EDA')
#plt.subplot(513)
#plt.plot(emg)
#plt.xlabel('EMG')
#plt.subplot(514)
#plt.plot(resp)
#plt.xlabel('RESP')
#plt.subplot(515)
#plt.plot(temp)
#plt.xlabel('TEMP')
#plt.show()

#x = acc[1000:2000, 0]
#y = acc[1000:2000, 1]
#z = acc[1000:2000, 2]

# axes = plt.axes()
# plt.figure()
# plt.subplot(311)
# plt.plot(x[1000:1500])
# # plt.ylim([0, 1])
# plt.xlabel('X-AXIS ACC')
# plt.subplot(312)
# plt.plot(y[1000:1500])
# # plt.ylim([0, 1])
# plt.xlabel('Y-AXIS ACC')
# plt.subplot(313)
# plt.plot(z[1000:1500])
# # plt.ylim([0, 1])
# plt.xlabel('Z-AXIS ACC')
# plt.show()
#
# plt.figure()
# plt.subplot(511)
# plt.plot(ecg[1000:1500])
# plt.ylim([0, 1])
# plt.xlabel('ECG')
# plt.subplot(512)
# plt.plot(eda[1000:1500])
# plt.ylim([0, 1])
# plt.xlabel('EDA')
# plt.subplot(513)
# plt.plot(emg[1000:1500])
# plt.ylim([0, 1])
# plt.xlabel('EMG')
# plt.subplot(514)
# plt.plot(resp[1000:1500])
# plt.ylim([0, 1])
# plt.xlabel('RESP')
# plt.subplot(515)
# plt.plot(temp[1000:1500])
# plt.ylim([0, 1])
# plt.xlabel('TEMP')
# plt.show()


# raw data
# train_acc = np.load('multi_result/epoch_20/val/train_acc.npy')
# train_loss = np.load('multi_result/epoch_20/val/train_loss.npy')
# val_acc = np.load('multi_result/epoch_20/val/val_acc.npy')
# val_loss = np.load('multi_result/epoch_20/val/val_loss.npy')

# Normalized data
train_acc = np.load('multi_result/epoch_20/val/train_acc_norm.npy')
train_loss = np.load('multi_result/epoch_20/val/train_loss_norm.npy')
val_acc = np.load('multi_result/epoch_20/val/val_acc_norm.npy')
val_loss = np.load('multi_result/epoch_20/val/val_loss_norm.npy')

epoch = [2,4,6,8,10,12,14,16,18,20]

lr1_ta = train_acc[1:21]
lr2_ta = train_acc[22:42]
lr3_ta = train_acc[43:]
lr1_tl = train_loss[1:21]
lr2_tl = train_loss[22:42]
lr3_tl = train_loss[43:]

lr1_va = val_acc[0:20]
lr2_va = val_acc[20:40]
lr3_va = val_acc[40:]
lr1_vl = val_loss[0:20]
lr2_vl = val_loss[20:40]
lr3_vl = val_loss[40:]

plt.figure()
plt.style.use('seaborn-darkgrid')
plt.plot(lr1_ta, label="lr=0.0020")
plt.plot(lr2_ta, label="lr=0.0025")
plt.plot(lr3_ta, label="lr=0.0030")
plt.xticks(epoch, epoch)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training Accuracy of Normalized data')
plt.show()
#plt.savefig('multi_result/epoch_20/val/norm_ta.png')

plt.figure()
plt.style.use('seaborn-darkgrid')
plt.plot(lr1_va, label="lr=0.0020")
plt.plot(lr2_va, label="lr=0.0025")
plt.plot(lr3_va, label="lr=0.0030")
plt.xticks(epoch, epoch)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Validation Accuracy of Normalized data')
plt.show()
#plt.savefig('multi_result/epoch_20/val/norm_va.png')

plt.figure()
plt.style.use('seaborn-darkgrid')
plt.plot(lr1_tl, label="lr=0.0020")
plt.plot(lr2_tl, label="lr=0.0025")
plt.plot(lr3_tl, label="lr=0.0030")
plt.xticks(epoch, epoch)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss of Normalized data')
plt.show()
#plt.savefig('multi_result/epoch_20/val/norm_tl.png')

plt.figure()
plt.style.use('seaborn-darkgrid')
plt.plot(lr1_vl, label="lr=0.0020")
plt.plot(lr2_vl, label="lr=0.0025")
plt.plot(lr3_vl, label="lr=0.0030")
plt.xticks(epoch, epoch)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Validation Loss of Normalized data')
plt.show()
#plt.savefig('multi_result/epoch_20/val/norm_vl.png')
