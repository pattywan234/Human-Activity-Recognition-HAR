import numpy as np
import random

# # data making
# data_1 = np.load('wesad/S2/Normalize/label_selected/norm_all_label_1.npy')
# data_2 = np.load('wesad/S2/Normalize/label_selected/norm_all_label_2.npy')
# data_3 = np.load('wesad/S2/Normalize/label_selected/norm_all_label_3.npy')
# data_4 = np.load('wesad/S2/Normalize/label_selected/norm_all_label_4.npy')
#
# # class 1
# x_1 = random.sample(list(data_1[:, 0]), 674100)
# y_1 = random.sample(list(data_1[:, 1]), 674100)
# z_1 = random.sample(list(data_1[:, 2]), 674100)
# ecg_1 = random.sample(list(data_1[:, 3]), 674100)
# eda_1 = random.sample(list(data_1[:, 4]), 674100)
# emg_1 = random.sample(list(data_1[:, 5]), 674100)
# resp_1 = random.sample(list(data_1[:, 6]), 674100)
# temp_1 = random.sample(list(data_1[:, 7]), 674100)
# class_1 = np.stack((x_1, y_1, z_1, ecg_1, eda_1, emg_1, resp_1, temp_1), axis=-1)
#
# # class 2
# x_2 = random.sample(list(data_2[:, 0]), 243600)
# y_2 = random.sample(list(data_2[:, 1]), 243600)
# z_2 = random.sample(list(data_2[:, 2]), 243600)
# ecg_2 = random.sample(list(data_2[:, 3]), 243600)
# eda_2 = random.sample(list(data_2[:, 4]), 243600)
# emg_2 = random.sample(list(data_2[:, 5]), 243600)
# resp_2 = random.sample(list(data_2[:, 6]), 243600)
# temp_2 = random.sample(list(data_2[:, 7]), 243600)
# rand_2 = np.stack((x_2, y_2, z_2, ecg_2, eda_2, emg_2, resp_2, temp_2), axis=-1)
# class_2 = np.concatenate((data_2, rand_2))
#
# # class 3
# i = 1
# rand_3 = []
# while i < 4:
#     x_3 = random.sample(list(data_3[:, 0]), 126700)
#     y_3 = random.sample(list(data_3[:, 1]), 126700)
#     z_3 = random.sample(list(data_3[:, 2]), 126700)
#     ecg_3 = random.sample(list(data_3[:, 3]), 126700)
#     eda_3 = random.sample(list(data_3[:, 4]), 126700)
#     emg_3 = random.sample(list(data_3[:, 5]), 126700)
#     resp_3 = random.sample(list(data_3[:, 6]), 126700)
#     temp_3 = random.sample(list(data_3[:, 7]), 126700)
#     rand_3.append((x_3, y_3, z_3, ecg_3, eda_3, emg_3, resp_3, temp_3))
#     i += 1
#
# rand_3_0 = np.asarray(rand_3[0]).reshape(126700, 8)
# rand_3_1 = np.asarray(rand_3[1]).reshape(126700, 8)
# rand_3_2 = np.asarray(rand_3[2]).reshape(126700, 8)
# rrand_3_1 = np.concatenate((rand_3_0, rand_3_1, rand_3_2))
#
# x_3 = random.sample(list(data_3[:, 0]), 40600)
# y_3 = random.sample(list(data_3[:, 1]), 40600)
# z_3 = random.sample(list(data_3[:, 2]), 40600)
# ecg_3 = random.sample(list(data_3[:, 3]), 40600)
# eda_3 = random.sample(list(data_3[:, 4]), 40600)
# emg_3 = random.sample(list(data_3[:, 5]), 40600)
# resp_3 = random.sample(list(data_3[:, 6]), 40600)
# temp_3 = random.sample(list(data_3[:, 7]), 40600)
# rrand_3_2 = np.stack((x_3, y_3, z_3, ecg_3, eda_3, emg_3, resp_3, temp_3), axis=-1)
# class_3 = np.concatenate((data_3, rrand_3_1, rrand_3_2))
#
# # class 4
# x_4 = random.sample(list(data_4[:, 0]), 136501)
# y_4 = random.sample(list(data_4[:, 1]), 136501)
# z_4 = random.sample(list(data_4[:, 2]), 136501)
# ecg_4 = random.sample(list(data_4[:, 3]), 136501)
# eda_4 = random.sample(list(data_4[:, 4]), 136501)
# emg_4 = random.sample(list(data_4[:, 5]), 136501)
# resp_4 = random.sample(list(data_4[:, 6]), 136501)
# temp_4 = random.sample(list(data_4[:, 7]), 136501)
# rand_4 = np.stack((x_4, y_4, z_4, ecg_4, eda_4, emg_4, resp_4, temp_4), axis=-1)
# class_4 = np.concatenate((data_4, rand_4))
#
# np.save('wesad/S2/Normalize/label_selected/train_keras/data_augment/data_aug_1.npy', class_1)
# np.save('wesad/S2/Normalize/label_selected/train_keras/data_augment/data_aug_2.npy', class_2)
# np.save('wesad/S2/Normalize/label_selected/train_keras/data_augment/data_aug_3.npy', class_3)
# np.save('wesad/S2/Normalize/label_selected/train_keras/data_augment/data_aug_4.npy', class_4)
# print('DONE!!')

# label making
label_1 = np.load('wesad/S2/Normalize/label_selected/label_1.npy')[0:674100]
label_2 = np.load('wesad/S2/Normalize/label_selected/label_2.npy')
label_3 = np.load('wesad/S2/Normalize/label_selected/label_3.npy')
label_4 = np.load('wesad/S2/Normalize/label_selected/label_4.npy')

f_2 = np.concatenate((label_2, label_2[0:243600]))

f_3 = np.concatenate((label_3, label_3, label_3[0:167300]))

f_4 = np.concatenate((label_4, label_4[0:136501]))

np.save('wesad/S2/Normalize/label_selected/train_keras/data_augment/label_1.npy', label_1)
np.save('wesad/S2/Normalize/label_selected/train_keras/data_augment/label_2.npy', f_2)
np.save('wesad/S2/Normalize/label_selected/train_keras/data_augment/label_3.npy', f_3)
np.save('wesad/S2/Normalize/label_selected/train_keras/data_augment/label_4.npy', f_4)

print('DONE!!')
