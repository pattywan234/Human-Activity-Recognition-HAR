import numpy as np
import tensorflow.compat.v1 as tf
from LSTM_model04 import create_LSTM_model, var_create_LSTM
from sklearn.model_selection import KFold
import pandas as pd

#tf.disable_v2_behavior()
tf.disable_resource_variables()

#import training and test data
data_1 = np.load('wesad/all_subjects/24May2021/extracted_all_1.npy')
data_2 = np.load('wesad/all_subjects/24May2021/extracted_all_2.npy')
data_3 = np.load('wesad/all_subjects/24May2021/extracted_all_3.npy')
data_4 = np.load('wesad/all_subjects/24May2021/extracted_all_4.npy')
label_1 = np.load('wesad/all_subjects/raw_data/label_all_subject_1.npy')[0:len(data_1)]
label_2 = np.load('wesad/all_subjects/raw_data/label_all_subject_2.npy')[0:len(data_2)]
label_3 = np.load('wesad/all_subjects/raw_data/label_all_subject_3.npy')[0:len(data_3)]
label_4 = np.load('wesad/all_subjects/raw_data/label_all_subject_4.npy')[0:len(data_4)]
#
all_data = np.concatenate((data_1, data_2, data_3, data_4))
all_label = np.concatenate((label_1, label_2, label_3, label_4))

step = 20
N_CLASSES = 4 #7
N_HIDDEN = 128
N_TIME_STEPS = 100
N_FEATURES = 13

segments = []
labelss = []
for i in range(0, len(all_data) - N_TIME_STEPS, step):
    d = all_data[i: i + N_TIME_STEPS]
    l = all_label[i: i + N_TIME_STEPS][0]
    segments.append(d)
    labelss.append(l)

reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
labeld = np.asarray(pd.get_dummies(labelss), dtype=np.float32)
print('DONE Data preparation!!')
# def cross_val(split_size, X_train, y_train):
#     kfold = KFold(n_splits=split_size, random_state=42, shuffle=True)
#     for train_idx, val_idx in kfold.split(X_train, y_train):
#         trainx = X_train[train_idx]
#         trainy = y_train[train_idx]
#         valx = X_train[val_idx]
#         valy = y_train[val_idx]
#     return trainx, trainy, valx, valy

#lr= 0.0020,0.0025,  0.0030
# LEARNING_RATE = [0.002, 0.0025,  0.0030]
lr = 0.001
N_EPOCHS = 1
BATCH_SIZE = 1024

train_acc = []
val_acc = []
# test_acc = []
train_loss = []
val_loss = []
# test_loss = []
# prediction = []


# train_count = len(X_train)

SPLIT_SIZE = 10
kfold = KFold(n_splits=SPLIT_SIZE, random_state=42, shuffle=True)

# train_x, train_y, val_x, val_y = cross_val(SPLIT_SIZE, X_train, y_train)


# train_count = len(train_x)
# val_count = len(val_x)
# test_count = len(X_test)

tf.reset_default_graph()
with tf.Session() as sess:
    X_lstm = tf.placeholder(tf.float32, [None, N_TIME_STEPS, N_FEATURES], name="input_acc")
    Y_lstm = tf.placeholder(tf.float32, [None, N_CLASSES], name="label_acc")

    #l2_acc = L2_LOSS * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

    w_lstm, b_lstm = var_create_LSTM(N_FEATURES, N_HIDDEN, N_CLASSES)
    pred_Y_lstm = create_LSTM_model(X_lstm, N_FEATURES, N_HIDDEN, N_TIME_STEPS, w_lstm, b_lstm)

    correct_pred_lstm = tf.equal(tf.argmax(pred_Y_lstm, 1), tf.argmax(Y_lstm, 1))
    accuracy_lstm = tf.reduce_mean(tf.cast(correct_pred_lstm, dtype=tf.float32))
    # loss_lstm = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_Y_lstm, labels=Y_lstm))
    loss_lstm = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_Y_lstm, labels=Y_lstm))


    optimizer_acc = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_lstm)
    # train_acc.append(lr)
    # train_loss.append(lr)
    # test_acc.append(lr)
    # test_loss.append(lr)

    sess.run(tf.global_variables_initializer())
    for train_idx, val_idx in kfold.split(reshaped_segments):
        train_x = reshaped_segments[train_idx]
        train_y = labeld[train_idx]
        val_x = reshaped_segments[val_idx]
        val_y = labeld[val_idx]
        train_count = len(train_x)
        val_count = len(val_x)
        for i in range(1, N_EPOCHS + 1):
            print('START TRAINING')
            for start, end in zip(range(0, train_count, BATCH_SIZE), range(BATCH_SIZE, train_count + 1, BATCH_SIZE)):
                sess.run(optimizer_acc, feed_dict={X_lstm: train_x[start:end], Y_lstm: train_y[start:end]})
                _lstm, train_acc_lstm, train_loss_lstm = sess.run([pred_Y_lstm, accuracy_lstm, loss_lstm],
                                                              feed_dict={X_lstm: train_x[start:end], Y_lstm: train_y[start:end]})

                # print(f'lr: {lr} epoch: {i} Train start: {start} end: {end}')
            train_acc.append(train_acc_lstm)
            train_loss.append(train_loss_lstm)
            print(f'lr: {lr} epoch: {i} train accuracy : {train_acc_lstm} loss: {train_loss_lstm}')
            for start, end in zip(range(0, val_count, BATCH_SIZE), range(BATCH_SIZE, val_count + 1, BATCH_SIZE)):
                sess.run(optimizer_acc, feed_dict={X_lstm: val_x[start:end], Y_lstm: val_y[start:end]})
                _val, val_acc_lstm, val_loss_lstm = sess.run([pred_Y_lstm, accuracy_lstm, loss_lstm],
                                                             feed_dict={X_lstm: val_x[start:end], Y_lstm: val_y[start:end]})
                # if i != 1 and i % 5 != 0:
                #     continue
            val_acc.append(val_acc_lstm)
            val_loss.append(val_loss_lstm)
            print(f'lr: {lr} epoch: {i} validation accuracy: {val_acc_lstm} loss: {val_loss_lstm}')
            print(f'DONE!! Training and validation epoch: {i}')
        np.save(f'wesad/all_subjects/24May2021/haft_data/train_acc_{N_EPOCHS}.npy', train_acc)
        np.save(f'wesad/all_subjects/24May2021/haft_data/train_loss_{N_EPOCHS}.npy', train_loss)
        np.save(f'wesad/all_subjects/24May2021/haft_data/val_acc_{N_EPOCHS}.npy', val_acc)
        np.save(f'wesad/all_subjects/24May2021/haft_data/val_loss_{N_EPOCHS}.npy', val_loss)
    sess.close()


print('DONE!!')
