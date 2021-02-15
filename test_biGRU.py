import numpy as np
import tensorflow.compat.v1 as tf
from GRU_model01 import biGRU_model, var_create_biGRU
from sklearn.model_selection import KFold

#tf.disable_v2_behavior()
tf.disable_resource_variables()

#import training and test data
X_train = np.load('wesad/all_subjects/timestep_100/X_train.npy')
X_test = np.load('wesad/all_subjects/timestep_100/X_test.npy')
y_train = np.load('wesad/all_subjects/timestep_100/y_train.npy')
y_test = np.load('wesad/all_subjects/timestep_100/y_test.npy')


def cross_val(split_size):
    kfold = KFold(n_splits=split_size, random_state=2, shuffle=True)
    for train_idx, val_idx in kfold.split(X_train, y_train):
        trainx = X_train[train_idx]
        trainy = y_train[train_idx]
        valx = X_train[val_idx]
        valy = y_train[val_idx]
    return trainx, trainy, valx, valy

N_CLASSES = 4 #7
N_HIDDEN = 128
N_TIME_STEPS = 100
N_FEATURES = len(X_train[2][1]) #13 #8
#lr= 0.0020,0.0025,  0.0030
# LEARNING_RATE = [0.002, 0.0025,  0.0030]
lr = 0.002
L2_LOSS = 0.002
N_EPOCHS = 1
BATCH_SIZE = 1024

train_acc = []
val_acc = []
test_acc = []
train_loss = []
val_loss = []
test_loss = []

SPLIT_SIZE = 5

train_x, train_y, val_x, val_y = cross_val(SPLIT_SIZE)

train_count = len(train_x)
val_count = len(val_x)

tf.reset_default_graph()
with tf.Session() as sess:
    X_gru = tf.placeholder(tf.float32, [None, N_TIME_STEPS, N_FEATURES], name="input_acc")
    Y_gru = tf.placeholder(tf.float32, [None, N_CLASSES], name="label_acc")

    #l2_acc = L2_LOSS * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # input, N_FEATURES, N_HIDDEN_UNITS, N_TIME_STEPS, N_LAYERS, W, biases

    w_gru, b_gru = var_create_biGRU(N_FEATURES, N_HIDDEN, N_CLASSES)
    pred_Y_gru = biGRU_model(X_gru, N_FEATURES, N_HIDDEN, N_TIME_STEPS, w_gru, b_gru)

    correct_pred_gru = tf.equal(tf.argmax(pred_Y_gru, 1), tf.argmax(Y_gru, 1))
    accuracy_gru = tf.reduce_mean(tf.cast(correct_pred_gru, dtype=tf.float32))
    # loss_gru = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_Y_gru, labels=Y_gru))
    loss_gru = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_Y_gru, labels=Y_gru))


    optimizer_acc = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_gru)
    # train_acc.append(lr)
    # train_loss.append(lr)
    # test_acc.append(lr)
    # test_loss.append(lr)

    sess.run(tf.global_variables_initializer())
    for i in range(1, N_EPOCHS + 1):
        for start, end in zip(range(0, train_count, BATCH_SIZE), range(BATCH_SIZE, train_count + 1, BATCH_SIZE)):
            sess.run(optimizer_acc, feed_dict={X_gru: train_x[start:end], Y_gru: train_y[start:end]})
            _gru, train_acc_gru, train_loss_gru = sess.run([pred_Y_gru, accuracy_gru, loss_gru],
                                                            feed_dict={X_gru: train_x[start:end], Y_gru: train_y[start:end]})
            print(f'lr: {lr} epoch: {i} Train start: {start} end: {end}')
        train_acc.append(train_acc_gru)
        train_loss.append(train_loss_gru)
        print(f'lr: {lr} epoch: {i} train accuracy : {train_acc_gru} loss: {train_loss_gru}')
        for start, end in zip(range(0, val_count, BATCH_SIZE), range(BATCH_SIZE, val_count + 1, BATCH_SIZE)):
            sess.run(optimizer_acc, feed_dict={X_gru: val_x[start:end], Y_gru: val_y[start:end]})
            _val, val_acc_gru, val_loss_gru = sess.run([pred_Y_gru, accuracy_gru, loss_gru],
                                                              feed_dict={X_gru: val_x[start:end], Y_gru: val_y[start:end]})
            print(f'lr: {lr} epoch: {i} Validate start: {start} end: {end}')
        val_acc.append(val_acc_gru)
        val_loss.append(val_loss_gru)
        print(f'lr: {lr} epoch: {i} validation accuracy : {val_acc_gru} loss: {val_loss_gru}')
        print(f'DONE!! Training and validation epoch: {i}')
    print('START Testing')
    predictions_gru, final_acc_gru, final_loss_gru = sess.run([pred_Y_gru, accuracy_gru, loss_gru],
                                                                     feed_dict={X_gru: X_test, Y_gru: y_test})
    test_acc.append(final_acc_gru)
    test_loss.append(final_loss_gru)
    print(f'lr: {lr} test accuracy : {final_acc_gru} loss: {final_loss_gru}')
    np.save(f'wesad/all_subjects/predict_{lr}.npy', predictions_gru)
    sess.close()

np.save(f'wesad/all_subjects/train_acc_{N_EPOCHS}_{lr}.npy', train_acc)
np.save(f'wesad/all_subjects/train_loss_{N_EPOCHS}_{lr}.npy', train_loss)
np.save(f'wesad/all_subjects/val_acc_{N_EPOCHS}_{lr}.npy', val_acc)
np.save(f'wesad/all_subjects/val_loss_{N_EPOCHS}_{lr}.npy', val_loss)
np.save(f'wesad/all_subjects/test_acc_{N_EPOCHS}_{lr}.npy', test_acc)
np.save(f'wesad/all_subjects/test_loss_{N_EPOCHS}_{lr}.npy', test_loss)

