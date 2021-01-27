import numpy as np
import tensorflow.compat.v1 as tf
from LSTM_model04 import biLSTM_model, var_create_biLSTM
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
LEARNING_RATE = 0.002
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
    X_lstm = tf.placeholder(tf.float32, [None, N_TIME_STEPS, N_FEATURES], name="input_acc")
    Y_lstm = tf.placeholder(tf.float32, [None, N_CLASSES], name="label_acc")

    #l2_acc = L2_LOSS * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # input, N_FEATURES, N_HIDDEN_UNITS, N_TIME_STEPS, N_LAYERS, W, biases

    w_lstm, b_lstm = var_create_biLSTM(N_FEATURES, N_HIDDEN, N_CLASSES)
    pred_Y_lstm = biLSTM_model(X_lstm, N_FEATURES, N_HIDDEN, N_TIME_STEPS, w_lstm, b_lstm)

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
    for i in range(1, N_EPOCHS + 1):
        for start, end in zip(range(0, train_count, BATCH_SIZE), range(BATCH_SIZE, train_count + 1, BATCH_SIZE)):
            sess.run(optimizer_acc, feed_dict={X_lstm: train_x[start:end], Y_lstm: train_y[start:end]})
            _lstm, train_acc_lstm, train_loss_lstm = sess.run([pred_Y_lstm, accuracy_lstm, loss_lstm],
                                                            feed_dict={X_lstm: train_x, Y_lstm: train_y})
            print(f'lr: {LEARNING_RATE} epoch: {i} Train start: {start} end: {end}')
        train_acc.append(train_acc_lstm)
        train_loss.append(train_loss_lstm)
        print(f'lr: {LEARNING_RATE} epoch: {i} train accuracy : {train_acc_lstm} loss: {train_loss_lstm}')
        for start, end in zip(range(0, train_count, BATCH_SIZE), range(BATCH_SIZE, val_count + 1, BATCH_SIZE)):
            sess.run(optimizer_acc, feed_dict={X_lstm: val_x[start:end], Y_lstm: val_y[start:end]})
            _val, val_acc_lstm, val_loss_lstm = sess.run([pred_Y_lstm, accuracy_lstm, loss_lstm],
                                                              feed_dict={X_lstm: val_x, Y_lstm: val_y})
            print(f'lr: {LEARNING_RATE} epoch: {i} Validate start: {start} end: {end}')
        val_acc.append(val_acc_lstm)
        val_loss.append(val_loss_lstm)
        print(f'lr: {LEARNING_RATE} epoch: {i} validation accuracy : {val_acc_lstm} loss: {val_loss_lstm}')
        print(f'DONE!! Training and validation epoch: {N_EPOCHS}')
    print('START Testing')
    predictions_lstm, final_acc_lstm, final_loss_lstm = sess.run([pred_Y_lstm, accuracy_lstm, loss_lstm],
                                                                     feed_dict={X_lstm: X_test, Y_lstm: y_test})
    print(f'lr: {LEARNING_RATE} test accuracy : {final_acc_lstm} loss: {final_loss_lstm}')
    test_acc.append(final_acc_lstm)
    test_loss.append(final_loss_lstm)
    np.save(f'wesad/all_subjects/predict_lr{LEARNING_RATE}.npy', predictions_lstm)
    sess.close()

np.save(f'wesad/all_subjects/train_acc_{N_EPOCHS}.npy', train_acc)
np.save(f'wesad/all_subjects/train_loss_{N_EPOCHS}.npy', train_loss)
np.save(f'wesad/all_subjects/val_acc_{N_EPOCHS}.npy', val_acc)
np.save(f'wesad/all_subjects/val_loss_{N_EPOCHS}.npy', val_loss)
np.save(f'wesad/all_subjects/test_acc_{N_EPOCHS}.npy', test_acc)
np.save(f'wesad/all_subjects/test_loss_{N_EPOCHS}.npy', test_loss)

