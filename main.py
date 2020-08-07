import numpy as np
import tensorflow.compat.v1 as tf
from LSTM_model04 import create_LSTM_model, var_create_LSTM, create_neural_net, var_create

tf.disable_v2_behavior()

#import data from PAMAP2 dataset
X_train_acc = np.load("acc_data/X_train_mix.npy")
X_test_acc = np.load("acc_data/X_test_mix.npy")
y_train_acc = np.load("acc_data/y_train_mix.npy")
y_test_acc = np.load("acc_data/y_test_mix.npy")
#import data from WESAD dataset
X_train_h = np.load('health_data/X_train-acc.npy')
X_test_h = np.load('health_data/X_test-acc.npy')
y_train_h = np.load('health_data/y_train-acc.npy')
y_test_h = np.load('health_data/y_test-acc.npy')

N_CLASSES = 19
N_HIDDEN = 128
N_TIME_STEPS = 200
N_FEATURES_ACC = 3
N_FEATURES_H = 5

#same L2 value and learning rate
L2_LOSS = 0.002

#l2 = L2_LOSS * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

LEARNING_RATE = 0.0015

N_EPOCHS = 1
BATCH_SIZE = 1024

train_count_acc = len(X_train_acc)
train_count_h = len(X_train_h)

#tf.reset_default_graph()
def main():
    tf.reset_default_graph()
    with tf.Session() as sess:
        X_ACC = tf.placeholder(tf.float32, [None, N_TIME_STEPS, N_FEATURES_ACC], name="input_acc")
        Y_ACC = tf.placeholder(tf.float32, [None, N_CLASSES], name="label_acc")

        l2_acc = L2_LOSS * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

        w_acc, b_acc = var_create_LSTM(N_FEATURES_ACC, N_HIDDEN, N_CLASSES)

        pred_Y_acc = create_LSTM_model(X_ACC, N_FEATURES_ACC, N_HIDDEN, N_TIME_STEPS, w_acc, b_acc)
        pred_softmax_acc = tf.nn.softmax(pred_Y_acc, name="y_acc")

        loss_acc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_Y_acc, labels=Y_ACC)) + l2_acc

        optimizer_acc = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss_acc)

        correct_pred_acc = tf.equal(tf.argmax(pred_softmax_acc, 1), tf.argmax(Y_ACC, 1))
        accuracy_acc = tf.reduce_mean(tf.cast(correct_pred_acc, dtype=tf.float32))
        sess.run(tf.global_variables_initializer())
        for i in range(1, N_EPOCHS + 1):
            for start, end in zip(range(0, train_count_acc, BATCH_SIZE), range(BATCH_SIZE, train_count_acc + 1, BATCH_SIZE)):
                sess.run(optimizer_acc, feed_dict={X_ACC: X_train_acc[start:end], Y_ACC: y_train_acc[start:end]})
            _acc, train_acc_acc, train_loss_acc = sess.run([pred_softmax_acc, accuracy_acc, loss_acc],
                                                           feed_dict={X_ACC: X_train_acc, Y_ACC: y_train_acc})
            if i != 1 and i % 5 != 0:
                continue
            print(f'epoch: {i} train accuracy accelerometer data: {train_acc_acc} loss: {train_loss_acc}')

        predictions_acc, final_acc_acc, final_loss_acc = sess.run([pred_softmax_acc, accuracy_acc, loss_acc],
                                                                  feed_dict={X_ACC: X_test_acc, Y_ACC: y_test_acc})
        print(f'test accuracy accelerometer data: {final_acc_acc} loss: {final_loss_acc}')
        np.save('after_train_acc.npy', _acc)
        sess.close()

    tf.reset_default_graph()
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        X_H = tf.placeholder(tf.float32, [None, N_TIME_STEPS, N_FEATURES_H], name="input_h")
        Y_H = tf.placeholder(tf.float32, [None, N_CLASSES], name="label_h")

        l2_h = L2_LOSS * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

        w_h, b_h = var_create_LSTM(N_FEATURES_H, N_HIDDEN, N_CLASSES)

        pred_Y_h = create_LSTM_model(X_H, N_FEATURES_H, N_HIDDEN, N_TIME_STEPS, w_h, b_h)
        pred_softmax_h = tf.nn.softmax(pred_Y_h, name="y_h")

        loss_h = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_Y_h, labels=Y_H)) + l2_h

        optimizer_h = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss_h)

        correct_pred_h = tf.equal(tf.argmax(pred_softmax_h, 1), tf.argmax(Y_H, 1))
        accuracy_h = tf.reduce_mean(tf.cast(correct_pred_h, dtype=tf.float32))
        sess.run(tf.global_variables_initializer())
        for i in range(1, N_EPOCHS + 1):
            for start, end in zip(range(0, train_count_h, BATCH_SIZE), range(BATCH_SIZE, train_count_h + 1, BATCH_SIZE)):
                sess.run(optimizer_h, feed_dict={X_H: X_train_h[start:end], Y_H: y_train_h[start:end]})
            _h, train_acc_h, train_loss_h = sess.run([pred_softmax_h, accuracy_h, loss_h],
                                                     feed_dict={X_H: X_train_h, Y_H: y_train_h})
            if i != 1 and i % 5 != 0:
                continue
            print(f'epoch: {i} train accuracy health data: {train_acc_h} loss: {train_loss_h}')
        predictions_h, final_acc_h, final_loss_h = sess.run([pred_softmax_h, accuracy_h, loss_h],
                                                            feed_dict={X_H: X_test_h, Y_H: y_test_h})
        print(f'test accuracy health data: {final_acc_h} loss: {final_loss_h}')
        np.save('after_train_h.npy', _h)
        sess.close()

    X_train_mix = np.concatenate((_acc, _h), axis=0)
    y_train_mix = np.concatenate((y_train_acc, y_train_h), axis=0)
    X_test_mix = np.concatenate((predictions_acc, predictions_h), axis=0)
    y_test_mix = np.concatenate((y_test_acc, y_test_h), axis=0)

    np.save('X_train_mix.npy', X_train_mix)
    np.save('y_train_mix.npy', y_train_mix)
    np.save('X_test_mix.npy', X_test_mix)
    np.save('y_test_mix.npy', y_test_mix)

    tf.reset_default_graph()
    with tf.Session() as sess:
        X_M = tf.placeholder(tf.float32, [None, N_CLASSES], name="input_m")
        Y_M = tf.placeholder(tf.float32, [None, N_CLASSES], name="label_m")

        l2_m = L2_LOSS * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

        w_m, b_m = var_create(N_HIDDEN, N_CLASSES)

        pred_Y_m = create_neural_net(X_train_mix, N_CLASSES, w_m, b_m)
        pred_softmax_m = tf.nn.softmax(pred_Y_m, name="y_m")

        loss_m = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_Y_m, labels=y_train_mix)) + l2_m

        correct_pred_m = tf.equal(tf.argmax(pred_softmax_m, 1), tf.argmax(Y_M, 1))
        accuracy_m = tf.reduce_mean(tf.cast(correct_pred_m, dtype=tf.float32))
        sess.run(tf.global_variables_initializer())
        for i in range(1, N_EPOCHS + 1):
            _m, train_acc_m, train_loss_m = sess.run([pred_softmax_m, accuracy_m, loss_m],
                                                     feed_dict={X_M: X_train_mix, Y_M: y_train_mix})
            if i != 1 and i % 5 != 0:
                continue
            print(f'epoch: {i} train accuracy health data: {train_acc_m} loss: {train_loss_m}')
        predictions_m, final_acc_m, final_loss_m = sess.run([pred_softmax_m, accuracy_m, loss_m],
                                                            feed_dict={X_M: X_test_mix, Y_M: y_test_mix})
        print(f'test accuracy health data: {final_acc_m} loss: {final_loss_m}')
        sess.close()

if __name__ == "__main__":
    main()

