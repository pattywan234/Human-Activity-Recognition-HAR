import numpy as np
from sklearn.model_selection import KFold
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


X_train = np.load("data/LSTM-data/X_train.npy")
X_test = np.load("data/LSTM-data/X_test.npy")
y_train = np.load("data/LSTM-data/y_train.npy")
y_test = np.load("data/LSTM-data/y_test.npy")

N_CLASSES = 6
N_HIDDEN_UNITS = 64
N_TIME_STEPS = 200
N_FEATURES = 3
SPLIT_SIZE = 5

def create_LSTM_model(inputs):
    W = {
        'hidden': tf.Variable(tf.random_normal([N_FEATURES, N_HIDDEN_UNITS])),
        'output': tf.Variable(tf.random_normal([N_HIDDEN_UNITS, N_CLASSES]))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([N_HIDDEN_UNITS], mean=1.0)),
        'output': tf.Variable(tf.random_normal([N_CLASSES]))
    }

    X = tf.transpose(inputs, [1, 0, 2])
    X = tf.reshape(X, [-1, N_FEATURES])
    hidden = tf.nn.relu(tf.matmul(X, W['hidden']) + biases['hidden'])
    hidden = tf.split(hidden, N_TIME_STEPS, 0)

    # Stack 2 LSTM layers
    lstm_layers = [tf.nn.rnn_cell.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0) for _ in range(2)]
    lstm_layers = tf.nn.rnn_cell.MultiRNNCell(lstm_layers)

    outputs, _ = tf.nn.static_rnn(lstm_layers, hidden, dtype=tf.float32)

    # Get output for the last time step
    lstm_last_output = outputs[-1]

    return tf.matmul(lstm_last_output, W['output']) + biases['output']


def cross_val(split_size):
    kfold = KFold(n_splits=split_size, random_state=2, shuffle=True)
    for train_idx, val_idx in kfold.split(X_train, y_train):
        trainx = X_train[train_idx]
        trainy = y_train[train_idx]
        val_x = X_train[val_idx]
        val_y = y_train[val_idx]
    return trainx, trainy, val_x, val_y


tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, N_TIME_STEPS, N_FEATURES], name="input")
Y = tf.placeholder(tf.float32, [None, N_CLASSES])

pred_Y = create_LSTM_model(X)
pred_softmax = tf.nn.softmax(pred_Y, name="y_")


L2_LOSS = 0.001

l2 = L2_LOSS * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_Y, labels=Y)) + l2

LEARNING_RATE = 0.0025

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

correct_pred = tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

N_EPOCHS = 20
BATCH_SIZE = 1024

saver = tf.train.Saver()

history = dict(train_loss=[],
               train_acc=[],)
               #test_loss=[],
               #test_acc=[])

train_x, train_y, val_x, val_y = cross_val(SPLIT_SIZE)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

train_count = len(train_x)


for i in range(1, N_EPOCHS + 1):
    for start, end in zip(range(0, train_count, BATCH_SIZE), range(BATCH_SIZE, train_count + 1, BATCH_SIZE)):
        sess.run(optimizer, feed_dict={X: train_x[start:end], Y: train_y[start:end]})


    _, acc_train, loss_train = sess.run([pred_softmax, accuracy, loss], feed_dict={X: train_x, Y: train_y})
    _, acc_val, loss_val = sess.run([pred_softmax, accuracy, loss], feed_dict={X: val_x, Y: val_y})
    #_, acc_test, loss_test = sess.run([pred_softmax, accuracy, loss], feed_dict={X: X_test, Y: y_test})

    history['train_loss'].append(loss_train)
    history['train_acc'].append(acc_train)
    #history['test_loss'].append(loss_test)
    #history['test_acc'].append(acc_test)

    if i != 1 and i % 10 != 0:
        continue
    print(f'epoch: {i} train accuracy: {acc_train} loss: {loss_train} validate accuracy: {acc_val} loss: {loss_val}')
    #print(f'epoch: {i} test accuracy: {acc_test} loss: {loss_test}')


predictions, acc_final, loss_final = sess.run([pred_softmax, accuracy, loss], feed_dict={X: X_test, Y: y_test})
print(f'test accuracy: {acc_final} loss: {loss_final}')

