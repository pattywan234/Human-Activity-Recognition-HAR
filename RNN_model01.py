"""
This file contains stack RNN and Bi-directional RNN.
"""
import tensorflow.compat.v1 as tf

# stack RNN
def var_create_RNN(N_FEATURES, N_HIDDEN_UNITS, N_CLASSES):
    W = {
        'hidden': tf.Variable(tf.random_normal([N_FEATURES, N_HIDDEN_UNITS])),
        'output': tf.Variable(tf.random_normal([N_HIDDEN_UNITS, N_CLASSES]))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([N_HIDDEN_UNITS], mean=1.0)),
        'output': tf.Variable(tf.random_normal([N_CLASSES]))
    }
    return W, biases

def create_RNN_model(inputs, N_FEATURES, N_HIDDEN_UNITS, N_TIME_STEPS, W, biases):
    X = tf.transpose(inputs, [1, 0, 2])
    X = tf.reshape(X, [-1, N_FEATURES])
    hidden = tf.nn.relu(tf.matmul(X, W['hidden']) + biases['hidden'])
    hidden = tf.split(hidden, N_TIME_STEPS, 0)

    # rnn layers
    rnn_layers = [tf.nn.rnn_cell.BasicRNNCell(N_HIDDEN_UNITS, activation='relu') for _ in range(2)]
    rnn_layers = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    outputs, _ = tf.nn.static_rnn(rnn_layers, hidden, dtype=tf.float32)

    # Get output for the last time step
    rnn_last_output = outputs[-1]

    return tf.matmul(rnn_last_output, W['output']) + biases['output']

# Bi-directional RNN
def var_create_biRNN(N_FEATURES, N_HIDDEN_UNITS, N_CLASSES):
    W = {
        'hidden': tf.Variable(tf.random_normal([N_FEATURES, N_HIDDEN_UNITS])),
        'output': tf.Variable(tf.random_normal([N_HIDDEN_UNITS*2, N_CLASSES]))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([N_HIDDEN_UNITS], mean=1.0)),
        'output': tf.Variable(tf.random_normal([N_CLASSES]))
    }
    return W, biases

def biRNN_model(input, N_FEATURES, N_HIDDEN_UNITS, N_TIME_STEPS, W, biases):
    X = tf.transpose(input, [1, 0, 2])
    X = tf.reshape(X, [-1, N_FEATURES])
    hidden = tf.nn.relu(tf.matmul(X, W['hidden']) + biases['hidden'])
    hidden = tf.split(hidden, N_TIME_STEPS, 0)

    # Bi-directional rnn
    rnn_fw_cell = tf.nn.rnn_cell.BasicRNNCell(N_HIDDEN_UNITS, activation='relu')
    rnn_bw_cell = tf.nn.rnn_cell.BasicRNNCell(N_HIDDEN_UNITS, activation='relu')

    rnn_fw_multicell = tf.nn.rnn_cell.MultiRNNCell([rnn_fw_cell])
    rnn_bw_multicell = tf.nn.rnn_cell.MultiRNNCell([rnn_bw_cell])

    output, state_fw, state_bw = tf.nn.static_bidirectional_rnn(rnn_fw_multicell, rnn_bw_multicell, hidden, dtype=tf.float32)

    last_output = output[-1]

    return tf.matmul(last_output, W['output']) + biases['output']