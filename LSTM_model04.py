import tensorflow.compat.v1 as tf

# Stack LSTM
def var_create_LSTM(N_FEATURES, N_HIDDEN_UNITS, N_CLASSES):
    W = {
        'hidden': tf.Variable(tf.random_normal([N_FEATURES, N_HIDDEN_UNITS])),
        'output': tf.Variable(tf.random_normal([N_HIDDEN_UNITS, N_CLASSES]))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([N_HIDDEN_UNITS], mean=1.0)),
        'output': tf.Variable(tf.random_normal([N_CLASSES]))
    }
    return W, biases

def create_LSTM_model(inputs, N_FEATURES, N_HIDDEN_UNITS, N_TIME_STEPS, W, biases):
    X = tf.transpose(inputs, [1, 0, 2])
    X = tf.reshape(X, [-1, N_FEATURES])
    hidden = tf.nn.relu(tf.matmul(X, W['hidden']) + biases['hidden'])
    hidden = tf.split(hidden, N_TIME_STEPS, 0)

    # LSTM layers
    lstm_layers = [tf.nn.rnn_cell.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0) for _ in range(2)]
    lstm_layers = tf.nn.rnn_cell.MultiRNNCell(lstm_layers)

    outputs, _ = tf.nn.static_rnn(lstm_layers, hidden, dtype=tf.float32)

    # Get output for the last time step
    lstm_last_output = outputs[-1]

    return tf.matmul(lstm_last_output, W['output']) + biases['output']

# Bi-directional LSTM
def var_create_biLSTM(N_FEATURES, N_HIDDEN_UNITS, N_CLASSES):
    W = {
        'hidden': tf.Variable(tf.random_normal([N_FEATURES, N_HIDDEN_UNITS])),
        'output': tf.Variable(tf.random_normal([N_HIDDEN_UNITS*2, N_CLASSES]))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([N_HIDDEN_UNITS], mean=1.0)),
        'output': tf.Variable(tf.random_normal([N_CLASSES]))
    }
    return W, biases

def biLSTM_model(input, N_FEATURES, N_HIDDEN_UNITS, N_TIME_STEPS, W, biases):
    X = tf.transpose(input, [1, 0, 2])
    X = tf.reshape(X, [-1, N_FEATURES])
    hidden = tf.nn.relu(tf.matmul(X, W['hidden']) + biases['hidden'])
    hidden = tf.split(hidden, N_TIME_STEPS, 0)

    # Bi-directional LSTM
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0)
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0)

    lstm_fw_multicell = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell])
    lstm_bw_multicell = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell])

    output, state_fw, state_bw = tf.nn.static_bidirectional_rnn(lstm_fw_multicell, lstm_bw_multicell, hidden, dtype=tf.float32)

    last_output = output[-1]

    return tf.matmul(last_output, W['output']) + biases['output']
