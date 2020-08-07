import tensorflow.compat.v1 as tf

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

def var_create(N_HIDDEN_UNITS, N_CLASSES):
    W = {
        'hidden': tf.Variable(tf.random_normal([N_CLASSES, N_HIDDEN_UNITS])),
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

def create_neural_net(inputs, N_FEATURES, W, biases):
    #X = tf.transpose(inputs, [1, 0, 2])
    #X = tf.reshape(X, [-1, N_FEATURES])
    hidden = tf.matmul(inputs, W['hidden']) + biases['hidden']
    output = tf.matmul(hidden, W['output']) + biases['output']
    return output

