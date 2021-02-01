import tensorflow.compat.v1 as tf

# Perceptron
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

def create_neural_net(inputs, W, biases):
    #X = tf.transpose(inputs, [1, 0, 2])
    #X = tf.reshape(X, [-1, N_FEATURES])
    hidden = tf.matmul(inputs, W['hidden']) + biases['hidden']
    output = tf.matmul(hidden, W['output']) + biases['output']
    return output

