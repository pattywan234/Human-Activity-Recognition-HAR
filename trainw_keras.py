# from numpy import array
import numpy as np
from LSTM_model05 import vanilla_LSTM, stacked_LSTM, bi_LSTM

#import training and test data
X_train = np.load('wesad/S2/Normalize/label_selected/train_keras/X_train.npy')
X_test = np.load('wesad/S2/Normalize/label_selected/train_keras/X_test.npy')
y_train = np.load('wesad/S2/Normalize/label_selected/train_keras/y_train.npy')
y_test = np.load('wesad/S2/Normalize/label_selected/train_keras/y_test.npy')

# def split_sequence(sequence, n_steps):
#     X, y = list(), list()
#     for i in range(len(sequence)):
#         # find the end of this pattern
#         end_ix = i + n_steps
#         # check if we are beyond the sequence
#         if end_ix > len(sequence)-1:
#             break
#         # gather input and output parts of the pattern
#         seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
#         X.append(seq_x)
#         y.append(seq_y)
#     return array(X), array(y)
#
# raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# time_step = 3
# X, y = split_sequence(raw_seq, time_step)
# n_feature = 1
# X = X.reshape((X.shape[0], X.shape[1], n_feature))
# n_hidden = 50
#
# x_input = array([70, 80, 90])
# x_input = x_input.reshape((1, time_step, n_feature))

n_hidden = 128
time_step = 200
n_feature = len(X_train[2][1])
epoch = 100

# vanilla LSTM
# training
vlstm = vanilla_LSTM(n_hidden, time_step, n_feature)
vlstm.fit(X_train, y_train, epochs=epoch, verbose=0)
# testing
vhat = vlstm.predict(X_test, verbose=0)
print('predicted result of vanilla LSTM', vhat)

# stacked LSTM
# training
slstm = stacked_LSTM(n_hidden, time_step, n_feature)
slstm.fit(X_train, y_train, epochs=100, verbose=0)
# testing
shat = slstm.predict(X_test, verbose=0)
print('predicted result of stacked LSTM', shat)

# Bidireectional LSTM
# training
blstm = bi_LSTM(n_hidden, time_step, n_feature)
blstm.fit(X_train, y_train, epochs=100, verbose=0)
# testing
bhat = blstm.predict(X_test, verbose=0)
print('predicted result of Bidireectional LSTM', bhat)

