from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import TimeDistributed

def vanilla_LSTM(n_hidden, time_step, n_feature):
    model = Sequential()
    model.add(LSTM(n_hidden, activation='relu', input_shape=(time_step, n_feature)))
    model.add(Dense(1))
    # model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mse')
    return model

def stacked_LSTM(n_hidden, time_step, n_feature):
    model = Sequential()
    model.add(LSTM(n_hidden, activation='relu', return_sequences=True, input_shape=(time_step, n_feature)))
    model.add(LSTM(n_hidden, activation='relu'))
    model.add(Dense(1))
    # model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mse')
    return model

def bi_LSTM(n_hidden, time_step, n_feature):
    model = Sequential()
    model.add(Bidirectional(LSTM(n_hidden, activation='relu', input_shape=(time_step, n_feature))))
    model.add(Dense(1))
    # model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mse')
    return model


