from tensorflow import keras
import numpy as np
import tensorflow as tf
import pickle
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


X_train = np.load('wesad/norm/range-1to1/wl/X_train_23Jun.npy')
X_test = np.load('wesad/norm/range-1to1/wl/X_test_23Jun.npy')
y_train = np.load('wesad/norm/range-1to1/wl/y_train_23Jun.npy')
y_test = np.load('wesad/norm/range-1to1/wl/y_test_23Jun.npy')

X_train_us = np.load('data/mix_wisdm_hhar.npy')
X_test_us = np.load('data/mix_wisdm_hhar_test.npy')
y_train_us = np.zeros((len(X_train_us), 2), dtype=np.float32)


N_CLASSES = 2
N_HIDDEN_UNITS = 8
N_TIME_STEPS = 100
N_FEATURES = 5
epoch = 100

f1, acc, con_mat = [], [], []

model = keras.Sequential()
model.add(keras.layers.Bidirectional(keras.layers.LSTM(5, return_sequences=True, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(5)))
# model.add(keras.layers.Dense(8))
# model.add(keras.layers.Dropout(0.5))
# model.add(keras.layers.Activation('softmax'))
model.add(keras.layers.Dense(N_CLASSES))
# model.add(keras.layers.Activation('softmax'))

opt = tf.keras.optimizers.Adam(learning_rate=0.001)

def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true))
# loss=keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer=opt, loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history_constant = model.fit(x=X_train, y=y_train, shuffle=True, epochs=epoch, callbacks=None, batch_size=1024, verbose=1, validation_split=0.2)

predicted = model.predict(X_test)

max_predict = np.argmax(predicted, axis=1)
max_y = np.argmax(y_test, axis=1)

f1.append(f1_score(max_y, max_predict, average=None))
acc.append(accuracy_score(max_y, max_predict))
con_mat.append(confusion_matrix(max_y, max_predict))

np.save(f'wesad/norm/f1_bi04_{epoch}.npy', f1)
np.save(f'wesad/norm/accuracy_bi04_{epoch}.npy', acc)
np.save(f'wesad/norm/con_mat_bi04_{epoch}.npy', con_mat)
with open(f'wesad/norm//histopy_bi04_{epoch}.pkl', 'wb') as file_pi:
    pickle.dump(history_constant.history, file_pi)

f1_us, acc_us, con_mat_us = [], [], []
model.trainable = False
model.compile(optimizer=opt, loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history_constant_us = model.fit(x=X_train_us, y=y_train_us, epochs=epoch, callbacks=None, batch_size=1024, verbose=1,  validation_split=0.2)

tran_labels = model.tranduction_

predicted_us = model.predict(X_test_us)

max_predict_us = np.argmax(predicted_us, axis=1)
max_tran_labels = np.argmax(tran_labels, axis=1)

f1_us.append(f1_score(max_tran_labels, max_predict_us, average=None))
acc_us.append(accuracy_score(max_tran_labels, max_predict_us))
con_mat_us.append(confusion_matrix(max_tran_labels, max_predict_us))

np.save(f'wesad/norm/f1_us_bi04_{epoch}.npy', f1)
np.save(f'wesad/norm/accuracy_us_bi04_{epoch}.npy', acc)
np.save(f'wesad/norm/con_mat_us_bi04_{epoch}.npy', con_mat)
with open(f'wesad/norm//histopy_us_bi04_{epoch}.pkl', 'wb') as file_pi:
    pickle.dump(history_constant_us.history, file_pi)
