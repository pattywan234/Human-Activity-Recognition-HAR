from tensorflow import keras
import numpy as np
import tensorflow as tf
import pickle
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


X_train = np.load('wesad/norm/range-1to1/wl/X_train_aug_22Jun.npy')
X_test = np.load('wesad/norm/range-1to1/wl/X_test_aug_22Jun.npy')
y_train = np.load('wesad/norm/range-1to1/wl/y_train_aug_22Jun.npy')
y_test = np.load('wesad/norm/range-1to1/wl/y_test_aug_22Jun.npy')


N_CLASSES = 4  # 7
N_HIDDEN_UNITS = 8
N_TIME_STEPS = 100
N_FEATURES = 5
epoch = 100

f1, prec, recall, acc, con_mat = [], [], [], [], []

model = keras.Sequential()
model.add(keras.layers.Bidirectional(keras.layers.LSTM(5, return_sequences=True, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02), input_shape=(N_TIME_STEPS, N_FEATURES))))
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
# prec.append(precision_score(max_y, max_predict, average=None))
# recall.append(recall_score(max_y, max_predict, average=None))
acc.append(accuracy_score(max_y, max_predict))
con_mat.append(confusion_matrix(max_y, max_predict))

np.save(f'wesad/norm/f1_bi03_{epoch}_4.npy', f1)
# np.save(f'wesad/all_subjects/6Jun2021/precision_model06_{epoch}_4.npy', prec)
# np.save(f'wesad/all_subjects/6Jun2021/recall_model06_{epoch}_4.npy', recall)
np.save(f'wesad/norm/accuracy_bi03_{epoch}_4.npy', acc)
np.save(f'wesad/norm/con_mat_bi03_{epoch}_4.npy', con_mat)
# np.save('wesad/all_subjects/24May2021/history.npy', history)
with open(f'wesad/norm//histopy_bi03_{epoch}_4.pkl', 'wb') as file_pi:
    pickle.dump(history_constant.history, file_pi)

