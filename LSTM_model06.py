# Normal cross validation (train/test: 70/30)
import numpy as np
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (recall_score, accuracy_score, f1_score, precision_score)
import pandas as pd

X_train = np.load('wesad/all_subjects/24May2021/X_train.npy')
X_test = np.load('wesad/all_subjects/24May2021/X_test.npy')
y_train = np.load('wesad/all_subjects/24May2021/y_train.npy')
y_test = np.load('wesad/all_subjects/24May2021/y_test.npy')

print("x_train shape: ", X_train.shape)
print("y_train shape:", y_train.shape)

N_CLASSES = 4  # 7
N_HIDDEN_UNITS = 13
N_TIME_STEPS = 100
N_FEATURES = 13
epoch = 50

class CustomLSTM(tf.keras.Model):
    def __init__(self, **kwargs):
        super(CustomLSTM, self).__init__(**kwargs)

    def build(self, num_of_lstms):
        self.w_hidden = tf.Variable(tf.random.normal([N_FEATURES, N_HIDDEN_UNITS], stddev=0.01))
        self.w_output = tf.Variable(tf.random.normal([N_HIDDEN_UNITS, N_CLASSES], stddev=0.01))
        self.b_hidden = tf.Variable(tf.random.normal([N_HIDDEN_UNITS], stddev=0.01))
        self.b_output = tf.Variable(tf.random.normal([N_CLASSES], stddev=0.01))

        self.lstm_layers = [tf.compat.v1.nn.rnn_cell.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0) for _ in range(2)]
        self.lstm_layers = tf.compat.v1.nn.rnn_cell.MultiRNNCell(self.lstm_layers)

    def call(self, x_input):
        y = tf.transpose(x_input, [1, 0, 2])
        y = tf.reshape(y, [-1, N_FEATURES])
        hidden = tf.nn.relu(tf.matmul(y, self.w_hidden) + self.b_hidden)
        hidden = tf.split(hidden, N_TIME_STEPS, 0)
        outputs, _ = tf.compat.v1.nn.static_rnn(self.lstm_layers, hidden, dtype=tf.float32)
        lstm_last_output = outputs[-1]
        return tf.matmul(lstm_last_output, self.w_output) + self.b_output


# history = []
f1, prec, recall, acc, ROC_AUC, conf = ([], [], [], [], [], [])
metric_cols = ['f1', 'precision', 'recall', 'accuracy', 'roc_auc']


model = CustomLSTM()

# loss_fn = tf.keras.losses.CategoricalCrossentropy()
opt = tf.keras.optimizers.Adamax(learning_rate=0.001)

train_x, val_x, train_y, val_y = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true))


model.compile(optimizer=opt, loss=custom_loss, metrics=['accuracy'])

history_constant = model.fit(train_x, train_y, epochs=epoch, batch_size=1024, verbose=1, validation_data=(val_x, val_y))

predicted = model.predict(X_test)

max_predict = np.argmax(predicted, axis=1)
max_y = np.argmax(y_test, axis=1)

f1.append(f1_score(max_y, max_predict, average='weighted'))
prec.append(precision_score(max_y, max_predict, average='weighted'))
recall.append(recall_score(max_y, max_predict, average='weighted'))
acc.append(accuracy_score(max_y, max_predict))
#
#performance = pd.DataFrame(zip(f1, prec, recall, acc, ROC_AUC, conf), columns=metric_cols).to_numpy()
# print(performance)
#np.save('wesad/all_subjects/24May2021/performance_matrix.npy', performance)

# evaluate_model = model.evaluate(predicted, y_test)
# history.append(history_constant)

#np.save('wesad/all_subjects/24May2021/predicted_keras.npy', predicted)
# np.save('wesad/all_subjects/24May2021/evaluate_keras.npy', evaluate_model)
np.save(f'wesad/all_subjects/24May2021/new_model/f1_model06_{epoch}', f1)
np.save(f'wesad/all_subjects/24May2021/new_model/precision_model06_{epoch}', prec)
np.save(f'wesad/all_subjects/24May2021/new_model/recall_model06_{epoch}', recall)
np.save(f'wesad/all_subjects/24May2021/new_model/accuracy_model06_{epoch}', acc)
# np.save('wesad/all_subjects/24May2021/history.npy', history)
with open('wesad/all_subjects/24May2021/histopy.pkl', 'wb') as file_pi:
    pickle.dump(history_constant.history, file_pi)
