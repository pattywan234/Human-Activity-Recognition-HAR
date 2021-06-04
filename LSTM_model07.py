# k-fold cross validation, when k=10
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import (recall_score, accuracy_score, f1_score, precision_score)


data_1 = np.load('wesad/all_subjects/24May2021/extracted_all_1.npy')
data_2 = np.load('wesad/all_subjects/24May2021/extracted_all_2.npy')
data_3 = np.load('wesad/all_subjects/24May2021/extracted_all_3.npy')
data_4 = np.load('wesad/all_subjects/24May2021/extracted_all_4.npy')
label_1 = np.load('wesad/all_subjects/raw_data/label_all_subject_1.npy')[0:len(data_1)]
label_2 = np.load('wesad/all_subjects/raw_data/label_all_subject_2.npy')[0:len(data_2)]
label_3 = np.load('wesad/all_subjects/raw_data/label_all_subject_3.npy')[0:len(data_3)]
label_4 = np.load('wesad/all_subjects/raw_data/label_all_subject_4.npy')[0:len(data_4)]

all_data = np.concatenate((data_1, data_2, data_3, data_4))
all_label = np.concatenate((label_1, label_2, label_3, label_4))
N_FEATURES = 13
N_TIME_STEPS = 100
step = 20

segments = []
labelss = []

for i in range(0, len(all_data) - N_TIME_STEPS, step):
    d = all_data[i: i + N_TIME_STEPS]
    l = all_label[i: i + N_TIME_STEPS][0]
    segments.append(d)
    labelss.append(l)

reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
labeld = np.asarray(pd.get_dummies(labelss), dtype=np.float32)

N_CLASSES = 4 #7
N_HIDDEN_UNITS = 13


class CustomLSTM(tf.keras.Model):
    def __init__(self, **kwargs):
        super(CustomLSTM, self).__init__(**kwargs)

    def build(self, N_HIDDEN_UNITS):
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

fold_no = 10

f1, prec, recall, acc, ROC_AUC, conf = ([], [], [], [], [], [])
metric_cols = ['f1', 'precision', 'recall', 'accuracy', 'roc_auc']
cv_kfold = KFold(n_splits=fold_no, random_state=42, shuffle=True)

model = CustomLSTM()
model.summary()
# print(type(model))
# exit()

history = []
def custom_loss(y_true, y_pred):
   return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true))

# loss_fn = tf.keras.losses.CategoricalCrossentropy()
opt = tf.keras.optimizers.Adam(learning_rate=0.001)

for train_index, val_index in cv_kfold.split(reshaped_segments):
    train_x = reshaped_segments[train_index]
    train_y = labeld[train_index]
    val_x = reshaped_segments[val_index]
    val_y = labeld[val_index]

    model.compile(optimizer=opt, loss=custom_loss, metrics=['accuracy'])

    history_constant = model.fit(train_x, train_y, epochs=10, batch_size=1024, verbose=1, validation_data=(val_x, val_y))
    history.append(history_constant)

    pred_test = model.predict(val_x, verbose=1)
    pred_test_probs = model.predict_proba(val_x)
    # fpr, tpr, thresholds = roc_curve(val_y, pred_test_probs)
    max_predict = np.argmax(pred_test_probs, axis=1)
    max_y = np.argmax(val_y, axis=1)

    f1.append(f1_score(max_y, max_predict, average='weighted'))
    prec.append(precision_score(max_y, max_predict, average='weighted'))
    recall.append(recall_score(max_y, max_predict, average='weighted'))
    acc.append(accuracy_score(max_y, max_predict))

    performance = pd.DataFrame(zip(f1, prec, recall, acc, ROC_AUC, conf), columns=metric_cols)
    print(performance)
    performance.to_numpy('wesad/all_subjects/24May2021/performance_matrix_keras.npy')

np.save('wesad/all_subjects/24May2021/history_constant.npy', history)

