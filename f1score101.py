from sklearn import metrics
import numpy as np

# Constants
C="Cat"
F="Fish"
H="Hen"

predictions = np.load('data/LSTM-data/evaluate/predictions100.npy')
y_test = np.load('data/LSTM-data/y_test.npy')
max_predict = np.argmax(predictions, axis=1)
max_test = np.argmax(y_test, axis=1)
labels = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']
# True values
y_true = [C,C,C,C,C,C, F,F,F,F,F,F,F,F,F,F, H,H,H,H,H,H,H,H,H]
# Predicted values
y_pred = [C,C,C,C,H,F, C,C,C,C,C,C,H,H,F,F, C,C,C,H,H,H,H,H,H]

# Print the confusion matrix
print(metrics.confusion_matrix(max_test, max_predict))

# Print the precision and recall, among other metrics
print(metrics.classification_report(max_test, max_predict, digits=3))

