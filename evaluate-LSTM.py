import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pickle
from sklearn.metrics import f1_score


history = pickle.load(open('data/LSTM-data/evaluate/histpry10.p', 'rb'))
predictions = np.load('data/LSTM-data/evaluate/predictions10.npy')
y_test = np.load('data/LSTM-data/y_test.npy')


max_test = np.argmax(y_test, axis=1)
max_predict = np.argmax(predictions, axis=1)
#f1 = f1_score(max_test, max_predict)
#print('F1 score', f1)

#1
plt.figure(1)
plt.subplot2grid((2, 1), (0, 0))
plt.title('Train Loss')
plt.plot(np.array(history['train_loss']), "bo-", markersize=3)
plt.subplot2grid((2, 1), (1, 0))
plt.title('Train Accuracy')
plt.plot(np.array(history['train_acc']), "bo-", markersize=3)

#2
plt.figure(2)
plt.subplot2grid((2, 1), (0, 0))
plt.title('Validation Loss')
plt.plot(np.array(history['val_loss']), "bo-", label="Val loss", markersize=3)
plt.subplot2grid((2, 1), (1, 0))
plt.title('Validation Accuracy')
plt.plot(np.array(history['val_acc']), "bo-", label="Val accuracy", markersize=3)

"""
#3
plt.subplot2grid((3, 1), (2, 0), colspan=2)
plt.title('Test')
plt.plot(np.array(history['test_loss']), "b-", label="Test loss")
plt.plot(np.array(history['test_acc']), "b--", label="Test accuracy")
"""
plt.gca().yaxis.set_minor_formatter(NullFormatter())
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.3, wspace=0.35)
plt.show()

"""
#Confusion matrix
labels = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']
conf_matrix = metrics.confusion_matrix(max_test, max_predict)
plt.figure(figsize=(16, 14))
sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt='d')
plt.title("Confusion matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
"""
