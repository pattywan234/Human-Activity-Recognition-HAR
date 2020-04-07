import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd


wisdm_lstm = pd.read_csv('data/LSTM-data/wisdm/result-accuracy.csv', header=None)
wisdm_cnn = pd.read_csv('data/CNN-data/wisdm/result-accuracy.csv', header=None)
hetero_lstm = pd.read_csv('data/LSTM-data/Heterogeneity/result-accuracy.csv', header=None)
hetero_cnn = pd.read_csv('data/CNN-data/Heterogeneity/result-accuracy.csv', header=None)

epoch_lstm = [10, 20, 50, 70, 100, 200]
epoch_cnn = [10, 20, 50, 70, 100]
font = {"family": "Times New Roman"}
fig, ax = plt.subplots()
ax.plot(epoch_lstm, hetero_lstm[0], marker='o', markersize=5, label='l2=0.0020, lr=0.0025')
ax.plot(epoch_lstm, hetero_lstm[1], marker='d', markersize=5, label='l2=0.0015, lr=0.0020')
ax.plot(epoch_lstm, hetero_lstm[2], marker='s', markersize=5, label='l2=0.0015, lr=0.0030')
ax.plot(epoch_lstm, hetero_lstm[3], marker='<', markersize=5, label='l2=0.0020, lr=0.0020')
ax.plot(epoch_lstm, hetero_lstm[4], marker='>', markersize=5, label='l2=0.0020, lr=0.0030')


legend = ax.legend(loc='lower right')
plt.xlabel('Epoch', **font)
plt.ylabel('Accuracy')
plt.title('LSTM with Heterogeneity dataset')
#plt.savefig('lstm-hetero.png')
plt.show()
print('finish')
