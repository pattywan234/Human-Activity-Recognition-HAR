import matplotlib
import matplotlib.pyplot as plt
#import numpy as np

x = [10, 20, 50, 70, 100]
#loss
y_lstm_loss = [0.643, 0.4466, 0.2442, 0.2159, 0.1918]
y_loss32 = [0.3958, 0.5896, 0.84659, 1.02906, 0.97725]
y_loss64 = [0.4536, 0.4491, 0.6121, 0.82872, 0.83755]
y_loss1024 = [0.57392, 0.51414, 0.43798, 0.47586, 0.5327]
#accuracy
#y_lstm_acc = [92.57, 95.84, 97.21, 97.60, 97.54]
y_acc32 = [87.19, 85.81, 87.66, 87.00, 87.47]
y_acc64 = [83.96, 87.11, 87.26, 87.68, 86.74]
y_acc1024 = [78.81, 81.12, 86.18, 86.72, 87.53]
fig, ax = plt.subplots()

line1, = ax.plot(x, y_loss32, label='CNN batch size = 32')
line1.set_dashes([2, 2, 10, 2])

line2, = ax.plot(x, y_loss64, dashes=[6, 2], label='CNN batch size = 64')

line3, = ax.plot(x, y_loss1024, label='CNN batch size = 1024')

line4, = ax.plot(x, y_lstm_loss, label='LSTM')

ax.set(xlabel='Epoch (times)', ylabel='test loss', title='Test loss (LSTM vs CNN)')

ax.legend()
plt.show()

