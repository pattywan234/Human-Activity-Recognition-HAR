import pandas as pd
from scipy import stats
import numpy as np


columns = ['user', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
df = pd.read_csv('data/WISDM_ar_v1.1_raw.txt', header=None, names=columns)
df = df.dropna()


N_TIME_STEPS = 200
N_FEATURES = 3
step = 20
segments = []
labels = []
for i in range(0, len(df) - N_TIME_STEPS, step):
    xs = df['x-axis'].values[i: i + N_TIME_STEPS]
    ys = df['y-axis'].values[i: i + N_TIME_STEPS]
    zs = df['z-axis'].values[i: i + N_TIME_STEPS]
    label = stats.mode(df['activity'][i: i + N_TIME_STEPS])[0][0]
    segments.append([xs, ys, zs])
    labels.append(label)

reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)

RANDOM_SEED = 42
#random_state is the seed used by the random number generator, if number is same random result will be same
X_train, X_test, y_train, y_test = train_test_split(
        reshaped_segments, labels, test_size=0.2, random_state=RANDOM_SEED)


np.save("data/LSTM-data/X_train.npy", X_train)
np.save("data/LSTM-data/X_test.npy", X_test)
np.save("data/LSTM-data/y_train.npy", y_train)
np.save("data/LSTM-data/y_test.npy", y_test)
print("finished")

