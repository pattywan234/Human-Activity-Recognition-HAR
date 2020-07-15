import pandas as pd
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv('PAMAP2/protocol/raw_data/subject101.csv', header=None)
df = df.dropna()
df_np = df.to_numpy()
dt_acc = []

#get chest accelerometer
for i in range(21, 24):
    dt_acc.append(df_np[:, i])
res_dt = np.asarray(dt_acc).reshape(len(df_np), 3)

#normalize data into -10 and 20
def norm_cal(input, min, max, a, b):
    output = ((b-a)*((input - min)/(max-min))) + a
    return output

norm_range = [-10, 20]
x_new = []
x_min = []
x_max = []
a = norm_range[0]
b = norm_range[1]

for i in range(0, len(res_dt[0])):
    x_test = res_dt[:, i]
    min = x_test.min()
    max = x_test.max()
    x_min.append(min)
    x_max.append(max)

for cc in range(0, len(x_min)):
    xmin = x_min[cc]
    xmax = x_max[cc]
    for rc in range(0, len(res_dt)):
        va = res_dt[rc, cc]
        after = norm_cal(va, xmin, xmax, a, b)
        x_new.append(after)

acc_reshape = np.asarray(x_new).reshape(len(res_dt), 3)

# train, test set split
N_TIME_STEPS = 200
N_FEATURES = 3
step = 20
segments = []
label = df_np[:, 1]

for i in range(0, len(acc_reshape) - N_TIME_STEPS, step):
    xs = acc_reshape[i: i + N_TIME_STEPS, 0]
    ys = acc_reshape[i: i + N_TIME_STEPS, 1]
    zs = acc_reshape[i: i + N_TIME_STEPS, 2]
    segments.append([xs, ys, zs])

reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
labels = np.asarray(pd.get_dummies(label), dtype=np.float32)

"""
RANDOM_SEED = 42
#random_state is the seed used by the random number generator, if number is same random result will be same
X_train, X_test, y_train, y_test = train_test_split(reshaped_segments, labels, test_size=0.3, random_state=RANDOM_SEED)

np.save("PAMAP2/res_dt.npy", res_dt)
np.save("PAMAP2/X_test03.npy", X_test)
np.save("PAMAP2/y_train07.npy", y_train)
np.save("PAMAP2/y_test03.npy", y_test)
"""
print("finished")

