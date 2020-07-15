from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split

data = np.load("wesad/S3/c_acc.npy") #

# normalize data into -10 and 20
def norm_cal(input, min, max, a, b):
    output = ((b-a)*((input - min)/(max-min))) + a
    return output

norm_range = [-10, 20]
x_new = []
x_min = []
x_max = []
a = norm_range[0]
b = norm_range[1]

for i in range(0, len(data[0])):
    x_test = data[:, i]
    min = x_test.min()
    max = x_test.max()
    x_min.append(min)
    x_max.append(max)

for cc in range(0, len(x_min)):
    xmin = x_min[cc]
    xmax = x_max[cc]
    for rc in range(0, len(data)):
        va = data[rc, cc]
        after = norm_cal(va, xmin, xmax, a, b)
        x_new.append(after)

x_reshape = np.asarray(x_new).reshape(len(data), 3)

N_TIME_STEPS = 200
N_FEATURES = 3
step = 20
segments = []
labels = []
for i in range(0, len(x_reshape) - N_TIME_STEPS, step):
    xs = x_reshape[i: i + N_TIME_STEPS, 0]
    ys = x_reshape[i: i + N_TIME_STEPS, 1]
    zs = x_reshape[i: i + N_TIME_STEPS, 2]
    #label = data[i: i + N_TIME_STEPS]
    segments.append([xs, ys, zs])
    #labels.append(label)

reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
#labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)

#RANDOM_SEED = 42
#random_state is the seed used by the random number generator, if number is same random result will be same
#X_train, X_test, y_train, y_test = train_test_split(reshaped_segments, labels, test_size=0.3, random_state=RANDOM_SEED)


np.save("wesad/S2/data_norm_cacc.npy", reshaped_segments)
#np.save("data/LSTM-data/PAMAP2/X_test03.npy", X_test)
#np.save("data/LSTM-data/PAMAP2/y_train07.npy", y_train)
#np.save("data/LSTM-data/PAMAP2/y_test03.npy", y_test)

print("finish")

