import numpy as np
import pandas as pd
from scipy import stats


def readData(filePath):
    # attributes of the dataset
    columnNames = ['user_id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    data = pd.read_csv(filePath, header=None, names=columnNames, na_values=';')
    return data
def windows(data,size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start+= (size/2)

# segmenting the time series
def segment_signal(data, window_size = 90):
    segments = np.empty((0,window_size,3))
    labels = np.empty((0))
    for (start, end) in windows(data['timestamp'],window_size):
        x = data['x-axis'][start:end]
        y = data['y-axis'][start:end]
        z = data['z-axis'][start:end]
        if(len(data['timestamp'][start:end])==window_size):
            segments = np.vstack([segments,np.dstack([x, y, z])])
            labels = np.append(labels,stats.mode(data['activity'][start:end])[0][0])
    return segments, labels


dataset = readData('data/WISDM_ar_v1.1_raw.txt')
segments, labels = segment_signal(dataset)
labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)
numOfRows = segments.shape[1]
numOfColumns = segments.shape[2]

reshapedSegments = segments.reshape(segments.shape[0], numOfRows, numOfColumns, 1)

np.save('data-cnn.npy', reshapedSegments)
np.save('label-cnn.npy', labels)
print("finished")

