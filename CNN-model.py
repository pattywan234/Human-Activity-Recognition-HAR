import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras import optimizers


reshapedSegments = np.load('data/CNN-data/data-cnn.npy')
labels = np.load('data/CNN-data/label-cnn.npy')
trainX = np.load('data/CNN-data/trainX.npy')
testX = np.load('data/CNN-data/testX.npy')
trainY = np.load('data/CNN-data/trainY.npy')
testY = np.load('data/CNN-data/testY.npy')

numOfRows = reshapedSegments.shape[1]
numOfColumns = reshapedSegments.shape[2]
numChannels = 1
numFilters = 128 # number of filters in Conv2D layer
# kernal size of the Conv2D layer
kernalSize1 = 2
# max pooling window size
poolingWindowSz = 2
# number of filters in fully connected layers
numNueronsFCL1 = 128
numNueronsFCL2 = 128
# split ratio for test and validation
trainSplitRatio = 0.8
# number of epochs
Epochs = 10
# batchsize
batchSize = 10
# number of total clases
numClasses = labels.shape[1]
# dropout ratio for dropout layer
dropOutRatio = 0.2

def cnnModel():
    model = Sequential()
    # adding the first convolutional layer
    model.add(Conv2D(numFilters, (kernalSize1, kernalSize1), input_shape=(numOfRows, numOfColumns, 1), activation='relu'))
    # adding a maxpooling layer
    model.add(MaxPooling2D(pool_size=(poolingWindowSz,poolingWindowSz), padding='valid'))
    # adding a dropout layer for the regularization and avoiding over fitting
    model.add(Dropout(dropOutRatio))
    # flattening the output in order to apply the fully connected layer
    model.add(Flatten())
    # adding first fully connected layer with 256 outputs
    model.add(Dense(numNueronsFCL1, activation='relu'))
    #adding second fully connected layer 128 outputs
    model.add(Dense(numNueronsFCL2, activation='relu'))
    # adding softmax layer for the classification
    model.add(Dense(numClasses, activation='softmax'))
    # Compiling the model to generate a model
    adam = optimizers.Adam(lr = 0.001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model
model = cnnModel()
for layer in model.layers:
    print(layer.name)
model.fit(trainX, trainY, validation_split=1-trainSplitRatio, epochs=Epochs, batch_size=batchSize, verbose=2)
predictions = model.predict(testX, verbose=2)
score = model.evaluate(testX, testY, verbose=2)
print('test loss, test accuracy', score)
