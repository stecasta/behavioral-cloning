import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D, Dropout, ELU
import matplotlib.image as mpimg


def GetData(data_dir):
    #This function takes the path of the data directory as
    #input and returns the images and the 
    #associated measurements
    lines = []
    with open(data_dir + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    images = []
    measurements = []
    # Set the correction for the left and right camera images
    correction = 0.15
    corrections = [correction, 0, -correction]
    for line in lines[1:]:
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = data_dir + '/IMG/' + filename
            img = cv2.imread(current_path)
            # When driving automnomously images are in RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            measurement = (float(line[3])) + corrections[i]
            measurements.append(measurement)
            # Augment data with flipped images
            images.append(cv2.flip(img, 1))
            measurements.append(measurement * - 1.0)
    return images, measurements

images, measurements = GetData('./data')

X_train = np.array(images)
Y_train = np.array(measurements)

# Model definition (NVIDIA-like)
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.25))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=1)
model.save('model.h5')
print("model saved")