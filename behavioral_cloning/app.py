import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D

def run():

    lines = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    images = []
    measurements = []
    correction = 0.1
    corrections = [-correction, 0, correction]
    for line in lines[1:]:
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = './data/IMG/' + filename
            img = cv2.imread(current_path)
            images.append(img)
            measurement = (float(line[3])) + corrections[i]
            measurements.append(measurement)
        
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image,1))
        augmented_measurements.append(measurement * - 1.0)

    X_train = np.array(augmented_images)
    Y_train = np.array(augmented_measurements)

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
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    print("started training")
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
    print("model trained")
    model.save('model.h5')
    print("model saved")


#     history_object = model.fit_generator(train_generator, samples_per_epoch =
#         len(train_samples), validation_data = 
#         validation_generator,
#         nb_val_samples = len(validation_samples), 
#         nb_epoch=5, verbose=1)

#     ### print the keys contained in the history object
#     print(history_object.history.keys())

#     ### plot the training and validation loss for each epoch
#     plt.plot(history_object.history['loss'])
#     plt.plot(history_object.history['val_loss'])
#     plt.title('model mean squared error loss')
#     plt.ylabel('mean squared error loss')
#     plt.xlabel('epoch')
#     plt.legend(['training set', 'validation set'], loc='upper right')
#     plt.show()

    return