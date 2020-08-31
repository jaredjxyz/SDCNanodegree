import numpy as np
import cv2
import pickle
import os
import tensorflow as tf
import json
import PIL
import argparse
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import Dense, Flatten, Dropout, Merge
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, Nadam
from keras.models import load_model


driving_log_data_names = ['center','steering','throttle','brake','speed']
def main():
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--remove_straight_angle', default=None, type=float, help="Remove all training data with steering angle less than this. Useful for getting rid of straight bias")
    parser.add_argument('--save_generated_images', action='store_true', "Location to save generated images to")
    parser.add_argument('--load_model', type=str, help="For transfer learning, here's the model to start with")
    parser.add_argument('--directory', type=str, default='data', help="Directory for training data")
    parser.add_argument('--learning_rate', type=float, default=.001)
    args = parser.parse_args()

    driving_log_filename = 'driving_log.csv'

    #Load data from computer into a dictionary
    data = get_driving_log_info(driving_log_filename, directory=args.directory, remove_straight_angle = args.remove_straight_angle)
    labels = np.array(data['steering'])

    train_test = train_test_split(data['center'], labels, test_size=.1, train_size=.4)
    center_train = np.array(train_test[0])
    center_val = np.array(train_test[1])
    y_train = np.array(train_test[2])
    y_val = np.array(train_test[3])

    #Get data set up for image generator
    train_datagen = ImageDataGenerator(width_shift_range = .1,
                                       height_shift_range = .1,
                                       rescale=1./255,
                                       fill_mode='constant',
                                       cval=0)

    val_datagen = ImageDataGenerator(rescale=1./255)

    center_generator = train_datagen.flow(
        center_train, y_train,
        batch_size=128, shuffle=True, save_to_dir='generated' if args.save_data else None)

    center_val_generator = val_datagen.flow(
        center_val, y_val,
        batch_size = 128, shuffle=True)

    if args.load_model:
    #Load previous model
        print('loading model')
        model = load_model(args.load_model)
    #Make a new model
    else:
        model = Sequential()

        #Convolution 1. Input: (?, 66, 200, 3) Output: (?, 31, 98, 24)
        model.add(Convolution2D(24,5,5, activation='relu', subsample=(2,2), input_shape=(66,200,3)))

        #Convolution 2. Input: (?, 31, 98, 24) Output: (?, 14, 47, 36)
        model.add(Convolution2D(36,5,5, activation='relu', subsample=(2,2)))

        #Convolution 3. Input: (?, 14, 47, 36) Output: (?, 5, 22, 48)
        model.add(Convolution2D(48,5,5, activation='relu', subsample=(2,2)))

        #Convolution 4. Input: (?, 5, 22, 48) Output: (?, 3, 20, 64)
        model.add(Convolution2D(64, 3, 3, activation='relu'))

        #Convolution 5. Input: (?, 3, 20, 64) Output: (?, 1, 18, 64)
        model.add(Convolution2D(64,3,3, activation='relu'))

        model.add(Dropout(.5))

        #Flatten the layers. Input: (?, 1, 18, 64) Output: (?, 1152)
        model.add(Flatten())

        #Fully Connected #1. Input: (?, 1152) Output: (?, 100)
        model.add(Dense(100, activation='relu'))

        #Fully connected #2. Input: (?, 100) Output: (?, 50)
        model.add(Dense(50, activation='relu'))

        #Fully connected #3. Input: (?, 50) Output: (?, 10)
        model.add(Dense(10, activation='relu'))

        model.add(Dropout(.5))

        #Output layer: 1 output
        model.add(Dense(1))

        for i in range(len(model.layers)):
            model.layers[i].name += str(i)

        optimizer = Nadam(lr=args.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

    #Save json file
    with open('model.json', 'w') as file:
        json_model = json.loads(model.to_json())
        json.dump(json_model, file)

    checkpoint_callback = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
    model.fit_generator(center_generator, nb_epoch=args.epochs, samples_per_epoch=len(center_train),
        validation_data = center_val_generator, nb_val_samples=len(y_val), callbacks=[checkpoint_callback]
        )

    model.save('model.md5')

    # model.fit(center_generator, y_train, batch_size=128, nb_epoch=args.epochs, validation_data=(center_val, y_val))



def load_driving_log_info(driving_log_filename, directory='data', save=True):
    """ Saves driving log info to file

    Given the name of the log file and optionally the directory, saves to multiple .p files inside of the directory

    Arguments:
        driving_log_filename {str} -- name of the file (ending in .csv)

    Keyword Arguments:
        directory {str} -- directory name (without '/' at the end) (default: {'data'})
    """

    print('Loading driving data')

    driving_log_filename = directory + '/' + driving_log_filename
    driving_log_data = np.genfromtxt(driving_log_filename, delimiter=',', names=True, dtype=None)
    log_data = {}
    #Decode names for files for the first three
    centerfilenames = [directory + '/IMG/' + os.path.basename(x.decode().strip()) for x in driving_log_data['center']]
    leftfilenames = [directory + '/IMG/' + os.path.basename(x.decode().strip()) for x in driving_log_data['left']]
    rightfilenames = [directory + '/IMG/' + os.path.basename(x.decode().strip()) for x in driving_log_data['right']]
    images = []
    for filename in centerfilenames + leftfilenames + rightfilenames:
        image = load_img(filename)
        image = image.resize((200,66), PIL.Image.BICUBIC)
        image_flipped = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        images.append(img_to_array(image))
        #Flip every other image
        images.append(img_to_array(image_flipped))
    log_data['center'] = images
    assert all(x is not None for x in log_data['center']), "A center image is missing"

    print('Finished loading {}'.format('center'))

    for name in driving_log_data_names[1:]:
        #Flip steering angle of every other image
        log_data[name] = np.tile(np.array([(x,x) if name is not 'steering' else (x,-x) for x in driving_log_data[name]]).flatten(), 3)
    print('Finished loading the rest of the data')

    num_data_points = len(driving_log_data)*6
    assert all([len(data) == num_data_points for data in log_data.values()]), "Data incorrect, missing some data"

    # Must save everything separately because of bug in saving large files with pickle :(
    if save:
        for name, data in log_data.items():
            output_file = directory + '/' + name + '_data.p'
            pickle.dump(data, open(output_file, 'wb'))
            print('Saved', output_file)

    return log_data


def get_driving_log_info(driving_log_filename='driving_log.csv', directory='data', remove_straight_angle = None, save=True):
    """
    Gets and preprocesses image data
    """
    data_dict = load_driving_log_info(driving_log_filename, directory=directory, save=save)

    if remove_straight_angle is not None:
        #Get indices where it's going straight
        straightIndices = np.array([index for index in range(len(data_dict['steering'])) if abs(data_dict['steering'][index]) <= remove_straight_angle])

        #Remove straight indices
        data_dict['center'] = np.delete(data_dict['center'], straightIndices, axis=0)
        data_dict['steering'] = np.delete(data_dict['steering'], straightIndices)
        data_dict['throttle'] = np.delete(data_dict['throttle'], straightIndices)
        data_dict['brake'] = np.delete(data_dict['brake'], straightIndices)
        data_dict['speed'] = np.delete(data_dict['speed'], straightIndices)
    return data_dict

if __name__ == '__main__':
    main()
