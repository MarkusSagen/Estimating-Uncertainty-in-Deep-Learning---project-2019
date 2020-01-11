# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras as tfk
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.compat.v1.keras.preprocessing.image import ImageDataGenerator
import cv2, random, time, os
from glob import glob
from sklearn.utils import class_weight


'''
Keract: Keras Activations + Gradients
https://github.com/philipperemy/keract
'''

def smooth_labels(y, smooth_factor=0):
    '''Convert a matrix of one-hot row-vector labels into smoothed versions.
    # Arguments
        y: matrix of one-hot row-vector labels to be smoothed
        smooth_factor: label smoothing factor (between 0 and 1)
    # Returns
        A matrix of smoothed labels.
    '''
    assert len(y.shape) == 2
    if 0 <= smooth_factor <= 1:
        # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
    else:
        raise Exception('Invalid label smoothing factor: ' + str(smooth_factor))
    return y

def aug_non_inter(img):
    ''' Interpolation-free augmentation function
    '''
    def ori(img):
        return img
    
    def fliph(img):
        return np.fliplr(img)
    
    def flipv(img):
        return np.flipud(img)
    
    def fliphv(img):
        return np.fliplr(np.flipud(img))
    
    def ori90(img):
        return np.rot90(img)
    
    def fliph90(img):
        return np.rot90(np.fliplr(img))
    
    def flipv90(img):
        return np.rot90(np.flipud(img))
    
    def fliphv90(img):
        return np.rot90(np.fliplr(np.flipud(img)))
    
    aug_functions = [ori, fliph, fliphv, fliphv, ori90, fliph90, fliphv90, fliphv90]   
    
    return random.choice(aug_functions)(img)

def viloss(y_true, y_pred):
    return tfk.losses.categorical_crossentropy(y_true, y_pred) + tfk.losses.KLD(y_true, y_pred)

def accuracy_curve(h, config, saveimg=False, verbose=1):
    acc, loss, val_acc, val_loss = h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']
    epoch = len(acc)
    plt.figure(figsize=(17, 5))
    plt.subplot(121)
    plt.plot(range(epoch), acc, label='Train')
    plt.plot(range(epoch), val_acc, label='Validation')
    plt.title('Accuracy over ' + str(epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(range(epoch), loss, label='Train')
    plt.plot(range(epoch), val_loss, label='Validation')
    plt.title('Loss over ' + str(epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    if saveimg:
        dir_imgs = './logs/imgs/'
        if not os.path.exists(dir_imgs):
            os.makedirs(dir_imgs)
        title = (f"{config['DATA_NAME']}_{config['ARCHI_NAME']}"
                    f"_LS{config['LABEL_SMOOTH']}_{config['METHOD_NAME']}_{config['CAL_NAME']}")
        plt.savefig(dir_imgs + f'{title}_leraningcurve.png', format='png', dpi=100, transparent=True, bbox_inches='tight')
        plt.savefig(dir_imgs + f'{title}_leraningcurve.pdf', format='pdf', transparent=True, bbox_inches='tight')
    if verbose == 2:
        plt.show()

def sample_data(X, Y, rate):
    ''' Randomly take rate% data
    '''
    # shuffle
    index = [i for i in range(len(X))]
    np.random.shuffle(index)
    X = X[index]
    Y = Y[index]
    
    # sample
    splitpoint = int(round(rate * len(X)))
    (X1, X2) = (X[0:splitpoint], X[splitpoint:]) 
    (Y1, Y2) = (Y[0:splitpoint], Y[splitpoint:])
    return X1, Y1, X2, Y2

def load_mnist(train_percentage, val_percentage, batch_size, label_smoothing):
    """
    Arguments:
        train_percentage: in [0, 1], percentage of training samples in whole dataset
        val_percentage: in [0, 1], percentage of validation samples in training set
        batch_size: batch size of ImageDataGenerator
        label_smoothing: in [0, 1], label smoothing factor. 0 means no smoothing.
    Returns:
        train_generator, X_val, Y_val, test_generator
    """
    from tensorflow.compat.v1.keras.datasets import mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    img_rows, img_cols = X_train.shape[1], X_train.shape[2]

    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

    Y_train = tfk.utils.to_categorical(Y_train)
    Y_test = tfk.utils.to_categorical(Y_test)

        
    _X_train, _Y_train, _, _  = sample_data(X_train, Y_train, train_percentage)
    X_train, Y_train, X_val, Y_val = sample_data(_X_train, _Y_train, (1 - val_percentage))
    
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')
    
    # normalise
    X_train = X_train / 255.
    X_val = X_val / 255.
    X_test = X_test / 255.
    
    # class weights
    class_weights = class_weight.compute_class_weight(
            'balanced',
            np.argmax(np.unique(Y_train, axis=0), axis=1),
            np.argmax(Y_train, axis=1)
            )
    class_weights = dict(enumerate(class_weights))
    
    # Label smoothing
    Y_train = smooth_labels(Y_train, smooth_factor=label_smoothing)

    # generators
    train_datagen = ImageDataGenerator(
        rescale=None,
        validation_split=0.0) # set validation split
    train_generator = train_datagen.flow(
            X_train, Y_train,
            batch_size=batch_size,
            subset=None) # set as training data
    
    test_datagen = ImageDataGenerator(rescale=None)
    test_generator = test_datagen.flow(
        X_test, Y_test, 
        shuffle=False, 
        batch_size=batch_size)

    print('train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_val.shape[0], 'validation samples')
    print(X_test.shape[0], 'test samples')

    return train_generator, class_weights, X_val, Y_val, test_generator

def load_OC(data_root, sample_shape, val_percentage, batch_size, label_smoothing):
    """
    Arguments:
        data_root: directory of data
        sample_shape: (h, w, c)
        val_percentage: in [0, 1], percentage of validation samples in training set
        batch_size: batch size of ImageDataGenerator
        label_smoothing: in [0, 1], label smoothing factor. 0 means no smoothing.
    Returns:
        train_generator, X_val, Y_val, test_generator
    """
    def load_dataset(subset):
        
        subset_dir = data_root + subset + '/'
        
        def read_sample(img_path):
            if sample_shape[-1] == 1:
                img = cv2.imread(img_path, 0)
            else:
                img = cv2.imread(img_path)
            if img.shape[:2] != sample_shape[:2]:
#                return
                img = cv2.resize(img, sample_shape[:2])
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
            return img
        
        cancer = np.array([read_sample(path) for path in glob(subset_dir + 'Cancer/*')])
        healthy = np.array([read_sample(path) for path in glob(subset_dir + 'Healthy/*')])
        X_test = np.concatenate((cancer, healthy), axis=0)
        
        Y_test = np.concatenate((np.zeros((len(cancer), 1)), np.ones((len(healthy), 1))), axis=0)
        Y_test = tfk.utils.to_categorical(Y_test)
        
        # shuffle testing set
        index = [i for i in range(len(X_test))]
        np.random.shuffle(index)
        X_test = X_test[index]
        Y_test = Y_test[index]
        
        return X_test, Y_test
    
    X_test, Y_test = load_dataset('test')
    
    X_test = X_test.astype('float32')
    X_test = X_test / 255.
    
    test_datagen = ImageDataGenerator(rescale=None)
    test_generator = test_datagen.flow(
        X_test, Y_test, 
        shuffle=False, 
        batch_size=batch_size)
    
    # training and validation set
    _X_train, _Y_train = load_dataset('train')
    X_train, Y_train, X_val, Y_val = sample_data(_X_train, _Y_train, (1 - val_percentage))  
    
    # class weights
    class_weights = class_weight.compute_class_weight(
            'balanced',
            np.argmax(np.unique(Y_train, axis=0), axis=1),
            np.argmax(Y_train, axis=1)
            )
    class_weights = dict(enumerate(class_weights))
    
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    # normalise
    X_train = X_train / 255.
    X_val = X_val / 255.
    
    # Label smoothing
    Y_train = smooth_labels(Y_train, smooth_factor=label_smoothing)
    
    train_datagen = ImageDataGenerator(
            rescale=None,
            preprocessing_function=aug_non_inter,
            validation_split=0.0) # set validation split
    train_generator = train_datagen.flow(
            X_train, Y_train,
            batch_size=batch_size,
            subset=None) # set as training data
    
    print('train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_val.shape[0], 'validation samples')
    print(X_test.shape[0], 'test samples')

    return train_generator, class_weights, X_val, Y_val, test_generator

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    
    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    # implementation 1
#    e_x = np.exp(x - np.max(x))
#    return e_x / e_x.sum(axis=1, keepdims=1)
    # implementation 2
#    prob = K.softmax(x, axis=-1)
#        with tf.Session() as sess:
#        return sess.run(prob)
    # implementation 3
#    prob = tf.nn.softmax(x, axis=-1)
#    with tf.Session() as sess:
#        return sess.run(prob)
    x = np.asarray(x, dtype=np.float64)
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

class TimeHistory(tfk.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def compute_train_time(ts):
    ''' training time per epoch
    ts: list of times during training, returned by utils.TimeHistory()
    '''
    t = ts.times[1:]
    return sum(t) / len(t)

def predict_logits(model, X_val, X_test):
    ''' compute logits from data
    model: trained model
    X: data, numpy array or keras_preprocessing.image.numpy_array_iterator.NumpyArrayIterator
    mc_samples: number of samples
    '''
    logits_model = tfk.models.Model(
            inputs=model.input,
            outputs=model.layers[-1].input)
    
    Y_logits_val = logits_model.predict(X_val)
    _t_test = time.time()
    Y_logits = logits_model.predict_generator(
            X_test,
            verbose=0)
    t_test = time.time() - _t_test

    return Y_logits_val, Y_logits, t_test

def calibrate(fn, logits_val, Y_val, logits_test, m_kwargs={}):
    """
    Calibrate models scores, using output from logits files and given function (fn). 
    There are implemented to different approaches "all" and "1-vs-K" for calibration,
    the approach of calibration should match with function used for calibration.
    
    TODO: split calibration of single and all into separate functions for more use cases.
    
    Parameters:
        fn (class): class of the calibration method used. It must contain methods "fit" and "predict", 
                    where first fits the models and second outputs calibrated probabilities.
        logits_val: predicted logits of validation set, shape of [samples, classes]
        Y_val: labels of test set, shape of [samples, classes]
        logits_test: predicted logits of test set, shape of [samples, classes]
        m_kwargs (dictionary): keyword arguments for the calibration class initialization
        
    Returns:
        probs_test: calibrated probabilities of test set.
    
    """
    # TODO: add support for MCsamples
    _t_cal = time.time()
    if fn.__name__ == 'NoCalibration':
        return softmax(logits_test), 0
    
    cal_model = fn(**m_kwargs)
    cal_model.fit(logits_val, Y_val)
    probs_test = cal_model.predict(logits_test)
    t_cal = time.time() - _t_cal
    return probs_test, t_cal