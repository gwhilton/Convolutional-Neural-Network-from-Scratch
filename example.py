import numpy as np
import tensorflow as tf
import pandas as pd
from network import *

# load in dataset 
def load_data():
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    x_val = x_val.astype('float32') / 255.

    # convert labels to categorical samples
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=10)
    return ((x_train, y_train), (x_val, y_val))

(x_train, y_train), (x_val, y_val) = load_data()


# flatten data
x_train_reshape = x_train.reshape(*x_train.shape[:1], -2)
x_val_reshape = x_val.reshape(*x_val.shape[:1], -2)


# initialise network
neural_net = network([3072, 400, 400, 400, 400, 400, 10])
# train the data, will take around 12 minutes
history = neural_net.train(x_train_reshape, y_train, x_val_reshape, y_val, 128, 40, 0.01)

# function to show 3 examples of picture and prediction probabilities
def network_estimate():
    labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    plt.figure(figsize=(20,8))
    plt.subplot(121)
    idx = np.random.randint(0, 49999)
    true_class = np.argmax(y_train[idx])
    plt.imshow(x_train[idx])
    plt.title('True Class is: {}'.format(labels[true_class]))

    plt.subplot(122)
    pred_class = np.squeeze(neural_net.predict(x_train_reshape[idx]))
    idx = range(1, 11)
    x_pos = [2*i for i in idx]
    plt.bar(x_pos, pred_class, width=1.5, color="blue", alpha=0.5)
    plt.xticks(x_pos, labels, rotation='vertical')
    plt.title('Class Prediction')
    plt.xlabel('Class')
    plt.ylabel('Probability')   
    plt.show()
    return

# function to produce plots of training/validation accuracy and training/validation loss
def history_plots():
    plt.figure(figsize=(20,8))
    
    plt.subplot(121)
    plt.plot(list(range(40)), history['train_acc'], color='r', label='Train Accuracy')
    plt.plot(list(range(40)), history['val_acc'], color='green', label='Validation Accuracy')
    plt.title('Accuracy Plots')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.subplot(122)
    plt.plot(list(range(40)), history['train_loss'], color='r', label='Train Loss')
    plt.plot(list(range(40)), history['val_loss'], color='green', label='Validation Loss')
    plt.title('Loss Plots')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()
    return


# plots etc
network_estimate()
network_estimate()
network_estimate()

history_plots()

print('Termination Training Accuracy is: {}'.format(history['train_acc'][-1]))
print('Termination Validation Accuracy is: {}'.format(history['val_acc'][-1]))
print('Termination Training Loss is: {}'.format(history['train_loss'][-1]))
print('Termination Validation Loss is: {}'.format(history['val_loss'][-1]))