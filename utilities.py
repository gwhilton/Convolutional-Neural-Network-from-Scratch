import numpy as np

# matrix mult in layers
def dense(h, w, b):
    return np.dot(h, w) + b

# derivative of tanh(x)
def dtanh(x):
    return 1 - (np.tanh(x))**2

def cross_entropy(y, a):
    N = a.shape[0]
    ce = -np.sum(y*np.log(a))/N
    return ce

def softmax(x):
    exp = np.exp(x-np.max(x))
    return exp/(exp.sum(axis=1, keepdims=True))

# performs the error at each level of the network
def backpropagate(delta, w, a):
    return dtanh(a)*np.dot(delta, w.T)

# grads function computes the gradients at each iteration
def grads(deltas, post_activations, x_batch):
    # initialise lists
    grad_W = [None]*len(deltas)
    grad_B = [None]*len(deltas)
    for i in range(len(deltas)):
        # I use matrix multiplication instead of np.newaxis as it is far more efficient
        grad_W[i] = np.dot(post_activations[i].T, deltas[i])/(x_batch.shape[0])
        grad_B[i] = np.mean(deltas[i], axis=0)
    return grad_W, grad_B

# accuracy function which takes the y_prediction and true value as its input
def accuracy_nn(y_pred, y_true):
    # find which class is predicted for each instance, this is stored as a list
    class_pred = np.argmax(y_pred, axis=1)
    class_true = np.argmax(y_true, axis=1)
    # compute the accuracy
    acc = sum(1 for x, y in zip(class_pred, class_true) if x == y) / float(len(class_pred))
    return acc

# create batches
def create_batches(x_train, y_train, batch_size):
    # shuffle data
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]
    x_batches = []
    y_batches = []
    # append lists with batches
    for i in range(int(np.ceil(x_train.shape[0]/batch_size))-1):
            x_batches.append(x_train[i*batch_size:(i+1)*batch_size])
            y_batches.append(y_train[i*batch_size:(i+1)*batch_size])
        
    x_batches.append(x_train[int(np.floor(x_train.shape[0]/batch_size))*batch_size:])
    y_batches.append(y_train[int(np.floor(x_train.shape[0]/batch_size))*batch_size:])

    return x_batches, y_batches