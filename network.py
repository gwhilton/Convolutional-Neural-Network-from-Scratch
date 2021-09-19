from utilities import *
import numpy as np

class network(object):
    # initialise the network, the variable names are self explanatory
    # sizes is a list of the number of nodes in each layer
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.input_dim = sizes[0]
        self.output_dim = sizes[-1]
        self.hidden_nodes = sizes[1] 
        # this initialises the weights of the model and the model structure, much like sequential does in tensorflow
        self.initialise()
        self.pre_activations=[None]*(len(self.weights)+1)
        self.post_activations=[None]*(len(self.weights)+1)

    # function used to initialise the weights of the model
    def initialise(self):
        self.weights = []
        self.biases = []
        limit = np.sqrt(6/(self.input_dim+self.hidden_nodes))
        # first layer
        self.weights.append(np.random.uniform(-limit, limit, (self.input_dim, self.hidden_nodes)))
        self.biases.append(np.zeros((1, self.hidden_nodes)))

        # hidden layers
        for i in range(self.num_layers-2):
            limit = np.sqrt(3/(self.hidden_nodes))
            self.weights.append(np.random.uniform(-limit, limit, (self.hidden_nodes, self.hidden_nodes)))
            self.biases.append(np.zeros((1, self.hidden_nodes)))

    
        # output layers
        limit = np.sqrt(6/(self.output_dim + self.hidden_nodes))
        self.weights.append(np.random.uniform(-limit, limit, (self.hidden_nodes, self.output_dim)))
        self.biases.append(np.zeros((1, self.output_dim)))
       
    # forward propagation function
    def forward_prop(self, x_batch):
        q = len(self.weights)
        # initialise 
        self.pre_activations[0] = x_batch
        self.post_activations[0] = x_batch

        # compute activations for tanh layers
        for i in range(q-1):
            self.pre_activations[i+1] = dense(self.post_activations[i], self.weights[i], self.biases[i])
            self.post_activations[i+1] = np.tanh(self.pre_activations[i+1])
    
        # compute final activations for the softmax layer
        self.pre_activations[q] = dense(self.post_activations[q-1], self.weights[q-1], self.biases[q-1])
        self.post_activations[q] = softmax(self.pre_activations[q])

    # function to compute the deltas, stored in a list
    def deltas(self, y_batch):
    
        delta_ls = [self.post_activations[-1] - y_batch]
        
        # we will iterate backwards
        for i in range(len(self.pre_activations)-2):
            w = self.weights[-1-i]
            pre = self.pre_activations[-2-i]
            err = backpropagate(delta_ls[-1], w, pre)
            delta_ls.append(err)

        return list(reversed(delta_ls))

    # this function updates the weights
    def update_weights(self, grads, learning_rate):
        grad_W, grad_B = grads
        for i, (dw, db) in enumerate(zip(grad_W, grad_B)):
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db

    # this function implements the stochastic gradient descent, utilising the prior defined functions
    def sgd(self, x_batch, y_batch, learning_rate):
        self.forward_prop(x_batch)
        delta_ls = self.deltas(y_batch)
        grad_ls = grads(delta_ls, self.post_activations, x_batch)
        self.update_weights(grad_ls, learning_rate)

    # evaluate function computes accuracy and loss when given and x input and true y value
    def evaluate(self, x, y):
        # predictions
        y_pred = self.predict(x)
        # accuracy
        acc = accuracy_nn(y_pred, y)
        # losses
        loss = cross_entropy(y, y_pred)
    
        return acc, loss
    
    # predict function, implements the feedforward processes more efficiently for use in the evaluate function
    def predict(self, a):
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            h = dense(a, w, b)
            a = np.tanh(h)
        h = dense(a, self.weights[-1], self.biases[-1])
        a = softmax(h)
        return a
    
    # the function which train this network
    def train(self, x_train, y_train, x_val, y_val, batch_size, epochs, learning_rate):
        # initialise lists
        train_accs = []
        val_accs = []
        train_losses = []
        val_losses = []

        
        for epoch in range(epochs):
            np.random.seed(5*epoch)
            x_batches, y_batches = create_batches(x_train, y_train, batch_size)
            for i in range(int(np.ceil(x_train.shape[0]/batch_size))):
                # select random batch
                x_batch = x_batches[i]
                y_batch = y_batches[i]
                self.sgd(x_batch, y_batch, learning_rate)
            
            # compute train and validation accuracy and loss
            train_acc, train_loss = self.evaluate(x_train, y_train)
            val_acc, val_loss = self.evaluate(x_val, y_val)

            # update the lists
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

        # return dictionary containing values
        return {'train_acc': train_accs, 'val_acc': val_accs, 'train_loss': train_losses, 'val_loss': val_losses}