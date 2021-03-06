
# Neural Computation (Extended)
# CW1: Backpropagation and Softmax
# Autumn 2020
#

import numpy as np
import time
import fnn_utils
import unittest
import time
import os

# Some activation functions with derivatives.
# Choose which one to use by updating the variable phi in the code below.

def sigmoid(x):
    # The logistic sigmoid function defined as (1/(1 + e^-x))
    # takes an input x of any real number and returns an output value in the range of -1 and 1.
    sig = 1/(1 + np.exp(-x))
    return sig

def sigmoid_d(x):
    sigma = sigmoid(x)
    return sigma*(1-sigma)

def relu(x): #Rectified Linear Unit
    return np.array([max(0,x_i) for x_i in x])

def relu_d(x):
    return np.array([1 if i > 0 else 0 for i in x])

def lrelu(x): #Leaky RELU
    return np.array([max(0.01*x_i,x_i) for x_i in x])

def lrelu_d(x):
    return np.array([1 if i > 0 else 0.01 for i in x])

def tanh(x): # Tanh function
    return np.array([np.tanh(x_i) for x_i in x])

def tanh_d(x):
    return np.array([(1- np.tanh(x_i)**2) for x_i in x])

class BackPropagation:

    # The network shape list describes the number of units in each
    # layer of the network. The input layer has 784 units (28 x 28
    # input pixels), and 10 output units, one for each of the ten
    # classes.

    def __init__(self,network_shape=[784,20,20,20,10]):

        # Network Size Overwrite
        #network_shape=[784,20,20,20,10]

        # Read the training and test data using the provided utility functions
        self.trainX, self.trainY, self.testX, self.testY = fnn_utils.read_data()
        # Number of layers in the network
        self.L = len(network_shape)

        self.crossings = [(1 if i < 1 else network_shape[i-1],network_shape[i]) for i in range(self.L)]

        # Create the network
        self.a             = [np.zeros(m) for m in network_shape]
        self.db            = [np.zeros(m) for m in network_shape]
        self.b             = [np.random.normal(0,1/10,m) for m in network_shape]
        self.z             = [np.zeros(m) for m in network_shape]
        self.delta         = [np.zeros(m) for m in network_shape]
        self.w             = [np.random.uniform(-1/np.sqrt(m0),1/np.sqrt(m0),(m1,m0)) for (m0,m1) in self.crossings]
        self.dw            = [np.zeros((m1,m0)) for (m0,m1) in self.crossings]
        self.nabla_C_out   = np.zeros(network_shape[-1])

        self.network_shape = network_shape
        self.starttime = time.time()

        # Choose activation function
        self.phi           = tanh
        self.phi_d         = tanh_d

        # Store activations over the batch for plotting
        self.batch_a       = [np.zeros(m) for m in network_shape]

    # Added function: Forward propagation for a single layer
    def forward_single(self, A_old, W_layer, B_layer):
        if np.shape(W_layer) == (784,1):

            Z_layer = np.matmul(np.transpose(W_layer), A_old) + B_layer
        else:
            Z_layer = np.matmul(W_layer, A_old) + B_layer
        A_layer = self.phi(Z_layer)

        return A_layer, Z_layer

    # Original function: Forward propagation for the entire Neural Network
    def forward(self, x):
        """ Set first activation in input layer equal to the input vector x (a 24x24 picture),
            feed forward through the layers, then return the activations of the last layer.
        """
        self.a[0] = (x/255.0) - 0.5      # Center the input values between [-0.5,0.5]

        for l in range(1,self.L):
            # Z_current = Sum(W_current x A_previous) + B_current
            self.z[l] = np.matmul(self.w[l], self.a[l-1]) + self.b[l]

            if l == self.L-1:
                self.a[l] = self.softmax(self.z[l])
            else:
                self.a[l] = self.phi(self.z[l])


        return self.a[self.L-1]

    def softmax(self,z):
        Q_i = np.exp(z)
        return Q_i / np.sum(Q_i)

    def loss(self, pred, y):
        target_idx = np.argmax(y)
        return -np.log(pred[target_idx])

    def backward(self,x, y):
        """ Compute local gradients, then return gradients of network.
        """
        # Compute the local gradient for output layer (Set the last layer error)
        for i in range(self.network_shape[-1]):
            self.delta[-1][i] = self.a[-1][i] - (np.argmax(y) == i)

        # Backpropagate local gradients for hidden kayers L-1 to 1
        for l in range(self.L-2, 0, -1):    # loop from L-1 to 1
            w_T_delta = np.matmul((self.w[l+1].transpose()),self.delta[l+1]) # w(l+1)T delta(l+1)

            self.delta[l] = self.phi_d(self.z[l]) * w_T_delta # delta(l) = {w(l+1) delta(l+1)} hadamard_prod phi_d{z(l)}
        # Return the partial derivatives
        for l in range(1,self.L):

            self.db[l] = self.delta[l]

            self.dw[l] = np.matmul(self.delta[l].reshape(len(self.delta[l]), 1), self.a[l-1].reshape(1, len(self.a[l-1])))
        pass

    # Return predicted image class for input x
    def predict(self, x):
        res = self.forward(x)
        return np.argmax(res)

    # Return predicted percentage for class j
    def predict_pct(self,j):
        return self.a[-1][j]


    def evaluate(self, X, Y, N):
        """ Evaluate the network on a random subset of size N. """
        num_data = min(len(X),len(Y))
        samples = np.random.randint(num_data,size=N)
        results = [(self.predict(x), np.argmax(y)) for (x,y) in zip(X[samples],Y[samples])]
        return sum([x==y for (x,y) in results])/N


    def sgd(self,
            batch_size=50,
            epsilon=0.01,
            epochs=1000):

        """ Mini-batch gradient descent on training data.

            batch_size: number of training examples between each weight update
            epsilon:    learning rate
            epochs:     the number of times to go through the entire training data
        """

        # Overwrite
        batch_size=5
        epsilon=0.01
        epochs=150

        # Compute the number of training examples and number of mini-batches.
        N = min(len(self.trainX), len(self.trainY))
        num_batches = int(N/batch_size)

        # Variables to keep track of statistics
        loss_log      = []
        test_acc_log  = []
        train_acc_log = []

        timestamp = time.time()
        timestamp2 = time.time()

        predictions_not_shown = True

        # In each "epoch", the network is exposed to the entire training set.
        for t in range(epochs):
            print("epoch ", t)
            # We will order the training data using a random permutation.

            permutation = np.random.permutation(N)
            # Evaluate the accuracy on 1000 samples from the training and test data
            test_acc_log.append( self.evaluate(self.testX, self.testY, 1000) )
            train_acc_log.append( self.evaluate(self.trainX, self.trainY, 1000))
            batch_loss = 0

            print(test_acc_log[-1])

            for k in range(num_batches):
                # Reset buffer containing updates
                self.dw = [np.zeros((m1,m0)) for (m0,m1) in self.crossings]
                self.db = [np.zeros(m) for m in self.network_shape]
                self.delta = [np.zeros(m) for m in self.network_shape]
                self.z = [np.zeros(m) for m in self.network_shape]
                self.a = [np.zeros(m) for m in self.network_shape]

                # Mini-batch loop
                for i in range(batch_size):
                    # Select the next training example (x,y)
                    x = self.trainX[permutation[k*batch_size+i]]
                    y = self.trainY[permutation[k*batch_size+i]]

                    # Feed forward inputs
                    x_pred = self.predict(x)

                    # Compute gradients
                    self.backward(x_pred, y)

                    # Update losgis log
                    batch_loss += self.loss(self.a[self.L-1], y)

                    for l in range(self.L):
                        self.batch_a[l] += self.a[l] / batch_size

                # Update the weights at the end of the mini-batch using gradient descent
                for l in range(1,self.L):
                    self.w[l] -= epsilon*self.dw[l]
                    self.b[l] -= epsilon*self.db[l]

                # Update logs
                loss_log.append( batch_loss / batch_size )
                batch_loss = 0

                # Update plot of statistics every 10 seconds.
                if time.time() - timestamp > 10:
                    timestamp = time.time()
                    fnn_utils.plot_stats(self.batch_a,
                                         loss_log,
                                         test_acc_log,
                                         train_acc_log)
                    with open("stats.txt", "a") as outfile:
                        outfile.write(
                            "Epoch: " + str(t) +
                            "\nTime elapsed: " + str(time.time()- self.starttime) + " seconds" +
                            "\nLoss: " + str(loss_log[-1]) +
                            "\nTest Accuracy: " + str(test_acc_log[-1]) +
                            "\nTrain Accuracy: " + str(train_acc_log[-1])+"\n\n")

                # Display predictions every 20 seconds.
                if (time.time() - timestamp2 > 20) or predictions_not_shown:
                    predictions_not_shown = False
                    timestamp2 = time.time()
                    fnn_utils.display_predictions(self,show_pct=True)

                # Reset batch average
                for l in range(self.L):
                    self.batch_a[l].fill(0.0)

# Start training with default parameters.

def main():
    if os.path.exists("stats.png"):
        os.remove("stats.png")
    if os.path.exists("predictions.png"):
        os.remove("predictions.png")
    if os.path.exists("stats.txt"):
        os.remove("stats.txt")

    bp = BackPropagation()
    bp.sgd(epsilon=0.0067)

if __name__ == "__main__":
    main()
