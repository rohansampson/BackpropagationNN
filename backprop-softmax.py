
# Neural Computation (Extended)
# CW1: Backpropagation and Softmax
# Autumn 2020
#

import numpy as np
import time
import fnn_utils

# Some activation functions with derivatives.
# Choose which one to use by updating the variable phi in the code below.

def sigmoid(x):
    return # TODO
def sigmoid_d(x):
    return # TODO
def relu(x):
    return # TODO
def relu_d(x):
    return # TODO
       
class BackPropagation:

    # The network shape list describes the number of units in each
    # layer of the network. The input layer has 784 units (28 x 28
    # input pixels), and 10 output units, one for each of the ten
    # classes.

    def __init__(self,network_shape=[784,20,20,20,10]):

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

        # Choose activation function
        self.phi           = relu
        self.phi_d         = relu_d
        
        # Store activations over the batch for plotting
        self.batch_a       = [np.zeros(m) for m in network_shape]
                
    def forward(self, x):
        """ Set first activation in input layer equal to the input vector x (a 24x24 picture), 
            feed forward through the layers, then return the activations of the last layer.
        """
        self.a[0] = x - 0.5      # Center the input values between [-0.5,0.5]
        # TODO
        
        return(self.a[self.L-1])

    def softmax(self, z):
        # TODO        
        return ... 

    def loss(self, pred, y):
        # TODO
    
    def backward(self,x, y):
        """ Compute local gradients, then return gradients of network.
        """
        # TODO

    # Return predicted image class for input x
    def predict(self, x):
        return # TODO

    # Return predicted percentage for class j
    def predict_pct(self, j):
        return # TODO 
    
    def evaluate(self, X, Y, N):
        """ Evaluate the network on a random subset of size N. """
        num_data = min(len(X),len(Y))
        samples = np.random.randint(num_data,size=N)
        results = [(self.predict(x), np.argmax(y)) for (x,y) in zip(X[samples],Y[samples])]
        return sum(int(x==y) for (x,y) in results)/N

    
    def sgd(self,
            batch_size=50,
            epsilon=0.01,
            epochs=1000):

        """ Mini-batch gradient descent on training data.

            batch_size: number of training examples between each weight update
            epsilon:    learning rate
            epochs:     the number of times to go through the entire training data
        """
        
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

            # We will order the training data using a random permutation.
            permutation = np.random.permutation(N)
            
            # Evaluate the accuracy on 1000 samples from the training and test data
            test_acc_log.append( self.evaluate(self.testX, self.testY, 1000) )
            train_acc_log.append( self.evaluate(self.trainX, self.trainY, 1000))
            batch_loss = 0

            for k in range(num_batches):
                
                # Reset buffer containing updates
                # TODO
                
                # Mini-batch loop
                for i in range(batch_size):

                    # Select the next training example (x,y)
                    x = self.trainX[permutation[k*batch_size+i]]
                    y = self.trainY[permutation[k*batch_size+i]]

                    # Feed forward inputs
                    # TODO
                    
                    # Compute gradients
                    # TODO

                    # Update loss log
                    batch_loss += self.loss(self.a[self.L-1], y)

                    for l in range(self.L):
                        self.batch_a[l] += self.a[l] / batch_size
                                    
                # Update the weights at the end of the mini-batch using gradient descent
                for l in range(1,self.L):
                    self.w[l] = # TODO
                    self.b[l] = # TODO
                
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
    bp = BackPropagation()
    bp.sgd()

if __name__ == "__main__":
    main()
    
