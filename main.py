from neural_network import NeuralNetwork
from data_generator import generate_digit_data
import numpy as np

def main():
    # Generate training data
    X_train, y_train = generate_digit_data(1000)

    # Create and train network
    nn = NeuralNetwork(input_size=64, hidden_size=128, output_size=10)

    # Training loop
    epochs = 100
    batch_size = 32

    # Lists to store metrics
    train_losses = []
    train_accuracies = []
    
    print("Starting training...")
    print("=" * 50)
    
    for epoch in range(epochs):
        epoch_losses = []
        epoch_accuracies = []
        
        # Training loop
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # Forward pass
            predictions = nn.forward(X_batch)
            # Backward pass with metrics
            loss, accuracy = nn.backward(X_batch, y_batch)
            
            epoch_losses.append(loss)
            epoch_accuracies.append(accuracy)

        # Calculate average metrics for the epoch
        avg_loss = np.mean(epoch_losses)
        avg_accuracy = np.mean(epoch_accuracies)
        
        train_losses.append(avg_loss)
        train_accuracies.append(avg_accuracy)

        # Print detailed progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{epochs}")
            print(f"├── Loss: {avg_loss:.4f}")
            print(f"├── Accuracy: {avg_accuracy:.4f}")
            print(f"└── Processed {(epoch+1)*len(X_train)} samples")
            print("-" * 30)
    
    print("\nTraining completed!")
    print(f"Final accuracy: {train_accuracies[-1]:.4f}")
    print(f"Final loss: {train_losses[-1]:.4f}")

if __name__ == "__main__":
    main()

"""
# 2 Layer Neural Network in NumPy



import numpy as np

# X = input of our 3 input XOR gate
# set up the inputs of the neural network (right from the table)
X = np.array(([0,0,0],[0,0,1],[0,1,0], \
    [0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]), dtype=float)
# y = our output of our neural network
y = np.array(([1], [0],  [0],  [0],  [0], \
     [0],  [0],  [1]), dtype=float)

# what value we want to predict
xPredicted = np.array(([0,0,1]), dtype=float)

X = X/np.amax(X, axis=0) # maximum of X input array
# maximum of xPredicted (our input data for the prediction)
xPredicted = xPredicted/np.amax(xPredicted, axis=0) 

# set up our Loss file for graphing

lossFile = open("SumSquaredLossList.csv", "w")

class Neural_Network (object):
  def __init__(self):
    #parameters
    self.inputLayerSize = 3  # X1,X2,X3 
    self.outputLayerSize = 1 # Y1
    self.hiddenLayerSize = 4 # Size of the hidden layer

    # build weights of each layer
    # set to random values
    # look at the interconnection diagram to make sense of this
    # 3x4 matrix for input to hidden
    self.W1 = \
            np.random.randn(self.inputLayerSize, self.hiddenLayerSize) 
    # 4x1 matrix for hidden layer to output
    self.W2 = \
            np.random.randn(self.hiddenLayerSize, self.outputLayerSize) 

  def feedForward(self, X):
    # feedForward propagation through our network
    # dot product of X (input) and first set of 3x4  weights
    self.z = np.dot(X, self.W1) 

    # the activationSigmoid activation function - neural magic
    self.z2 = self.activationSigmoid(self.z) 

    # dot product of hidden layer (z2) and second set of 4x1 weights
    self.z3 = np.dot(self.z2, self.W2) 

    # final activation function - more neural magic
    o = self.activationSigmoid(self.z3) 
    return o

  def backwardPropagate(self, X, y, o):
    # backward propagate through the network
    # calculate the error in output
    self.o_error = y - o 

    # apply derivative of activationSigmoid to error
    self.o_delta = self.o_error*self.activationSigmoidPrime(o) 

    # z2 error: how much our hidden layer weights contributed to output error
    self.z2_error = self.o_delta.dot(self.W2.T) 

    # applying derivative of activationSigmoid to z2 error
    self.z2_delta = self.z2_error*self.activationSigmoidPrime(self.z2) 

    # adjusting first set (inputLayer --> hiddenLayer) weights
    self.W1 += X.T.dot(self.z2_delta) 
    # adjusting second set (hiddenLayer --> outputLayer) weights 
    self.W2 += self.z2.T.dot(self.o_delta) 

  def trainNetwork(self, X, y):
    # feed forward the loop
    o = self.feedForward(X)
    # and then back propagate the values (feedback)
    self.backwardPropagate(X, y, o)


  def activationSigmoid(self, s):
    # activation function
    # simple activationSigmoid curve as in the book
    return 1/(1+np.exp(-s))

  def activationSigmoidPrime(self, s):
    # First derivative of activationSigmoid
    # calculus time!
    return s * (1 - s)


  def saveSumSquaredLossList(self,i,error):
    lossFile.write(str(i)+","+str(error.tolist())+'\n')

  def saveWeights(self):
    # save this in order to reproduce our cool network
    np.savetxt("weightsLayer1.txt", self.W1, fmt="%s")
    np.savetxt("weightsLayer2.txt", self.W2, fmt="%s")

  def predictOutput(self):
    print ("Predicted XOR output data based on trained weights: ")
    print ("Expected (X1-X3): \n" + str(xPredicted))
    print ("Output (Y1): \n" + str(self.feedForward(xPredicted)))

myNeuralNetwork = Neural_Network()
trainingEpochs = 1000
trainingEpochs = 100000

for i in range(trainingEpochs): # train myNeuralNetwork 1,000 times
  print ("Epoch # " + str(i) + "\n")
  print ("Network Input : \n" + str(X))
  print ("Expected Output of XOR Gate Neural Network: \n" + str(y))
  print ("Actual  Output from XOR Gate Neural Network: \n" + \
          str(myNeuralNetwork.feedForward(X)))
  # mean sum squared loss
  Loss = np.mean(np.square(y - myNeuralNetwork.feedForward(X))) 
  myNeuralNetwork.saveSumSquaredLossList(i,Loss)
  print ("Sum Squared Loss: \n" + str(Loss))
  print ("\n")
  myNeuralNetwork.trainNetwork(X, y)

myNeuralNetwork.saveWeights()
myNeuralNetwork.predictOutput()
"""

