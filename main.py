from neural_network import NeuralNetwork
from data_generator import generate_digit_data
import numpy as np
import os

def color_text(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"

def cool_progress_bar(progress, total, width=30):
    filled = int(width * progress // total)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{color_text(bar, '36')}] {progress}/{total}"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_menu():
    print(color_text("\n=== 🤖 Neural Network Learning Lab! ===", "33"))
    print("\nChoose your experiment:")
    
    print(color_text("\n1. AND Gate", "32"))
    print(color_text("\n2. OR Gate", "32"))
    print(color_text("\n3. XOR Gate", "35"))
    print(color_text("\n4. Number Recognition", "36"))
    print(color_text("\n0. Exit", "31"))
    print("=" * 50)

def run_digit_classifier():
    print("\n=== Welcome to the Number Recognition Demo! ===")
    print("\nWhat's happening here:")
    print("1. We're teaching the computer to recognize numbers")
    print("2. It will look at patterns that represent digits 0-9")
    print("3. Over time, it will get better at guessing which number it sees")
    print("\nWatch these numbers as the computer learns:")
    print("- Accuracy: Higher is better (1.0 = perfect)")
    print("- Loss: Lower is better (0.0 = perfect)")
    input("\nReady to start? Press Enter to begin...")

    # Generate training data
    X_train, y_train = generate_digit_data(1000)
    nn = NeuralNetwork(input_size=64, hidden_size=128, output_size=10)

    epochs = 100
    batch_size = 32
    train_losses = []
    train_accuracies = []

    print("\nStarting training...")
    print("=" * 50)

    print(color_text("\n🚀 Training started! Hold tight...", "32"))
    for epoch in range(epochs):
        epoch_losses = []
        epoch_accuracies = []

        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            predictions = nn.forward(X_batch)
            loss, accuracy = nn.backward(X_batch, y_batch)
            epoch_losses.append(loss)
            epoch_accuracies.append(accuracy)

        avg_loss = np.mean(epoch_losses)
        avg_accuracy = np.mean(epoch_accuracies)
        train_losses.append(avg_loss)
        train_accuracies.append(avg_accuracy)

        if epoch % 10 == 0:
            clear_screen()
            print(color_text(f"🤖 Training Progress - Epoch {epoch}/{epochs}", "33"))
            print(cool_progress_bar(epoch, epochs))
            print(f"\n📊 Metrics:")
            print(color_text(f"├── 📉 Loss: {avg_loss:.4f}", "35"))
            print(color_text(f"├── 📈 Accuracy: {avg_accuracy:.4f}", "32"))
            print(color_text(f"└── 🔄 Processed {(epoch+1)*len(X_train)} samples", "36"))
            print("\n" + "✨" * 20)

    print("\nTraining completed!")
    print(f"Final accuracy: {train_accuracies[-1]:.4f}")
    print(f"Final loss: {train_losses[-1]:.4f}")

def run_xor_gate():
    print("\n=== Welcome to the Logic Puzzle Learning Demo! ===")
    print("\nWhat's happening here:")
    print("1. We're teaching the computer a simple logic puzzle")
    print("2. The computer will try to learn when to answer 'yes' or 'no'")
    print("3. You'll see it practice and get better over time!")
    print("\nDon't worry about the numbers you'll see - just watch how")
    print("the 'Loss' number gets smaller as the computer learns!")
    input("\nReady to start? Press Enter to begin...")

    X = np.array(([0,0,0],[0,0,1],[0,1,0],[0,1,1],
                  [1,0,0],[1,0,1],[1,1,0],[1,1,1]), dtype=float)
    y = np.array(([1],[0],[0],[0],[0],[0],[0],[1]), dtype=float)
    xPredicted = np.array(([0,0,1]), dtype=float)

    X = X/np.amax(X, axis=0)
    xPredicted = xPredicted/np.amax(xPredicted, axis=0)

    class XORNeuralNetwork:
        def __init__(self):
            self.inputLayerSize = 3
            self.outputLayerSize = 1
            self.hiddenLayerSize = 4
            self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
            self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

        def sigmoid(self, s):
            return 1/(1+np.exp(-s))

        def sigmoid_prime(self, s):
            return s * (1 - s)

        def forward(self, X):
            self.z = np.dot(X, self.W1)
            self.z2 = self.sigmoid(self.z)
            self.z3 = np.dot(self.z2, self.W2)
            return self.sigmoid(self.z3)

        def backward(self, X, y, o):
            self.o_error = y - o
            self.o_delta = self.o_error * self.sigmoid_prime(o)
            self.z2_error = self.o_delta.dot(self.W2.T)
            self.z2_delta = self.z2_error * self.sigmoid_prime(self.z2)
            self.W1 += X.T.dot(self.z2_delta)
            self.W2 += self.z2.T.dot(self.o_delta)

        def train(self, X, y):
            o = self.forward(X)
            self.backward(X, y, o)
            return o

    nn = XORNeuralNetwork()
    for i in range(1000):
        if i % 100 == 0:
            clear_screen()
            print(color_text("🧠 XOR Neural Network Training", "33"))
            print(cool_progress_bar(i, 1000))
            print(color_text("\n📊 Network Status:", "36"))
            loss = np.mean(np.square(y - nn.forward(X)))
            print(color_text(f"└── 📉 Loss: {loss:.6f}", "35"))
            print("\n" + "✨" * 20)
        nn.train(X, y)

    print("\nTraining Complete!")
    print("Final prediction for [0,0,1]:", nn.forward(xPredicted))

def main():
    while True:
        clear_screen()
        print_menu()
        choice = input("\nEnter your choice (0-4): ")
        
        if choice == '0':
            print(color_text("\nGoodbye! 👋", "31"))
            break
            
        epochs = int(input("\nEnter number of epochs: "))
        save_csv = input("Save results to CSV? (y/n): ").lower() == 'y'
        
        if choice == '1':
            run_logic_gate('AND', epochs=epochs, save_results=save_csv)
        elif choice == '2':
            run_logic_gate('OR', epochs=epochs, save_results=save_csv)
        elif choice == '3':
            run_xor_gate(epochs=epochs, save_results=save_csv)
        elif choice == '4':
            run_digit_classifier(epochs=epochs, save_results=save_csv)
        else:
            print("\nInvalid choice. Please try again.")

        input("\nPress Enter to continue...")

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