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

    for epoch in range(epochs):
        # Simple batch training
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # Forward and backward pass
            predictions = nn.forward(X_batch)
            nn.backward(X_batch, y_batch)

        # Print progress
        if epoch % 10 == 0:
            predictions = nn.forward(X_train)
            accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_train, axis=1))
            print(f"Epoch {epoch}, Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()