
import numpy as np

def generate_digit_data(num_samples=1000):
    # Generate synthetic digit data (0-9)
    X = np.random.randn(num_samples, 64) * 0.1
    y = np.zeros((num_samples, 10))
    
    for i in range(num_samples):
        digit = i % 10
        # Create simple patterns for digits
        if digit == 0:
            X[i, :32] = 1.0
        elif digit == 1:
            X[i, 32:] = 1.0
        elif digit == 2:
            X[i, ::2] = 1.0
        # ... and so on for other digits
        
        y[i, digit] = 1
    
    return X, y
