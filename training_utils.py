
import csv
import os
from datetime import datetime
import numpy as np

def save_training_results(metrics, model_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{model_type}_{timestamp}.csv"
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'accuracy'])
        for epoch, (loss, acc) in enumerate(metrics):
            writer.writerow([epoch, loss, acc])
    return filename

def get_training_params():
    print("\n=== Training Configuration ===")
    epochs = int(input("Number of epochs (10-10000): ").strip() or "100")
    batch_size = int(input("Batch size (1-128): ").strip() or "32")
    save_results = input("Save results to CSV? (y/n): ").strip().lower() == 'y'
    return epochs, batch_size, save_results
