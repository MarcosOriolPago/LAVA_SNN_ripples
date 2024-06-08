import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, 'liset_tk/')
from signal_aid import y_discretize_1Dsignal, cutoff_amplitude
import os

# Load the saved dataset
true_positives = np.load('train/dataset/true_positives.npy')
true_negatives = np.load('train/dataset/true_negatives.npy')
y_size = int(sys.argv[1])


# Define parameters
samples_len = true_positives.shape[1]
y_num_samples = y_size
cutoff = cutoff_amplitude(true_positives)
num_samples = 1794 # length of the true positives

n_true_positives = np.zeros((num_samples, y_num_samples, samples_len))
n_true_negatives = np.zeros((num_samples, y_num_samples, samples_len))

for idx in range(true_positives.shape[0]):
    n_true_positives[idx, :, :] = y_discretize_1Dsignal(true_positives[idx], y_num_samples, cutoff)
    n_true_negatives[idx, :, :] = y_discretize_1Dsignal(true_negatives[idx], y_num_samples, cutoff)  

# Save the arrays
parent = f'train/n_dataset/{y_size}/'
os.makedirs(parent, exist_ok = True)
np.save(f'{parent}n_true_positives.npy', arr=n_true_positives, allow_pickle=True)
np.save(f'{parent}n_true_negatives.npy', arr=n_true_negatives, allow_pickle=True)