import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
# import slayer from lava-dl
import lava.lib.dl.slayer as slayer

# Import custom code
from network import RipplesNetwork
from train_aux import torchSlayer_Dataset

# Import subprocess for external commands
import subprocess as sh


# Set general variables
epochs = int(sys.argv[1])
y_size = int(sys.argv[3])
folder = sys.argv[4]
trained_folder = f'{folder}/{sys.argv[2]}/E{epochs}_Y{y_size}/'
os.makedirs(trained_folder, exist_ok=True)

# Assign the runtime method
device = torch.device('cpu')

# Test-Train data preparation
print('Loading data ...', end='')
true_positives = np.load(f'../dataset/{y_size}/n_true_positives.npy')
true_negatives = np.load(f'../dataset/{y_size}/n_true_negatives.npy')

# Create labels for the datasets (1 for true positives, 0 for true negatives)
labels_positives = np.ones(len(true_positives)).astype(int)
labels_negatives = np.zeros(len(true_negatives)).astype(int)

# Concatenate the datasets and labels
data = np.concatenate((true_positives, true_negatives), axis=0)
labels = np.concatenate((labels_positives, labels_negatives))

# Split the dataset into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True)

train = torchSlayer_Dataset(train_data, train_labels)
test = torchSlayer_Dataset(test_data, test_labels)

train_loader = DataLoader(dataset=train, batch_size=32, shuffle=True)
test_loader  = DataLoader(dataset=test , batch_size=32, shuffle=True)
print('Done!')

# Define the structure of the model
# Create network object and training objects
net = RipplesNetwork(y_input_size=y_size, str_layers=sys.argv[2]).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
error = slayer.loss.SpikeRate(true_rate=0.2, false_rate=0.03, reduction='sum').to(device) 
# error = slayer.loss.SpikeTime(time_constant=5, length=100, filter_order=1, reduction='sum').to(device)
stats = slayer.utils.LearningStats()
assistant = slayer.utils.Assistant(net, error, optimizer, stats, classifier=slayer.classifier.Rate.predict)
# assistant = slayer.utils.Assistant(net, error, optimizer, stats) # Regression mode

# Train the model

print('Starting training!')
samples = 0
for epoch in range(epochs):
    samples = 0
    for i, (input, label) in enumerate(train_loader): # training loop
        samples += input.shape[0]
        print(samples, end='\r', flush=True)
        output = assistant.train(input, label)
    print(f'\r[Epoch {epoch:2d}/{epochs}] {stats}', end='')
        
    for i, (input, label) in enumerate(test_loader): # training loop
        output = assistant.test(input, label)
    print(f'\r[Epoch {epoch:2d}/{epochs}] {stats}', end='')
        
    if epoch%20 == 19: # cleanup display
        print('\r', ' '*len(f'\r[Epoch {epoch:2d}/{epochs}] {stats}'))
        stats_str = str(stats).replace("| ", "\n")
        print(f'[Epoch {epoch:2d}/{epochs}]\n{stats_str}')
    
    if stats.testing.best_accuracy:
        net.export_hdf5(trained_folder + '/network.net')
        torch.save(net.state_dict(), trained_folder + '/network.pt')
    stats.update()
    stats.save(trained_folder + '/')
    # stats.plot(figsize=(15, 5), path=trained_folder)
    # net.grad_flow(trained_folder + '/')

    sh.run(['python3.9', 'plot_accuracies.py', folder])
    sh.run(['bash', 'update_git.sh'])

