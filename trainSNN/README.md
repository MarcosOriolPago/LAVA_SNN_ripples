# trainSNNripples

## Walkthough
In this notebook, you will learn to define and train a model using pytorch and SLAYER.

**Imports**

```python
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import tensor
from sklearn.model_selection import train_test_split
# import slayer from lava-dl
import lava.lib.dl.slayer as slayer

# Import custom code
from network import RipplesNetwork
from train_aux import torchSlayer_Dataset
```

**Define variables**

```python
# Set general variables
epochs = 10
y_size = 50
folder = 'trial_output'
trained_folder = f'{folder}/256_128/E{epochs}_Y{y_size}/'
os.makedirs(trained_folder, exist_ok=True)
```

**Assign training on CPU**
```python
# Assign the runtime method
device = torch.device('cpu')
#device = torch.device('cuda')
```


**Prepare data for the training**

```python

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
```

**Define the network (see `network.py`)**

```python
# Define the structure of the model
# Create network object and training objects
net = RipplesNetwork(y_input_size=y_size, str_layers="256_128").to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
error = slayer.loss.SpikeRate(true_rate=0.2, false_rate=0.03, reduction='sum').to(device) 
stats = slayer.utils.LearningStats()
assistant = slayer.utils.Assistant(net, error, optimizer, stats, classifier=slayer.classifier.Rate.predict)
```

**Training Loop**
```python
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
```

**Load the last saved pretrained values**
```python
net.load_state_dict(torch.load(trained_folder + '/network.pt'))
```

## Test the model
**Load the input from the `extract_Nripples` repository**
```python
trial_input = tensor(np.load('../report_util/input_spikes.npy'), dtype=torch.float32)
print(trial_input.shape)
batches = int(trial_input.shape[1] / input.shape[2])
trial_input2network = torch.zeros((batches, input.shape[1], input.shape[2]))

for i in range(batches):
    trial_input2network[i, :, :] = trial_input[:, i*input.shape[2]:(i+1)*input.shape[2]]

print(trial_input2network.shape)
```

**Perform a prediction**
```python
output = net(trial_input2network.to(device))
print(output.shape)
```

## Visualize the output
```python 
predictions = np.zeros((output.shape[1], output.shape[0] * output.shape[2]))
for i, batch in enumerate(output):
    predictions[:, batch.shape[1]*i:batch.shape[1]*(i+1)] = batch.detach().numpy()

print(predictions.shape)
```

**Use builtin `event.anim` method from `lava.lib.dl.slayer` to create animations**

```python
sys.path.insert(0, '../report_util/extract_Nripples/')
from anim import slayer_gif

slayer_gif(predictions, filename='../report_util/gifs/output', figsize=(30, 5), fps=5)
```
<table>
<tr><td align="center"><b>Input</b></td></tr>
<tr><td> <img src="gifs/ripple.gif" alt="Drawing" style="height: 250px;"/> </td></tr>
<tr><td align="center"><b>Output</b></td></tr>
<tr><td> <img src="gifs/output.gif" alt="Drawing" style="height: 250px;"/> </td></tr>
</table>

The network has two outputs neurons:
- **1** will fire when the network predicts a ripple event
- **0** will fire when the network predicts a non-ripple event

As it can be obseerved in the animation, when there is high frequency oscillations (ripples), the network fires at higher frequency in the output neuron 1, otherwise, neuron 0 increases firing frequency.