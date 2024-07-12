# extract_Nripples

Extract N-ripples refer to extract the events tagged in each recording session to build a dataset for the training of the model.

Activate the environment:
```bash
conda activate lava_snn_ripples 
```

### Extract the events and save them as a dataset

```bash
python3.9 extract_ripples_for_training.py 4000 "100_250"
```
4000 refers to the sampling frequency to be downsampled to (original 30000), and "100_250" is the bandpass filter, which is set to the ripple band. <br>
This will create a dataset that would be suitable for training a normal network. However, the SNN needs to work in a spike domain, so the data needs to be converted to spikes.

### Convert the data to spikes

```bash
python3.9 convert_dataset_to_neuromorphic.py 50
```
50 is the resoltion of the N-dataset. It will have 50 levels of discretization, meaning that the network will have 50 neurons in the input layer.