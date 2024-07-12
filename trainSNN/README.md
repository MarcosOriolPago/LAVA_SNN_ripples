# trainSNNripples
In this section, the SNN is defined and trained. 
## Training
**Model**: Defined in network.py. <br>
**Training**: Performed by train_model.py <br>
**Main** script orchetsrates the definition of the network and training all in one to make it easier.<br><br>
For the training, define the inner architecture (layer1_layer2_layer3...), and the dataset to use (10, 20, 30...) in the main.sh script and run:

First activate the environment:
```bash
conda activate lava_snn_ripples 
```

Then run the training.
```bash
bash main.sh
```