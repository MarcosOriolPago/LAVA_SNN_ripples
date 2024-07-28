# Liset ripples example usage

## Create a liset_tk instance
Liset_tk object has all the functionalities for loading, visualizing and doing predictions on the data. <br><br>
#### <span style="color:#24938b;">***Class***</span> `liset_tk()`
- **Inputs**
    - *data_path:* should be the path to the folder where the **.dat** file and **info.mat** live in. Optionally it can have a .csv file with the ground truth events of the desired signals. 
    - *shank:* The shank of electrodes chosen to get from the recording. _(Only one shank at a time)_
    - *downsample:* Downsamples the data from its original FS to **1250Hz**
    - *numSamples (Optional):* If reading a dataset too big, it can be only loaded X number of samples with this parameter.


```python
# Import the class from the liset_tk.py file
import sys
from liset_tk import liset_tk

# Define the path to your data
path = r'../../data/CNN_TRAINING_SESSIONS/Amigo2_1_hippo_2019-07-11_11-57-07_1150um' # Modify this to your data path folder
liset = liset_tk(data_path=path, shank=3, downsample=1250, verbose=True, numSamples=1000000)
```

    WARNING:tensorflow:From c:\Users\marco\miniconda3\envs\lava_snn_ripples\lib\site-packages\keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.
    
    Downsampling data from 30000 Hz to 1250 Hz... Done!
    Normalizing data... Done!
    Shape of loaded data after downsampling and z-score:  (41667, 8)
    

## Data visualization
### Overview data features


```python
# Explore the loaded data.

print(f'Original Sampling Frequency: {liset.original_fs} Hz')
print(f'Current Sampling Frequency: {liset.fs} Hz')
print(f'Shape of the loaded data: {liset.data.shape}')
print(f'Duration of the loaded data: {liset.duration} seconds')
print('\n')
if liset.has_ripples:
    print(f'Number of loaded GT ripples: {len(liset.ripples_GT)}')
    print(f'Overview of the ripples:\n\n{liset.ripples_GT[0:5]}\n...')
```

    Original Sampling Frequency: 30000 Hz
    Current Sampling Frequency: 4000 Hz
    Shape of the loaded data: (26667, 8)
    Duration of the loaded data: 6.66675 seconds
    
    
    Number of loaded GT ripples: 5
    Overview of the ripples:
    
    [[ 6667  6832]
     [ 7131  7327]
     [ 5400  5565]
     [ 4758  5062]
     [14450 14632]]
    ...
    

## Visualize the data

#### <span style="color:#24938b;">***function***</span> `liset.plot_event()`
- **Inputs**
    - *time_window*: ***tuple*** --> (start, stop)
    - *offset*: ***float***, default: ***0***
    - *extend*: ***bool***, default: ***False***
    - *delimiter*: ***bool***, default: ***False***
    - *show*: ***bool***, default: ***True***
    - *filtered*: ***iterable***, default: ***[]***
    - *title*: ***str***, default: ***''***
    - *label*: ***str***, default: ***''***
    - *show_ground_truth*: ***bool***, default: ***False***
    - *show_predictions*: ***bool***, default: ***False***


```python
# Plot the loaded channels in a time window.
window = [4000, 6000]

# Play with the offset and extend parameters to zoom in and out of the data.
offset = 3
extend = 100
liset.plot_event(window, 
                 offset=offset, 
                 extend=extend, 
                 show_ground_truth=True, 
                 label='Ripple Activity',
                 title='Ripple Activity in the Hippocampus'
                    )
```


    
![png](img/output_7_0.png)
    


## Load CNN or SNN network models
(Other models to be implemented... ) <br><br>
- CNN model folder should have this structure: <br>
<img src="img/model_structure.png" alt="Model Structure" width="600"/>

- parent path of the SNN trained model should be inserted with ```sys.path.insert()``` method.

#### <span style="color:#24938b;">***function***</span> `liset.load_model()`
- **Inputs**
    - *model_path:* ***str***

Load *CNN* model <br>
As the model was trained in 1250 Hz, we have to load the data at that frequency


```python
# Put the correct path to your tensorflow model
liset = liset_tk(data_path=path, shank=3, downsample=1250, verbose=True, start=1000000, numSamples=1000000)
model_path = '../runSNN/CNN_vs_SNN/CNN/' # Change this to your path
liset.load_model(model_path)

# Overview the structure of the loaded model.
# print(liset.model.summary()) # See the layers of the model
```

    Downsampling data from 30000 Hz to 1250 Hz... Done!
    Normalizing data... Done!
    Shape of loaded data after downsampling and z-score:  (41667, 8)
    Loading model... 
    Running on:  CPU
    Done!
    

Alternatively, load SNN model <br>
As the model was trained in 4000 Hz, we have to load the data at that frequency


```python
sys.path.insert(0, '../trainSNN')
model_path = '../trainSNN/optimized_model/pytorchNetwork/256_128/E50_Y50/network.pt'
liset.load_model(model_path)
```

    Loading model... 
    Running on:  CPU
    Done!
    

## Predict with the loaded model

#### <span style="color:#24938b;">***function***</span> `liset.predict()`
- **Inputs**
    - *threshold (Default=0.7):* ***float***

Use the <span style="color:#3498db;">***predict function***</span> to predict the ripple times. 
The function gets the input size of the loaded model and prepares the data to iterate all through, so the number of predictions will depend on the model input size and the lenght of the data.

**CNN**


```python
# The CNN model has a threshold for deterining whether a ripple is detected or not.
liset.predict(threshold=0.3)
print(liset.prediction_times)
```

    66/66 [==============================] - 0s 3ms/step
    [[ 1.328   1.3912]
     [ 1.792   1.8392]
     [ 7.792   7.8232]
     [ 8.912   8.9432]
     [ 9.312   9.3432]
     [ 9.92    9.9512]
     [10.032  10.0792]
     [10.48   10.5112]
     [11.168  11.2312]
     [11.376  11.4072]
     [11.616  11.6792]
     [15.344  15.3752]
     [16.     16.0472]
     [17.728  17.7912]
     [18.272  18.3032]
     [18.832  18.8792]
     [19.68   19.7272]
     [19.728  19.7752]
     [19.856  19.8872]
     [20.592  20.6232]
     [21.056  21.0872]
     [25.408  25.4392]
     [27.056  27.0872]
     [27.6    27.6632]
     [27.76   27.8072]
     [28.32   28.3512]
     [29.136  29.1992]
     [33.024  33.0872]]
    

**SNN**


```python
liset.predict()
print(liset.prediction_times)
```

    [[ 1.3185   1.38125]
     [ 1.662    1.723  ]
     [ 1.772    1.83175]
     [ 3.5975   3.65675]
     [ 9.56975  9.60775]
     [ 9.75475  9.81325]
     [ 9.9185   9.977  ]
     [10.0325  10.085  ]
     [10.31    10.36325]
     [10.4725  10.53275]
     [10.67075 10.70725]
     [11.11575 11.161  ]
     [11.171   11.2355 ]
     [11.47825 11.52975]
     [11.62    11.67475]
     [11.83075 11.86525]
     [15.24725 15.38975]
     [15.99425 16.06   ]
     [16.731   16.78975]
     [17.068   17.12775]
     [17.7215  17.783  ]
     [17.976   18.02475]
     [18.26975 18.32075]
     [18.73475 18.78675]
     [19.1545  19.2195 ]
     [19.5645  19.61675]
     [19.66    19.729  ]
     [19.82825 19.88625]
     [20.522   20.58325]
     [23.1585  23.218  ]
     [27.583   27.65025]
     [27.70925 27.7905 ]
     [28.2185  28.266  ]
     [28.31    28.3595 ]
     [28.528   28.58425]
     [28.745   28.7985 ]
     [28.88525 28.9435 ]
     [29.13525 29.19125]
     [32.9335  32.989  ]
     [33.01225 33.08375]]
    

## Plot the predictions on the data

#### <span style="color:#24938b;">***use***</span> `liset.plot_event()` 
Use the functon ```plot_event```
<br><br>

Overview the accuracy of the model compared to the ground truth events.


```python
liset.plot_event(
    event=[13000, 16000], 
    offset=offset, 
    extend=extend, 
    show_ground_truth=True, 
    show_predictions=True,
    label='Ripple Activity',
    title='Ripple Activity in the Hippocampus'
)
```


    
![png](img/output_21_0.png)
    


## Check the performance of the model

#### <span style="color:#24938b;">***function***</span> `liset.evaluate()`
- **Inputs**
    - *preds*: ***iterable***, *default*: ***```None```***
    - *chart*: ***bool*** (whether to show or not the plot) <br>
    - *model_type*: ***str***, *default*: ***''***

Provide the Precision, Recall and F1 Score. If no preds are provided, it will use the obtained from liset.predict().


```python
liset.evaluate(chart=True)
```


    
![png](img/output_24_0.png)
    

