import numpy as np
import math
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd



def load_ripple_times(path):
    """
    Loads ripple times from a CSV file located at the specified path.

    Parameters:
    - path (str): The path to the directory containing the ripple CSV file.

    Returns:
    - ripples (numpy.ndarray): A NumPy array containing the ripple times, where each row represents a ripple event  (start, end).
    """

    ripples_file_path = f'{path}/ripples.csv'
    if os.path.exists(ripples_file_path):
        ripples = pd.read_csv(ripples_file_path, sep = ' ').to_numpy()
        if type(ripples[0][0]) is str:
            ripples = pd.read_csv(ripples_file_path, sep = ',').to_numpy()
        return ripples
    else:
        print(f'File {ripples_file_path} does not exist. \nEnsure to put the path where riplpes.csv live in.') 
        return None  
    
    
def RAW2ORDERED(data, channels, num_channels_raw = 43):
    """
    Reorganizes raw data into an ordered format based on the specified channels.

    Parameters:
    - data (numpy.ndarray): The raw data to be reorganized.
    - channels (list): A list of channel IDs to be included in the ordered data. (Use to be the channels of the shank)
    - num_channels_raw (int): The total number of channels in the raw data. Default is 43.

    Returns:
    - ordered (numpy.ndarray): A NumPy array containing the ordered data, where each column represents 
    a channel and each row represents a sample.
    """

    num_channels = len(channels)
    channel_len = int(data.shape[0]/43)
    ordered = np.zeros((channel_len, num_channels))

    for i, chan in enumerate(channels):
        channel_indices = np.arange(chan, int(data.shape[0]), num_channels_raw, dtype=int)
        ordered[:,i] = data[channel_indices]

    return ordered


def hide_y_ticks_on_offset(func, verbose = True):
    """
    Decorator function to hide y-ticks when an offset is applied to signals.

    Parameters:
    - func (function): The function to be decorated.
    - verbose (bool, optional): Whether to print error messages. Default is False.

    Returns:
    - wrapper function
    """

    def wrapper(*args, **kwargs):
        # Call the function and get the figure and axis objects
        if isinstance(kwargs, dict):
            # Call the function and get the figure and axis objects
            try:
                fig, ax = func(*args, **kwargs)
            except Exception as err:
                if verbose:
                    print(err)
                return
        else:
            print('kwargs is not a dict')
        # Check if the offset is nonzero
        offset = kwargs.get('offset', 0)
        filtered = kwargs.get('filtered', 0)

        if offset:
            # Hide the y-ticks
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_ylabel('')
            ax.grid(False)

        plt.show()        

    return wrapper


def custom_performance_sigmoid(x, k=2):
    """
    Computes the performance of the model using a sigmoid function.

    Parameters:
    - x (float): The input value to the sigmoid function. The input must be a sum of missed ripples 
    and false positives related to the total amount of events (expected from ~ 0.1-2)
    - k (float): The slope parameter of the sigmoid function. Default is 2.

    Returns:
    - result (float): The output of the sigmoid function, representing the custom performance metric (from 0 - 1).
    """

    return  2 - (1 / (1 + math.exp(-k * x))) * 2


def middle(ripple, time=False):
    """
    Gives the middle between an event with 2 increasing positions.
    """
    if time:
        return (ripple[0] + ripple[1])/2
    else:
        return int((ripple[0] + ripple[1])/2)


def to_sample(ripple, fs=30000):
    """
    Converts the time values to sample values.
    """
    return [int(ripple[0]*fs), int(ripple[1]*fs)]


def format_training_ripples(ripples, std, mean):
    """
    Converts the values of the ripple positions so that all ripples have the same length and can be 
    used for training.
    The length of the ripples will depend on the mean lean and standard deviation of the ripple population. 
    """
    half_window_size = (std + mean)/2
    formatted_ripples = np.zeros_like(ripples)
    for i, ripple in enumerate(ripples):
        middle_ripple = middle(ripple, time=True)
        formatted_ripples[i, :] = np.array([middle_ripple - half_window_size, middle_ripple + half_window_size])

    return formatted_ripples


def ripples_std(parent_path):
    """
    Reads the ripples.csv file from each dataset in the parent path, and calculates the std and the mean of the ripple lengths.
    """
    ripples_list = []  # Create an empty list to store arrays

    for i in os.listdir(parent_path):
        ripples_list.append(load_ripple_times(f'{parent_path}{i}'))  # Append arrays to the list

    # Concatenate arrays in the list into a single numpy array
    ripples = np.concatenate(ripples_list)

    # Duration is the stop - start
    durations = ripples[:, 1] - ripples[:, 0]

    # Extract features
    mean_duration = np.mean(durations)
    std = np.std(durations)

    return std, mean_duration

