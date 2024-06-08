###########################################################################################
#                                                                                         #
#                         Deloped by: Marcos Oriol Pagonabarraga                          #
#                           Contact: marcos.oriol.p@gmail.com                             #
#                                                                                         #
###########################################################################################

#
#  This module is developed for 3 main reasons:
#           - Loading correctly the data from liset labs (Instituto Cajal)
#           - Testing different models.
#           - Visualizing performance and data quality.
#
#

# Suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as kr
from scipy.io import loadmat
from copy import deepcopy
from format_predictions import get_predictions_indexes
from liset_aux import *
from load_data import *
from signal_aid import *




class liset_tk():
    """
    Class for handling data processing and visualization related to ripple events.

    Parameters:
    - data_path (str): Path to the directory containing the data.
    - shank (int): Shank of the electrode.
    - downsample (bool, optional): Whether to downsample the data. Default is False.
    - normalize (bool, optional): Whether to normalize the data. Default is True.
    - numSamples (int, optional): Number of samples. Default is False.
    - verbose (bool, optional): Whether to display verbose output. Default is True.
    """
     
    def __init__(self, data_path, shank, downsample = False, normalize = True, numSamples = False, start = 0, verbose=True, original_fs=30000):

        # Set the verbose
        self.verbose = verbose
        self.numSamples = numSamples
        self.start = start
        self.original_fs = original_fs
        if downsample:
            self.downsampled_fs = downsample
            self.fs_conv_fact = self.original_fs/self.downsampled_fs
        else:
            self.fs_conv_fact = 1

        # Initialize class variables.
        self.prediction_times = []
        self.model = None
        self.default_channels = [16, 4, 1, 13, 15, 3, 2, 14]

        # Load the data.
        self.load(data_path, shank, downsample = downsample, normalize=normalize)

        # Try to load the ripples if the path contain a file with ripple times.
        # Only load ripples in the interval selected of data (chunk)
        if hasattr(self, 'fs'):
            self.ripples_GT = self.ripples_in_chunk(load_ripple_times(data_path), start, numSamples, self.fs, self.fs_conv_fact)
        
            if type(self.ripples_GT) is not np.ndarray:
                self.has_ripples = False
                self.has_ripplesGT = False
            else:
                self.has_ripples = True
                self.has_ripplesGT = True
                self.num_ripples = len(self.ripples_GT)
                self.ripples_GT = (self.ripples_GT * self.fs - start / self.fs_conv_fact).astype(int)


    @hide_y_ticks_on_offset
    def plot_event(self, event, offset=0, extend=0, delimiter=False, show=True, filtered=False, title=False, label=False, show_ground_truth=False, show_predictions=False):
        """
        Plot the ripple signal number idx.

        Parameters:
        - idx (int): Index of the ripple to plot.
        - offset (float): Offset between channels for visualization.
        - extend (float): Extend the plotted time range before and after the ripple.
        - delimiter (bool): Whether to highlight the ripple area.

        Returns:
        - fig (matplotlib.figure.Figure): The generated figure.
        - ax (matplotlib.axes.Axes): The axes object containing the plot.
        """
            
        prop = self.fs_conv_fact
        event_copy = deepcopy(event)
        handles = []
        labels = []

        mask = (self.ripples_GT[:, 1] >= event[0]) & (self.ripples_GT[:, 0] <= event[1])
        self.window_ripples = self.ripples_GT[mask]

        try:
            if extend != 0:
                if (event_copy[0] - extend) < 0:
                    event_copy[0] = int(self.start / prop)
                else:
                    event_copy[0] = event[0] - extend

                if (event_copy[1] + extend) > self.numSamples/prop:
                    event_copy[1] = int((self.start + self.numSamples)/prop)
                else:
                    event_copy[1] = event_copy[1] + extend

        except IndexError:
            print('IndexError')
            print(f'There no data available for the selected samples.\nLength of loaded data: {int(self.numSamples/self.fs_conv_fact)}')
            return None, None

        data_event = self.data[event_copy[0]: event_copy[1]][:]
        self.window = deepcopy(data_event)
        
        time_vector = np.linspace(event_copy[0] / self.fs, event_copy[1] / self.fs, data_event.shape[0])
        if show:
            fig, ax = plt.subplots(figsize=(10, 6))
        for i, chann in enumerate(data_event.transpose()):
            if filtered:
                bandpass = filtered
                chann = bandpass_filter(chann, bandpass, self.fs)
                self.window[:, i] = chann
            if show:
                ax.plot(time_vector, chann + i * offset)

        max_val = np.max(self.window.reshape((self.window.shape[0]*self.window.shape[1]))) + offset*8
        min_val = np.min(self.window.reshape((self.window.shape[0]*self.window.shape[1])))

        if delimiter and show:
            if extend > 0:
                ripple_area = [time_vector[round(extend)], time_vector[-round(extend)]]
                if not label:
                    label='Event area'
                fill_DEL = ax.fill_between(ripple_area, min_val, max_val, color="tab:blue", alpha=0.2)
                handles.append(fill_DEL)
                labels.append(label)
            else:
                if self.verbose:
                    print('Delimiter not applied because there is no extend.')

        if show_ground_truth:
            if hasattr(self.ripples_GT, 'dtype'):
                for ripple in self.window_ripples:
                    fill_GT = ax.fill_between([ripple[0] / self.fs, ripple[1] / self.fs],  min_val, max_val, color="tab:red", alpha=0.3)

            if 'fill_GT' in locals():
                handles.append(fill_GT)
                labels.append('Ground truth')

        if show_predictions:
            if hasattr(self, 'prediction_samples_from_window'):
                for times in self.prediction_samples_from_window:
                    fill_PRED = ax.fill_between([(times[0] + event_copy[0]) / self.fs, (times[1] + event_copy[0]) / self.fs],  min_val, max_val, color="tab:blue", alpha=0.3)

            if 'fill_PRED' in locals():
                handles.append(fill_PRED)
                labels.append(f'{self.model_type} predict')

        # Figure styles
        if filtered and not title:
            title = f'Filtered channels\nEvent {event_copy}\nBandpass: {bandpass[0]}-{bandpass[1]}'
        if not title:
            title = f'Channels for samples {event_copy}'

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (mV)')
        if not len(handles) == 0:        
            ax.legend(handles, labels)

        text = ax.set_title(title, loc='center', fontfamily='serif', fontsize=12, fontweight='bold')
        ax.grid(True)
        self.fig = fig
        self.ax = ax

        if show:
            return fig, ax
        
    
    def load_predictions(self, preds):
        self.prediction_samples_from_window = preds
    

    def ripples_in_chunk(self, ripples, start, numSamples, fs, prop):
        if not numSamples:
            numSamples = self.file_samples - self.start

        in_chunk = ripples[(ripples[:,0] > start/prop/fs) & (ripples[:,0] < (start + numSamples)/prop/fs)]

        return in_chunk


    def load_dat(self, path, channels, numSamples = False, verbose=False):
        """
        Load data from a .dat file.

        Parameters:
        - path (str): Path to the directory containing the .dat file.
        - channels (list): List of channel IDs to load.
        - numSamples (int, optional): Number of samples to load. Default is False (load all samples).
        - sampleSize (int, optional): Size of each sample in bytes. Default is 2.
        - verbose (bool, optional): Whether to display verbose output. Default is False.

        Returns:
        - data (numpy.ndarray): Loaded data as a NumPy array.
        """

        try:
            filename = f"{path}/{[i for i in os.listdir(path) if i.endswith('.dat')][0]}"
            self.file_len = os.path.getsize(filename=filename)
            self.file_samples = self.file_len / 43 / 2
        except:
            if self.verbose:
                print('.dat file not in path')
            return False
        
        nChannels = len(channels)
        if (len(channels) > nChannels):
            if self.verbose:
                print("Cannot load specified channels (listed channel IDs inconsistent with total number of channels).")
            return False
        
        start = self.start * 43 * 2
        numSamples = self.numSamples * 43 * 2

        if start > self.file_len:
            if self.verbose:
                print(f'the start must be lower than the total file samples.\nTotal file samples: {self.file_samples}')
            return False
        if (numSamples + start) > self.file_len:
            numSamples = self.file_len - start


        if (self.file_len < numSamples) or ((numSamples + self.start) > self.file_len):
            if self.verbose:
                print(f'file has only {self.file_samples} samples')
            return False
            
        with open(filename, "rb") as f:
            # Put the reader at the starting index
            f.seek(start)

            if numSamples:
                raw = f.read(numSamples)
            else:
                raw = f.read(self.file_len - self.start)
            data = np.frombuffer(raw, dtype=np.int16)
            data = RAW2ORDERED(data, channels)
            return data
            

    def load(self, data_path, shank, downsample, normalize):
        """
        Load all, optionally downsample and normalize it.

        Parameters:
        - data_path (str): Path to the data directory.
        - shank (int): Shank of the electrode.
        - downsample (float): Downsample factor.
        - normalize (bool): Whether to normalize the data.
    
        Returns:
        - data (numpy.ndarray): Loaded and processed data.
        """

        try:
            info = loadmat(f'{data_path}/info.mat')
        except:
            try:
                info = loadmat(f'{data_path}/neurospark.mat')
            except:
                print('.mat file cannot be opened or is not in path.')
                return
            
        try:
            channels = info['neurosparkmat']['channels'][0][0][8 * (shank -1):8 * shank]
        except Exception as err:
            print(f'No data available for shank {shank}\n\n{err}')
            return 
        raw_data = self.load_dat(data_path, channels, numSamples=self.numSamples)
        if hasattr(raw_data, 'shape'):
            self.data = self.clean(raw_data, downsample, normalize)
            self.duration = self.data.shape[0]/self.fs


    def clean(self, data, downsample, normalize):
        """
        Clean the loaded data by downsampling and normalizing it.

        Parameters:
        - data (numpy.ndarray): Raw data to be cleaned.
        - downsample (bool): Whether to downsample the data.
        - normalize (bool): Whether to normalize the data.

        Returns:
        - data (numpy.ndarray): Cleaned data after downsampling and normalization.
        """

        if downsample:
            self.fs = self.downsampled_fs
            # Downsample data
            if self.verbose:
                print("Downsampling data from %d Hz to %d Hz..."%(self.original_fs, self.downsampled_fs), end=" ")
            data = downsample_data(data, self.original_fs, self.downsampled_fs)
            if self.verbose:
                print("Done!")

            if normalize:
                # Normalize it with z-score
                if self.verbose:
                    print("Normalizing data...", end=" ")
                data = z_score_normalization(data)
                if self.verbose:
                    print("Done!")
                    print("Shape of loaded data after downsampling and z-score: ", np.shape(data))
        else:
            self.fs = self.original_fs

        return data


    def load_model(self, model_path, y_samples=50):
        """
        Load a trained tf.keras model from the specified path.

        Parameters:
        - model_path (str): Path to the saved model file.

        Returns:
        - None
        """
        if self.verbose:
            print("Loading model...", end=" ")

        if model_path.endswith('.net'):
            pass

        else:
            optimizer = kr.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
            model = kr.models.load_model(model_path, compile=False)
            model.compile(loss="binary_crossentropy", optimizer=optimizer)
            self.model = model
            self.model_type = 'CNN'

            # Try to get the input shape of the model, otherwise most probable is 0.0128.
            try:
                self.model_window_for_input = self.model.input_shape[1] / self.fs
            except:
                self.model_window_for_input = 0.0128

        if self.verbose:
            print("Done!")



    def predict(self, threshold = 0.7):
        """
        Predict events in the data using the loaded model.

        Parameters:
        - threshold (float, optional): Threshold for event prediction confidence. Default is 0.7.

        Returns:
        - None
        """


        if self.model_type == 'CNN':
            window = self.model_window_for_input
            X = generate_overlapping_windows(self.data, window, window/2, self.fs)
            raw_predictions = self.model.predict(X, verbose=self.verbose)
            self.prediction_idxs = get_predictions_indexes(raw_predictions, window, window/2, self.fs, threshold)
            self.prediction_times = self.prediction_idxs / self.fs

            if len(self.prediction_times) > 0:
                self.has_ripples = True 

        else:
            print('No model loaded.')
            return
        
    def check_performance(self, show=True, criteria=False):
        """
        Check the performance of the ripples prediction compared to ground truth events.

        Parameters:
        - show (bool, optional): Whether to display a pie chart illustrating the performance evaluation. Default is False.

        Returns:
        - None
        """

        if not hasattr(self.prediction_times, 'dtype'):
            print('No predictions found. Use a model to predict ripples and then check the performance.')
            return 
        elif not hasattr(self, 'ripples_GT'):
            print('No Ground truth ripple events could be loaded.')
            return
        else:
            if not hasattr(self.ripples_GT, 'dtype'):
                GT = np.array(self.ripples_GT)
            else:
                GT = self.ripples_GT

            # If the loaded ripples are in a format with 5 columns, then the values are in frames. 
            # So they need to be converted to seconds.
            if GT.shape[1] == 5:
                GT = GT / self.original_fs

            PT = self.prediction_times

            # Only compare with the GT events in the range of loaded data. (Only a part of the dataset may be loaded)
            GT = GT[GT[:, 0] < self.duration]
            # Round the values to get rid of overlapped predictions, and to compare easily if a prediction is correct.
            GT = set([round(i[0], 1) for i in GT])
            PT = set([round(i[0], 1) for i in PT])

            true_positives = len(GT.intersection(PT)) # When a PT has the same value as GT.
            false_positives = len(PT) - true_positives
            missed = len(GT) - true_positives
            labels = ['True positives', 'False positives', 'Missed ripples']
            total = true_positives + false_positives + missed

            val = missed/total * 1.3 + false_positives/total * 0.8
            self.performance = custom_performance_sigmoid(val)

            if show:
                plt.figure(figsize=(8, 8))
                plt.pie([true_positives, false_positives, missed], labels=labels, autopct='%1.1f%%', startangle=140)
                plt.title('Performance Evaluation')
                plt.text(0, 0, f'Performance: {round(self.performance, 2)}', horizontalalignment='center', verticalalignment='center', fontsize=12)
                plt.show()
            
            if criteria:
                print(f"""val = missed_ripples/total * 1.3 + false_positives/total * 0.8\n\n
From liset_aux.py
function custom_performance_sigmoid(val, k=2) --> {self.performance}\n
    return  2 - (1 / (1 + math.exp(-k * x))) * 2\n\n
Where k is the slope parameter (default is 2)\n

                """)

    
    def savefig(self, fname, background=False):
        if fname.endswith('.svg'):
            self.fig.savefig(fname, transparent=not background, format='svg')
        else:
            self.fig.savefig(fname, transparent=not background)
                

