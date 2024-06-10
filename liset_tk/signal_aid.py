import numpy as np
from scipy.signal import butter, filtfilt
import pandas as pd
import matplotlib.pyplot as plt


def bandpass_filter(signal, bandpass, fs, order=4):
    """
    Filters the signal with butterworth bandpass filter.
    """
    # Definir la frecuencia de corte del filtro pasa-bandas
    low_cutoff = bandpass[0]  # Frecuencia de corte inferior en Hz
    high_cutoff = bandpass[1] # Frecuencia de corte superior en Hz

    # Calcular las frecuencias de corte normalizadas
    nyquist_freq = 0.5 * fs  # Frecuencia de Nyquist para una señal con frecuencia de muestreo de 1000 Hz
    low = low_cutoff / nyquist_freq
    high = high_cutoff / nyquist_freq

    # Diseñar el filtro pasa-bandas de Butterworth
    b, a = butter(order, [low, high], btype='band')

    # Aplicar el filtro a la señal
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal


def most_active_channel(window, chans = 8):
    max_distance = 0

    for i, channel in enumerate(window.transpose()):
        max_amp = np.max(channel)
        min_amp = np.min(channel)
        distance = np.abs(max_amp - min_amp)
        if distance > max_distance:
            max_distance = distance
            best_channel = i

    return window[:, best_channel]


def y_discretize_1Dsignal(signal, y_num_samples, cutoff=1.648):
    """
    Converts the 1D signal into a 2D binary array, like an image representing the shape of the signal.
    """
    discretized_signal = np.zeros((y_num_samples, signal.shape[0]))
    for idx, value in enumerate(signal):
        if np.abs(value) < cutoff:  
            y_val = (y_num_samples - 1) - int(value/cutoff * y_num_samples/2 + y_num_samples/2)
            discretized_signal[y_val, idx] = 1
        else:
            if value < 0:
                y_val = y_num_samples - 1
            else:
                y_val = 0
            discretized_signal[y_val, idx] = 1

    return discretized_signal


def y_discretize_to_compressed(signal, y_num_samples, cutoff=1.648):
    discretized_signal = np.zeros(signal.shape[0])
    for idx, value in enumerate(signal):
        if np.abs(value) < cutoff:  
            y_val = (y_num_samples - 1) - int(value/cutoff * y_num_samples/2 + y_num_samples/2)
            discretized_signal[idx] = y_val
        else:
            if value < 0:
                y_val = y_num_samples - 1
            else:
                y_val = 0
            discretized_signal[idx] = y_val

    return discretized_signal


def compress_2D_signals(spikes):
    return np.argmax(spikes, axis=0)


def discretize_compressed(compressed, y_num_samples=50):
    discretized_signal = np.zeros((y_num_samples, compressed.shape[0]))

    for idx, value in enumerate(compressed):
        discretized_signal[int(value), idx] = 1

    return discretized_signal


def cutoff_amplitude(signals):
    maxAmplitudes = np.max(np.abs(signals), axis=1)
    meanMaxAmpl = np.mean(maxAmplitudes)
    std = np.std(maxAmplitudes)

    return meanMaxAmpl + std


def merge_overlapping_intervals(intervals):
    sorted_indices = np.argsort(intervals[:, 0])
    sorted_intervals = intervals[sorted_indices]
    
    merged_intervals = []
    start, end = sorted_intervals[0]
    
    for interval in sorted_intervals[1:]:
        if interval[0] <= end:
            end = max(end, interval[1])
        else:
            merged_intervals.append([start, end])
            start, end = interval
    
    merged_intervals.append([start, end])
    
    return np.array(merged_intervals).astype(int)


def detect_rate_increase(spike_train, window_size=130, threshold=25):
    spike_count = pd.Series(spike_train).rolling(window=window_size, min_periods=1, center=False).sum()

    # Detect periods where spike count exceeds threshold
    rate_increase_periods = []
    in_increase_period = False
    for i, count in enumerate(spike_count):
        if count > threshold:
            if not in_increase_period:
                start_index = i
                in_increase_period = True
        else:
            if in_increase_period:
                end_index = i
                rate_increase_periods.append((start_index - window_size, end_index))
                in_increase_period = False

    return merge_overlapping_intervals(np.array(rate_increase_periods))


def window_to_spikes(window, y_size=50):
    channel = most_active_channel(window)
    spikes = y_discretize_1Dsignal(channel, y_size)

    return spikes


def scatter(input, save = False, prediction=False, title=None, labels=[], dot_size=8):
    if prediction:
        signal = input
    else:
        signal = np.zeros(input.shape[1])
        for i, row in enumerate(input.transpose()):
            y_val = np.where(row == 1)[0]
            if len(y_val) == 1:
                signal[i] = y_val[0]
            else:
                signal[i] = None

    axis_vector = [i for i in range(signal.shape[0])]
    sizes = [dot_size for i in range(signal.shape[0])]

    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(1, figsize=(10,2), dpi=300)
    plt.scatter(axis_vector, signal, s=sizes)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim([0, signal.shape[0]])
    
    if prediction: 
        ax.set_ylim([-1, 2])
        plt.yticks([0, 1])
    else:
        ax.set_ylim([0, 50])

    if labels:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])

    if title:
        plt.title(title, fontsize=12, fontweight='bold')
    
    if save:
        if save.endswith('.svg'):
            plt.savefig(save, format='svg', transparent=True)
        else:
            plt.savefig(save)
            
    plt.show()