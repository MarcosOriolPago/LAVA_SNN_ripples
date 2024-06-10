import sys
sys.path.insert(0, 'liset_tk/')
from liset_aux import load_ripple_times
import os
import numpy as np
import matplotlib.pyplot as plt

parent = 'C:/__NeuroSpark_Liset_Dataset__/neurospark_mat/CNN TRAINING SESSIONS/'
ripples_list = []  # Create an empty list to store arrays

for i in os.listdir(parent):
    ripples_list.append(load_ripple_times(f'{parent}{i}'))  # Append arrays to the list

# Concatenate arrays in the list into a single numpy array
ripples = np.concatenate(ripples_list)

# Duration is the stop - start
durations = ripples[:, 1] - ripples[:, 0]



# PLot the histogram of the ditribution length
# Set up figure and axis
fig, ax = plt.subplots()

# Plot histogram
ax.hist(durations, bins=20, color='#49D197', edgecolor='black')

# Set labels and title
ax.set_xlabel('Ripple Duration (s)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Ripple Durations', fontsize=14)

# Set grid
ax.grid(True, linestyle='--', alpha=0.7)

# Set axis ticks font size
ax.tick_params(axis='both', which='major', labelsize=10)

# Add vertical lines for mean and median
mean_duration = np.mean(durations)
median_duration = np.median(durations)
ax.axvline(mean_duration, color='red', linestyle='--', linewidth=1, label=f'Mean: {mean_duration:.3f}s')
ax.axvline(median_duration, color='green', linestyle='-.', linewidth=1, label=f'Median: {median_duration:.3f}s')

# Add legend for standard deviation
std_dev = np.std(durations)
std_dev_label = f'std: {std_dev:.3f}s'
ax.plot([], [], color='black', linestyle='-', linewidth=1, label=std_dev_label)

# Add legend
ax.legend(fontsize=10)

# Adjust layout to prevent clipping of labels
plt.tight_layout()

# Save or display the plot
plt.savefig('img/ripple_duration_histogram.svg', format='svg', transparent=True)