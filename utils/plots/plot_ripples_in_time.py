import sys
sys.path.insert(0, 'liset_tk/')
from liset_tk import liset_tk
import matplotlib.pyplot as plt
import os
import numpy as np

# Define the path to your data
parent = r'C:\__NeuroSpark_Liset_Dataset__\neurospark_mat\CNN TRAINING SESSIONS' # Modify this to your data path folder

# Collect the sessions
sessions = os.listdir(parent)
num_sessions = len(sessions)
colors = ['#1f77b4', '#23943b']
extend = 1000

# Create subplots
fig, axes = plt.subplots(num_sessions, 1, figsize=(8 * num_sessions, 4))  # Adjust the figsize as needed
plt.style.use('seaborn-v0_8-paper')

for ax, session in zip(axes, sessions):
    path = os.path.join(parent, session)
    liset = liset_tk(data_path=path, shank=3, downsample=4000, verbose=False)

    # Ripple times (np.array each have start-end)
    ripples = liset.ripples_GT
    all_data_points = liset.data.shape

    # Calculate the duration of the recording
    recording_duration = all_data_points[0] / liset.fs

    # Initialize an empty array for bar colors
    bar_colors = np.zeros(int(recording_duration))

    # Update bar colors where ripples occur
    for start, end in ripples:
        bar_colors[int(start / liset.fs) : int(end / liset.fs) + 1] = 1

    color_array = np.where(bar_colors, colors[1], colors[0])

    x_positions = np.arange(len(color_array))

    ax.scatter(x=x_positions, y=np.ones_like(x_positions), c=color_array, s=1000, marker='|')
    ax.set_ylim(0.95, 1.05)
    ax.set_xlim(0, len(color_array))
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel('Sample')
    ax.set_title(session.split('_')[0], fontweight='bold', fontsize=16, fontfamily='georgia', color='black')

    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set labels and title
    ax.set_xlabel('Time (s)')
    legend_elements = [plt.Line2D([0], [0], marker='|', color=color, markersize=10, label=['No Ripple', 'Ripple'][i]) for i, color in enumerate(np.unique(color_array))]
    ax.legend(handles=legend_elements, loc='upper left')
    ax.text(1, 1.2, f'Ripple events: {ripples.shape[0]}',
        transform=ax.transAxes, fontsize=8, verticalalignment='top',
        horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))



# Set a common y-axis label
fig.subplots_adjust(hspace=0.5)
# Show the plot
plt.tight_layout()
# Save the figure
plt.savefig('img/ripples_in_time.svg', format='svg', transparent=True)