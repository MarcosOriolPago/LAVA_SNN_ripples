import sys
sys.path.insert(0, 'liset_tk/')
from liset_tk import liset_tk
import matplotlib.pyplot as plt
import os

# Define the path to your data
parent = r'C:\__NeuroSpark_Liset_Dataset__\neurospark_mat\CNN TRAINING SESSIONS' # Modify this to your data path folder

# Collect the sessions
sessions = os.listdir(parent)
num_sessions = len(sessions)

# Create subplots
fig, axes = plt.subplots(1, num_sessions, figsize=(5 * num_sessions, 5))  # Adjust the figsize as needed
plt.style.use('seaborn-v0_8-paper')

for ax, session in zip(axes, sessions):
    path = os.path.join(parent, session)
    liset = liset_tk(data_path=path, shank=3, downsample=4000, verbose=False)

    total_ripples = 0
    for i in liset.ripples_GT:
        total_ripples += (i[1] - i[0])

    rest = liset.file_samples / liset.fs_conv_fact - total_ripples
    duration_no_ripple = rest / liset.fs
    duration_ripple = total_ripples / liset.fs

    colors = ['#23943b', '#1f77b4']
    wedges, texts, autotexts = ax.pie(
        [total_ripples, rest],
        labels=['Ripples', 'No ripples'],
        autopct='%1.2f%%',
        startangle=130,
        wedgeprops=dict(width=0.25),
        colors=colors
    )

    ax.set_title(session.split('_')[0], fontsize=16, fontweight='bold', fontfamily='georgia', color='black')

    # Adjust the legend to be in the center gap of the pie chart
    ax.legend(wedges, [f'{duration_ripple :.2f} s', f'{duration_no_ripple :.2f} s'],
              loc="center", bbox_to_anchor=(1, 0.9), fontsize=10)

    for wedge in wedges:
        wedge.set_edgecolor('white')

# Save the figure
plt.savefig('img/AllSessions_DatasetDist.svg', format='svg', transparent=True)

