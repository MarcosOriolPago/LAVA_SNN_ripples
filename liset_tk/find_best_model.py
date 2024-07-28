# Suppress warnings
from liset_tk import liset_tk
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import matplotlib.pyplot as plt


try:
    # Check if the required number of arguments are provided
    if len(sys.argv) < 3:
        raise ValueError("Usage: python find_best_model.py <data_path> <models_path>")
    
    # Get the command-line arguments
    data_path = sys.argv[1]
    models_path = sys.argv[2]

except IndexError:
    print("Error: Insufficient command-line arguments provided.")
    sys.exit()
except ValueError as ve:
    print(ve)
    sys.exit()
except Exception as e:
    print("An error occurred:", e)
    sys.exit()

# Create a liset object instance
liset = liset_tk(data_path, shank=3, downsample=True, numSamples=5000000, verbose=False)

# Define variables for the iteration
models = [models_path + '/' + i for i in os.listdir(models_path)]
model_types = set([os.path.split(i)[1].split('_')[0] for i in models])
print(model_types)
thresh_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
bad_model_idxs = []
performances = {}

# Iterate through each model and test performance for different threshold values.
for m_i, model in enumerate(models):
    model_performances = []

    try:
        print(model.split('/')[-1])
        liset.load_model(model)

        for th in thresh_range:
            try:
                liset.predict(threshold=th)
            except Exception as err:
                model_performances.append(0)
                continue

            if len(liset.prediction_times) > 0:
                liset.check_performance(show=False)
                model_performances.append(liset.performance)
            else:
                model_performances.append(0)

        performances[model] = model_performances

    except Exception as err:
        bad_model_idxs.append(m_i)


for i in bad_model_idxs:
    performances[models[i]] = [0 for i in range(len(thresh_range))]


th_range_str = [str(i) for i in thresh_range]

# Plot each model's performance on a separate subplot
for model_type in model_types:
    type_models = [i for i in models if model_type in i]
    fig, ax = plt.subplots(figsize=(10, 6))    
    fig.suptitle(model_type) 
    for i, model in enumerate(type_models): # Get the current axes
        if set(performances[model]) != {0}:
            ax.plot(thresh_range, performances[model], label=model.split('/')[-1])

    ax.legend()
    ax.set_ylabel('Performance')
    ax.set_xlabel('Threshold')
    ax.set_xticks(thresh_range)


plt.show()
