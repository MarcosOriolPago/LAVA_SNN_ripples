# Define the epochs, model architectures, and Y_num_samples
architectures=("256_128")
y_num_samples=("50")

# Iterate through each combination and run the Python script

for architecture in "${architectures[@]}"
do
    for y_num_sample in "${y_num_samples[@]}"
    do
        echo "Running train_model.py with architecture=$architecture, Y_num_samples=$y_num_sample"
        python3.9 train_model.py "50" "$architecture" "$y_num_sample" "trial_output"
    done
done

python3.9 plot_accuracies.py "trial_output"

./update_git.sh
