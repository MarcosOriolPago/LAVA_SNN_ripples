# Parse command-line arguments
while getopts a:y: flag
do
    case "${flag}" in
        a) architectures=(${OPTARG//,/ });;
        y) y_num_samples=(${OPTARG//,/ });;
    esac
done

# Iterate through each combination and run the Python script
for architecture in "${architectures[@]}"
do
    for y_num_sample in "${y_num_samples[@]}"
    do
        echo "Running train_model.py with architecture=$architecture, Y_num_samples=$y_num_sample"
        python3.9 train_model.py "50" "$architecture" "$y_num_sample" "models"
    done
done

python3.9 plot_accuracies.py "models"
