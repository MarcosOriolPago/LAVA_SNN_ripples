# **LAVA_SNN_ripples**

## Introduction

This repository contains code and resources for comparing Spiking Neural Networks (SNNs) and Convolutional Neural Networks (CNNs) in the analysis of Local Field Potentials (LFPs) recorded from mice. The primary objective of this project is to validate the effectiveness of SNNs and CNNs in identifying and classifying patterns in LFP data, which provides valuable insights into neural activity at the population level.

## Objectives

- Implement and train a Spiking Neural Network (SNN) model to classify patterns in LFP data.
- Develop a Convolutional Neural Network (CNN) model for the same classification task and compare the results.
- Evaluate the performance metrics such as accuracy, precision, recall, and F1-score.
- Discuss the strengths and limitations of each model in analyzing LFPs.

## Dataset

The dataset consists of LFP recordings obtained from experimental sessions with mice performing cognitive tasks or under different physiological conditions. Each sample contains multi-channel LFP traces over time.

## Installation

To set up the environment and install the necessary dependencies, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/MarcosOriolPago/LAVA_SNN_ripples.git
    cd LAVA_SNN_ripples
    ```

2. Create and activate the Conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate lava_snn_ripples
    ```

## Usage

1. **Data Preparation**: Load and preprocess the LFP data for model training.
    ```python
    # Example code to load and preprocess data
    from data_preparation import load_data, preprocess_data
    data = load_data('path_to_data')
    preprocessed_data = preprocess_data(data)
    ```

2. **Train SNN Model**: Implement and train the SNN model.
    ```python
    # Example code to train SNN
    from models import SNN
    snn_model = SNN()
    snn_model.train(preprocessed_data)
    ```

3. **Train CNN Model**: Implement and train the CNN model.
    ```python
    # Example code to train CNN
    from models import CNN
    cnn_model = CNN()
    cnn_model.train(preprocessed_data)
    ```

4. **Evaluate Models**: Compare the performance of SNN and CNN models.
    ```python
    # Example code to evaluate models
    from evaluation import evaluate
    snn_results = evaluate(snn_model, preprocessed_data)
    cnn_results = evaluate(cnn_model, preprocessed_data)
    ```

## Results and Discussion

The results of the analysis will include comparative metrics such as accuracy, precision, recall, and F1-score for both SNN and CNN models. Visualizations and discussions will be provided to interpret the model predictions on LFP data.

## Contributing

We welcome contributions to enhance this project. If you would like to contribute, please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

We would like to thank the contributors and the neuroscience research community for their valuable inputs and support.