# **LAVA_SNN_ripples**

## Introduction

This repository contains code and resources for an implementation of Spiking Neural Networks (SNNs), using LAVA Neuromorphic Computing Framework, for the analysis of Local Field Potentials (LFPs) recorded from mice. The primary objective of this project is to validate the effectiveness of SNNs in signal processing tasks, paving the way for their ultimate potential of analising in real-time with a low energy cost and high computation speed.

## Objectives

- Implement and train a Spiking Neural Network (SNN) model to classify patterns in LFP data.
    - Load LFP data --> `liset_tk`
    - Build a n-dataset for training the model. --> `extract_Nripples`
    - Train the model --> `trainSNN`

- Evaluate the performance metrics such as accuracy, precision, recall, and F1-score. --> `runSNN`


## Dataset

The dataset used to extract the evebts consisted of LFP recordings obtained from experimental sessions with mice performing cognitive tasks or under different physiological conditions. Each sample contains multi-channel LFP traces over time.
	- Amigo2_1_hippo_2019-07-11_11-57-07_1150um
	- Som_2_hippo_2019-07-24_12-01-49_1530um

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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

I would like to thank NCN group for the support and company through this project. 
