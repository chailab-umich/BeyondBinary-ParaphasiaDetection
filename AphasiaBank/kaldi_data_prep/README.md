# Data Preparation for AphasiaBank using Kaldi

This repository contains scripts to preprocess the AphasiaBank data using Kaldi. The goal is to create a unified data preparation framework for consistent dataset preprocessing and research collaboration.

## Prerequisites

Make sure you have a working version of Kaldi installed before running the scripts.

## Steps for Data Preparation

1. **Generate Cross Validation Folds and Partitions**

    Run the script to generate cross validation folds and partitions:
    ```
    ./create_kaldi_data.sh
    ```

2. **Generate Data CSV Files for SpeechBrain**

    Run the script to generate the data CSV files needed for SpeechBrain:
    ```
    python create_speechbrain_data.py
    ```

## Contributing

Feel free to contribute to this project by submitting issues or pull requests.