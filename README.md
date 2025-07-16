# George B. Moody PhysioNet Challenge Solution

## Overview
This repository contains a solution for the George B. Moody PhysioNet Challenge, which focuses on automated detection of Chagas disease from ECG signals. The challenge involves developing models that can accurately classify ECG records as Chagas-positive or Chagas-negative.

## Solution Approach
The solution uses two main machine learning approaches:

1. **Random Forest Classifier**
   - Extracts statistical and demographic features from ECG records (age, sex, heart rate, heart rate variance).
   - Handles class imbalance using SMOTE.
   - Trains and evaluates a Random Forest model, reporting ROC curves, confusion matrix, and classification metrics.

2. **Convolutional Neural Network (CNN)**
   - Extracts fixed-length ECG signal segments for deep learning.
   - Defines a custom PyTorch dataset and CNN architecture using PyTorch Lightning.
   - Trains, validates, and tests the CNN, reporting confusion matrix, ROC curves, and classification metrics.

## File Descriptions
- `team_code.py`: Main code for data loading, feature extraction, model training, evaluation, and utility functions. Contains consistent print statements and comments for clarity.
- `helper_code.py`: Helper functions for loading ECG data and labels (provided by the challenge).
- `train_model.py`, `evaluate_model.py`: Scripts for training and evaluating models.

## Usage
1. Place the ECG data in the appropriate folder.
2. Run the training script to train both models and save the Random Forest model.
3. Use the evaluation script to test the trained models on new data.

## Requirements
- Python 3.8+
- numpy, pandas, scipy, scikit-learn, imbalanced-learn, matplotlib, seaborn
- torch, torchvision, lightning, torchmetrics

Install dependencies with:
```bash
pip install numpy pandas scipy scikit-learn imbalanced-learn matplotlib seaborn torch torchvision lightning torchmetrics
```


## Challenge Reference
For more details on the challenge, see the [PhysioNet Challenge website](https://physionet.org/challenge/2022/).

## Contact
For questions, please contact the repository owner.
