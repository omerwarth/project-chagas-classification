#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
from scipy.signal import find_peaks
import os
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sys

from helper_code import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    features = np.zeros((num_records, 6), dtype=np.float64)
    labels = np.zeros(num_records, dtype=bool)

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        features[i] = extract_features(record)
        labels[i] = load_label(record)

    # Train the models.
    if verbose:
        print('Training the model on the data...')

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    # This very simple model trains a random forest model with very simple features.

    if verbose:
        print('Applying SMOTE to handle class imbalance...')

    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_sampled, y_trained_sampled = smote.fit_resample(X_train, y_train)

    #run_grid_param(features, labels)

    # Define the parameters for the random forest classifier and regressor.
    n_estimators = 10  # Number of trees in the forest.
    max_leaf_nodes = 34  # Maximum number of leaf nodes in each tree.
    random_state = 56  # Random state; set for reproducibility.

    # Fit the model.
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state, class_weight='balanced').fit(X_train_sampled, y_trained_sampled)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_model(model_folder, model)

    test_accuracy = model.score(X_test, y_test)

    if verbose:
        print(f"Test Accuracy: {test_accuracy:.4f}")

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model.sav')
    model = joblib.load(model_filename)
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    # Load the model.
    model = model['model']

    # Extract the features.
    features = extract_features(record)
    features = features.reshape(1, -1)

    # Get the model outputs.
    binary_output = model.predict(features)[0]
    probability_output = model.predict_proba(features)[0][1]

    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################
    

# Modify the grid search to exclude 'l1_ratio' when penalty is not 'elasticnet'
def filter_params(params):
    if params.get('penalty') != 'elasticnet':
        params.pop('l1_ratio', None)  # Remove l1_ratio if it's not needed
    return params

def run_grid_param(features, labels):
    # create a model dictionary
    SEED = 654321

    models = {
        'logistic_regression': LogisticRegression(random_state=SEED, max_iter=1000, tol=0.001),
        'decision_tree': DecisionTreeClassifier(random_state=SEED),
        'random_forest': RandomForestClassifier(random_state=SEED)
    }

    # create a hyperparameter dictionary for the models
    param_grid = {
        'logistic_regression': {'C': [.5, 1, 5],
                                'penalty': ['l2']},
        'decision_tree': {'max_depth': [3, 5, None],
                          'criterion': ['gini', 'entropy']},
        'random_forest': {'n_estimators': [10, 50, 100], 
                        'max_depth': [None, 3, 5],
                        'criterion': ['gini', 'entropy']}
    }

    # declare variables to store the best model and score
    model_best = None
    score_best = 0.0

    ########### START YOUR CODE HERE #############
    K_folds = 10

    score = 'balanced_accuracy'

    for name, model in models.items():
        grid_search = GridSearchCV(model, param_grid[name], cv=K_folds, scoring=score, refit=True, n_jobs=-1)
        grid_search.fit(features, labels)

        if grid_search.best_score_ > score_best:
            model_best = grid_search.best_estimator_
            score_best = grid_search.best_score_

    print(f'The best model is {model_best} with a score of {score_best}')

# Extract your features.
def extract_features(record):
    header = load_header(record)
    age = get_age(header)
    sex = get_sex(header)

    one_hot_encoding_sex = np.zeros(3, dtype=bool)
    if sex == 'Female':
        one_hot_encoding_sex[0] = 1
    elif sex == 'Male':
        one_hot_encoding_sex[1] = 1
    else:
        one_hot_encoding_sex[2] = 1

    signal, fields = load_signals(record)

    fs  = fields.get('fs', 500)
    lead = signal[:, 0]

    peaks, _ = find_peaks(lead, distance = fs * .6)
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / fs
        heart_rates = 60 / rr_intervals

        avg_heart_rate = np.mean(heart_rates)
        heart_rate_variance = np.var(heart_rates)

    else:
        avg_heart_rate = 0.0
        heart_rate_variance = 0.0

    # TO-DO: Update to compute per-lead features. Check lead order and update and use functions for reordering leads as needed.
    # num_finite_samples = np.size(np.isfinite(signal))
    # if num_finite_samples > 0:
    #     signal_mean = np.nanmean(signal)
    # else:
    #     signal_mean = 0.0
    # if num_finite_samples > 1:
    #     signal_std = np.nanstd(signal)
    # else:
    #     signal_std = 0.0

    features = np.concatenate(([age], one_hot_encoding_sex, [avg_heart_rate, heart_rate_variance]))

    return np.asarray(features, dtype=np.float32)

# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)