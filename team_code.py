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
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
import sys
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torchvision import transforms
import torchvision.transforms.functional as F
from collections import Counter
import seaborn as sns
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import torchmetrics as TM

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

    features = np.zeros((num_records, 5000), dtype=np.float64)
    labels = np.zeros(num_records, dtype=bool)

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        
        # CHANGE BACK - 2
        features[i] = extract_features_CNN(record)
        labels[i] = load_label(record)
        
        print(labels[i])

    # Train the models.
    if verbose:
        print('Training the CNN model on the data...')
        
    dataset = ECGDataset(features, labels)

    if verbose:
        print('Printing the CNN model results on the data...')
        
    trainCNN(dataset)
        
    if verbose:
        print('Printing the Radnom Forest model results on the data...')

    # Split data into train/test split for random forest
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Applying SMOTE to the training data
    if verbose:
        print('Applying SMOTE to handle class imbalance...')

    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_sampled, y_trained_sampled = smote.fit_resample(X_train, y_train)

    # Uncomment to run grid parameter search
    # run_grid_param(features, labels)

    # Define the parameters for the random forest classifier and regressor
    n_estimators = 10  # Number of trees in the forest.
    max_leaf_nodes = 34  # Maximum number of leaf nodes in each tree
    random_state = 56  # Random state; set for reproducibility

    # Fit the model
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state, class_weight='balanced').fit(X_train_sampled, y_trained_sampled)

    # Create a folder for the model if it does not already exist
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_model(model_folder, model)

    # Make model predictions
    y_pred = model.predict(X_test)
    
    if verbose:        
        # Make a confusion matrix for our data
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=["Actual False", "Actual True"], columns=["Pred False", "Pred True"])
        
        probs = model.predict_proba(X_test)
        fpr = dict()
        tpr = dict()
        roc_auc = {0:0, 1:0, 2:0}
        for i in range(2):
            fpr[i], tpr[i], _ = roc_curve(y_test, probs[:, i], pos_label=i)
            y_test_i = np.array(y_test == i).astype(int)
            roc_auc[i] = roc_auc_score(y_test_i, probs[:, i])
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

        # Making ROC curves for the data 
        print('Printing ROC curves for Random Forest model...') 
        
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve For Iris Data Using Random Forest Model')
        plt.legend()
        plt.show()

        print('Printing confusion matrix for random forest model...') 
        print(cm_df)

        # we can use the ConfusionMatrixDisplay to plot the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(cmap='Blues')
        plt.title('Confusion Matrix For ECG Data Using Random Forest')
        plt.show()

        print('Printing classificagtion report for Random Forest model...') 
        print(classification_report(y_test, y_pred))

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
    
    
# Extracts features and prepares for CNN imaging
def extract_features_CNN(record):
    signal, fields = load_signals(record)

    fs  = fields.get('fs', 500)
    lead = signal[:, 0]
    
    if len(lead) >= 5000:
        segment = lead[:5000]
    else:
        segment = np.pad(lead, (0, 5000 - len(lead)), 'constant')
        
    return np.asarray(segment, dtype=np.float32)

    
# Custom ECG dataset
class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        signal = self.features[idx]
        label = self.labels[idx]

        # Ensuring features are in a torch
        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal).float()
            
        signal = signal.float()
        signal = signal.view(-1)
        signal = signal.unsqueeze(0)
    
        label = torch.tensor(label, dtype=torch.long)
        
        return signal, label


class CNNEncoder(nn.Module):
    def __init__(self, num_channels, num_kernels, kernel_size=5, stride=1, padding=0):
        super().__init__()
        self.encoder = nn.Sequential(
            # Using three hidden layers with max pooling and ReLU
            nn.Conv1d(num_channels, num_kernels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(num_kernels, num_kernels * 2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(num_kernels * 2, num_kernels * 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

    def forward(self, x):
        x = self.encoder(x)
        #x = x.view(x.size(0), -1)
        return x
    
class CNNOutput(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x
    
class ECGClassifier(L.LightningModule):
    def __init__(self, encoder, decoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

       # validation metrics - we will use these to compute the metrics at the end of the validation epoch
        self.val_metrics_tracker = TM.wrappers.MetricTracker(TM.MetricCollection([TM.classification.MulticlassAccuracy(num_classes=num_classes)]), maximize=True)
        self.validation_step_outputs = []
        self.validation_step_targets = []

        # test metrics - we will use these to compute the metrics at the end of the test epoch
        self.test_roc = TM.ROC(task="multiclass", num_classes=num_classes) # roc and cm have methods we want to call so store them in a variable
        self.test_cm = TM.ConfusionMatrix(task='multiclass', num_classes=num_classes)
        self.test_metrics_tracker = TM.wrappers.MetricTracker(TM.MetricCollection([TM.classification.MulticlassAccuracy(num_classes=num_classes),
                                                            self.test_roc, self.test_cm]), maximize=True)

        # test outputs and targets - we will store the outputs and targets for the test step
        self.test_step_outputs = []
        self.test_step_targets = []

    # the forward method applies the encoder and output to the input
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    # the training step. pass the batch through the model and compute the loss
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        # where is our softmax? We don't need it here because we are using cross_entropy which includes the softmax for efficiency
        loss = nn.functional.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss

    # the validation step. pass the batch through the model and compute the loss. Store the outputs and targets for the epoch end step and log the loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # x = self.encoder(x)
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True)

        # store the outputs and targets for the epoch end step
        self.validation_step_outputs.append(logits)
        self.validation_step_targets.append(y)
        return loss

    # the test step. pass the batch through the model and compute the loss. Store the outputs and targets for the epoch end step and log the loss
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True)
        self.test_step_outputs.append(logits)
        self.test_step_targets.append(y)
        return loss

    # at the end of the epoch compute the metrics
    def on_validation_epoch_end(self):
        # stack all the outputs and targets into a single tensor
        all_preds = torch.vstack(self.validation_step_outputs)
        all_targets = torch.hstack(self.validation_step_targets)

        # compute the metrics
        loss = nn.functional.cross_entropy(all_preds, all_targets)
        self.val_metrics_tracker.increment()
        self.val_metrics_tracker.update(all_preds, all_targets)
        self.log('val_loss_epoch_end', loss)

        # clear the validation step outputs
        self.validation_step_outputs.clear()
        self.validation_step_targets.clear()

    def on_test_epoch_end(self):
        all_preds = torch.vstack(self.test_step_outputs)
        all_targets = torch.hstack(self.test_step_targets)

        self.test_metrics_tracker.increment()
        self.test_metrics_tracker.update(all_preds, all_targets)
        # clear the test step outputs
        self.test_step_outputs.clear()
        self.test_step_targets.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# Train CNN model method to make instances of the loaders, encoder, and decoder
def trainCNN(dataset):
    batch_size = 32
    num_kernels = 64

    # Splitting dataset into train/test/validation sets
    val_size = int(len(dataset) * .15)
    train_size = int(len(dataset) * .7)
    test_size = len(dataset) - val_size - train_size
    ecg_train, ecg_val, ecg_test = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    # Initialize encoder to flatten output
    encoder = CNNEncoder(num_channels = 1, num_kernels = num_kernels)
    encoder.eval()
    
    # Set device variable to cuda or mps if available otherwise cpu
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        
    encoder.to(device)

    # Making data loaders
    train_loader = DataLoader(ecg_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ecg_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ecg_test, batch_size=batch_size, shuffle=False)

    # Create a dummy input to inspect encoder output shape
    dummy_input = torch.randn(1, 1, 5000).to(device)
    encoder.eval()
    with torch.no_grad():
        encoded_output = encoder(dummy_input)
        flattened_dim = encoded_output.view(1, -1).shape[1]
    
    decoder = CNNOutput(input_dim = flattened_dim, num_classes=2)

    # Instantiating model
    model = ECGClassifier(encoder = encoder, decoder = decoder, num_classes = 2)
    model.to(device)

    # Training + fitting model
    trainer = L.Trainer(
                    max_epochs=10,
                    callbacks=[EarlyStopping(monitor="val_loss_epoch_end", mode="min", patience=3)],)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    device = torch.device("cpu")
    
    # put the model in evaluation mode so that the parameters are fixed and we don't compute gradients
    model.eval()   

    # Plotting confusion matrices and ROC curves for model using seaborn
    trainer.test(model=model, dataloaders=test_loader)
    rslt = model.test_metrics_tracker.compute()

    plt.figure()    
    cmp = sns.heatmap(rslt['MulticlassConfusionMatrix'], annot=True, fmt='d', cmap='Blues')
    cmp.set_xlabel('Predicted Label')
    cmp.set_xticklabels(['False', 'True'], rotation=0)
    cmp.set_yticklabels(['False', 'True'], rotation=0)
    cmp.set_ylabel('Actual Label');
    plt.title('Confusion Matrix')
    plt.savefig('plots/confusion_matrix.png')
    plt.show()
    plt.close()

    fpr, tpr, thresholds = rslt['MulticlassROC']
    
    plt.figure()
    ROC_label=['Class 0', 'Class 1']
    for i in range(2):
        plt.plot(fpr[i], tpr[i], label=ROC_label[i])
    plt.xlabel('False Positive Rate')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid()
    plt.title('Multiclass ROC Curve')
    plt.savefig('plots/roc_curve.png')
    plt.show()
    plt.close()

    device = torch.device("cpu")
    
    # put the model in evaluation mode so that the parameters are fixed and we don't compute gradients
    model.eval()
    y_true=[]
    y_pred=[]
    
    # use torch.no_grad() to disable gradient computation
    with torch.no_grad():
        # iterate over the test loader minibatches
        for test_data in test_loader:
            # get the images and labels from the test loader and move them to the cpu. this will make it easier to use them with sklearn
            test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
            pred = model(test_images).argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(test_labels[i].item())
                y_pred.append(pred[i].item())
                
    print('Printing classificagtion report for CNN model...')            
    print(classification_report(y_true,y_pred,target_names=['chagas-negative', 'chagas-positive'], digits=4))
