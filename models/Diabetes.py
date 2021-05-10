import os
import sys

import numpy as np
import pandas as pd
from sklearn.svm import SVC

sys.path.append('/home/farzaneh/DataScientist/LearnPython/Diabetes/diabetes/src/features/')
import build_features
from build_features import *
sys.path.append('/home/farzaneh/DataScientist/LearnPython/Diabetes/diabetes/src/models/')
import train_model
from train_model import *
sys.path.append('/home/farzaneh/DataScientist/LearnPython/Diabetes/diabetes/src/visualization/')
import visualize
from visualize import *
#https://stackabuse.com/creating-and-importing-modules-in-python/


#reading Data
data = pd.read_csv('/home/farzaneh/DataScientist/LearnPython/Diabetes/diabetes/data/processed/diabetes_data_upload.csv', header = 0)

#preparing Data for modelling
preprocessor = Preprocessor(data)
X_train, X_test, y_train, y_test, X, y = preprocessor.get_data()

#Data Statistics
count_positive, count_negative, percentage_positive, percentage_negative = preprocessor.statistica()

#choosing the model
algo="rf" # algo = ["DT","rf","knn","svm"]

#training the model
trained_model, model_confusion_mat = train_model(X_train, X_test, y_train, y_test, algo)

#model optimisation
if algo == "svm":
    best_model_parameters = SVC(C=0.1, gamma='auto', kernel='poly', probability=True)
else:
    best_model_parameters = optimize_model(trained_model, X_train, y_train, algo)

best_model, best_model_confusion_mat = kfoldevaluate_optimzed_model(algo, best_model_parameters, X, y, X_train, X_test, y_train, y_test)

#ROC calculation
fpr, tpr, threshold, roc_auc = roc_vorbereitung(best_model, X_test, y_test)

#graphical output
pl_ROC(algo, roc_auc, fpr, tpr)
if algo=="DT": Diabetes_tree(best_model, X)
