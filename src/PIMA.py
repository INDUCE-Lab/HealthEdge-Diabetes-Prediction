#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np


dataset=pd.read_csv('../dataset/diabetes_PIMA_preprocessed.csv')
dataset.shape


y=dataset['diabetic/non-diabetic']
percent_pos = sum(y)/len(y)
print('PIMA Percentage Diabetes cases %.02f %%' %(percent_pos * 100))
print('PIMA Percentage Diabetes cases  %d %.02f%%  %d  %.02f%% ' % 
      (sum(y) , percent_pos * 100,  len(y) - sum(y), (1-percent_pos)*100 ))

dataset.columns


import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve,f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit


from sklearn.preprocessing import  LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

def pima_data_prep(df = None, do_balance = False):
    X = df.drop(["diabetic/non-diabetic"], axis = 1)
    y = df["diabetic/non-diabetic"]
        
    if do_balance:
        # transform the dataset
        oversample = SMOTE(random_state=123)
        X, y = oversample.fit_resample(X, y)        
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.30, 
                                                    random_state = 123)

    return X, y ,X_train, X_test, y_train, y_test


# Utility function to train and evaluate a model on PIMA dataset
def train_and_evaluate_model(model,X,y, verbose=False, n_splits=10):
    acc = 0
    auc = 0
    f1 = 0
    prec = 0
    recall = 0
    
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=123)
    
    start_time = time.time()
    for train_index, test_index in sss.split(X, y):
        if X is pd.DataFrame:
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        else:
            X_train, X_test = X[train_index], X[test_index]
          
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc1 = accuracy_score(y_test, y_pred)
        auc1 = roc_auc_score(y_test, y_pred, average="macro")
        f11 = f1_score(y_test, y_pred, average="macro")
        prec1 = precision_score(y_test, y_pred, average="macro")
        recall1 = recall_score(y_test, y_pred, average="macro")
        if verbose:
            print ('acc', acc1)
            print('f1', f11)
            print('recall1', recall1)
            print('auc1', auc1)
        acc += acc1
        auc += auc1
        f1 += f11
        prec += prec1
        recall += recall1
        
    spent_time = time.time() - start_time
    print("Acc      F-Meas   Precis   Recall   AUC      Time")
    print("%.04f\t%.04f\t%.04f\t%.04f\t%.04f\t%0.4f" % (acc/n_splits, f1/n_splits, prec/n_splits, 
                                            recall/n_splits, auc/n_splits, spent_time))
    

# Create tuned model for RF
rf_tuned_nofs_nobl = RandomForestClassifier(max_depth = 5,  max_features = None, 
                        criterion = 'entropy', n_estimators = 100, random_state=123)


# Prepare dataset with No feature Selection No balancing
X,y, X_train, X_test, y_train, y_test = pima_data_prep(dataset, do_balance=False)

#  Train and evaluate the Random Forest Model
train_and_evaluate_model(rf_tuned_nofs_nobl, X, y, verbose=False)


# Do prediction (example) with the trained model
y_pred = rf_tuned_nofs_nobl.predict(X_test)

# if We have the labels with can also evaluate the prediction        
acc1 = accuracy_score(y_test, y_pred)
auc1 = roc_auc_score(y_test, y_pred, average="macro")
f11 = f1_score(y_test, y_pred, average="macro")
prec1 = precision_score(y_test, y_pred, average="macro")
recall1 = recall_score(y_test, y_pred, average="macro")
print ('acc', acc1)
print('f1', f11)
print('recall1', recall1)
print('auc1', auc1)

