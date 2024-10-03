# Common modules
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import interp
from itertools import cycle
import matplotlib.pyplot as plt

# Data Preparation
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import LabelEncoder, label_binarize

# Modelling
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Testing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score #, plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc, roc_curve, roc_auc_score, classification_report

# DATA LOAD

answer_columns =    [f"EXT{i}" for i in range(1, 11)] + \
                    [f"EST{i}" for i in range(1, 11)] + \
                    [f"AGR{i}" for i in range(1, 11)] + \
                    [f"CSN{i}" for i in range(1, 11)] + \
                    [f"OPN{i}" for i in range(1, 11)]

time_columns =      [f"EXT{i}_E" for i in range(1, 11)] + \
                    [f"EST{i}_E" for i in range(1, 11)] + \
                    [f"AGR{i}_E" for i in range(1, 11)] + \
                    [f"CSN{i}_E" for i in range(1, 11)] + \
                    [f"OPN{i}_E" for i in range(1, 11)]

float32_columns = answer_columns + time_columns

float32_types = {k: 'float32' for k in float32_columns}
dtype = {**float32_types}

df = pd.read_csv('data/BigFivePersonalityTest-TrainSet.csv', parse_dates = ['dateload'], dtype = dtype)
df.rename(columns = {'Unnamed: 0' : 'id'}, inplace = True)

pd.options.display.float_format = "{:,.2f}".format
pd.options.display.max_columns = 999

# METADATA/VOLUME

print(df.head())
print(df.info())
print(f'The dataset has {df.shape[0]} records and {df.shape[1]} features.')

# FEATURES

def filter_IPC(df):
    return df[df['IPC']==1]

df = filter_IPC(df)

# DUPLICATES

def drop_dupl(df):
    df.drop_duplicates(keep = 'first', inplace = True)
    return df

df = drop_dupl(df)

# MISSING

def missing_answers(df,answer_columns):
    df[answer_columns] = df[answer_columns].replace(to_replace = 0, value = np.nan)
    return df

df = missing_answers(df,answer_columns)
df.isnull().sum()[df.isnull().sum()>0].to_frame('Nulls %').sort_values(by = 'Nulls %')/len(df)*100