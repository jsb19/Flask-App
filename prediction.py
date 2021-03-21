#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  19 23:06:33 2021

@author: jujharbedi
"""
# Import libraries and variables
import numpy as np 
import pandas as pd 
import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Read dataset
dataset = pd.read_csv('./static/data/shot_data.csv')

# Remove unnecessary columns from dataset
dataset.drop("team_id", inplace=True, axis=1)
dataset.drop("team_name", inplace=True, axis=1)
dataset.drop("matchup", inplace=True, axis=1)
dataset.drop("opponent", inplace=True, axis=1)
dataset.drop("game_date", inplace=True, axis=1)


#### Encoding categorical variables ####

# List of all categorical variables
categoricalVars = ["action_type", "combined_shot_type", "shot_zone_area", "shot_zone_basic",
                   "shot_zone_range", "season", "shot_type"]
# Creating encoder
enc = OrdinalEncoder()
# Encode Categorical Variables
dataset[categoricalVars] = enc.fit_transform(dataset[categoricalVars])

# Removing nan values for shot_flag 
dataset = dataset[pd.notnull(dataset['shot_made_flag'])]

# Creating matrix of features
X = dataset.drop('shot_made_flag',axis=1)
y = dataset.shot_made_flag

#### Splitting dataset into training and test sets ####
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Building model using xgboost algorithm
model = XGBClassifier(n_estimators=20, random_state=71)
model.fit(X_train, Y_train)

# Calculating the importance of each feature in the model
def featImp(model):
    coeffImp = model.feature_importances_
    coeffImp = [round(i,2) * 100 for i in coeffImp]
    return coeffImp

featureImportance = pd.DataFrame(
    {'Feature': X_train.columns.values,
     'Importance': featImp(model),
    })

# Removing features that have minimal impact on the model
for feat, coeff in zip(featureImportance.Feature, featureImportance.Importance):
    if coeff < 2.5:
        X.drop(feat, inplace=True, axis=1)

# Recreating training and test sets with low impact features removed
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Building new model
newModel = XGBClassifier(n_estimators=20, random_state=71)
newModel.fit(X_train, Y_train)

# Creating new dataframe for feature importance based on new model
newFeatureImportance = pd.DataFrame(
    {'Feature': X_train.columns.values,
     'Importance': featImp(newModel),
    })

# Plot of features and their importance to the model
featureImportanceFig, ax = plt.subplots(figsize=[11,8])
ax = sns.barplot(x="Feature", y="Importance", data=newFeatureImportance, capsize=.2)
featureImportanceFig.autofmt_xdate()

# Saving figure as a image file 
featureImportanceFig.savefig('./static/img/featureImportance.png')

# Calculate prediction based on new model
pred = newModel.predict_proba(X_test)[:, 1]

# Round predictions from model to either 1 or 0
roundedPred = []
for i in pred:
    if round(i, 2) > .50:
        i = 1
        roundedPred.append(i)
    else:
        i = 0
        roundedPred.append(i)

# Convert to array
roundedPred = np.array(roundedPred)

rawPredData = pd.DataFrame({'shot_id':X_test['shot_id'], 'shot_made_flag':pred})

predData = pd.DataFrame({'shot_id':X_test['shot_id'], 'shot_made_flag':roundedPred})

# Calculate accuracy of model
def modelAccuracy():
    countTrue = 0
    for index, row in predData.iterrows():
        if dataset.loc[dataset['shot_id'] == row["shot_id"]]["shot_made_flag"].item() == row["shot_made_flag"]:
            countTrue+=1
    return round(countTrue/len(predData) * 100, 2)


try:
  accuracy
except NameError:
 accuracy = str(modelAccuracy()) + "%"
except:
  pass