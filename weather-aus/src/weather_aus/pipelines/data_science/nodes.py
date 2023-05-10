import logging
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_test_split(X_training_le,y_training):
  """Splitting of the x_traing data into X_train,X_test,y_train,y_test data for predicting the output.

  Arg=X_training_le,y_training

  output=Get splitted data in the form of train and test data."""
  from sklearn.model_selection import train_test_split
  X_train,X_test,y_train,y_test=train_test_split(X_training_le,y_training,random_state=0,test_size=0.20)
  return X_train,X_test,y_train,y_test

def logReg(X_train,X_test,y_train,y_test):
  logreg=LogisticRegression()
  logreg = logreg.fit(X_train,y_train)
  return logreg

def pred(logreg,X_train:pd.DataFrame,X_test:pd.DataFrame) -> pd.DataFrame:
  y_pred_train=logreg.predict(X_train)
  y_pred_test=logreg.predict(X_test)
  y_pred_train=pd.DataFrame(y_pred_train)
  y_pred_test=pd.DataFrame(y_pred_test)
  return y_pred_train,y_pred_test

def r2_score(y_train,y_test,y_pred_train,y_pred_test) -> pd.DataFrame:
  """calculate r2_score

  Arg=y_train,y_test,y_pred_train,y_pred_test

  output=get r2 score to evaluate the model performance on train and test data."""
  r2_score_train=accuracy_score(y_pred_train,y_train)
  r2_score_test=accuracy_score(y_pred_test,y_test)
  r2_score_train=pd.Series(r2_score_train)
  r2_score_test=pd.Series(r2_score_train)
  return r2_score_train,r2_score_test


