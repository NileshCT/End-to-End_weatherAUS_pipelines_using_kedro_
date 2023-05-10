import logging
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def inference_data(df):
    """function is to extract the inference data from dataframe.
    
    Arg=df
    
    output=get inference data"""
    df=df.drop("Date",axis=1)
    inference_data=df[df["RainTomorrow"].isna()]

    return inference_data

def infer_missing_treat(inference_data):
    """function is to trate missing values in the inference data.
    Arg=inference_data
    output=missing value treatment we will fill all null values here."""
    inference_data_treat=inference_data.fillna(method="ffill",axis=0).fillna(method="bfill",axis=0)
    return inference_data_treat

def inference_data_split(inference_data_treat):
  """inference data spliting ie.separating predictors and response variables.

  Arg=inference_data_treat

  Output=Target variable get separated from main traing data."""
  X_infer=inference_data_treat.drop(["RainTomorrow"],axis=1)
  return X_infer

def lebal_encoder_infer(X_infer):
  """Lebal encoding on the discrite varibales.

  Arg=X_training 

  Output=converting categorical data into numeric variable."""


  from sklearn import preprocessing
  L_Encoder=preprocessing.LabelEncoder()
  X_infer["Location"]=L_Encoder.fit_transform(X_infer["Location"])

  X_infer["WindGustDir"]=L_Encoder.fit_transform(X_infer["WindGustDir"])
  X_infer["WindDir9am"]=L_Encoder.fit_transform(X_infer["WindDir9am"])
  X_infer["WindDir3pm"]=L_Encoder.fit_transform(X_infer["WindDir3pm"])
  X_infer["RainToday"]=L_Encoder.fit_transform(X_infer["RainToday"])
  X_infer_le=X_infer.copy()
  return X_infer_le

def pred_infer(logreg,X_infer_le):
   """ function is used to predict the target varibale of inference data
   Arg=logreg,X_infer_le
   output=get predictions for inference data."""
   y_pred_infer=logreg.predict(X_infer_le)
   y_pred_infer=pd.Series(y_pred_infer)
   return y_pred_infer

def concat_infer(y_pred_infer,X_infer):
   """ function used to concat the predictions with original predicted varibales.
   Arg=y_pred_infer,inference_data
   output=y_pred_infer
   """
   X_infer=pd.DataFrame(X_infer)
   concate_infer=pd.concat([X_infer,y_pred_infer],axis=1)
   return concate_infer

