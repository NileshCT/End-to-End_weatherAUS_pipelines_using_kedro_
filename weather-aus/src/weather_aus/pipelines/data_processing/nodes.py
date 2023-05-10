import pandas as pd
from sklearn import preprocessing

def extract_training_data(df):
    """Extracting the training data from raw data and droping date columns
    Args=df,here raw data
    output=Dataframe"""
    df=df.drop("Date",axis=1)
    df1=df[df["RainTomorrow"].notna()]
    return df1

def treat_missing(df1):
  """treat missing values with ffill and bfill method

  Arg=df_training_data

  output=nan values from training data get fill with the ffill and bfill method."""
  df1_treat_training_data=df1.fillna(method="ffill",axis=0).fillna(method="bfill",axis=0)
  return df1_treat_training_data

def training_data_split(df1_treat_training_data):
  """Training data spliting ie.separating predictors and response variables.

  Arg=df_treat_training_data

  Output=Target variable get separated from main traing data."""
  X_training=df1_treat_training_data.drop(["RainTomorrow"],axis=1)
  # y_training=pd.Series(y_training)
  # X_training=pd.DataFrame(X_training)
  return X_training

def y_training(df1_treat_training_data,X_training):
    y_training=df1_treat_training_data["RainTomorrow"]
    return y_training

def lebal_encoder(X_training):
  """Lebal encoding on the discrite varibales.

  Arg=X_training 

  Output=converting categorical data into numeric variable."""

  from sklearn import preprocessing
  L_Encoder=preprocessing.LabelEncoder()
 
  X_training["Location"]=L_Encoder.fit_transform(X_training["Location"])

  X_training["WindGustDir"]=L_Encoder.fit_transform(X_training["WindGustDir"])
  X_training["WindDir9am"]=L_Encoder.fit_transform(X_training["WindDir9am"])
  X_training["WindDir3pm"]=L_Encoder.fit_transform(X_training["WindDir3pm"])
  X_training["RainToday"]=L_Encoder.fit_transform(X_training["RainToday"])
  X_training_le=X_training.copy()
  return X_training_le



