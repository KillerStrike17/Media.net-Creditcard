from sklearn.preprocessing import LabelEncoder
import pandas as pd

def Encoder(df, columns):
  le = LabelEncoder()
  for column in columns:
    df[column] = le.fit_transform(df[column])
  return df
