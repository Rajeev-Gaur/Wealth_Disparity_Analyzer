# preprocessing_data.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from config import DATA_FILE

def load_data():
    # Load the dataset
    df = pd.read_csv(DATA_FILE, header=None)
    
    # Set the column names
    df.columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary'
    ]
    
    # Handle missing values
    df.replace('?', pd.NA, inplace=True)
    df.dropna(inplace=True)
    
    return df

def preprocess_data(df):
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df

