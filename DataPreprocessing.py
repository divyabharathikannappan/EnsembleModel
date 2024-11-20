import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    data = pd.read_csv(file_path, sep="\t")
    return data

def preprocess_data(data):
    data.replace('?', np.nan, inplace= True)
    missing_percentage = data.isnull().mean() * 100
    columns_to_drop = missing_percentage[missing_percentage > 50].index
    data.drop(columns=columns_to_drop, inplace=True)

    # Numeric columns: fill with median
    data['AMOUNT_ORDER'] = data['AMOUNT_ORDER'].fillna(data['AMOUNT_ORDER'].median())
    data['VALUE_ORDER'] = data['VALUE_ORDER'].fillna(data['VALUE_ORDER'].median())
    data['B_BIRTHDATE'].fillna('Unknown', inplace=True)  # Placeholder for missing dates
    data['TIME_ORDER'].fillna('00:00', inplace=True)  # Default time

    # Drop rows where 'class' is missing
    data.dropna(subset=['CLASS'], inplace=True)
    # Encode the target variable
    label_encoder = LabelEncoder()
    data['CLASS'] = label_encoder.fit_transform(data['CLASS'])
    
    numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
    #Removing Outlier using IQR techmique
    Q1 = data[numerical_features].quantile(0.25)
    Q3 = data[numerical_features].quantile(0.75)
    IQR = Q3 - Q1

    # Defining limits for outlier removal
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Removing outliers from the dataset
    data = data[~((data[numerical_features] < lower_bound) | (data[numerical_features] > upper_bound)).any(axis=1)]
   
    # Scaling numerical features
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
   

    categorical_columns = data.select_dtypes(include=['object']).columns
    data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])
    label_encoders = {}
    # Encoding categorical variables to numeric using LabelEncoder
    for column in categorical_columns:
        le = LabelEncoder()
        # Use a temporary variable to store encoded values
        data[column] = le.fit_transform(data[column].values)
        label_encoders[column] = le  # Store the encoder for potential future use

    return data
