import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Handle missing values
    data.fillna(data.mean(), inplace=True)
    
    # Feature engineering (example: converting categorical variables)
    data = pd.get_dummies(data, drop_first=True)
    
    # Separate features and target variable
    X = data.drop('SalePrice', axis=1)
    y = data['SalePrice']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test