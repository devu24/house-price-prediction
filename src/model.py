from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

class HousePriceModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        predictions = self.model.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        return mse, r2

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Implement preprocessing steps such as handling missing values and feature engineering
    return data

def main():
    # Example usage
    data = load_data('path_to_your_data.csv')
    processed_data = preprocess_data(data)
    X = processed_data.drop('target_column', axis=1)
    y = processed_data['target_column']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = HousePriceModel()
    model.train(X_train, y_train)
    
    mse, r2 = model.evaluate(X_test, y_test)
    print(f'Mean Squared Error: {mse}, R^2 Score: {r2}')

if __name__ == "__main__":
    main()