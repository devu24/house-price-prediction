import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
from src.data_preprocessing import load_and_preprocess_data
from src.model import create_model

def train_model():
    # Load and preprocess the data
    X, y = load_and_preprocess_data('data/house_prices.csv')  # Adjust the path as necessary

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = create_model()
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    # Save the trained model
    joblib.dump(model, 'model/house_price_model.pkl')

if __name__ == '__main__':
    train_model()