# House Price Prediction Project

This project aims to predict house prices using a linear regression model. The model is trained on a dataset obtained from Kaggle, which contains various features related to house characteristics.

## Project Structure

- **data/**: Contains the dataset and a README file with details about the dataset.
- **src/**: Contains the source code for data preprocessing, model definition, training, and prediction.
  - **data_preprocessing.py**: Functions for loading and preprocessing the dataset.
  - **model.py**: Defines the linear regression model and evaluation metrics.
  - **train.py**: Responsible for training the model on the dataset.
  - **predict.py**: Functions for making predictions with the trained model.
- **deployment/**: Contains files for deploying the model as a web application.
  - **app.py**: Sets up the web application and defines API endpoints.
  - **requirements.txt**: Lists the dependencies required for the deployment application.
- **notebooks/**: Contains Jupyter notebooks for exploratory data analysis (EDA).
- **.gitignore**: Specifies files and directories to be ignored by Git.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd house-price-prediction
   ```

2. Install the required dependencies:
   ```
   pip install -r deployment/requirements.txt
   ```

3. Run the training script to train the model:
   ```
   python src/train.py
   ```

4. Start the web application for predictions:
   ```
   python deployment/app.py
   ```

## Usage

Once the application is running, you can send a POST request to the API endpoint with the necessary input features to receive a predicted house price.

## Dataset Information

For detailed information about the dataset, including its source, structure, and preprocessing steps, please refer to the `data/README.md` file.