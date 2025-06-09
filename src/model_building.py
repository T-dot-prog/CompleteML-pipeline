import os
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from data_ingestion import logger, load_params, load_data



def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    """
    Train the RandomForest model.
    
    :param X_train: Training features
    :param y_train: Training labels
    :param params: Dictionary of hyperparameters
    :return: Trained RandomForestClassifier
    """
    try:


        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be the same.")
        
        logger.debug('Initializing RandomForest model with parameters: %s', params)
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
        
        logger.debug('Model training started with %d samples', X_train.shape[0])
        clf.fit(X_train, y_train)
        logger.debug('Model training completed')
        
        return clf
    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise


def main():
    try:
        params = load_params('E:/CompleteML-pipeline/params.yaml')['model_building']
        train_data = load_data('E:/CompleteML-pipeline/data/processed/train_tfidf.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train, params)
        
        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()