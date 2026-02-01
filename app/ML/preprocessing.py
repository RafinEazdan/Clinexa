"""
Preprocessing Module for Dengue Prediction
Handles data loading, cleaning, feature engineering, and transformation
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
import os


class DenguePreprocessor:
    """
    Preprocessing pipeline for Dengue prediction dataset
    """
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
        self.feature_columns = None
        self.columns_to_impute = ['ESR', 'Lymphocyte', 'Monocyte', 'Eosinophil', 'Basophil', 'RBC', 'Neutrophil']
        self.is_fitted = False
        
    def load_data(self, filepath):
        """
        Load data from Excel file
        
        Args:
            filepath: Path to the Excel file
            
        Returns:
            pandas DataFrame
        """
        data = pd.read_excel(filepath) if filepath.endswith('.xlsx') else pd.read_csv(filepath)
        return data
    
    def clean_data(self, data):
        """
        Clean the dataset by removing unnecessary columns
        
        Args:
            data: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Drop unnecessary columns
        columns_to_drop = ['Serial', 'Date']
        data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
        return data
    
    def handle_missing_values(self, data, fit=True):
        """
        Handle missing values using mean imputation
        
        Args:
            data: Input DataFrame
            fit: Whether to fit the imputer or just transform
            
        Returns:
            DataFrame with imputed values
        """
        columns_present = [col for col in self.columns_to_impute if col in data.columns]
        
        if fit:
            self.imputer = self.imputer.fit(data[columns_present])
        
        data[columns_present] = self.imputer.transform(data[columns_present])
        return data
    
    def encode_labels(self, data, fit=True):
        """
        Encode the Result column
        
        Args:
            data: Input DataFrame
            fit: Whether to fit the encoder or just transform
            
        Returns:
            DataFrame with encoded labels
        """
        if 'Result' in data.columns:
            if fit:
                data['Result_cat'] = self.label_encoder.fit_transform(data['Result'])
            else:
                data['Result_cat'] = self.label_encoder.transform(data['Result'])
            data = data.drop(columns=['Result'])
        return data
    
    def encode_categorical(self, data):
        """
        One-hot encode categorical variables
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical variables
        """
        if 'Gender' in data.columns:
            data = pd.get_dummies(data=data, drop_first=True, columns=['Gender'])
        return data
    
    def split_features_target(self, data):
        """
        Split data into features and target
        
        Args:
            data: Input DataFrame
            
        Returns:
            X (features), y (target)
        """
        if 'Result_cat' in data.columns:
            X = data.drop(['Result_cat'], axis=1)
            y = data['Result_cat']
            return X, y
        else:
            return data, None
    
    def scale_features(self, X, fit=True):
        """
        Scale features using StandardScaler
        
        Args:
            X: Feature DataFrame
            fit: Whether to fit the scaler or just transform
            
        Returns:
            Scaled features as numpy array
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.feature_columns = X.columns.tolist()
        else:
            X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def fit_transform(self, data):
        """
        Fit and transform the entire preprocessing pipeline
        
        Args:
            data: Input DataFrame
            
        Returns:
            X_scaled, y
        """
        data = self.clean_data(data)
        data = self.handle_missing_values(data, fit=True)
        data = self.encode_labels(data, fit=True)
        data = self.encode_categorical(data)
        X, y = self.split_features_target(data)
        X_scaled = self.scale_features(X, fit=True)
        self.is_fitted = True
        return X_scaled, y
    
    def transform(self, data):
        """
        Transform new data using fitted preprocessing pipeline
        
        Args:
            data: Input DataFrame
            
        Returns:
            X_scaled (and y if available)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Use fit_transform first.")
        
        data = self.clean_data(data)
        data = self.handle_missing_values(data, fit=False)
        
        # For inference, Result column might not be present
        has_target = 'Result' in data.columns
        if has_target:
            data = self.encode_labels(data, fit=False)
        
        data = self.encode_categorical(data)
        X, y = self.split_features_target(data)
        
        # Ensure columns match training
        if self.feature_columns is not None:
            missing_cols = set(self.feature_columns) - set(X.columns)
            for col in missing_cols:
                X[col] = 0
            X = X[self.feature_columns]
        
        X_scaled = self.scale_features(X, fit=False)
        
        if has_target:
            return X_scaled, y
        else:
            return X_scaled
    
    def save(self, directory='models'):
        """
        Save the preprocessor objects
        
        Args:
            directory: Directory to save the preprocessor
        """
        os.makedirs(directory, exist_ok=True)
        
        joblib.dump(self.imputer, os.path.join(directory, 'imputer.pkl'))
        joblib.dump(self.scaler, os.path.join(directory, 'scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join(directory, 'label_encoder.pkl'))
        joblib.dump(self.feature_columns, os.path.join(directory, 'feature_columns.pkl'))
        joblib.dump(self.columns_to_impute, os.path.join(directory, 'columns_to_impute.pkl'))
        
        print(f"Preprocessor saved to {directory}")
    
    def load(self, directory='models'):
        """
        Load the preprocessor objects
        
        Args:
            directory: Directory to load the preprocessor from
        """
        self.imputer = joblib.load(os.path.join(directory, 'imputer.pkl'))
        self.scaler = joblib.load(os.path.join(directory, 'scaler.pkl'))
        self.label_encoder = joblib.load(os.path.join(directory, 'label_encoder.pkl'))
        self.feature_columns = joblib.load(os.path.join(directory, 'feature_columns.pkl'))
        self.columns_to_impute = joblib.load(os.path.join(directory, 'columns_to_impute.pkl'))
        self.is_fitted = True
        
        print(f"Preprocessor loaded from {directory}")


def prepare_train_test_split(X, y, test_size=0.2, cv_size=0.1, random_state=23):
    """
    Split data into train, test, and cross-validation sets
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of data for testing
        cv_size: Proportion of data for cross-validation
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, X_cv, y_train, y_test, y_cv
    """
    # First split: train + remaining
    X_train, temp_X, y_train, temp_y = train_test_split(
        X, y, test_size=test_size + cv_size, random_state=random_state, stratify=y
    )
    
    # Second split: test + cv
    test_ratio = test_size / (test_size + cv_size)
    X_test, X_cv, y_test, y_cv = train_test_split(
        temp_X, temp_y, test_size=1-test_ratio, random_state=random_state, stratify=temp_y
    )
    
    return X_train, X_test, X_cv, y_train, y_test, y_cv


if __name__ == "__main__":
    # Example usage
    preprocessor = DenguePreprocessor()
    
    # Load and process data
    data = preprocessor.load_data('CBC Report.csv')
    X_scaled, y = preprocessor.fit_transform(data)
    
    print(f"Processed data shape: {X_scaled.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Split data
    X_train, X_test, X_cv, y_train, y_test, y_cv = prepare_train_test_split(X_scaled, y)
    
    print(f"\nTrain set: {X_train.shape}, {y_train.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    print(f"CV set: {X_cv.shape}, {y_cv.shape}")
    
    # Save preprocessor
    preprocessor.save('models')
