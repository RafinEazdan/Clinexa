"""
Ensemble Model for Dengue Prediction
Combines multiple models: TabPFN, MLP, Perceptron, XGBoost, LGBM
Uses StackingClassifier for ensemble predictions
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tabpfn import TabPFNClassifier
import joblib
import os
from safetensors.torch import save_file, load_file
import json


class TabPFNWrapper:
    """
    Wrapper for TabPFN to handle safetensors saving
    """
    def __init__(self, device='cpu', N_ensemble_configurations=8):
        self.device = device
        self.N_ensemble_configurations = N_ensemble_configurations
        self.model = TabPFNClassifier(device=device, N_ensemble_configurations=N_ensemble_configurations)
        self.is_fitted = False
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def save(self, directory):
        """Save TabPFN model configuration"""
        os.makedirs(directory, exist_ok=True)
        config = {
            'device': self.device,
            'N_ensemble_configurations': self.N_ensemble_configurations,
            'is_fitted': self.is_fitted
        }
        with open(os.path.join(directory, 'tabpfn_config.json'), 'w') as f:
            json.dump(config, f)
        # Note: TabPFN is pretrained, we just save its config
        print(f"TabPFN config saved to {directory}")
    
    def load(self, directory):
        """Load TabPFN model configuration"""
        with open(os.path.join(directory, 'tabpfn_config.json'), 'r') as f:
            config = json.load(f)
        self.device = config['device']
        self.N_ensemble_configurations = config['N_ensemble_configurations']
        self.is_fitted = config['is_fitted']
        if self.is_fitted:
            self.model = TabPFNClassifier(device=self.device, N_ensemble_configurations=self.N_ensemble_configurations)
        print(f"TabPFN config loaded from {directory}")


class PyTorchMLPWrapper(nn.Module):
    """
    PyTorch MLP wrapper compatible with sklearn
    """
    def __init__(self, input_size, hidden_sizes=[100, 50], output_size=1, learning_rate=0.001):
        super(PyTorchMLPWrapper, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Build network
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = None
        self.is_fitted = False
    
    def forward(self, x):
        return self.network(x)
    
    def fit(self, X, y, epochs=50, batch_size=32):
        """Fit the model"""
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y.values if hasattr(y, 'values') else y).unsqueeze(1)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}')
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict class labels"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self(X_tensor)
            predictions = (torch.sigmoid(outputs) > 0.5).float().squeeze().numpy()
        return predictions.astype(int)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self(X_tensor)
            probs = torch.sigmoid(outputs).squeeze().numpy()
        # Return probabilities for both classes
        return np.column_stack([1 - probs, probs])
    
    def save_safetensors(self, filepath):
        """Save model to safetensors format"""
        state_dict = self.state_dict()
        # Convert all tensors to contiguous format
        safe_state_dict = {k: v.contiguous() for k, v in state_dict.items()}
        save_file(safe_state_dict, filepath)
        
        # Save model config
        config = {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'is_fitted': self.is_fitted
        }
        config_path = filepath.replace('.safetensors', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        print(f"PyTorch MLP saved to {filepath}")
    
    def load_safetensors(self, filepath):
        """Load model from safetensors format"""
        # Load config first
        config_path = filepath.replace('.safetensors', '_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.input_size = config['input_size']
        self.hidden_sizes = config['hidden_sizes']
        self.output_size = config['output_size']
        self.learning_rate = config['learning_rate']
        self.is_fitted = config['is_fitted']
        
        # Rebuild network with loaded config
        self.__init__(self.input_size, self.hidden_sizes, self.output_size, self.learning_rate)
        
        # Load weights
        state_dict = load_file(filepath)
        self.load_state_dict(state_dict)
        
        print(f"PyTorch MLP loaded from {filepath}")


class DengueEnsembleModel:
    """
    Ensemble model combining multiple classifiers for Dengue prediction
    """
    
    def __init__(self, input_size=None):
        self.input_size = input_size
        
        # Initialize base models
        self.tabpfn = TabPFNWrapper(device='cpu', N_ensemble_configurations=8)
        self.pytorch_mlp = None  # Will be initialized with input_size
        self.perceptron = Perceptron(random_state=42)
        self.xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        self.lgbm = LGBMClassifier(
            boosting_type='gbdt',
            objective='binary',
            metric='binary_logloss',
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.9,
            n_estimators=1000,
            random_state=42
        )
        self.sklearn_mlp = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            max_iter=500,
            random_state=42
        )
        
        # Stacking ensemble
        self.stacking_classifier = None
        self.is_fitted = False
    
    def initialize_pytorch_mlp(self, input_size):
        """Initialize PyTorch MLP with given input size"""
        self.input_size = input_size
        self.pytorch_mlp = PyTorchMLPWrapper(input_size=input_size, hidden_sizes=[100, 50])
    
    def fit(self, X_train, y_train, use_pytorch_mlp=True):
        """
        Fit the ensemble model
        
        Args:
            X_train: Training features
            y_train: Training labels
            use_pytorch_mlp: Whether to use PyTorch MLP (True) or sklearn MLP (False)
        """
        if self.input_size is None:
            self.input_size = X_train.shape[1]
        
        print("Training ensemble models...")
        
        # Initialize PyTorch MLP if needed
        if use_pytorch_mlp and self.pytorch_mlp is None:
            self.initialize_pytorch_mlp(self.input_size)
        
        # Fit TabPFN
        print("\nTraining TabPFN...")
        self.tabpfn.fit(X_train.astype(np.float32), y_train.astype(np.float32))
        
        # Fit PyTorch MLP
        if use_pytorch_mlp:
            print("\nTraining PyTorch MLP...")
            self.pytorch_mlp.fit(X_train, y_train, epochs=50)
        
        # Define base models for stacking
        if use_pytorch_mlp:
            base_models = [
                ('xgb', self.xgb),
                ('lgbm', self.lgbm),
                ('perceptron', self.perceptron)
            ]
        else:
            base_models = [
                ('xgb', self.xgb),
                ('lgbm', self.lgbm),
                ('mlp', self.sklearn_mlp),
                ('perceptron', self.perceptron)
            ]
        
        # Meta-model
        meta_model = LogisticRegression(max_iter=1000, random_state=42)
        
        # Create stacking classifier
        print("\nTraining Stacking Ensemble...")
        self.stacking_classifier = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5
        )
        
        self.stacking_classifier.fit(X_train, y_train)
        self.is_fitted = True
        
        print("\nEnsemble training completed!")
    
    def predict(self, X):
        """
        Make predictions using all models
        
        Args:
            X: Features
            
        Returns:
            Dictionary with predictions from all models
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = {}
        
        # TabPFN predictions
        predictions['tabpfn'] = self.tabpfn.predict(X.astype(np.float32))
        
        # PyTorch MLP predictions
        if self.pytorch_mlp is not None:
            predictions['pytorch_mlp'] = self.pytorch_mlp.predict(X)
        
        # Stacking predictions
        predictions['stacking'] = self.stacking_classifier.predict(X)
        
        # Individual base model predictions
        predictions['xgb'] = self.xgb.predict(X)
        predictions['lgbm'] = self.lgbm.predict(X)
        predictions['perceptron'] = self.perceptron.predict(X)
        
        if 'mlp' in dict(self.stacking_classifier.estimators_):
            predictions['sklearn_mlp'] = self.sklearn_mlp.predict(X)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict probabilities using all models
        
        Args:
            X: Features
            
        Returns:
            Dictionary with probability predictions from all models
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        probabilities = {}
        
        # TabPFN probabilities
        probabilities['tabpfn'] = self.tabpfn.predict_proba(X.astype(np.float32))
        
        # PyTorch MLP probabilities
        if self.pytorch_mlp is not None:
            probabilities['pytorch_mlp'] = self.pytorch_mlp.predict_proba(X)
        
        # Stacking probabilities
        probabilities['stacking'] = self.stacking_classifier.predict_proba(X)
        
        return probabilities
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate all models
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        results = {}
        
        for model_name, y_pred in predictions.items():
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            results[model_name] = {
                'accuracy': accuracy,
                'classification_report': report
            }
            
            # Add ROC AUC if probabilities available
            if model_name in probabilities:
                roc_auc = roc_auc_score(y_test, probabilities[model_name][:, 1])
                results[model_name]['roc_auc'] = roc_auc
            
            print(f"\n{'='*60}")
            print(f"{model_name.upper()} Results")
            print(f"{'='*60}")
            print(f"Accuracy: {accuracy:.4f}")
            if 'roc_auc' in results[model_name]:
                print(f"ROC AUC: {results[model_name]['roc_auc']:.4f}")
            print(f"\nClassification Report:")
            print(classification_report(y_test, y_pred, digits=4))
        
        return results
    
    def save(self, directory='models'):
        """
        Save all models
        
        Args:
            directory: Directory to save models
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save TabPFN
        self.tabpfn.save(os.path.join(directory, 'tabpfn'))
        
        # Save PyTorch MLP as safetensors
        if self.pytorch_mlp is not None:
            self.pytorch_mlp.save_safetensors(os.path.join(directory, 'pytorch_mlp.safetensors'))
        
        # Save sklearn models
        joblib.dump(self.xgb, os.path.join(directory, 'xgb_model.pkl'))
        joblib.dump(self.lgbm, os.path.join(directory, 'lgbm_model.pkl'))
        joblib.dump(self.perceptron, os.path.join(directory, 'perceptron_model.pkl'))
        joblib.dump(self.sklearn_mlp, os.path.join(directory, 'sklearn_mlp_model.pkl'))
        joblib.dump(self.stacking_classifier, os.path.join(directory, 'stacking_classifier.pkl'))
        
        # Save config
        config = {
            'input_size': self.input_size,
            'is_fitted': self.is_fitted,
            'has_pytorch_mlp': self.pytorch_mlp is not None
        }
        with open(os.path.join(directory, 'ensemble_config.json'), 'w') as f:
            json.dump(config, f)
        
        print(f"\nAll models saved to {directory}")
    
    def load(self, directory='models'):
        """
        Load all models
        
        Args:
            directory: Directory to load models from
        """
        # Load config
        with open(os.path.join(directory, 'ensemble_config.json'), 'r') as f:
            config = json.load(f)
        
        self.input_size = config['input_size']
        self.is_fitted = config['is_fitted']
        
        # Load TabPFN
        self.tabpfn.load(os.path.join(directory, 'tabpfn'))
        
        # Load PyTorch MLP
        if config['has_pytorch_mlp']:
            self.pytorch_mlp = PyTorchMLPWrapper(input_size=self.input_size)
            self.pytorch_mlp.load_safetensors(os.path.join(directory, 'pytorch_mlp.safetensors'))
        
        # Load sklearn models
        self.xgb = joblib.load(os.path.join(directory, 'xgb_model.pkl'))
        self.lgbm = joblib.load(os.path.join(directory, 'lgbm_model.pkl'))
        self.perceptron = joblib.load(os.path.join(directory, 'perceptron_model.pkl'))
        self.sklearn_mlp = joblib.load(os.path.join(directory, 'sklearn_mlp_model.pkl'))
        self.stacking_classifier = joblib.load(os.path.join(directory, 'stacking_classifier.pkl'))
        
        print(f"\nAll models loaded from {directory}")


if __name__ == "__main__":
    # Example usage
    from preprocessing import DenguePreprocessor, prepare_train_test_split
    
    # Load and preprocess data
    preprocessor = DenguePreprocessor()
    data = preprocessor.load_data('CBC Report.csv')
    X_scaled, y = preprocessor.fit_transform(data)
    
    # Split data
    X_train, X_test, X_cv, y_train, y_test, y_cv = prepare_train_test_split(X_scaled, y)
    
    # Initialize and train ensemble
    ensemble = DengueEnsembleModel(input_size=X_train.shape[1])
    ensemble.fit(X_train, y_train, use_pytorch_mlp=True)
    
    # Evaluate
    results = ensemble.evaluate(X_test, y_test)
    
    # Save models
    ensemble.save('models')
    preprocessor.save('models')
