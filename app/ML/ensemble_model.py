"""
Ensemble Model for Dengue Prediction with GPU Support
Combines multiple models: TabPFN, MLP, Perceptron, XGBoost, LGBM
Uses StackingClassifier for ensemble predictions
Supports both CPU and GPU (CUDA) for training and inference
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
    Wrapper for TabPFN with GPU support
    """
    def __init__(self, device='auto', N_ensemble_configurations=8):
        # Auto-detect device if not specified
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"TabPFN using device: {self.device}")
        self.N_ensemble_configurations = N_ensemble_configurations
        self.model = TabPFNClassifier(device=self.device, N_ensemble_configurations=N_ensemble_configurations)
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
        print(f"TabPFN config saved to {directory}")
    
    def load(self, directory, device=None):
        """Load TabPFN model configuration"""
        with open(os.path.join(directory, 'tabpfn_config.json'), 'r') as f:
            config = json.load(f)
        
        # Use specified device or saved device
        if device is not None:
            self.device = device
        else:
            self.device = config['device']
        
        self.N_ensemble_configurations = config['N_ensemble_configurations']
        self.is_fitted = config['is_fitted']
        if self.is_fitted:
            self.model = TabPFNClassifier(device=self.device, N_ensemble_configurations=self.N_ensemble_configurations)
        print(f"TabPFN config loaded on device: {self.device}")


class PyTorchMLPWrapper(nn.Module):
    """
    PyTorch MLP wrapper with GPU support
    """
    def __init__(self, input_size, hidden_sizes=[100, 50], output_size=1, learning_rate=0.001, device='auto'):
        super(PyTorchMLPWrapper, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Auto-detect device if not specified
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"PyTorch MLP using device: {self.device}")
        
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
        self.to(self.device)  # Move model to device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = None
        self.is_fitted = False
    
    def forward(self, x):
        return self.network(x)
    
    def fit(self, X, y, epochs=50, batch_size=32):
        """Fit the model"""
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y.values if hasattr(y, 'values') else y).unsqueeze(1).to(self.device)
        
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
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self(X_tensor)
            predictions = (torch.sigmoid(outputs) > 0.5).float().squeeze().cpu().numpy()
        return predictions.astype(int)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self(X_tensor)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
        # Return probabilities for both classes
        return np.column_stack([1 - probs, probs])
    
    def save_safetensors(self, filepath):
        """Save model to safetensors format"""
        state_dict = self.state_dict()
        # Convert all tensors to contiguous format and move to CPU
        safe_state_dict = {k: v.contiguous().cpu() for k, v in state_dict.items()}
        save_file(safe_state_dict, filepath)
        
        # Save model config
        config = {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'is_fitted': self.is_fitted,
            'device': str(self.device)
        }
        config_path = filepath.replace('.safetensors', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        print(f"PyTorch MLP saved to {filepath}")
    
    def load_safetensors(self, filepath, device='auto'):
        """Load model from safetensors format
        
        Args:
            filepath: Path to safetensors file
            device: Device to load model on ('cpu', 'cuda', or 'auto')
        """
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
        self.__init__(self.input_size, self.hidden_sizes, self.output_size, self.learning_rate, device=device)
        
        # Load weights
        state_dict = load_file(filepath)
        self.load_state_dict(state_dict)
        self.to(self.device)  # Move to target device
        
        print(f"PyTorch MLP loaded on device: {self.device}")


class DengueEnsembleModel:
    """
    Ensemble model with GPU support for training and inference
    """
    
    def __init__(self, input_size=None, device='auto'):
        self.input_size = input_size
        
        # Auto-detect device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"\n{'='*60}")
        print(f"Initializing Ensemble")
        print(f"Device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        print(f"{'='*60}\n")
        
        # Initialize base models
        self.tabpfn = TabPFNWrapper(device=self.device, N_ensemble_configurations=8)
        self.pytorch_mlp = None  # Will be initialized with input_size
        self.perceptron = Perceptron(random_state=42)
        
        # XGBoost with GPU support if available
        xgb_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
        if self.device == 'cuda' and torch.cuda.is_available():
            xgb_params['tree_method'] = 'gpu_hist'
            xgb_params['gpu_id'] = 0
            print("XGBoost: Using GPU acceleration")
        self.xgb = XGBClassifier(**xgb_params)
        
        # LightGBM with GPU support if available
        lgbm_params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'n_estimators': 1000,
            'random_state': 42,
            'verbose': -1
        }
        if self.device == 'cuda' and torch.cuda.is_available():
            lgbm_params['device'] = 'gpu'
            print("LightGBM: Using GPU acceleration")
        self.lgbm = LGBMClassifier(**lgbm_params)
        
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
        self.pytorch_mlp = PyTorchMLPWrapper(
            input_size=input_size,
            hidden_sizes=[100, 50],
            device=self.device
        )
    
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
        """Make predictions using all models"""
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
        """Predict probabilities using all models"""
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
        """Evaluate all models"""
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
        """Save all models"""
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
            'has_pytorch_mlp': self.pytorch_mlp is not None,
            'device': self.device
        }
        with open(os.path.join(directory, 'ensemble_config.json'), 'w') as f:
            json.dump(config, f)
        
        print(f"\nAll models saved to {directory}")
    
    def load(self, directory='models', device='cpu'):
        """Load all models
        
        Args:
            directory: Directory to load models from
            device: Device to load models on ('cpu', 'cuda', or 'auto')
        """
        # Auto-detect device if requested
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"\n{'='*60}")
        print(f"Loading models")
        print(f"Device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"{'='*60}\n")
        
        # Load config
        with open(os.path.join(directory, 'ensemble_config.json'), 'r') as f:
            config = json.load(f)
        
        self.input_size = config['input_size']
        self.is_fitted = config['is_fitted']
        
        # Load TabPFN with specified device
        self.tabpfn = TabPFNWrapper(device=self.device, N_ensemble_configurations=8)
        self.tabpfn.load(os.path.join(directory, 'tabpfn'), device=self.device)
        
        # Load PyTorch MLP with specified device
        if config['has_pytorch_mlp']:
            self.pytorch_mlp = PyTorchMLPWrapper(input_size=self.input_size, device=self.device)
            self.pytorch_mlp.load_safetensors(
                os.path.join(directory, 'pytorch_mlp.safetensors'),
                device=self.device
            )
        
        # Load sklearn models
        self.xgb = joblib.load(os.path.join(directory, 'xgb_model.pkl'))
        self.lgbm = joblib.load(os.path.join(directory, 'lgbm_model.pkl'))
        self.perceptron = joblib.load(os.path.join(directory, 'perceptron_model.pkl'))
        self.sklearn_mlp = joblib.load(os.path.join(directory, 'sklearn_mlp_model.pkl'))
        self.stacking_classifier = joblib.load(os.path.join(directory, 'stacking_classifier.pkl'))
        
        print(f"All models loaded successfully!")


if __name__ == "__main__":
    # Example usage
    from preprocessing import DenguePreprocessor, prepare_train_test_split
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Load and preprocess data
    preprocessor = DenguePreprocessor()
    data = preprocessor.load_data('CBC Report.csv')
    X_scaled, y = preprocessor.fit_transform(data)
    
    # Split data
    X_train, X_test, X_cv, y_train, y_test, y_cv = prepare_train_test_split(X_scaled, y)
    
    # Initialize and train ensemble (use 'auto' for automatic device selection)
    ensemble = DengueEnsembleModel(input_size=X_train.shape[1], device='auto')
    ensemble.fit(X_train, y_train, use_pytorch_mlp=True)
    
    # Evaluate
    results = ensemble.evaluate(X_test, y_test)
    
    # Save models
    ensemble.save('models')
    preprocessor.save('models')
