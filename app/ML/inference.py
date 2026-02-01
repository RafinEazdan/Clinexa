"""
Inference Script for Dengue Prediction
Loads trained models and makes predictions on new data
"""

import numpy as np
import pandas as pd
import argparse
import json
import torch
from preprocessing import DenguePreprocessor
from ensemble_model import DengueEnsembleModel


class DenguePredictor:
    """
    Prediction interface for Dengue classification
    """
    
    def __init__(self, model_directory='models', device='cpu'):
        """
        Initialize predictor with saved models
        
        Args:
            model_directory: Directory containing saved models
            device: Device for inference ('cpu', 'cuda', or 'auto')
        """
        self.model_directory = model_directory
        self.device = device
        
        # Auto-detect device if requested
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Inference device: {self.device.upper()}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.preprocessor = DenguePreprocessor()
        self.ensemble = DengueEnsembleModel(device=self.device)
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load preprocessor and ensemble models"""
        print("Loading models...")
        self.preprocessor.load(self.model_directory)
        self.ensemble.load(self.model_directory)
        print("Models loaded successfully!")
    
    def predict_from_file(self, filepath, output_file=None):
        """
        Make predictions from a file
        
        Args:
            filepath: Path to input data file (CSV or Excel)
            output_file: Optional path to save predictions
            
        Returns:
            DataFrame with predictions
        """
        # Load data
        print(f"Loading data from {filepath}...")
        data = self.preprocessor.load_data(filepath)
        
        # Check if data has Result column (labeled data)
        has_labels = 'Result' in data.columns
        
        # Preprocess
        print("Preprocessing data...")
        if has_labels:
            X_scaled, y_true = self.preprocessor.transform(data)
        else:
            X_scaled = self.preprocessor.transform(data)
            y_true = None
        
        # Make predictions
        print("Making predictions...")
        predictions = self.ensemble.predict(X_scaled)
        probabilities = self.ensemble.predict_proba(X_scaled)
        
        # Create results DataFrame
        results_df = data.copy()
        
        # Add predictions from all models
        for model_name, preds in predictions.items():
            results_df[f'pred_{model_name}'] = preds
        
        # Add probabilities for key models
        for model_name in ['stacking', 'tabpfn']:
            if model_name in probabilities:
                results_df[f'prob_{model_name}_negative'] = probabilities[model_name][:, 0]
                results_df[f'prob_{model_name}_positive'] = probabilities[model_name][:, 1]
        
        # Add ensemble prediction (majority vote)
        pred_array = np.array([preds for preds in predictions.values()])
        ensemble_pred = np.round(pred_array.mean(axis=0)).astype(int)
        results_df['pred_ensemble_majority'] = ensemble_pred
        
        # Map predictions back to original labels
        label_mapping = {v: k for k, v in enumerate(self.preprocessor.label_encoder.classes_)}
        for col in results_df.columns:
            if col.startswith('pred_'):
                results_df[f'{col}_label'] = results_df[col].map(label_mapping)
        
        # Calculate accuracy if labels available
        if y_true is not None:
            print("\n" + "="*60)
            print("EVALUATION METRICS")
            print("="*60)
            for model_name, preds in predictions.items():
                accuracy = (preds == y_true).mean()
                print(f"{model_name.upper()}: {accuracy:.4f}")
            
            ensemble_accuracy = (ensemble_pred == y_true).mean()
            print(f"ENSEMBLE MAJORITY: {ensemble_accuracy:.4f}")
        
        # Save results if output file specified
        if output_file:
            if output_file.endswith('.csv'):
                results_df.to_csv(output_file, index=False)
            else:
                results_df.to_excel(output_file, index=False)
            print(f"\nPredictions saved to {output_file}")
        
        return results_df
    
    def predict_single(self, patient_data):
        """
        Make prediction for a single patient
        
        Args:
            patient_data: Dictionary with patient features
            
        Returns:
            Dictionary with predictions from all models
        """
        # Convert to DataFrame
        df = pd.DataFrame([patient_data])
        
        # Preprocess
        X_scaled = self.preprocessor.transform(df)
        
        # Make predictions
        predictions = self.ensemble.predict(X_scaled)
        probabilities = self.ensemble.predict_proba(X_scaled)
        
        # Format results
        result = {
            'predictions': {},
            'probabilities': {},
            'ensemble_prediction': None
        }
        
        for model_name, preds in predictions.items():
            pred_label = self.preprocessor.label_encoder.inverse_transform([preds[0]])[0]
            result['predictions'][model_name] = {
                'class': int(preds[0]),
                'label': pred_label
            }
        
        for model_name in ['stacking', 'tabpfn']:
            if model_name in probabilities:
                result['probabilities'][model_name] = {
                    'negative': float(probabilities[model_name][0, 0]),
                    'positive': float(probabilities[model_name][0, 1])
                }
        
        # Ensemble majority vote
        pred_values = [preds[0] for preds in predictions.values()]
        ensemble_pred = int(np.round(np.mean(pred_values)))
        ensemble_label = self.preprocessor.label_encoder.inverse_transform([ensemble_pred])[0]
        
        result['ensemble_prediction'] = {
            'class': ensemble_pred,
            'label': ensemble_label,
            'confidence': float(np.mean([pred == ensemble_pred for pred in pred_values]))
        }
        
        return result
    
    def predict_batch(self, patients_data):
        """
        Make predictions for multiple patients
        
        Args:
            patients_data: List of dictionaries with patient features
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for patient_data in patients_data:
            result = self.predict_single(patient_data)
            results.append(result)
        return results


def main():
    """Command-line interface for inference"""
    parser = argparse.ArgumentParser(description='Dengue Prediction Inference')
    parser.add_argument('--input', type=str, required=True, help='Input data file (CSV or Excel)')
    parser.add_argument('--output', type=str, help='Output file for predictions')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory containing saved models')
    parser.add_argument('--single', action='store_true', help='Predict single patient from JSON input')
    parser.add_argument('--device', type=str, default='cpu', 
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device for inference: auto (detect GPU), cpu, or cuda (default: cpu)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = DenguePredictor(model_directory=args.model_dir, device=args.device)
    
    if args.single:
        # Load single patient data from JSON
        with open(args.input, 'r') as f:
            patient_data = json.load(f)
        
        result = predictor.predict_single(patient_data)
        
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(json.dumps(result, indent=2))
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to {args.output}")
    else:
        # Batch prediction from file
        results_df = predictor.predict_from_file(args.input, args.output)
        
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        print(f"Total samples: {len(results_df)}")
        
        # Show prediction distribution for stacking model
        if 'pred_stacking_label' in results_df.columns:
            print("\nPrediction distribution (Stacking Model):")
            print(results_df['pred_stacking_label'].value_counts())


if __name__ == "__main__":
    # Example usage without command line
    import sys
    
    if len(sys.argv) == 1:
        # Demo \n# Batch prediction (CPU - Recommended for inference)")
        print("python inference.py --input data.csv --output predictions.csv --device cpu")
        print("\n# Batch prediction (GPU)")
        print("python inference.py --input data.csv --output predictions.csv --device cuda")
        print("\n# Single prediction (CPU)")
        print("python inference.py --input patient.json --output result.json --single --device cpu")
        print("\n# Auto device detection")
        print("python inference.py --input data.csv --output predictions.csv --device auto")
        print("python inference.py --input data.csv --output predictions.csv")
        print("\nFor single prediction:")
        print("python inference.py --input patient.json --output result.json --single")
        
        # Example single prediction
        try:
            predictor = DenguePredictor(model_directory='models')
            
            # Example patient data
            example_patient = {
                'Age': 25,
                'WBC': 5.5,
                'Hb': 13.5,
                'Hct': 40.0,
                'Platelet': 150000,
                'ESR': 15,
                'Lymphocyte': 35,
                'Monocyte': 6,
                'Eosinophil': 2,
                'Basophil': 1,
                'RBC': 4.5,
                'Neutrophil': 56,
                'Gender_Male': 1  # 1 for Male, 0 for Female
            }
            
            print("\nExample prediction:")
            result = predictor.predict_single(example_patient)
            print(json.dumps(result, indent=2))
            
        except Exception as e:
            print(f"\nNote: Models not found. Train models first using ensemble_model.py")
            print(f"Error: {e}")
    else:
        # Command-line mode
        main()
