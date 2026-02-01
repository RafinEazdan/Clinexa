"""
Training Script for Dengue Prediction Ensemble Model
Trains all models and saves them in safetensors format
"""

import argparse
import numpy as np
import torch
from preprocessing import DenguePreprocessor, prepare_train_test_split
from ensemble_model import DengueEnsembleModel


def train_models(data_path, model_dir='models', test_size=0.2, cv_size=0.1, use_pytorch_mlp=True, device='auto'):
    """
    Complete training pipeline
    
    Args:
        data_path: Path to input data file
        model_dir: Directory to save trained models
        test_size: Proportion of data for testing
        cv_size: Proportion of data for cross-validation
        use_pytorch_mlp: Whether to use PyTorch MLP (with safetensors) or sklearn MLP
    """
    print("="*60)
    print("DENGUE PREDICTION MODEL TRAINING")
    print("="*60)
    
    # Show GPU info
    print(f"\nCUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    if device == 'auto':
        selected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        selected_device = device
    print(f"Training Device: {selected_device.upper()}")
    
    # Step 1: Load and preprocess data
    print("\n[1/4] Loading and preprocessing data...")
    preprocessor = DenguePreprocessor()
    data = preprocessor.load_data(data_path)
    print(f"Loaded {len(data)} samples")
    
    X_scaled, y = preprocessor.fit_transform(data)
    print(f"Preprocessed data shape: {X_scaled.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Step 2: Split data, device=device
    print(f"\n[2/4] Splitting data (test={test_size}, cv={cv_size})...")
    X_train, X_test, X_cv, y_train, y_test, y_cv = prepare_train_test_split(
        X_scaled, y, test_size=test_size, cv_size=cv_size
    )
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"CV set: {X_cv.shape}")
    
    # Step 3: Train ensemble
    print(f"\n[3/4] Training ensemble model...")
    ensemble = DengueEnsembleModel(input_size=X_train.shape[1])
    ensemble.fit(X_train, y_train, use_pytorch_mlp=use_pytorch_mlp)
    
    # Step 4: Evaluate
    print(f"\n[4/4] Evaluating models on test set...")
    results = ensemble.evaluate(X_test, y_test)
    
    # Also evaluate on CV set
    print("\n" + "="*60)
    print("CROSS-VALIDATION SET EVALUATION")
    print("="*60)
    cv_results = ensemble.evaluate(X_cv, y_cv)
    
    # Step 5: Save models
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    ensemble.save(model_dir)
    preprocessor.save(model_dir)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nModels saved to: {model_dir}")
    print("\nTo make predictions, use:")
    print(f"python inference.py --input data.csv --output predictions.csv --model-dir {model_dir}")
    
    return ensemble, preprocessor, results


def main():
    """Command-line interface for training"""
    parser = argparse.ArgumentParser(description='Train Dengue Prediction Ensemble Model')
    parser.add_argument('--data', type=str, required=True, help='Path to training data (CSV or Excel)')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set proportion (default: 0.2)')
    parser.add_argument('--cv-size', type=float, default=0.1, help='CV set proportion (default: 0.1)')
    parser.add_argument('--no-pytorch-mlp', action='store_true', 
                        help='Use sklearn MLP instead of PyTorch MLP')
    parser.add_argument('--device', type=str, default='auto', 
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device for training: auto (detect GPU), cpu, or cuda (default: auto)')
    
    args = parser.parse_args()
    
    # Train models
    ensemble, preprocessor, results = train_models(
        data_path=args.data,
        model_dir=args.model_dir,
        device=args.device,
        test_size=args.test_size,
        cv_size=args.cv_size,
        use_pytorch_mlp=not args.no_pytorch_mlp
    )
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest Model: {best_model[0].upper()}")
    print(f"Accuracy: {best_model[1]['accuracy']:.4f}")
    if 'roc_auc' in best_model[1]:
        print(f"ROC AUC: {best_model[1]['roc_auc']:.4f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # Demo mode with example
        print("Training Script for Dengue Prediction")
        print("\nUsage:")
        print("python train.py --data CBC_Report.csv --model-dir models")
        print("\nOptions:")
        print("  --data           Path to training data (required)")
        print("  --device         Device for training: auto, cpu, or cuda (default: auto)")
        print("\nExamples:")
        print("# Train with auto device detection (recommended for T4 GPU)")
        print("python train.py --data 'CBC Report.csv' --model-dir models")
        print("\n# Force GPU training")
        print("python train.py --data 'CBC Report.csv' --device cuda")
        print("\n# Force CPU training")
        print("python train.py --data 'CBC Report.csv' --device cpu")
        print("  --cv-size        CV set proportion (default: 0.1)")
        print("  --no-pytorch-mlp Use sklearn MLP instead of PyTorch MLP")
        print("\nExample:")
        print("python train.py --data 'CBC Report.csv' --model-dir models --test-size 0.2")
    else:
        main()
