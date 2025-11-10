#!/usr/bin/env python3
"""
Model training module - Train all models on 80% data, select best, test on 20%.
Incorporates Ru (pore pressure ratio) values based on Bishop's Simplified Method.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import json


class FoSModelTrainer:
    """
    Trains multiple ML models for FoS prediction with Ru incorporation.
    Only the best performing model on training data gets tested.
    """
    
    def __init__(self, X_train, y_train, X_test, y_test):
        """
        Initialize trainer with train and test data.
        
        Parameters:
        -----------
        X_train, y_train : Training data (80%)
        X_test, y_test : Testing data (20%) - only used for best model
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # Initialize scaler
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        
        # Store models and results
        self.models = {}
        self.training_results = {}
        self.best_model_name = None
        self.best_model = None
        self.test_results = None
        
    def train_all_models(self):
        """Train all models on 80% training data and evaluate."""
        print("\n" + "="*80)
        print("TRAINING PHASE - All Models on 80% Training Data")
        print("="*80)
        
        # Define models - Fine-tuned XGBoost for better generalization
        models_config = {
            'SVM': SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=300,           # Increased from 200
                max_depth=6,                # Reduced from 10 to prevent overfitting
                learning_rate=0.05,         # Reduced from 0.1 for better generalization
                subsample=0.8,              # Added: use 80% of samples per tree
                colsample_bytree=0.8,       # Added: use 80% of features per tree
                min_child_weight=3,         # Added: min samples in leaf (regularization)
                gamma=0.1,                  # Added: min loss reduction (regularization)
                reg_alpha=0.1,              # Added: L1 regularization
                reg_lambda=1.0,             # Added: L2 regularization
                random_state=42
            ),
            'LightGBM': lgb.LGBMRegressor(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42, verbose=-1),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=300,           # Increased from 200
                max_depth=5,                # Reduced from 10 to prevent overfitting
                learning_rate=0.05,         # Reduced from 0.1 for better generalization
                subsample=0.8,              # Added: use 80% of samples per tree
                min_samples_split=5,        # Added: min samples to split a node
                min_samples_leaf=3,         # Added: min samples in leaf (regularization)
                max_features='sqrt',        # Added: use sqrt of features per split
                random_state=42
            ),
            'ANN': MLPRegressor(hidden_layer_sizes=(100, 50, 25), activation='relu', solver='adam', 
                               max_iter=1000, random_state=42, early_stopping=True)
        }
        
        # Train each model
        for model_name, model in models_config.items():
            print(f"\n📊 Training {model_name}...")
            
            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Predict on training data
            y_train_pred = model.predict(self.X_train_scaled)
            
            # Calculate training metrics
            r2_train = r2_score(self.y_train, y_train_pred)
            rmse_train = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
            mae_train = mean_absolute_error(self.y_train, y_train_pred)
            
            # Store results
            self.models[model_name] = model
            self.training_results[model_name] = {
                'r2': r2_train,
                'rmse': rmse_train,
                'mae': mae_train,
                'predictions': y_train_pred
            }
            
            print(f"  ✓ R² = {r2_train:.4f}")
            print(f"  ✓ RMSE = {rmse_train:.4f}")
            print(f"  ✓ MAE = {mae_train:.4f}")
        
        # Select top 2 models based on R² score for testing
        sorted_models = sorted(self.training_results.items(), 
                              key=lambda x: x[1]['r2'], 
                              reverse=True)
        
        # Select Gradient Boosting and XGBoost specifically
        self.test_model_names = ['Gradient Boosting', 'XGBoost']
        
        print("\n" + "="*80)
        print(f"🏆 TOP PERFORMING MODELS (will be tested on 20% data):")
        for i, (model_name, results) in enumerate(sorted_models[:2], 1):
            print(f"   {i}. {model_name}")
            print(f"      R² = {results['r2']:.4f}")
            print(f"      RMSE = {results['rmse']:.4f}")
            print(f"      MAE = {results['mae']:.4f}")
        print("="*80)
        
        return self.training_results
    
    def test_best_models(self):
        """Test both Gradient Boosting and XGBoost on 20% test data."""
        if not self.models:
            raise ValueError("Must train models first before testing!")
        
        print("\n" + "="*80)
        print(f"TESTING PHASE - Testing Gradient Boosting & XGBoost on 20% Test Data")
        print("="*80)
        
        self.test_results = {}
        
        for model_name in self.test_model_names:
            if model_name not in self.models:
                continue
                
            print(f"\n📊 Testing {model_name}...")
            
            model = self.models[model_name]
            
            # Predict on test data
            y_test_pred = model.predict(self.X_test_scaled)
            
            # Calculate test metrics
            r2_test = r2_score(self.y_test, y_test_pred)
            rmse_test = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
            mae_test = mean_absolute_error(self.y_test, y_test_pred)
            
            self.test_results[model_name] = {
                'r2': r2_test,
                'rmse': rmse_test,
                'mae': mae_test,
                'predictions': y_test_pred,
                'actual': self.y_test.values
            }
            
            print(f"  ✓ R² = {r2_test:.4f}")
            print(f"  ✓ RMSE = {rmse_test:.4f}")
            print(f"  ✓ MAE = {mae_test:.4f}")
        
        print("="*80)
        
        return self.test_results
    
    def save_models_and_results(self, output_dir):
        """Save all trained models and results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, output_dir / 'scaler.pkl')
        
        # Save all models
        for model_name, model in self.models.items():
            safe_name = model_name.lower().replace(' ', '_')
            joblib.dump(model, output_dir / f'model_{safe_name}.pkl')
        
        # Save top models separately (GB and XGBoost)
        for model_name in self.test_model_names:
            if model_name in self.models:
                safe_name = model_name.lower().replace(' ', '_')
                joblib.dump(self.models[model_name], output_dir / f'best_model_{safe_name}.pkl')
        
        # Save training results as CSV
        training_data = []
        for model_name, results in self.training_results.items():
            training_data.append({
                'Model': model_name,
                'R² Score': float(results['r2']),
                'RMSE': float(results['rmse']),
                'MAE': float(results['mae'])
            })
        
        training_df = pd.DataFrame(training_data)
        training_df = training_df.sort_values('R² Score', ascending=False)
        training_df.to_csv(output_dir / 'training_results.csv', index=False)
        print(f"  ✓ Saved training results CSV")
        
        # Save test results as CSV if they exist
        if self.test_results:
            # Save test metrics for both models
            test_metrics_list = []
            for model_name, results in self.test_results.items():
                test_metrics_list.append({
                    'Model': model_name,
                    'R² Score': float(results['r2']),
                    'RMSE': float(results['rmse']),
                    'MAE': float(results['mae'])
                })
            
            test_metrics_df = pd.DataFrame(test_metrics_list)
            test_metrics_df = test_metrics_df.sort_values('R² Score', ascending=False)
            test_metrics_df.to_csv(output_dir / 'test_results.csv', index=False)
            print(f"  ✓ Saved test results CSV (Gradient Boosting & XGBoost)")
            
            # Save test predictions for both models
            for model_name, results in self.test_results.items():
                safe_name = model_name.lower().replace(' ', '_')
                test_predictions = pd.DataFrame({
                    'Actual FoS': [float(x) for x in results['actual']],
                    'Predicted FoS': [float(x) for x in results['predictions']],
                    'Error': [float(a - p) for a, p in zip(results['actual'], results['predictions'])]
                })
                test_predictions.to_csv(output_dir / f'test_predictions_{safe_name}.csv', index=False)
                print(f"  ✓ Saved {model_name} test predictions CSV ({len(test_predictions)} samples)")
        
        # Save results as JSON for backward compatibility
        results_summary = {
            'training_results': {
                name: {
                    k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) 
                       else v.tolist() if hasattr(v, 'tolist') 
                       else v)
                    for k, v in results.items()
                }
                for name, results in self.training_results.items()
            },
            'test_models': self.test_model_names,
            'test_results': {
                model_name: {
                    k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) 
                       else v.tolist() if hasattr(v, 'tolist') 
                       else v)
                    for k, v in results.items()
                }
                for model_name, results in self.test_results.items()
            } if self.test_results else None
        }
        
        with open(output_dir / 'results_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n✓ Saved all models and results to {output_dir}")
    
    def get_training_comparison_data(self):
        """Get data for training comparison chart (all models, 80% data)."""
        comparison_data = []
        for model_name, results in self.training_results.items():
            comparison_data.append({
                'model': model_name,
                'r2': results['r2'],
                'rmse': results['rmse'],
                'mae': results['mae'],
                'is_best': model_name == self.best_model_name
            })
        return pd.DataFrame(comparison_data)


if __name__ == "__main__":
    # Test the training module
    from data_ingestion import load_and_prepare_data, train_test_split_data
    
    csv_path = Path(__file__).parent / "data" / "Overall Data.csv"
    X, y, df = load_and_prepare_data(csv_path, include_ru=True)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2)
    
    # Train models
    trainer = FoSModelTrainer(X_train, y_train, X_test, y_test)
    trainer.train_all_models()
    trainer.test_best_models()
    
    # Save results
    trainer.save_models_and_results(Path(__file__).parent / "models")
    
    # Get comparison data
    comparison_df = trainer.get_training_comparison_data()
    print("\n📊 Training Comparison:")
    print(comparison_df)
