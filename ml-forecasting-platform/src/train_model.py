"""
Main Training Pipeline for ML Forecasting Platform
Orchestrates data loading, preprocessing, model training, and evaluation
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pytorch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from config import ConfigManager
from data.preprocessing import DataLoader, TimeSeriesPreprocessor, DataSplitter
from models.forecasters import ModelFactory
from evaluation.metrics import ForecastEvaluator
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

class ForecastingPipeline:
    """Main forecasting pipeline orchestrator"""
    
    def __init__(self, config_path: str):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()
        self.models = {}
        self.results = {}
        
        # Setup MLflow
        self._setup_experiment_tracking()
        
    def _setup_experiment_tracking(self):
        """Setup experiment tracking"""
        tracking_config = self.config.experiment_tracking
        
        if tracking_config.get('backend') == 'mlflow':
            mlflow.set_tracking_uri(tracking_config.get('tracking_uri', 'sqlite:///mlruns.db'))
            mlflow.set_experiment(tracking_config.get('experiment_name', 'forecasting_experiment'))
            
    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete forecasting pipeline"""
        logger.info("Starting forecasting pipeline...")
        
        # 1. Load and preprocess data
        logger.info("Loading and preprocessing data...")
        train_data, val_data, test_data = self._load_and_preprocess_data()
        
        # 2. Train models
        logger.info("Training models...")
        self._train_models(train_data, val_data)
        
        # 3. Evaluate models
        logger.info("Evaluating models...")
        evaluation_results = self._evaluate_models(test_data)
        
        # 4. Select best model
        best_model_name = self._select_best_model(evaluation_results)
        
        logger.info(f"Pipeline completed. Best model: {best_model_name}")
        
        return {
            'best_model': best_model_name,
            'evaluation_results': evaluation_results,
            'models': self.models
        }
        
    def _load_and_preprocess_data(self) -> tuple:
        """Load and preprocess the data"""
        # Load data
        data_loader = DataLoader(self.config.data.source_type)
        raw_data = data_loader.load_data(self.config.data.source_path)
        
        logger.info(f"Loaded {len(raw_data)} records")
        
        # Preprocess data
        preprocessor = TimeSeriesPreprocessor(
            target_column=self.config.data.target_column,
            time_column=self.config.data.time_column,
            feature_columns=self.config.data.feature_columns
        )
        
        processed_data = preprocessor.preprocess(raw_data, self.config.data.preprocessing)
        
        # Split data
        train_data, val_data, test_data = DataSplitter.time_series_split(
            processed_data, 
            self.config.data.time_column,
            train_ratio=0.7,
            val_ratio=0.15
        )
        
        logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
        
    def _prepare_model_data(self, data: pd.DataFrame, algorithm: str) -> tuple:
        """Prepare data for specific model type"""
        
        feature_columns = [col for col in data.columns 
                          if col not in [self.config.data.time_column, self.config.data.target_column]]
        
        if algorithm.lower() == 'prophet':
            # Prophet expects specific format
            prophet_data = data[[self.config.data.time_column, self.config.data.target_column]].copy()
            prophet_data.columns = ['ds', 'y']
            return prophet_data, None
            
        elif algorithm.lower() in ['lstm']:
            # Deep learning models need sequences
            feature_data = data[feature_columns].values
            target_data = data[self.config.data.target_column].values
            
            # Create sequences (assuming we want to predict 1 step ahead)
            sequence_length = 48  # Default sequence length
            X, y = DataSplitter.create_sequences(
                np.column_stack([target_data, feature_data]), 
                sequence_length,
                target_column_idx=0
            )
            return X, y
            
        else:
            # Traditional ML models
            X = data[feature_columns].values
            y = data[self.config.data.target_column].values
            return X, y
            
    def _train_models(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
        """Train all configured models"""
        
        for model_config in self.config.models:
            logger.info(f"Training model: {model_config.name}")
            
            with mlflow.start_run(run_name=model_config.name):
                try:
                    # Create model
                    model = ModelFactory.create_model(
                        model_config.algorithm,
                        model_config.name,
                        model_config.hyperparameters
                    )
                    
                    # Prepare data
                    X_train, y_train = self._prepare_model_data(train_data, model_config.algorithm)
                    X_val, y_val = self._prepare_model_data(val_data, model_config.algorithm)
                    
                    # Train model
                    model.fit(X_train, y_train, X_val, y_val)
                    
                    # Log parameters
                    mlflow.log_params(model_config.hyperparameters)
                    
                    # Save model
                    model_path = f"models/{model_config.name}"
                    os.makedirs(model_path, exist_ok=True)
                    model.save_model(f"{model_path}/model.pkl")
                    
                    self.models[model_config.name] = model
                    
                    logger.info(f"Successfully trained {model_config.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to train {model_config.name}: {e}")
                    continue
                    
    def _evaluate_models(self, test_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models"""
        
        evaluation_results = {}
        evaluator = ForecastEvaluator()
        
        for model_name, model in self.models.items():
            logger.info(f"Evaluating model: {model_name}")
            
            try:
                # Get model config
                model_config = next(m for m in self.config.models if m.name == model_name)
                
                # Prepare test data
                X_test, y_test = self._prepare_model_data(test_data, model_config.algorithm)
                
                # Make predictions
                if model_config.algorithm.lower() == 'prophet':
                    # For Prophet, we need future dates
                    future_dates = model.model.make_future_dataframe(periods=len(X_test))
                    predictions = model.predict(future_dates[-len(X_test):])
                else:
                    predictions = model.predict(X_test)
                
                # Calculate metrics
                metrics = evaluator.calculate_all_metrics(y_test, predictions)
                evaluation_results[model_name] = metrics
                
                # Log metrics to MLflow
                with mlflow.start_run(run_name=f"{model_name}_evaluation"):
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(metric_name, metric_value)
                
                logger.info(f"Model {model_name} - RMSE: {metrics.get('rmse', 'N/A'):.4f}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                continue
                
        return evaluation_results
        
    def _select_best_model(self, evaluation_results: Dict[str, Dict[str, float]]) -> str:
        """Select the best performing model based on RMSE"""
        
        if not evaluation_results:
            logger.warning("No evaluation results available")
            return None
            
        best_model = min(evaluation_results.items(), 
                        key=lambda x: x[1].get('rmse', float('inf')))
        
        return best_model[0]

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="ML Forecasting Pipeline")
    parser.add_argument("--config", 
                       default="config/default_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--log-level", 
                       default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Generate sample data if it doesn't exist
    from data.preprocessing import generate_sample_data
    sample_data_path = "data/sample_energy_data.csv"
    if not os.path.exists(sample_data_path):
        logger.info("Generating sample data...")
        generate_sample_data(sample_data_path)
    
    # Run pipeline
    try:
        pipeline = ForecastingPipeline(args.config)
        results = pipeline.run_pipeline()
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Best model: {results['best_model']}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
