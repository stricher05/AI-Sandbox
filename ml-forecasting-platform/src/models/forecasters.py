"""
Model Factory and Base Classes for ML Forecasting Platform
Supports multiple ML frameworks and algorithms
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import joblib
import logging

# Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Tree-based models
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

# Time series specific
from prophet import Prophet

logger = logging.getLogger(__name__)

class BaseForecaster(ABC):
    """Base class for all forecasting models"""
    
    def __init__(self, name: str, hyperparameters: Dict[str, Any]):
        self.name = name
        self.hyperparameters = hyperparameters
        self.model = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train the model"""
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
        
    def save_model(self, path: str):
        """Save trained model"""
        joblib.dump(self.model, path)
        
    def load_model(self, path: str):
        """Load trained model"""
        self.model = joblib.load(path)
        self.is_fitted = True

class LSTMModel(nn.Module):
    """PyTorch LSTM model"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output from the last time step
        out = self.fc(out[:, -1, :])
        
        return out

class LSTMForecaster(BaseForecaster):
    """LSTM-based forecasting model using PyTorch"""
    
    def __init__(self, name: str, hyperparameters: Dict[str, Any]):
        super().__init__(name, hyperparameters)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train LSTM model"""
        
        # Model parameters
        sequence_length = X_train.shape[1]
        n_features = X_train.shape[2]
        hidden_size = self.hyperparameters.get('hidden_size', 64)
        num_layers = self.hyperparameters.get('num_layers', 2)
        dropout = self.hyperparameters.get('dropout', 0.2)
        learning_rate = self.hyperparameters.get('learning_rate', 0.001)
        epochs = self.hyperparameters.get('epochs', 100)
        batch_size = self.hyperparameters.get('batch_size', 32)
        
        # Create model
        self.model = LSTMModel(n_features, hidden_size, num_layers, dropout).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Prepare data
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if epoch % 10 == 0:
                avg_loss = total_loss / len(train_loader)
                logger.info(f'Epoch {epoch}, Loss: {avg_loss:.6f}')
                
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with LSTM"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
            
        return predictions.cpu().numpy().flatten()

class XGBoostForecaster(BaseForecaster):
    """XGBoost-based forecasting model"""
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train XGBoost model"""
        
        # XGBoost parameters
        params = {
            'n_estimators': self.hyperparameters.get('n_estimators', 1000),
            'max_depth': self.hyperparameters.get('max_depth', 6),
            'learning_rate': self.hyperparameters.get('learning_rate', 0.1),
            'subsample': self.hyperparameters.get('subsample', 0.8),
            'colsample_bytree': self.hyperparameters.get('colsample_bytree', 0.8),
            'reg_alpha': self.hyperparameters.get('reg_alpha', 0.1),
            'reg_lambda': self.hyperparameters.get('reg_lambda', 1.0),
            'random_state': 42
        }
        
        # Handle 3D input (flatten for XGBoost)
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
            if X_val is not None:
                X_val = X_val.reshape(X_val.shape[0], -1)
        
        self.model = xgb.XGBRegressor(**params)
        
        # Training with validation
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=self.hyperparameters.get('early_stopping_rounds', 50),
            verbose=False
        )
        
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with XGBoost"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        # Handle 3D input
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
            
        return self.model.predict(X)

class RandomForestForecaster(BaseForecaster):
    """Random Forest forecasting model"""
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train Random Forest model"""
        
        # Handle 3D input (flatten for sklearn)
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
        
        params = {
            'n_estimators': self.hyperparameters.get('n_estimators', 200),
            'max_depth': self.hyperparameters.get('max_depth', 10),
            'min_samples_split': self.hyperparameters.get('min_samples_split', 5),
            'min_samples_leaf': self.hyperparameters.get('min_samples_leaf', 2),
            'random_state': self.hyperparameters.get('random_state', 42),
            'n_jobs': self.hyperparameters.get('n_jobs', -1)
        }
        
        self.model = RandomForestRegressor(**params)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with Random Forest"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        # Handle 3D input
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
            
        return self.model.predict(X)

class ProphetForecaster(BaseForecaster):
    """Prophet forecasting model"""
    
    def fit(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """Train Prophet model (expects DataFrame with 'ds' and 'y' columns)"""
        
        params = {
            'seasonality_mode': self.hyperparameters.get('seasonality_mode', 'additive'),
            'yearly_seasonality': self.hyperparameters.get('yearly_seasonality', True),
            'weekly_seasonality': self.hyperparameters.get('weekly_seasonality', True),
            'daily_seasonality': self.hyperparameters.get('daily_seasonality', True),
            'holidays_prior_scale': self.hyperparameters.get('holidays_prior_scale', 10.0),
            'changepoint_prior_scale': self.hyperparameters.get('changepoint_prior_scale', 0.05)
        }
        
        self.model = Prophet(**params)
        
        # Prophet expects specific column names
        if isinstance(X_train, pd.DataFrame):
            if 'ds' not in X_train.columns or 'y' not in X_train.columns:
                raise ValueError("Prophet requires DataFrame with 'ds' and 'y' columns")
            self.model.fit(X_train)
        else:
            raise ValueError("Prophet requires pandas DataFrame input")
            
        self.is_fitted = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with Prophet"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        forecast = self.model.predict(X)
        return forecast['yhat'].values

class ModelFactory:
    """Factory class for creating forecasting models"""
    
    @staticmethod
    def create_model(algorithm: str, name: str, hyperparameters: Dict[str, Any]) -> BaseForecaster:
        """Create a forecasting model based on algorithm type"""
        
        if algorithm.lower() == 'lstm':
            return LSTMForecaster(name, hyperparameters)
        elif algorithm.lower() == 'xgboost':
            return XGBoostForecaster(name, hyperparameters)
        elif algorithm.lower() == 'random_forest':
            return RandomForestForecaster(name, hyperparameters)
        elif algorithm.lower() == 'prophet':
            return ProphetForecaster(name, hyperparameters)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    @staticmethod
    def get_supported_algorithms() -> list:
        """Get list of supported algorithms"""
        return ['lstm', 'xgboost', 'random_forest', 'prophet']
