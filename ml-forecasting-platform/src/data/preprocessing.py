"""
Data Processing Module for ML Forecasting Platform
Handles data loading, preprocessing, and feature engineering
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import boto3
from azure.storage.blob import BlobServiceClient
from google.cloud import storage as gcs
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles data loading from various sources"""
    
    def __init__(self, source_type: str = "local"):
        self.source_type = source_type
        
    def load_data(self, source_path: str, **kwargs) -> pd.DataFrame:
        """Load data from specified source"""
        if self.source_type == "local":
            return self._load_local(source_path, **kwargs)
        elif self.source_type == "s3":
            return self._load_s3(source_path, **kwargs)
        elif self.source_type == "azure_blob":
            return self._load_azure_blob(source_path, **kwargs)
        elif self.source_type == "gcs":
            return self._load_gcs(source_path, **kwargs)
        else:
            raise ValueError(f"Unsupported source type: {self.source_type}")
            
    def _load_local(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from local file"""
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path, **kwargs)
        elif file_path.endswith('.parquet'):
            return pd.read_parquet(file_path, **kwargs)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
            
    def _load_s3(self, s3_path: str, **kwargs) -> pd.DataFrame:
        """Load data from AWS S3"""
        try:
            s3_client = boto3.client('s3')
            # Parse S3 path
            bucket, key = s3_path.replace('s3://', '').split('/', 1)
            
            if key.endswith('.csv'):
                obj = s3_client.get_object(Bucket=bucket, Key=key)
                return pd.read_csv(obj['Body'], **kwargs)
            elif key.endswith('.parquet'):
                return pd.read_parquet(f's3://{bucket}/{key}', **kwargs)
            else:
                raise ValueError(f"Unsupported S3 file format: {key}")
        except Exception as e:
            logger.error(f"Failed to load data from S3: {e}")
            raise

class TimeSeriesPreprocessor:
    """Handles time series preprocessing and feature engineering"""
    
    def __init__(self, 
                 target_column: str, 
                 time_column: str,
                 feature_columns: List[str] = None):
        self.target_column = target_column
        self.time_column = time_column
        self.feature_columns = feature_columns or []
        self.scalers = {}
        
    def preprocess(self, 
                   data: pd.DataFrame, 
                   config: Dict[str, Any]) -> pd.DataFrame:
        """Main preprocessing pipeline"""
        df = data.copy()
        
        # Convert time column to datetime
        df[self.time_column] = pd.to_datetime(df[self.time_column])
        df = df.sort_values(self.time_column).reset_index(drop=True)
        
        # Handle missing values
        if config.get('handle_missing') == 'interpolate':
            df = df.interpolate(method='time')
        elif config.get('handle_missing') == 'forward_fill':
            df = df.fillna(method='ffill')
        elif config.get('handle_missing') == 'drop':
            df = df.dropna()
            
        # Add time-based features
        df = self._add_time_features(df)
        
        # Add lag features
        if 'feature_lag_windows' in config:
            df = self._add_lag_features(df, config['feature_lag_windows'])
            
        # Add rolling statistics
        if 'rolling_windows' in config:
            df = self._add_rolling_features(df, config['rolling_windows'])
            
        # Scale features
        if config.get('scale_features', False):
            df = self._scale_features(df, config.get('scaling_method', 'standard'))
            
        # Remove rows with NaN values created by lag features
        df = df.dropna()
        
        return df
        
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df['hour'] = df[self.time_column].dt.hour
        df['day_of_week'] = df[self.time_column].dt.dayofweek
        df['day_of_month'] = df[self.time_column].dt.day
        df['month'] = df[self.time_column].dt.month
        df['quarter'] = df[self.time_column].dt.quarter
        df['year'] = df[self.time_column].dt.year
        
        # Cyclical encoding for better representation
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
        
    def _add_lag_features(self, df: pd.DataFrame, lag_windows: List[int]) -> pd.DataFrame:
        """Add lag features for target and other columns"""
        for lag in lag_windows:
            # Target lags
            df[f'{self.target_column}_lag_{lag}'] = df[self.target_column].shift(lag)
            
            # Feature lags
            for col in self.feature_columns:
                if col in df.columns:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    
        return df
        
    def _add_rolling_features(self, df: pd.DataFrame, rolling_windows: List[int]) -> pd.DataFrame:
        """Add rolling statistics features"""
        for window in rolling_windows:
            # Rolling mean
            df[f'{self.target_column}_rolling_mean_{window}'] = (
                df[self.target_column].rolling(window=window).mean()
            )
            # Rolling std
            df[f'{self.target_column}_rolling_std_{window}'] = (
                df[self.target_column].rolling(window=window).std()
            )
            # Rolling min/max
            df[f'{self.target_column}_rolling_min_{window}'] = (
                df[self.target_column].rolling(window=window).min()
            )
            df[f'{self.target_column}_rolling_max_{window}'] = (
                df[self.target_column].rolling(window=window).max()
            )
            
        return df
        
    def _scale_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Scale numerical features"""
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and time column from scaling
        columns_to_scale = [col for col in numerical_columns 
                           if col not in [self.target_column, self.time_column]]
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
            
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        self.scalers['feature_scaler'] = scaler
        
        # Scale target separately if needed
        target_scaler = StandardScaler()
        df[self.target_column] = target_scaler.fit_transform(
            df[[self.target_column]]
        ).flatten()
        self.scalers['target_scaler'] = target_scaler
        
        return df

class DataSplitter:
    """Handles time series data splitting"""
    
    @staticmethod
    def time_series_split(df: pd.DataFrame, 
                         time_column: str,
                         train_ratio: float = 0.7,
                         val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split time series data chronologically"""
        df_sorted = df.sort_values(time_column).reset_index(drop=True)
        n = len(df_sorted)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df_sorted.iloc[:train_end]
        val_df = df_sorted.iloc[train_end:val_end]
        test_df = df_sorted.iloc[val_end:]
        
        return train_df, val_df, test_df
        
    @staticmethod
    def create_sequences(data: np.ndarray, 
                        sequence_length: int, 
                        target_column_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for deep learning models"""
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i, target_column_idx])
            
        return np.array(X), np.array(y)

def generate_sample_data(file_path: str, n_samples: int = 8760) -> None:
    """Generate sample energy demand data for testing"""
    np.random.seed(42)
    
    # Create hourly timestamps for one year
    timestamps = pd.date_range(
        start='2023-01-01', 
        periods=n_samples, 
        freq='H'
    )
    
    # Create synthetic features
    temperature = 20 + 15 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 365)) + \
                  5 * np.sin(2 * np.pi * np.arange(n_samples) / 24) + \
                  np.random.normal(0, 2, n_samples)
    
    humidity = 50 + 20 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 365 / 4)) + \
               np.random.normal(0, 5, n_samples)
    
    # Create demand with patterns
    hour_pattern = 0.3 * np.sin(2 * np.pi * (timestamps.hour - 6) / 24)
    day_pattern = 0.2 * np.sin(2 * np.pi * timestamps.dayofweek / 7)
    seasonal_pattern = 0.4 * np.sin(2 * np.pi * timestamps.dayofyear / 365)
    
    base_demand = 1000
    demand = (base_demand + 
              200 * hour_pattern + 
              100 * day_pattern + 
              300 * seasonal_pattern + 
              0.5 * temperature + 
              0.2 * humidity + 
              np.random.normal(0, 50, n_samples))
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'demand': demand,
        'temperature': temperature,
        'humidity': humidity,
        'day_of_week': timestamps.dayofweek,
        'hour': timestamps.hour
    })
    
    # Save to file
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)
    logger.info(f"Sample data generated and saved to {file_path}")
