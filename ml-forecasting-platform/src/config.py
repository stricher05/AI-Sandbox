"""
ML Forecasting Platform - Core Configuration Module
Manages configuration for different environments and cloud providers
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class ModelConfig:
    """Model configuration settings"""
    name: str
    algorithm: str  # lstm, xgboost, prophet, etc.
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_params: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class DataConfig:
    """Data configuration settings"""
    source_type: str  # local, s3, azure_blob, gcs
    source_path: str
    target_column: str
    time_column: str
    feature_columns: list = field(default_factory=list)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class DeploymentConfig:
    """Deployment configuration settings"""
    platform: str  # local, aws, azure, gcp
    compute_type: str  # cpu, gpu
    scaling: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class ForecastConfig:
    """Main forecasting configuration"""
    project_name: str
    data: DataConfig
    models: list[ModelConfig] = field(default_factory=list)
    deployment: DeploymentConfig = None
    experiment_tracking: Dict[str, Any] = field(default_factory=dict)

class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/default_config.yaml"
        self.config = None
        
    def load_config(self) -> ForecastConfig:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
            
        return self._dict_to_config(config_dict)
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ForecastConfig:
        """Convert dictionary to configuration objects"""
        # Parse data config
        data_config = DataConfig(**config_dict['data'])
        
        # Parse model configs
        model_configs = [
            ModelConfig(**model_dict) 
            for model_dict in config_dict.get('models', [])
        ]
        
        # Parse deployment config
        deployment_config = None
        if 'deployment' in config_dict:
            deployment_config = DeploymentConfig(**config_dict['deployment'])
            
        return ForecastConfig(
            project_name=config_dict['project_name'],
            data=data_config,
            models=model_configs,
            deployment=deployment_config,
            experiment_tracking=config_dict.get('experiment_tracking', {})
        )

def get_cloud_config(platform: str) -> Dict[str, Any]:
    """Get cloud-specific configuration"""
    cloud_configs = {
        'aws': {
            'compute_service': 'sagemaker',
            'storage_service': 's3',
            'model_registry': 'sagemaker_model_registry',
            'instance_types': ['ml.t3.medium', 'ml.m5.large', 'ml.c5.xlarge']
        },
        'azure': {
            'compute_service': 'azure_ml',
            'storage_service': 'blob_storage',
            'model_registry': 'azure_ml_registry',
            'instance_types': ['Standard_DS2_v2', 'Standard_DS3_v2', 'Standard_NC6']
        },
        'gcp': {
            'compute_service': 'vertex_ai',
            'storage_service': 'cloud_storage',
            'model_registry': 'vertex_model_registry',
            'instance_types': ['n1-standard-2', 'n1-standard-4', 'n1-highmem-2']
        },
        'local': {
            'compute_service': 'local',
            'storage_service': 'local_filesystem',
            'model_registry': 'mlflow',
            'instance_types': ['cpu', 'gpu']
        }
    }
    
    return cloud_configs.get(platform, cloud_configs['local'])
