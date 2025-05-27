"""
Evaluation Metrics Module for ML Forecasting Platform
Provides comprehensive metrics for forecast evaluation
"""

import numpy as np
from typing import Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class ForecastEvaluator:
    """Comprehensive forecasting evaluation metrics"""
    
    def __init__(self):
        self.metrics = {}
        
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all available forecasting metrics"""
        
        metrics = {}
        
        # Basic regression metrics
        metrics['mse'] = self.mean_squared_error(y_true, y_pred)
        metrics['rmse'] = self.root_mean_squared_error(y_true, y_pred)
        metrics['mae'] = self.mean_absolute_error(y_true, y_pred)
        metrics['r2'] = self.r2_score(y_true, y_pred)
        
        # Forecasting-specific metrics
        metrics['mape'] = self.mean_absolute_percentage_error(y_true, y_pred)
        metrics['smape'] = self.symmetric_mean_absolute_percentage_error(y_true, y_pred)
        metrics['mase'] = self.mean_absolute_scaled_error(y_true, y_pred)
        metrics['wape'] = self.weighted_absolute_percentage_error(y_true, y_pred)
        
        # Statistical tests
        metrics['bias'] = self.bias(y_true, y_pred)
        metrics['forecast_accuracy'] = self.forecast_accuracy(y_true, y_pred)
        
        return metrics
    
    def mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error"""
        return float(mean_squared_error(y_true, y_pred))
    
    def root_mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    def mean_absolute_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return float(mean_absolute_error(y_true, y_pred))
    
    def r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared Score"""
        return float(r2_score(y_true, y_pred))
    
    def mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error"""
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return float('inf')
        
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    
    def symmetric_mean_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error"""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        
        # Avoid division by zero
        mask = denominator != 0
        if not np.any(mask):
            return float('inf')
            
        return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100)
    
    def mean_absolute_scaled_error(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  seasonal_period: int = 1) -> float:
        """Mean Absolute Scaled Error"""
        # Calculate naive forecast error (seasonal naive)
        if len(y_true) <= seasonal_period:
            # Fall back to simple naive forecast
            naive_error = np.mean(np.abs(np.diff(y_true)))
        else:
            naive_forecast = y_true[:-seasonal_period]
            naive_actual = y_true[seasonal_period:]
            naive_error = np.mean(np.abs(naive_actual - naive_forecast))
        
        if naive_error == 0:
            return float('inf')
            
        mae = self.mean_absolute_error(y_true, y_pred)
        return float(mae / naive_error)
    
    def weighted_absolute_percentage_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Weighted Absolute Percentage Error"""
        sum_actual = np.sum(np.abs(y_true))
        if sum_actual == 0:
            return float('inf')
            
        return float(np.sum(np.abs(y_true - y_pred)) / sum_actual * 100)
    
    def bias(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Forecast Bias"""
        return float(np.mean(y_pred - y_true))
    
    def forecast_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Forecast Accuracy (100 - MAPE)"""
        mape = self.mean_absolute_percentage_error(y_true, y_pred)
        if mape == float('inf'):
            return 0.0
        return float(max(0, 100 - mape))
    
    def directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Directional Accuracy - percentage of correct direction predictions"""
        if len(y_true) < 2:
            return 0.0
            
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        
        return float(np.mean(true_direction == pred_direction) * 100)
    
    def create_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               model_name: str = "Model") -> Dict[str, Any]:
        """Create comprehensive evaluation report"""
        
        metrics = self.calculate_all_metrics(y_true, y_pred)
        
        # Add directional accuracy
        metrics['directional_accuracy'] = self.directional_accuracy(y_true, y_pred)
        
        # Create summary
        report = {
            'model_name': model_name,
            'metrics': metrics,
            'sample_size': len(y_true),
            'summary': {
                'performance_tier': self._get_performance_tier(metrics),
                'key_insights': self._generate_insights(metrics, y_true, y_pred)
            }
        }
        
        return report
    
    def _get_performance_tier(self, metrics: Dict[str, float]) -> str:
        """Categorize model performance"""
        mape = metrics.get('mape', float('inf'))
        
        if mape < 5:
            return "Excellent"
        elif mape < 10:
            return "Good"
        elif mape < 20:
            return "Acceptable"
        else:
            return "Poor"
    
    def _generate_insights(self, metrics: Dict[str, float], y_true: np.ndarray, y_pred: np.ndarray) -> list:
        """Generate key insights from metrics"""
        insights = []
        
        # Bias analysis
        bias = metrics.get('bias', 0)
        if abs(bias) > np.std(y_true) * 0.1:
            if bias > 0:
                insights.append("Model tends to overpredict")
            else:
                insights.append("Model tends to underpredict")
        
        # Accuracy analysis
        r2 = metrics.get('r2', 0)
        if r2 > 0.9:
            insights.append("Excellent fit to historical data")
        elif r2 < 0.5:
            insights.append("Poor fit to historical data - consider feature engineering")
        
        # Direction accuracy
        dir_acc = metrics.get('directional_accuracy', 0)
        if dir_acc > 70:
            insights.append("Good at predicting trend direction")
        elif dir_acc < 50:
            insights.append("Poor at predicting trend direction")
        
        return insights
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        title: str = "Forecast vs Actual", save_path: str = None):
        """Plot predictions vs actual values"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Time series plot
        axes[0, 0].plot(y_true, label='Actual', alpha=0.8)
        axes[0, 0].plot(y_pred, label='Predicted', alpha=0.8)
        axes[0, 0].set_title('Time Series Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[0, 1].scatter(y_true, y_pred, alpha=0.6)
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[0, 1].set_xlabel('Actual')
        axes[0, 1].set_ylabel('Predicted')
        axes[0, 1].set_title('Actual vs Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_pred - y_true
        axes[1, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residuals Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.8)
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Residuals Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
