"""
Direct aggregation module for multi-regional load forecasting.
This module implements the baseline method of direct accumulation.
"""

import pandas as pd
import numpy as np


class DirectAggregation:
    """
    Implements direct aggregation of load forecasts from multiple regions.
    This serves as the baseline for comparison with weighted fusion.
    """
    
    def __init__(self):
        """Initialize the direct aggregation calculator."""
        pass
    
    def aggregate(self, forecasts):
        """
        Aggregate forecasts by direct summation.
        
        Args:
            forecasts: Dictionary with forecasted load data for each region
            
        Returns:
            DataFrame with aggregated forecast
        """
        # Get all regions
        regions = list(forecasts.keys())
        if not regions:
            return pd.DataFrame()
        
        # Start with first region
        aggregated = forecasts[regions[0]].copy()
        
        # Add remaining regions
        for region in regions[1:]:
            aggregated['load'] += forecasts[region]['load']
        
        return aggregated
    
    def evaluate_performance(self, actual_aggregate, forecast_aggregate):
        """
        Evaluate performance of aggregated forecast.
        
        Args:
            actual_aggregate: DataFrame with actual aggregated load
            forecast_aggregate: DataFrame with forecasted aggregated load
            
        Returns:
            Dictionary with performance metrics
        """
        # Ensure both dataframes have the same length
        min_length = min(len(actual_aggregate), len(forecast_aggregate))
        actual_load = actual_aggregate['load'].iloc[:min_length]
        forecast_load = forecast_aggregate['load'].iloc[:min_length]
        
        # Calculate metrics
        mape = np.mean(np.abs((actual_load - forecast_load) / actual_load))
        rmse = np.sqrt(np.mean((actual_load - forecast_load) ** 2))
        mae = np.mean(np.abs(actual_load - forecast_load))
        
        return {
            "MAPE": mape,
            "RMSE": rmse,
            "MAE": mae
        }