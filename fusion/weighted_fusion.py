"""
Weighted fusion module for multi-regional load forecasting.
"""

import numpy as np
import pandas as pd


class WeightedFusion:
    """
    Implements the weighted fusion methodology for integrating provincial forecasts.
    """
    
    def __init__(self, base_adjustment_coefficient=0.5):
        """
        Initialize the weighted fusion calculator.
        
        Args:
            base_adjustment_coefficient: Base adjustment coefficient (beta)
        """
        self.beta = base_adjustment_coefficient
    
    def direct_aggregation(self, forecasts):
        """
        Perform direct aggregation of forecasts (baseline method).
        
        Args:
            forecasts: Dictionary with forecasted load data for each region
            
        Returns:
            DataFrame with aggregated forecast
        """
        # Initialize with first region's data
        regions = list(forecasts.keys())
        if not regions:
            return pd.DataFrame()
        
        # Start with first region
        aggregated = forecasts[regions[0]].copy()
        
        # Add remaining regions
        for region in regions[1:]:
            aggregated['load'] += forecasts[region]['load']
        
        return aggregated
    
    def calculate_dynamic_adjustment(self, weights, historical_accuracy):
        """
        Calculate dynamic adjustment coefficient alpha.
        
        Args:
            weights: Dictionary with weights for each region
            historical_accuracy: Dictionary with historical accuracy for each region
            
        Returns:
            Dynamic adjustment coefficient
        """
        adjustment = self.beta * sum(
            historical_accuracy.get(region, 0.8) * weight  # Default accuracy 0.8
            for region, weight in weights.items()
        )
        
        # Ensure adjustment is within reasonable range
        adjustment = max(0.3, min(adjustment, 0.7))
        
        return adjustment
    
    def weighted_fusion_v0(self, provincial_forecasts, main_grid_forecasts, weights, historical_accuracy=None):
        """
        Perform weighted fusion of provincial and main grid forecasts.
        
        Args:
            provincial_forecasts: Dictionary with provincial forecasted load data
            main_grid_forecasts: Dictionary with main grid forecasted load data for provinces
            weights: Dictionary or DataFrame with time-varying weights for each province
            historical_accuracy: Dictionary with historical accuracy for each province
            
        Returns:
            DataFrame with fused forecast
        """
        if historical_accuracy is None:
            # Default historical accuracy
            historical_accuracy = {region: 0.8 for region in provincial_forecasts.keys()}
        
        # Get all regions
        regions = list(provincial_forecasts.keys())
        if not regions:
            return pd.DataFrame()
        
        # Use first region's timepoints as reference
        reference_data = provincial_forecasts[regions[0]]
        
        # Check if weights is a DataFrame (time-varying)
        is_time_varying = isinstance(weights, pd.DataFrame)
        
        if is_time_varying:
            print(f"Using time-varying weights with shape {weights.shape}")
            # Debug info - show first few rows
            print("First 3 rows of time-varying weights:")
            print(weights.head(3))
        else:
            print("Using static weights")
            print(weights)
        
        # Initialize result DataFrame and weight tracking
        fused_forecast = pd.DataFrame(index=reference_data.index, columns=['load'])
        fused_forecast['load'] = 0.0
        
        if is_time_varying:
            self.tracked_weights = pd.DataFrame(index=reference_data.index, columns=regions)
        
        # For each time point
        for idx, timestamp in enumerate(reference_data.index):
            # Get weights for this timestamp
            if is_time_varying:
                try:
                    # Get time-varying weights - match by exact timestamp
                    if timestamp in weights.index:
                        timestamp_weights = {r: weights.loc[timestamp, r] for r in regions if r in weights.columns}
                    else:
                        # Try to find nearest timestamp
                        nearest_idx = weights.index.get_indexer([timestamp], method='nearest')[0]
                        nearest_timestamp = weights.index[nearest_idx]
                        timestamp_weights = {r: weights.loc[nearest_timestamp, r] for r in regions if r in weights.columns}
                    
                    # Track weights for debugging
                    for r in regions:
                        if r in timestamp_weights:
                            self.tracked_weights.loc[timestamp, r] = timestamp_weights[r]
                except Exception as e:
                    print(f"Error getting time-varying weights for timestamp {timestamp}: {e}")
                    # Fall back to uniform weights
                    timestamp_weights = {r: 1.0/len(regions) for r in regions}
            else:
                # Static weights
                timestamp_weights = weights
            
            # Calculate alpha (dynamic adjustment coefficient)
            alpha = self.calculate_dynamic_adjustment(timestamp_weights, historical_accuracy)
            
            # Calculate forecast for each region individually
            weighted_forecast = 0.0
            total_region_contribution = 0.0
            
            for region in regions:
                # Get weight for this region
                region_weight = timestamp_weights.get(region, 1.0/len(regions))
                
                # Get load values safely
                try:
                    prov_load = provincial_forecasts[region].loc[timestamp, 'load']
                    if region in main_grid_forecasts:
                        main_load = main_grid_forecasts[region].loc[timestamp, 'load']
                    else:
                        main_load = prov_load
                except Exception as e:
                    print(f"Error accessing forecast data for {region} at {timestamp}: {e}")
                    continue
                
                # Calculate region's contribution
                denominator = alpha * region_weight + (1 - alpha) * (1 - region_weight)
                if denominator > 0:
                    region_contrib = (alpha * region_weight * prov_load + 
                                    (1 - alpha) * (1 - region_weight) * main_load) / denominator
                    weighted_forecast += region_contrib
                    total_region_contribution += 1
            
            # Store the weighted forecast
            if total_region_contribution > 0:
                fused_forecast.loc[timestamp, 'load'] = weighted_forecast
        
        return fused_forecast
    
    def weighted_fusion(self, provincial_forecasts, main_grid_forecasts, weights, historical_accuracy=None):
        """
        Perform weighted fusion of provincial and main grid forecasts.
        
        Args:
            provincial_forecasts: Dictionary with provincial forecasted load data
            main_grid_forecasts: Dictionary with main grid forecasted load data for provinces
            weights: Dictionary or DataFrame with time-varying weights for each province
            historical_accuracy: Dictionary with historical accuracy for each province
            
        Returns:
            DataFrame with fused forecast
        """
        if historical_accuracy is None:
            # Default historical accuracy
            historical_accuracy = {region: 0.8 for region in provincial_forecasts.keys()}
        
        # Get all regions
        regions = list(provincial_forecasts.keys())
        if not regions:
            return pd.DataFrame()
        
        # Use first region's timepoints as reference
        reference_data = provincial_forecasts[regions[0]]
        
        # Check if weights is a DataFrame (time-varying)
        is_time_varying = isinstance(weights, pd.DataFrame)
        
        if is_time_varying:
            print(f"Using time-varying weights with shape {weights.shape}")
            # Debug info - show first few rows
            print("First 3 rows of time-varying weights:")
            print(weights.head(3))
        else:
            print("Using static weights")
            print(weights)
        
        # Initialize result DataFrame and weight tracking
        fused_forecast = pd.DataFrame(index=reference_data.index, columns=['load'])
        fused_forecast['load'] = 0.0
        
        if is_time_varying:
            self.tracked_weights = pd.DataFrame(index=reference_data.index, columns=regions)
        
        # For each time point
        for idx, timestamp in enumerate(reference_data.index):
            # Get weights for this timestamp
            if is_time_varying:
                try:
                    # Get time-varying weights - match by exact timestamp
                    if timestamp in weights.index:
                        timestamp_weights = {r: weights.loc[timestamp, r] for r in regions if r in weights.columns}
                    else:
                        # Try to find nearest timestamp
                        nearest_idx = weights.index.get_indexer([timestamp], method='nearest')[0]
                        nearest_timestamp = weights.index[nearest_idx]
                        timestamp_weights = {r: weights.loc[nearest_timestamp, r] for r in regions if r in weights.columns}
                    
                    # Track weights for debugging
                    for r in regions:
                        if r in timestamp_weights:
                            self.tracked_weights.loc[timestamp, r] = timestamp_weights[r]
                except Exception as e:
                    print(f"Error getting time-varying weights for timestamp {timestamp}: {e}")
                    # Fall back to uniform weights
                    timestamp_weights = {r: 1.0/len(regions) for r in regions}
            else:
                # Static weights
                timestamp_weights = weights
            
            # Calculate alpha (dynamic adjustment coefficient)
            alpha = self.calculate_dynamic_adjustment(timestamp_weights, historical_accuracy)
            
            # Calculate forecast for each region individually
            weighted_forecast = 0.0
            total_region_contribution = 0.0
            valid_regions = 0
            
            for region in regions:
                # Get weight for this region
                region_weight = timestamp_weights.get(region, 1.0/len(regions))
                
                # Get load values safely
                try:
                    # First check if timestamp exists in the DataFrame
                    if timestamp in provincial_forecasts[region].index:
                        prov_load = provincial_forecasts[region].loc[timestamp, 'load']
                        
                        if region in main_grid_forecasts and timestamp in main_grid_forecasts[region].index:
                            main_load = main_grid_forecasts[region].loc[timestamp, 'load']
                        else:
                            main_load = prov_load
                            
                        # Calculate region's contribution
                        denominator = alpha * region_weight + (1 - alpha) * (1 - region_weight)
                        if denominator > 0:
                            region_contrib = (alpha * region_weight * prov_load + 
                                            (1 - alpha) * (1 - region_weight) * main_load) / denominator
                            weighted_forecast += region_contrib
                            total_region_contribution += 1
                            valid_regions += 1
                    else:
                        print(f"Timestamp {timestamp} not found in data for {region} - skipping")
                except Exception as e:
                    print(f"Error processing {region} at {timestamp}: {e}")
                    continue
            
            # Only store the weighted forecast if we have valid data from at least one region
            if valid_regions > 0:
                fused_forecast.loc[timestamp, 'load'] = weighted_forecast
        
        return fused_forecast

    def generate_main_grid_forecasts(self, actual_data, provincial_forecasts, error_level=0.03):
        """
        Generate simulated main grid forecasts for each province.
        
        Args:
            actual_data: Dictionary with actual load data for each province
            provincial_forecasts: Dictionary with provincial forecasted load data
            error_level: Error level for main grid forecasts (MAPE)
            
        Returns:
            Dictionary with main grid forecasted load data
        """
        main_grid_forecasts = {}
        
        for region, actual in actual_data.items():
            # Add random errors to actual data
            np.random.seed(42 + list(actual_data.keys()).index(region) + 100)  # Different seed than provincial
            errors = np.random.normal(1, error_level, len(actual))
            
            # Create forecast with errors
            forecast = actual.copy()
            forecast['load'] *= errors
            
            main_grid_forecasts[region] = forecast
        
        return main_grid_forecasts
    
    def evaluate_forecast(self, actual, forecast):
        """
        Evaluate forecast performance.
        
        Args:
            actual: DataFrame with actual load data
            forecast: DataFrame with forecasted load data
            
        Returns:
            Dictionary with performance metrics
        """
        # Ensure both dataframes have the same length
        min_length = min(len(actual), len(forecast))
        actual_load = actual['load'].iloc[:min_length]
        forecast_load = forecast['load'].iloc[:min_length]
        
        # Calculate APE (Absolute Percentage Error)
        ape = np.abs((actual_load - forecast_load) / actual_load)
        
        # Calculate metrics
        mape = ape.mean()  # Mean Absolute Percentage Error
        mspe = (ape ** 2).mean()  # Mean Squared Percentage Error
        mae = np.abs(actual_load - forecast_load).mean()  # Mean Absolute Error
        rmse = np.sqrt(((actual_load - forecast_load) ** 2).mean())  # Root Mean Square Error
        
        # Calculate R² (coefficient of determination)
        ss_tot = ((actual_load - actual_load.mean()) ** 2).sum()
        ss_res = ((actual_load - forecast_load) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            "MAPE": mape,
            "MSPE": mspe,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }
    
    def compare_methods(self, actual_data, provincial_forecasts, weights):
        """
        Compare direct aggregation vs weighted fusion methods.
        
        Args:
            actual_data: Dictionary with actual load data for each province
            provincial_forecasts: Dictionary with provincial forecasted load data
            weights: Dictionary or DataFrame with weights for each province
            
        Returns:
            Dictionary with comparison results
        """
        # Generate aggregate of actual data (ground truth)
        actual_aggregate = self.direct_aggregation(actual_data)
        
        # Method 1: Direct aggregation
        direct_aggregate = self.direct_aggregation(provincial_forecasts)
        direct_performance = self.evaluate_forecast(actual_aggregate, direct_aggregate)
        
        # Generate main grid forecasts
        main_grid_forecasts = self.generate_main_grid_forecasts(actual_data, provincial_forecasts)
        
        # Calculate historical accuracy (based on provincial forecasts vs actual)
        historical_accuracy = {}
        for region in provincial_forecasts.keys():
            metrics = self.evaluate_forecast(actual_data[region], provincial_forecasts[region])
            historical_accuracy[region] = 1 - metrics["MAPE"]  # Convert MAPE to accuracy
        
        # Method 2: Weighted fusion
        fused_forecast = self.weighted_fusion(
            provincial_forecasts, 
            main_grid_forecasts, 
            weights, 
            historical_accuracy
        )
        fusion_performance = self.evaluate_forecast(actual_aggregate, fused_forecast)
        
        # Calculate improvement metrics
        improvement = {}
        for metric in direct_performance.keys():
            if metric in ["MAPE", "MSPE", "MAE", "RMSE"]:
                # For these metrics, lower is better
                improvement[metric] = (direct_performance[metric] - fusion_performance[metric]) / direct_performance[metric]
            else:
                # For metrics like R2, higher is better
                improvement[metric] = (fusion_performance[metric] - direct_performance[metric]) / direct_performance[metric]
        
        return {
            "actual_aggregate": actual_aggregate,
            "direct_aggregate": direct_aggregate,
            "fused_forecast": fused_forecast,
            "direct_performance": direct_performance,
            "fusion_performance": fusion_performance,
            "improvement": improvement
        }