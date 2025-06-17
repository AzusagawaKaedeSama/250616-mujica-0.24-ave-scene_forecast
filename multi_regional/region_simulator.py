"""
Improved Region simulator module for multi-regional load forecasting.
This module simulates multiple regions using load data from different time periods.
"""

import pandas as pd
import numpy as np
from datetime import timedelta


class RegionSimulator:
    """
    Simulates multiple regions by using load data from different time periods.
    """
    
    def __init__(self, data_loader_adapter, n_regions=6):
        """
        Initialize the region simulator.
        
        Args:
            data_loader_adapter: DataLoaderAdapter instance to load data
            n_regions: Number of regions to simulate
        """
        self.data_loader = data_loader_adapter
        self.n_regions = n_regions
        
        # Define region characteristics (based on the paper's description)
        self.region_characteristics = {
            "Region1": {"name": "Shanghai", "type": "service", "weekend_factor": 1.2, "scale": 0.85, "year_offset": 0},
            "Region2": {"name": "Jiangsu", "type": "manufacturing", "weekend_factor": 0.95, "scale": 1.5, "year_offset": 1},
            "Region3": {"name": "Zhejiang", "type": "mixed", "weekend_factor": 1.05, "scale": 1.1, "year_offset": 2},
            "Region4": {"name": "Anhui", "type": "industrial", "weekend_factor": 0.9, "scale": 0.7, "year_offset": 3},
            "Region5": {"name": "Fujian", "type": "coastal", "weekend_factor": 1.1, "scale": 0.8, "year_offset": 1},
            "Region6": {"name": "Jiangxi", "type": "rural", "weekend_factor": 1.0, "scale": 0.6, "year_offset": 2}
        }
            
    def _apply_regional_characteristics(self, data, region_type):
        """
        Apply regional characteristics to the load data.
        
        Args:
            data: DataFrame with load data
            region_type: Type of region to simulate
            
        Returns:
            DataFrame with modified load data
        """
        # Safety check: make sure we have data
        if data is None or data.empty:
            print(f"Warning: No data to apply characteristics for {region_type}")
            return pd.DataFrame(columns=['load'])
        
        modified_data = data.copy()
        
        # Get region settings
        region_setting = self.region_characteristics[region_type]
        
        # Apply weekend factor
        modified_data['dayofweek'] = modified_data.index.dayofweek
        weekend_mask = (modified_data['dayofweek'] >= 5)  # Saturday and Sunday
        modified_data.loc[weekend_mask, 'load'] *= region_setting['weekend_factor']
        
        # Apply scale factor
        modified_data['load'] *= region_setting['scale']
        
        # Add random variations to represent regional characteristics
        np.random.seed(42 + list(self.region_characteristics.keys()).index(region_type))
        random_factor = np.random.normal(1, 0.05, len(modified_data))
        modified_data['load'] *= random_factor
        
        # Drop auxiliary columns
        modified_data = modified_data.drop(columns=['dayofweek'])
        
        return modified_data
    
    def get_date_with_offset(self, base_date, region_key):
        """
        Get date with year offset for a specific region.
        Ensures the resulting date is within the valid data range (2009-01-01 to 2014-12-31).
        
        Args:
            base_date: Base date string or datetime
            region_key: Region identifier
            
        Returns:
            Datetime with year offset applied and constrained to valid range
        """
        base_dt = pd.to_datetime(base_date)
        year_offset = self.region_characteristics[region_key].get("year_offset", 0)
        
        # Valid data range
        min_date = pd.to_datetime('2009-01-01')
        max_date = pd.to_datetime('2014-12-31')
        
        # Apply year offset
        offset_dt = base_dt.replace(year=base_dt.year - year_offset)
        
        # Check if resulting date is within valid range
        if offset_dt < min_date:
            # If too early, use the earliest valid year but keep month/day
            print(f"Warning: Adjusted date for {region_key} from {offset_dt} to valid range")
            offset_dt = offset_dt.replace(year=min_date.year)
            
            # If still too early (because of month/day), use min_date
            if offset_dt < min_date:
                offset_dt = min_date
        
        elif offset_dt > max_date:
            # If too late, use the latest valid year but keep month/day
            print(f"Warning: Adjusted date for {region_key} from {offset_dt} to valid range")
            offset_dt = offset_dt.replace(year=max_date.year)
            
            # If still too late (because of month/day), use max_date
            if offset_dt > max_date:
                offset_dt = max_date - pd.Timedelta(days=1)  # Subtract a day to ensure we're within range
        
        return offset_dt
    
    def generate_regional_data(self, start_date, end_date):
        """
        Generate simulated regional data for the specified date range.
        
        Args:
            start_date: Start date for data generation
            end_date: End date for data generation
            
        Returns:
            Dictionary with regional data and metadata
        """
        regional_data = {}
        metadata = {}
        
        # Target date range for result filtering
        target_start = pd.to_datetime(start_date)
        target_end = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        target_days = (target_end - target_start).days + 1
        
        # For each region
        for i in range(min(self.n_regions, len(self.region_characteristics))):
            region_key = f"Region{i+1}"
            
            # Get region-specific date range with year offset
            region_start = self.get_date_with_offset(start_date, region_key)
            region_end = region_start + timedelta(days=target_days)
            
            print(f"Loading data for {region_key} from {region_start.strftime('%Y-%m-%d')} to {region_end.strftime('%Y-%m-%d')}")
            
            # Load data for this region's time period
            region_data = self.data_loader.load_data_for_period(
                region_start.strftime('%Y-%m-%d'),
                region_end.strftime('%Y-%m-%d')
            )
            
            if region_data is None or region_data.empty:
                print(f"Warning: No data available for {region_key} in period {region_start} to {region_end}")
                continue
            
            # Apply regional characteristics
            modified_data = self._apply_regional_characteristics(region_data, region_key)
            
            # Adjust timestamp to match target date range
            if len(modified_data) > 0:
                # Calculate the time difference between source and target
                time_diff = target_start - region_start
                
                # Shift timestamps
                modified_data.index = modified_data.index + time_diff
                
                # Filter to exact target range
                filtered_data = modified_data[
                    (modified_data.index >= target_start) & 
                    (modified_data.index <= target_end)
                ]
                
                if filtered_data.empty:
                    print(f"Warning: No data available for {region_key} after filtering")
                    continue
                
                regional_data[region_key] = filtered_data
                
                # Store metadata
                metadata[region_key] = {
                    "original_period": f"{region_start} to {region_end}",
                    "characteristics": self.region_characteristics[region_key]
                }
                
                print(f"Successfully generated data for {region_key} with {len(filtered_data)} data points")
            else:
                print(f"Warning: Empty data for {region_key}")
        
        return {
            "data": regional_data,
            "metadata": metadata
        }
    
    def generate_forecast_data(self, start_date, end_date, error_levels=None):
        """
        Generate simulated forecasts for each region.
        
        Args:
            start_date: Start date for forecast data
            end_date: End date for forecast data
            error_levels: Dictionary mapping region keys to error levels (MAPE)
            
        Returns:
            Dictionary with forecasted data and actual data
        """
        # Default error levels based on paper
        if error_levels is None:
            error_levels = {
                "Region1": 0.0524,  # Shanghai: 5.24%
                "Region2": 0.0707,  # Jiangsu: 7.07% 
                "Region3": 0.1551,  # Zhejiang: 15.51%
                "Region4": 0.0386,  # Anhui: 3.86%
                "Region5": 0.0784,  # Fujian: 7.84%
                "Region6": 0.0720   # Jiangxi: 7.20%
            }
        
        # Generate actual regional data
        actual_data = self.generate_regional_data(start_date, end_date)
        
        if not actual_data["data"]:
            print("Error: No actual data generated")
            return {
                "actual": {},
                "forecast": {},
                "metadata": {}
            }
        
        # Generate forecasts by adding errors to actual data
        forecasts = {}
        for region_key, region_data in actual_data["data"].items():
            error_level = error_levels.get(region_key, 0.05)  # Default 5% error
            
            # Add random errors
            np.random.seed(42 + list(self.region_characteristics.keys()).index(region_key))
            errors = np.random.normal(1, error_level, len(region_data))
            
            # Create forecast with errors
            forecast = region_data.copy()
            forecast['load'] *= errors
            
            forecasts[region_key] = forecast
        
        return {
            "actual": actual_data["data"],
            "forecast": forecasts,
            "metadata": actual_data["metadata"]
        }