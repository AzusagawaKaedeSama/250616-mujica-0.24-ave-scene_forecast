"""
Adapter for existing DataLoader to ensure compatibility with RegionSimulator.
Customized for the specific DataLoader implementation.
"""

import pandas as pd


class DataLoaderAdapter:
    """
    Adapter class to make the existing DataLoader compatible with the RegionSimulator.
    This works with the provided DataLoader class that has load_raw_data and preprocess methods.
    """
    
    def __init__(self, data_loader):
        """
        Initialize with an existing DataLoader.
        
        Args:
            data_loader: Existing DataLoader instance
        """
        self.data_loader = data_loader
        self._load_data = None
    
    def _ensure_data_loaded(self):
        """Make sure the data is loaded and preprocessed."""
        if self._load_data is None:
            # Load and preprocess data using the provided methods
            self._load_data = self.data_loader.preprocess()
            
            # Make sure we have a DataFrame with 'load' column
            if isinstance(self._load_data, pd.Series):
                self._load_data = self._load_data.to_frame(name='load')
    
    def load_data_for_period(self, start_date, end_date):
        """
        Load data for a specific period.
        
        Args:
            start_date: Start date string
            end_date: End date string
            
        Returns:
            DataFrame with load data
        """
        # Make sure data is loaded
        self._ensure_data_loaded()
        
        # Convert dates to datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        # Filter by date range
        filtered_data = self._load_data[(self._load_data.index >= start) & 
                                        (self._load_data.index <= end)]
        
        if filtered_data.empty:
            print(f"Warning: No data found for period {start_date} to {end_date}")
        
        return filtered_data