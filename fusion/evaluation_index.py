"""
Evaluation index module for multi-regional load forecasting.
Implements the hierarchical evaluation index system described in the paper.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy


class EvaluationIndexCalculator:
    """
    Calculator for the three-level evaluation index system:
    1. Forecast Reliability
    2. Provincial Load Impact
    3. Forecasting Complexity
    """
    
    def __init__(self):
        # Define the index structure with default weights
        self.index_structure = {
            "ForecastReliability": {
                "weight": 0.35,
                "sub_indices": {
                    "HistoricalForecastPerformance": {
                        "weight": 0.5,
                        "metrics": ["day_ahead_accuracy", "realtime_accuracy", "extreme_weather_accuracy"]
                    },
                    "ForecastingSystemStability": {
                        "weight": 0.3,
                        "metrics": ["system_update_frequency", "forecast_result_continuity", "abnormal_forecast_proportion"]
                    },
                    "DataQualityLevel": {
                        "weight": 0.2,
                        "metrics": ["data_completeness_rate", "data_timeliness", "data_consistency"]
                    }
                }
            },
            "ProvincialLoadImpact": {
                "weight": 0.40,
                "sub_indices": {
                    "LoadScaleProportion": {
                        "weight": 0.6,
                        "metrics": ["maximum_load_proportion", "average_load_proportion", "peak_load_contribution_rate"]
                    },
                    "RegulationCapability": {
                        "weight": 0.4,
                        "metrics": ["peak_regulation_capacity_ratio", "renewable_energy_installation_ratio", "demand_response_capability"]
                    }
                }
            },
            "ForecastingComplexity": {
                "weight": 0.25,
                "sub_indices": {
                    "LoadFluctuationCharacteristics": {
                        "weight": 0.4,
                        "metrics": ["daily_load_fluctuation_rate", "weekly_load_fluctuation_rate", "seasonal_fluctuation_intensity"]
                    },
                    "ExternalFactorSensitivity": {
                        "weight": 0.3,
                        "metrics": ["temperature_sensitivity", "humidity_sensitivity", "holiday_sensitivity"]
                    },
                    "ElectricityConsumptionStructureComplexity": {
                        "weight": 0.3,
                        "metrics": ["industrial_electricity_proportion", "key_users_count", "user_type_diversity"]
                    }
                }
            }
        }
    
    def calculate_data_completeness_rate(self, data):
        """
        Calculate data completeness rate.
        
        Args:
            data: DataFrame with load data
            
        Returns:
            Data completeness rate (0-1)
        """
        try:
            # Count missing values
            missing_values = data['load'].isnull().sum()
            total_values = len(data)
            
            # Calculate completeness rate
            completeness_rate = 1 - (missing_values / total_values) if total_values > 0 else 0
            
            return completeness_rate
        except Exception as e:
            print(f"Error calculating data completeness rate: {e}")
            return 0.9  # Default value
    
    def calculate_data_timeliness(self, data, expected_interval='15min'):
        """
        Calculate data timeliness based on time intervals.
        
        Args:
            data: DataFrame with time-indexed data
            expected_interval: Expected time interval between records
            
        Returns:
            Data timeliness score (0-1)
        """
        try:
            # Check for actual time intervals
            time_diffs = data.index.to_series().diff().dropna()
            expected_diff = pd.Timedelta(expected_interval)
            
            # Calculate timeliness score
            valid_intervals = (time_diffs == expected_diff).sum()
            total_intervals = len(time_diffs)
            
            timeliness_score = valid_intervals / total_intervals if total_intervals > 0 else 0
            
            return timeliness_score
        except Exception as e:
            print(f"Error calculating data timeliness: {e}")
            return 0.8  # Default value
    
    def calculate_data_consistency(self, data):
        """
        Calculate data consistency based on logical patterns.
        
        Args:
            data: DataFrame with load data
            
        Returns:
            Data consistency score (0-1)
        """
        try:
            # Check for logical consistency - load should be positive and within reasonable range
            logical_checks = (data['load'] >= 0) & (data['load'] < data['load'].quantile(0.999) * 1.5)
            
            consistency_score = logical_checks.mean()
            
            return consistency_score
        except Exception as e:
            print(f"Error calculating data consistency: {e}")
            return 0.85  # Default value
    
    def calculate_historical_accuracy(self, actual, forecast):
        """
        Calculate historical forecast accuracy.
        
        Args:
            actual: DataFrame with actual load data
            forecast: DataFrame with forecasted load data
            
        Returns:
            Accuracy score (0-1)
        """
        try:
            if len(actual) != len(forecast):
                min_len = min(len(actual), len(forecast))
                actual = actual.iloc[:min_len]
                forecast = forecast.iloc[:min_len]
            
            # Calculate MAPE
            absolute_percentage_errors = np.abs((actual['load'] - forecast['load']) / actual['load'])
            absolute_percentage_errors = absolute_percentage_errors.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(absolute_percentage_errors) == 0:
                return 0.8  # Default if no valid calculations
                
            mape = absolute_percentage_errors.mean()
            
            # Convert to accuracy (0-1 scale, where 1 is perfect)
            accuracy = max(0, 1 - mape)
            
            return accuracy
        except Exception as e:
            print(f"Error calculating historical accuracy: {e}")
            return 0.8  # Default value
    
    def calculate_load_proportion(self, region_data, all_regions_data):
        """
        Calculate load proportion metrics.
        
        Args:
            region_data: DataFrame with region load data
            all_regions_data: Dictionary with all regions' load data
            
        Returns:
            Dictionary with load proportion metrics
        """
        try:
            # Calculate total load across all regions
            total_load = pd.DataFrame()
            for region, data in all_regions_data.items():
                if total_load.empty:
                    total_load = data.copy()
                else:
                    total_load['load'] += data['load']
            
            # Calculate maximum load proportion
            region_max_load = region_data['load'].max()
            total_max_load = total_load['load'].max()
            max_load_proportion = region_max_load / total_max_load if total_max_load > 0 else 0
            
            # Calculate average load proportion
            region_avg_load = region_data['load'].mean()
            total_avg_load = total_load['load'].mean()
            avg_load_proportion = region_avg_load / total_avg_load if total_avg_load > 0 else 0
            
            # Calculate peak load contribution
            # Identify the time of system peak
            system_peak_time = total_load['load'].idxmax()
            # Get region's load at that time
            if system_peak_time in region_data.index:
                region_load_at_peak = region_data.loc[system_peak_time, 'load']
                peak_load_contribution = region_load_at_peak / total_load.loc[system_peak_time, 'load']
            else:
                peak_load_contribution = avg_load_proportion  # Fallback
            
            return {
                "maximum_load_proportion": max_load_proportion,
                "average_load_proportion": avg_load_proportion,
                "peak_load_contribution_rate": peak_load_contribution
            }
        except Exception as e:
            print(f"Error calculating load proportion: {e}")
            return {
                "maximum_load_proportion": 0.25,
                "average_load_proportion": 0.25,
                "peak_load_contribution_rate": 0.25
            }
    
    def calculate_load_fluctuation(self, data):
        """
        Calculate load fluctuation metrics.
        
        Args:
            data: DataFrame with load data
            
        Returns:
            Dictionary with load fluctuation metrics
        """
        try:
            # Daily load fluctuation rate
            daily_std = data.groupby(data.index.date)['load'].std()
            daily_mean = data.groupby(data.index.date)['load'].mean()
            daily_fluctuation = (daily_std / daily_mean).mean()
            
            # Weekly load fluctuation rate
            weekly_std = data.groupby(data.index.isocalendar().week)['load'].std()
            weekly_mean = data.groupby(data.index.isocalendar().week)['load'].mean()
            weekly_fluctuation = (weekly_std / weekly_mean).mean()
            
            # Seasonal fluctuation (estimate based on available data)
            monthly_mean = data.groupby(data.index.month)['load'].mean()
            overall_mean = data['load'].mean()
            seasonal_fluctuation = monthly_mean.std() / overall_mean
            
            # Handle NaN values
            daily_fluctuation = 0.2 if np.isnan(daily_fluctuation) else daily_fluctuation
            weekly_fluctuation = 0.15 if np.isnan(weekly_fluctuation) else weekly_fluctuation
            seasonal_fluctuation = 0.1 if np.isnan(seasonal_fluctuation) else seasonal_fluctuation
            
            return {
                "daily_load_fluctuation_rate": daily_fluctuation,
                "weekly_load_fluctuation_rate": weekly_fluctuation,
                "seasonal_fluctuation_intensity": seasonal_fluctuation
            }
        except Exception as e:
            print(f"Error calculating load fluctuation: {e}")
            return {
                "daily_load_fluctuation_rate": 0.2,
                "weekly_load_fluctuation_rate": 0.15,
                "seasonal_fluctuation_intensity": 0.1
            }
    
    def calculate_entropy_metrics(self, data):
        """
        Calculate complexity metrics based on entropy.
        
        Args:
            data: DataFrame with load data
            
        Returns:
            Entropy score (0-1)
        """
        try:
            # Normalize load to 0-1 scale
            load_min = data['load'].min()
            load_max = data['load'].max()
            
            if load_max == load_min:  # Handle case where all values are the same
                return 0.0
                
            normalized_load = (data['load'] - load_min) / (load_max - load_min)
            
            # Discretize into bins
            bins = 20
            hist, _ = np.histogram(normalized_load, bins=bins, range=(0, 1), density=True)
            
            # Ensure hist has non-zero values for entropy calculation
            hist = hist + 1e-10  # Add small epsilon to avoid log(0)
            hist = hist / hist.sum()  # Re-normalize
            
            # Calculate entropy
            entropy_score = entropy(hist) / np.log(bins)  # Normalize to 0-1
            
            return entropy_score
        except Exception as e:
            print(f"Error calculating entropy metrics: {e}")
            return 0.5  # Default value
    
    def simulate_evaluation_metrics(self, region_key, actual_data, forecast_data, all_regions_actual):
        """
        Simulate evaluation metrics based on actual and forecast data.
        
        Args:
            region_key: Region identifier
            actual_data: DataFrame with actual load data for the region
            forecast_data: DataFrame with forecasted load data for the region
            all_regions_actual: Dictionary with all regions' actual load data
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}
        
        # ------------------------
        # Forecast Reliability metrics
        # ------------------------
        
        # Historical forecast performance
        accuracy = self.calculate_historical_accuracy(actual_data, forecast_data)
        metrics["day_ahead_accuracy"] = accuracy
        metrics["realtime_accuracy"] = accuracy * np.random.uniform(0.95, 1.05)
        
        # Simulate extreme weather accuracy (lower than normal accuracy)
        metrics["extreme_weather_accuracy"] = accuracy * np.random.uniform(0.7, 0.9)
        
        # Forecasting system stability
        metrics["system_update_frequency"] = np.random.uniform(0.8, 1.0)
        metrics["forecast_result_continuity"] = self.calculate_data_timeliness(forecast_data)
        
        # Abnormal forecast proportion (percentage of outliers in forecast)
        forecast_mean = forecast_data['load'].mean()
        forecast_std = forecast_data['load'].std()
        outliers = ((forecast_data['load'] - forecast_mean).abs() > 3 * forecast_std).mean()
        metrics["abnormal_forecast_proportion"] = 1 - outliers  # Convert to goodness score
        
        # Data quality metrics
        metrics["data_completeness_rate"] = self.calculate_data_completeness_rate(actual_data)
        metrics["data_timeliness"] = self.calculate_data_timeliness(actual_data)
        metrics["data_consistency"] = self.calculate_data_consistency(actual_data)
        
        # ------------------------
        # Provincial Load Impact metrics
        # ------------------------
        
        # Load scale proportion
        load_proportion = self.calculate_load_proportion(actual_data, all_regions_actual)
        metrics.update(load_proportion)
        
        # Regulation capability (simulated)
        metrics["peak_regulation_capacity_ratio"] = np.random.uniform(0.3, 0.7)
        metrics["renewable_energy_installation_ratio"] = np.random.uniform(0.1, 0.5)
        metrics["demand_response_capability"] = np.random.uniform(0.2, 0.6)
        
        # ------------------------
        # Forecasting Complexity metrics
        # ------------------------
        
        # Load fluctuation characteristics
        fluctuation_metrics = self.calculate_load_fluctuation(actual_data)
        metrics.update(fluctuation_metrics)
        
        # External factor sensitivity (simulated)
        metrics["temperature_sensitivity"] = np.random.uniform(0.4, 0.9)
        metrics["humidity_sensitivity"] = np.random.uniform(0.3, 0.7)
        metrics["holiday_sensitivity"] = np.random.uniform(0.5, 0.9)
        
        # Electricity consumption structure complexity
        metrics["industrial_electricity_proportion"] = np.random.uniform(0.2, 0.8)
        metrics["key_users_count"] = np.random.uniform(0.3, 0.7)
        metrics["user_type_diversity"] = self.calculate_entropy_metrics(actual_data)
        
        # Replace any NaN values with default values
        for key, value in metrics.items():
            if np.isnan(value):
                if "proportion" in key or "ratio" in key or "rate" in key:
                    metrics[key] = 0.25  # Default proportion
                elif "accuracy" in key:
                    metrics[key] = 0.8   # Default accuracy
                else:
                    metrics[key] = 0.5   # Default for other metrics
        
        return metrics
    
    def calculate_indices(self, region_metrics):
        """
        Calculate hierarchical indices from metrics.
        
        Args:
            region_metrics: Dictionary with region metrics
            
        Returns:
            Dictionary with calculated indices
        """
        indices = {}
        
        # Calculate scores for each primary indicator
        for primary_key, primary_data in self.index_structure.items():
            sub_indices_scores = {}
            
            # Calculate scores for secondary indicators
            for sub_key, sub_data in primary_data["sub_indices"].items():
                metric_scores = []
                
                # Calculate scores for tertiary indicators (metrics)
                for metric in sub_data["metrics"]:
                    if metric in region_metrics:
                        value = region_metrics[metric]
                        if not np.isnan(value):  # Skip NaN values
                            metric_scores.append(value)
                
                # Calculate secondary indicator score as average of metrics
                if metric_scores:
                    sub_indices_scores[sub_key] = np.mean(metric_scores) * sub_data["weight"]
                else:
                    # Default value if no valid metrics
                    sub_indices_scores[sub_key] = 0.5 * sub_data["weight"]
            
            # Calculate primary indicator score as sum of weighted secondary indicators
            indices[primary_key] = sum(sub_indices_scores.values())
        
        # Calculate final score as weighted sum of primary indicators
        indices["FinalScore"] = sum(
            indices[key] * data["weight"] 
            for key, data in self.index_structure.items() 
            if key in indices
        )
        
        return indices
    
    def evaluate_regions(self, actual_data, forecast_data):
        """
        Evaluate all regions based on actual and forecast data.
        
        Args:
            actual_data: Dictionary with actual load data for each region
            forecast_data: Dictionary with forecasted load data for each region
            
        Returns:
            Dictionary with evaluation results for each region
        """
        evaluation_results = {}
        
        for region_key in actual_data.keys():
            # Get data for this region
            region_actual = actual_data[region_key]
            region_forecast = forecast_data[region_key]
            
            # Calculate metrics
            metrics = self.simulate_evaluation_metrics(
                region_key, 
                region_actual, 
                region_forecast, 
                actual_data
            )
            
            # Calculate indices
            indices = self.calculate_indices(metrics)
            
            # Store results
            evaluation_results[region_key] = {
                "metrics": metrics,
                "indices": indices
            }
        
        return evaluation_results
    
# 1. 创建评价指标管理器
class EvaluationManager:
    def __init__(self):
        self.metrics = {
            'standard': ['MAE', 'RMSE', 'MAPE'],
            'probabilistic': ['Pinball', 'Winkler', 'CRPS'],
            'scenarios': {
                'peak_hours': {'hours': [9, 10, 11, 12, 19, 20, 21], 'weight': 1.5},
                'extreme_weather': {'condition': lambda temp: temp > 35 or temp < 0, 'weight': 2.0},
                'holidays': {'days': [...], 'weight': 1.2}
            }
        }
    
    def evaluate_by_scenario(self, y_true, y_pred, scenario_data):
        """按不同场景评估预测性能"""
        results = {}
        
        # 标准指标
        for metric in self.metrics['standard']:
            results[metric] = self._calculate_metric(y_true, y_pred, metric)
        
        # 场景化评估
        for scenario, config in self.metrics['scenarios'].items():
            # 筛选符合场景的数据点
            mask = self._create_scenario_mask(scenario_data, config)
            
            # 计算场景内的指标
            scenario_results = {}
            for metric in self.metrics['standard']:
                scenario_results[metric] = self._calculate_metric(
                    y_true[mask], y_pred[mask], metric)
            
            results[scenario] = scenario_results
        
        return results

# 2. SHAP值分析
def analyze_feature_importance(model, X_test, feature_names):
    """使用SHAP值分析特征重要性"""
    import shap
    
    # 创建背景数据集
    explainer = shap.DeepExplainer(model, X_test[:100])
    
    # 计算SHAP值
    shap_values = explainer.shap_values(X_test)
    
    # 绘制摘要图
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)
    
    # 返回特征重要性
    importance = np.abs(shap_values).mean(0).mean(1)
    return dict(zip(feature_names, importance))