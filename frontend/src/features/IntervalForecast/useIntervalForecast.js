import { useState, useCallback } from 'react';
import { runIntervalForecast } from '../../utils/api';

const DEFAULT_PROVINCES = ['上海', '安徽', '浙江', '江苏', '福建'];
const DEFAULT_FORECAST_START_DATE = '2024-03-01';
const DEFAULT_FORECAST_END_DATE = '2024-03-02';

const useIntervalForecast = () => {
  const [params, setParams] = useState({
    forecast_type: 'load',
    forecast_date: DEFAULT_FORECAST_START_DATE,
    forecast_end_date: DEFAULT_FORECAST_END_DATE,
    province: DEFAULT_PROVINCES[0],
    confidence_level: 0.9,
    model_type: 'peak_aware',
    historical_days: 15
  });
  
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleRunIntervalForecast = useCallback(async (submittedParams) => {
    setParams(submittedParams);
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      // The API call is specific for interval forecasts
      const apiResponse = await runIntervalForecast(submittedParams);
      
      // The interval forecast API returns the data structure directly
      // We might still need to normalize field names for the chart component
      if (apiResponse && apiResponse.predictions) {
        const needsMapping = apiResponse.predictions.some(p => 
          p.predicted === undefined && p.point_forecast !== undefined
        );
        
        if (needsMapping) {
          console.log('Hook: Mapping point_forecast to predicted for interval chart');
          apiResponse.predictions.forEach(p => {
            if (p.predicted === undefined && p.point_forecast !== undefined) {
              p.predicted = p.point_forecast;
            }
          });
        }
      }
      
      setResult(apiResponse);

    } catch (err) {
      setError(err.message);
      console.error('Interval Forecast Failed:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    params,
    result,
    isLoading,
    error,
    runIntervalForecast: handleRunIntervalForecast,
  };
};

export default useIntervalForecast; 