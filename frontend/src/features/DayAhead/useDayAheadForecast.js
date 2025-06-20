import { useState, useCallback } from 'react';
import { runPrediction } from '../../utils/api';

const DEFAULT_PROVINCES = ['上海', '安徽', '浙江', '江苏', '福建'];
const DEFAULT_FORECAST_START_DATE = '2024-03-01';
const DEFAULT_FORECAST_END_DATE = '2024-03-02';

const useDayAheadForecast = () => {
  const [params, setParams] = useState({
    forecastType: 'load',
    forecastDate: DEFAULT_FORECAST_START_DATE,
    forecastEndDate: DEFAULT_FORECAST_END_DATE,
    province: DEFAULT_PROVINCES[0],
    predictionType: 'day-ahead',
    historicalDays: 15,
    modelType: 'torch',
    weatherAware: false, // Add weather aware flag
  });
  
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleRunPrediction = useCallback(async (submittedParams) => {
    setParams(submittedParams);
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const chartData = await runPrediction(submittedParams);
      
      setResult(chartData);

    } catch (err) {
      setError(err.message);
      console.error('Day Ahead Forecast Failed:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    params,
    result,
    isLoading,
    error,
    runPrediction: handleRunPrediction,
  };
};

export default useDayAheadForecast; 