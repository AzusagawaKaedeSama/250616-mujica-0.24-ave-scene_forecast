import { useState, useCallback, useEffect, useRef } from 'react';
import { trainModel, getTrainingStatus } from '../../utils/api';

const DEFAULT_PROVINCES = ['上海', '安徽', '浙江', '江苏', '福建'];
const DEFAULT_TRAIN_START_DATE = '2024-01-01';
const DEFAULT_TRAIN_END_DATE = '2024-02-28';

const useTraining = () => {
  const [params, setParams] = useState({
    forecast_type: 'load',
    train_start: DEFAULT_TRAIN_START_DATE,
    train_end: DEFAULT_TRAIN_END_DATE,
    province: DEFAULT_PROVINCES[0],
    epochs: 100,
    batch_size: 32,
    train_prediction_type: 'deterministic'
  });

  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const pollingRef = useRef(null);

  const pollStatus = useCallback(async (taskId) => {
    try {
      const statusResult = await getTrainingStatus(taskId);
      setResult(statusResult);

      if (statusResult.status === 'completed' || statusResult.status === 'failed') {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
        setIsLoading(false); // Also stop loading indicator on completion/failure
      }
    } catch (err) {
      console.error(`Failed to get training status for task ${taskId}:`, err);
      // If status check fails (e.g., 404), stop polling
      setError(`获取任务 ${taskId} 状态失败: ${err.message}`);
      clearInterval(pollingRef.current);
      pollingRef.current = null;
      setIsLoading(false);
    }
  }, []);

  const handleStartTraining = useCallback(async (submittedParams) => {
    // Stop any previous polling
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
    }
    
    setParams(submittedParams);
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const apiResponse = await trainModel(submittedParams);
      if (apiResponse && apiResponse.task_id) {
        const { task_id } = apiResponse;
        setResult({ ...apiResponse, status: 'pending' }); // Initial status

        // Start polling
        pollingRef.current = setInterval(() => {
          pollStatus(task_id);
        }, 5000); // Poll every 5 seconds

        // Immediate first check
        pollStatus(task_id);

      } else {
        throw new Error("API did not return a task_id.");
      }
    } catch (err) {
      setError(err.message);
      setIsLoading(false);
      console.error('Failed to start training:', err);
    }
  }, [pollStatus]);

  // Cleanup effect to clear interval on unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
    };
  }, []);

  return {
    params,
    result,
    isLoading,
    error,
    startTraining: handleStartTraining,
  };
};

export default useTraining; 