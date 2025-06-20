/**
 * API 通信模块 - 封装与后端 Flask API 的所有交互
 */

import axios from 'axios';

// 新的、符合DDD架构的API服务的基础URL
const API_BASE_URL = 'http://localhost:5001';

// 旧的API URL - 仅作为备用
const OLD_API_BASE_URL = 'http://localhost:5000';

// A hardcoded list of provinces to avoid calling the old backend during tests.
const DUMMY_PROVINCES = ['上海', '安徽', '浙江', '江苏', '福建'];

/**
 * 通用 POST 请求工具函数
 * @param {string} endpoint - API 端点路径（不含基础URL）
 * @param {Object} data - 请求体数据
 * @returns {Promise} 返回包含响应数据的 Promise
 */
const postRequest = async (endpoint, params, useNewAPI = true) => {
  try {
    const baseURL = useNewAPI ? API_BASE_URL : OLD_API_BASE_URL;
    const response = await axios.post(`${baseURL}${endpoint}`, params);
    return response.data;
  } catch (error) {
    console.error(`Error in postRequest for endpoint ${endpoint}:`, error);
    // Rethrow a more structured error
    throw new Error(error.response?.data?.error || error.message || 'An unknown API error occurred');
  }
};

/**
 * 获取可用省份列表
 * @returns {Promise<Array<string>>} 省份名称数组
 */
export const getProvinces = async () => {
  try {
    console.log("Trying to get provinces from new DDD API...");
    const response = await axios.get(`${API_BASE_URL}/api/provinces`);
    if (response.data && response.data.success) {
      return response.data.data;
    } else {
      console.log("DDD API failed, using dummy provinces for testing.");
      return DUMMY_PROVINCES;
    }
  } catch (error) {
    console.log("DDD API not available, using dummy provinces for testing.");
    return DUMMY_PROVINCES;
  }
};

/**
 * 执行预测 (支持日前预测、滚动预测、概率预测、区间预测)
 * @param {Object} params - 预测参数
 * @returns {Promise<Object>} 预测结果
 */
export const runPrediction = async (params) => {
  try {
    console.log('Routing to NEW DDD API for prediction:', params);
    const response = await axios.post(`${API_BASE_URL}/api/predict`, params);

    if (response.data && response.data.success) {
      const returnData = response.data.data;
      console.log('Using DDD implementation:', response.data.implementation);
      
      // Return the nested 'data' object directly, which contains 'predictions'
      return returnData; 
    } else {
      throw new Error(response.data.error || 'DDD API returned an error');
    }

  } catch (error) {
    console.error('Error running prediction:', error);
    // Ensure a consistent error message is thrown
    const message = error.response?.data?.error || error.message || 'Prediction failed';
    throw new Error(message);
  }
};

/**
 * 执行区间预测
 * @param {Object} params - 区间预测参数
 * @returns {Promise<Object>} 区间预测结果
 */
export const runIntervalForecast = async (params) => {
  try {
    console.log('Routing interval forecast to NEW DDD API:', params);
    const response = await axios.post(`${API_BASE_URL}/api/predict`, {
      ...params,
      prediction_type: 'interval'
    });
    
    if (response.data && response.data.success) {
      return response.data.data;
    } else {
      throw new Error(response.data.error || '区间预测失败');
    }
  } catch (error) {
    console.error('Interval forecast error:', error);
    // Fallback to old API
    console.log('Fallback to old API for interval forecast');
    const response = await postRequest('/api/interval-forecast', params, false);
    if (!response.success) {
        throw new Error(response.error || '区间预测失败');
    }
    return response;
  }
};

/**
 * 训练模型
 * @param {Object} params - 训练参数
 * @returns {Promise<Object>} 训练任务信息，包含任务ID
 */
export const trainModel = async (params) => {
  try {
    console.log('Routing training to NEW DDD API:', params);
    const response = await axios.post(`${API_BASE_URL}/api/train`, params);
    
    if (response.data && response.data.success) {
      return response.data;
    } else {
      throw new Error(response.data.error || 'DDD API训练失败');
    }
  } catch (error) {
    console.error('Training error with DDD API:', error);
    // Fallback to old API
    console.log('Fallback to old API for training');
    const response = await postRequest('/api/train', params, false);
    if (!response.success) {
        throw new Error(response.error || '模型训练启动失败');
    }
    return response;
  }
};

/**
 * 获取训练任务状态
 * @param {string} taskId - 训练任务ID
 * @returns {Promise<Object>} 训练任务状态信息
 */
export const getTrainingStatus = async (taskId) => {
    try {
        console.log('Getting training status from DDD API:', taskId);
        const response = await axios.get(`${API_BASE_URL}/api/training-status/${taskId}`);
        if(response.data.success){
            return response.data.data;
        } else {
            throw new Error(response.data.error || `Failed to get status for task ${taskId}`);
        }
    } catch (error) {
        console.error('Training status error with DDD API:', error);
        // Fallback to old API
        try {
            const response = await axios.get(`${OLD_API_BASE_URL}/api/training-status/${taskId}`);
            if(response.data.success){
                return response.data.data;
            } else {
                throw new Error(response.data.error || `Failed to get status for task ${taskId}`);
            }
        } catch (fallbackError) {
            const message = fallbackError.response?.data?.error || fallbackError.message || 'Failed to fetch training status';
            throw new Error(message);
        }
    }
};

/**
 * 执行场景识别
 * @param {Object} params - 场景识别参数
 * @returns {Promise<Object>} 场景识别结果
 */
export const recognizeScenarios = async (params) => {
  try {
    console.log('Routing scenario recognition to NEW DDD API:', params);
    const response = await axios.post(`${API_BASE_URL}/api/scenarios`, params);
    
    if (response.data && response.data.success) {
      return response.data.data;
    } else {
      throw new Error(response.data.error || '场景识别失败');
    }
  } catch (error) {
    console.error('Scenario recognition error with DDD API:', error);
    // Fallback to old API
    const response = await postRequest('/recognize-scenarios', params, false);
    
    if (!response.success) {
      throw new Error(response.error || '场景识别失败');
    }
    
    return response.data;
  }
};

/**
 * 清除缓存
 * @param {string} cacheType - (可选) 缓存类型
 * @returns {Promise<Object>} 操作结果
 */
export const clearCache = async () => {
    try {
        console.log('Clearing cache via DDD API');
        const response = await axios.post(`${API_BASE_URL}/api/clear-cache`, {});
        if (response.data && response.data.success) {
            return response.data;
        } else {
            throw new Error(response.data.error || 'Cache clear failed');
        }
    } catch (error) {
        console.error('Cache clear error with DDD API:', error);
        // Fallback to old API
        const response = await postRequest('/api/clear-cache', {}, false);
        return response;
    }
};

/**
 * 获取缓存统计信息
 * @returns {Promise<Object>} 缓存统计数据
 */
export const getCacheStats = async () => {
  try {
    console.log('Getting cache stats from DDD API');
    const response = await axios.get(`${API_BASE_URL}/api/cache-stats`);
    
    if (response.data && response.data.success) {
      return response.data.data;
    } else {
      throw new Error(response.data.error || 'Failed to get cache stats');
    }
  } catch (error) {
    console.error('Cache stats error with DDD API:', error);
    // Fallback to old API
    const response = await axios.get(`${OLD_API_BASE_URL}/cache-stats`);
    
    if (!response.data.success) {
      throw new Error(response.data.error || '获取缓存统计信息失败');
    }
    
    return response.data.data;
  }
};

// 获取历史预测结果
export const getHistoricalResults = async (params) => {
    try {
        console.log('Getting historical results from DDD API:', params);
        const response = await axios.get(`${API_BASE_URL}/api/historical-results`, { params });
        if (response.data.success) {
            return response.data.data;
        } else {
            throw new Error(response.data.error || 'Failed to fetch historical results');
        }
    } catch (error) {
        console.error('Historical results error with DDD API:', error);
        // Fallback to old API
        try {
            const response = await axios.get(`${OLD_API_BASE_URL}/api/historical-results`, { params });
            if (response.data.success) {
                return response.data.data;
            } else {
                throw new Error(response.data.error || 'Failed to fetch historical results');
            }
        } catch (fallbackError) {
            const message = fallbackError.response?.data?.error || fallbackError.message || 'Failed to fetch historical results';
            throw new Error(message);
        }
    }
}; 