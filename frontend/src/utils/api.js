/**
 * API 通信模块 - 封装与后端 Flask API 的所有交互
 */

// API 基础 URL - 可根据开发/生产环境配置不同的值
const API_BASE_URL = 'http://localhost:5001/api';

/**
 * 通用 POST 请求工具函数
 * @param {string} endpoint - API 端点路径（不含基础URL）
 * @param {Object} data - 请求体数据
 * @returns {Promise} 返回包含响应数据的 Promise
 */
async function postRequest(endpoint, data = {}) {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    // 检查响应状态
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.error || `请求失败，状态码: ${response.status}`
      );
    }

    return await response.json();
  } catch (error) {
    console.error(`API请求错误 (${endpoint}):`, error);
    throw error; // 重新抛出错误，让调用者处理
  }
}

/**
 * 通用 GET 请求工具函数
 * @param {string} endpoint - API 端点路径（不含基础URL）
 * @returns {Promise} 返回包含响应数据的 Promise
 */
async function getRequest(endpoint) {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // 检查响应状态
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.error || `请求失败，状态码: ${response.status}`
      );
    }

    return await response.json();
  } catch (error) {
    console.error(`API请求错误 (${endpoint}):`, error);
    throw error; // 重新抛出错误，让调用者处理
  }
}

/**
 * 获取可用省份列表
 * @returns {Promise<Array<string>>} 省份名称数组
 */
export async function getProvinces() {
  const response = await getRequest('/provinces');
  
  if (!response.success) {
    throw new Error(response.error || '获取省份列表失败');
  }
  
  return response.provinces || response.data || [];
}

/**
 * 执行预测 (支持日前预测、滚动预测、概率预测、区间预测)
 * @param {Object} params - 预测参数
 * @returns {Promise<Object>} 预测结果
 */
export async function runPrediction(params) {
  const response = await postRequest('/predict', params);
  
  if (!response.success) {
    throw new Error(response.error || '预测失败');
  }
  
  return response.data;
}

/**
 * 执行区间预测
 * @param {Object} params - 区间预测参数
 * @returns {Promise<Object>} 区间预测结果
 */
export async function runIntervalForecast(params) {
  // 确保有前导斜杠，以便与 API_BASE_URL 正确拼接
  const response = await postRequest('/interval-forecast', params);
  
  if (!response.success) {
    throw new Error(response.error || '区间预测失败');
  }
  
  return response.data;
}

/**
 * 训练模型
 * @param {Object} params - 训练参数
 * @returns {Promise<Object>} 训练任务信息，包含任务ID
 */
export async function trainModel(params) {
  const response = await postRequest('/train', params);
  
  if (!response.success) {
    throw new Error(response.error || '模型训练启动失败');
  }
  
  return response; // 包含 task_id 和 message
}

/**
 * 获取训练任务状态
 * @param {string} taskId - 训练任务ID
 * @returns {Promise<Object>} 训练任务状态信息
 */
export async function getTrainingStatus(taskId) {
  const response = await getRequest(`/training-status/${taskId}`);
  
  if (!response.success) {
    throw new Error(response.error || '获取训练状态失败');
  }
  
  return response.data;
}

/**
 * 执行场景识别
 * @param {Object} params - 场景识别参数
 * @returns {Promise<Object>} 场景识别结果
 */
export async function recognizeScenarios(params) {
  const response = await postRequest('/recognize-scenarios', params);
  
  if (!response.success) {
    throw new Error(response.error || '场景识别失败');
  }
  
  return response.data;
}

/**
 * 清除缓存
 * @param {string} cacheType - (可选) 缓存类型
 * @returns {Promise<Object>} 操作结果
 */
export async function clearCache(cacheType = null) {
  const response = await postRequest('/clear-cache', cacheType ? { cache_type: cacheType } : {});
  
  if (!response.success) {
    throw new Error(response.error || '清除缓存失败');
  }
  
  return response; // 包含 message
}

/**
 * 获取缓存统计信息
 * @returns {Promise<Object>} 缓存统计数据
 */
export async function getCacheStats() {
  const response = await getRequest('/cache-stats');
  
  if (!response.success) {
    throw new Error(response.error || '获取缓存统计信息失败');
  }
  
  return response.data;
}

// 获取历史预测结果
export async function getHistoricalResults(params) {
  try {
    const queryParams = new URLSearchParams({
      forecastType: params.forecastType,
      province: params.province,
      startDate: params.startDate,
      endDate: params.endDate,
      predictionType: params.predictionType,
      modelType: params.modelType
    });
    
    const response = await fetch(`/api/historical-results?${queryParams.toString()}`);
    if (!response.ok) {
      throw new Error(`API响应错误: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('获取历史预测结果错误:', error);
    throw error;
  }
} 