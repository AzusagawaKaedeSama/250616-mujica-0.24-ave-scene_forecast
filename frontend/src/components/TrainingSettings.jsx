import React, { useState } from 'react';

function TrainingSettings({ availableProvinces, onSubmit, isLoading, initialParams }) {
  const [params, setParams] = useState({
    forecast_type: initialParams?.forecast_type || 'load',
    province: initialParams?.province || (availableProvinces && availableProvinces.length > 0 ? availableProvinces[0] : ''),
    train_start: initialParams?.train_start || '2024-01-01',
    train_end: initialParams?.train_end || '2024-02-28',
    epochs: initialParams?.epochs || 100,
    batch_size: initialParams?.batch_size || 32,
    learning_rate: initialParams?.learning_rate || 0.001,
    train_prediction_type: initialParams?.train_prediction_type || 'deterministic',
    retrain: initialParams?.retrain || false,
    peak_aware: initialParams?.peak_aware || false,
    peak_start: initialParams?.peak_start || 0,
    peak_end: initialParams?.peak_end || 0,
    valley_start: initialParams?.valley_start || 0,
    valley_end: initialParams?.valley_end || 0,
    peak_weight: initialParams?.peak_weight || 1.0,
    valley_weight: initialParams?.valley_weight || 1.0,
    historical_days: initialParams?.historical_days || 1,
    weather_aware: initialParams?.weather_aware || false,
  });

  const today = new Date().toISOString().split('T')[0];

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(params);
  };

  const handleChange = (e) => {
    const { name, type, checked, value } = e.target;
    setParams((prevParams) => ({
      ...prevParams,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  return (
    <div className="bg-neutral-800 rounded-lg border border-neutral-700 shadow-xl p-6">
      <h2 className="text-2xl text-neutral-100 mb-6">参数设置 - 模型训练</h2>
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Row 1: Forecast Type & Province */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label htmlFor="train-forecastType" className="block text-base font-medium text-neutral-300 mb-1">
              预测类型
            </label>
            <select
              id="train-forecastType"
              name="forecast_type"
              value={params.forecast_type}
              onChange={handleChange}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            >
              <option value="load">电力负荷</option>
              <option value="pv">光伏发电</option>
              <option value="wind">风电出力</option>
            </select>
          </div>
          <div>
            <label htmlFor="train-province" className="block text-base font-medium text-neutral-300 mb-1">
              省份
            </label>
            <select
              id="train-province"
              name="province"
              value={params.province}
              onChange={handleChange}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            >
              {availableProvinces && availableProvinces.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
              {(!availableProvinces || availableProvinces.length === 0) && <option value="上海">上海 (默认)</option>}
            </select>
          </div>
        </div>

        {/* Row 2: Train Start Date & Train End Date */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label htmlFor="train-trainStartDate" className="block text-base font-medium text-neutral-300 mb-1">
              训练开始日期
            </label>
            <input
              type="date"
              id="train-trainStartDate"
              name="train_start"
              value={params.train_start}
              max={params.train_end}
              onChange={handleChange}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            />
          </div>
          <div>
            <label htmlFor="train-trainEndDate" className="block text-base font-medium text-neutral-300 mb-1">
              训练结束日期
            </label>
            <input
              type="date"
              id="train-trainEndDate"
              name="train_end"
              value={params.train_end}
              min={params.train_start}
              max={today}
              onChange={handleChange}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            />
          </div>
        </div>

        {/* Row 3: Epochs & Batch Size */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label htmlFor="train-epochs" className="block text-base font-medium text-neutral-300 mb-1">
              训练轮数 (Epochs)
            </label>
            <input
              type="number"
              id="train-epochs"
              name="epochs"
              value={params.epochs}
              min="1"
              onChange={handleChange}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            />
          </div>
          <div>
            <label htmlFor="train-batchSize" className="block text-base font-medium text-neutral-300 mb-1">
              批处理大小 (Batch Size)
            </label>
            <input
              type="number"
              id="train-batchSize"
              name="batch_size"
              value={params.batch_size}
              min="16"
              step="16"
              onChange={handleChange}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            />
          </div>
        </div>
        
        {/* Row 4: Learning Rate & Historical Days */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label htmlFor="train-learningRate" className="block text-base font-medium text-neutral-300 mb-1">
              学习率
            </label>
            <input
              type="number"
              id="train-learningRate"
              name="learning_rate"
              value={params.learning_rate}
              min="0.00001"
              max="0.1"
              step="0.00001"
              onChange={handleChange}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            />
          </div>
          <div>
            <label htmlFor="train-historicalDays" className="block text-base font-medium text-neutral-300 mb-1">
              历史数据天数 (特征工程)
            </label>
            <input
              type="number"
              id="train-historicalDays"
              name="historical_days"
              value={params.historical_days}
              min="1"
              max="30"
              onChange={handleChange}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            />
          </div>
        </div>

        {/* Row 5: Training Prediction Type & Model Type (Removed Model Type, fixed in App.jsx) */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
           <div>
            <label htmlFor="train-trainPredictionType" className="block text-base font-medium text-neutral-300 mb-1">
              训练目标类型
            </label>
            <select
              id="train-trainPredictionType"
              name="train_prediction_type"
              value={params.train_prediction_type}
              onChange={handleChange}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            >
              <option value="deterministic">确定性预测</option>
              <option value="probabilistic">概率性预测</option>
            </select>
          </div>
        </div>

        {/* Row 6: Checkboxes for Retrain & Peak Aware */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
          <div className="flex items-center">
            <input
              id="train-retrain"
              type="checkbox"
              name="retrain"
              checked={params.retrain}
              onChange={handleChange}
              className="h-4 w-4 text-red-500 border-neutral-600 rounded bg-neutral-700 focus:ring-red-600 focus:ring-offset-neutral-800"
            />
            <label htmlFor="train-retrain" className="ml-2 text-base text-neutral-300">
              重新训练 (覆盖现有模型)
            </label>
          </div>
          {params.forecast_type === 'load' && params.train_prediction_type === 'deterministic' && (
            <div className="flex items-center">
              <input
                id="train-peakAware"
                type="checkbox"
                name="peak_aware"
                checked={params.peak_aware}
                onChange={handleChange}
                className="h-4 w-4 text-red-500 border-neutral-600 rounded bg-neutral-700 focus:ring-red-600 focus:ring-offset-neutral-800"
              />
              <label htmlFor="train-peakAware" className="ml-2 text-base text-neutral-300">
                启用峰谷感知 (仅负荷确定性)
              </label>
            </div>
          )}
        </div>

        {/* Row 7: Weather Aware Option (for all forecast types) */}
        <div className="grid grid-cols-1 gap-6">
          <div className="flex items-center">
            <input
              id="train-weatherAware"
              type="checkbox"
              name="weather_aware"
              checked={params.weather_aware}
              onChange={handleChange}
              className="h-4 w-4 text-red-500 border-neutral-600 rounded bg-neutral-700 focus:ring-red-600 focus:ring-offset-neutral-800"
            />
            <label htmlFor="train-weatherAware" className="ml-2 text-base text-neutral-300">
              启用天气感知训练 (使用天气数据增强{params.forecast_type === 'load' ? '负荷' : params.forecast_type === 'pv' ? '光伏' : '风电'}预测)
            </label>
          </div>
          {params.weather_aware && (
            <div className="bg-neutral-750 p-4 rounded-md border border-neutral-600">
              <p className="text-sm text-neutral-400 mb-2">
                <strong>天气感知训练说明：</strong>
              </p>
              <ul className="text-sm text-neutral-400 space-y-1 list-disc list-inside">
                {params.forecast_type === 'load' && (
                  <>
                    <li>将使用集成的天气负荷数据进行训练</li>
                    <li>包含温度、湿度、风速、降水等气象特征</li>
                    <li>自动计算供暖度日(HDD)和制冷度日(CDD)</li>
                    <li>提升模型对天气变化的敏感性和预测准确性</li>
                  </>
                )}
                {params.forecast_type === 'pv' && (
                  <>
                    <li>将使用集成的天气光伏数据进行训练</li>
                    <li>重点关注温度、湿度、降水、露点等影响光伏发电的因素</li>
                    <li>自动识别光照强度和云层覆盖对发电的影响</li>
                    <li>提高光伏发电量预测在不同天气条件下的准确性</li>
                  </>
                )}
                {params.forecast_type === 'wind' && (
                  <>
                    <li>将使用集成的天气风电数据进行训练</li>
                    <li>重点关注风速、风向、温度、湿度等影响风电的因素</li>
                    <li>捕捉风速变化模式和大气稳定性对发电的影响</li>
                    <li>提升风电出力预测在复杂气象条件下的精度</li>
                  </>
                )}
                <li>支持确定性和概率性预测</li>
                <li>采用与传统训练一致的模型架构，确保兼容性</li>
              </ul>
            </div>
          )}
        </div>
        
        {/* Conditional Peak/Valley parameters if peak_aware is true */}
        {params.peak_aware && params.forecast_type === 'load' && params.train_prediction_type === 'deterministic' && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label htmlFor="train-peakStart" className="block text-base font-medium text-neutral-300 mb-1">峰时段开始 (小时)</label>
                <input type="number" id="train-peakStart" name="peak_start" value={params.peak_start} onChange={handleChange} min="0" max="23" className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm" />
              </div>
              <div>
                <label htmlFor="train-peakEnd" className="block text-base font-medium text-neutral-300 mb-1">峰时段结束 (小时)</label>
                <input type="number" id="train-peakEnd" name="peak_end" value={params.peak_end} onChange={handleChange} min="0" max="23" className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm" />
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label htmlFor="train-valleyStart" className="block text-base font-medium text-neutral-300 mb-1">谷时段开始 (小时)</label>
                <input type="number" id="train-valleyStart" name="valley_start" value={params.valley_start} onChange={handleChange} min="0" max="23" className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm" />
              </div>
              <div>
                <label htmlFor="train-valleyEnd" className="block text-base font-medium text-neutral-300 mb-1">谷时段结束 (小时)</label>
                <input type="number" id="train-valleyEnd" name="valley_end" value={params.valley_end} onChange={handleChange} min="0" max="23" className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm" />
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label htmlFor="train-peakWeight" className="block text-base font-medium text-neutral-300 mb-1">峰时段损失权重</label>
                <input type="number" id="train-peakWeight" name="peak_weight" value={params.peak_weight} onChange={handleChange} step="0.1" className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm" />
              </div>
              <div>
                <label htmlFor="train-valleyWeight" className="block text-base font-medium text-neutral-300 mb-1">谷时段损失权重</label>
                <input type="number" id="train-valleyWeight" name="valley_weight" value={params.valley_weight} onChange={handleChange} step="0.1" className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm" />
              </div>
            </div>
          </>
        )}

        {/* Submit Button */}
        <div className="flex flex-col space-y-4 pt-4">
          {/* 单个模型训练 */}
          <div className="flex space-x-4">
            <button
              type="submit"
              disabled={isLoading}
              className="btn-primary w-full md:w-auto"
            >
              {isLoading ? '正在训练...' : `训练${params.forecast_type === 'load' ? '负荷' : params.forecast_type === 'pv' ? '光伏' : '风电'}模型${params.weather_aware ? ' (天气感知)' : ''}`}
            </button>
          </div>
          
          {/* 一键训练所有模型 */}
          <div className="flex space-x-4">
            <button
              type="button"
              disabled={isLoading}
              onClick={() => {
                // 依次训练负荷、光伏和风电三种传统模型
                const trainAllModels = async () => {
                  const modelTypes = ['load', 'pv', 'wind'];
                  const baseParams = { ...params, weather_aware: false };
                  
                  for (let i = 0; i < modelTypes.length; i++) {
                    const modelType = modelTypes[i];
                    try {
                      // 更新当前训练的模型类型
                      const currentParams = { 
                        ...baseParams,
                        forecast_type: modelType 
                      };
                      
                      // 提交训练请求
                      console.log(`开始训练${modelType}传统模型...`);
                      const response = await onSubmit(currentParams);
                      
                      // 如果不是最后一个模型，等待一段时间再训练下一个
                      if (i < modelTypes.length - 1) {
                        console.log(`${modelType}传统模型训练任务已提交，等待5秒后继续训练下一个模型...`);
                        await new Promise(resolve => setTimeout(resolve, 5000));
                      }
                    } catch (error) {
                      console.error(`训练${modelType}传统模型时出错:`, error);
                    }
                  }
                  
                  console.log('所有传统模型训练任务已提交完成');
                };
                
                trainAllModels();
              }}
              className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded w-full md:w-auto"
            >
              {isLoading ? '训练进行中...' : '一键训练所有传统模型'}
            </button>
            
            <button
              type="button"
              disabled={isLoading}
              onClick={() => {
                // 依次训练负荷、光伏和风电三种天气感知模型
                const trainAllWeatherAwareModels = async () => {
                  const modelTypes = ['load', 'pv', 'wind'];
                  const baseParams = { ...params, weather_aware: true };
                  
                  for (let i = 0; i < modelTypes.length; i++) {
                    const modelType = modelTypes[i];
                    try {
                      // 更新当前训练的模型类型
                      const currentParams = { 
                        ...baseParams,
                        forecast_type: modelType 
                      };
                      
                      // 提交训练请求
                      console.log(`开始训练${modelType}天气感知模型...`);
                      const response = await onSubmit(currentParams);
                      
                      // 如果不是最后一个模型，等待一段时间再训练下一个
                      if (i < modelTypes.length - 1) {
                        console.log(`${modelType}天气感知模型训练任务已提交，等待8秒后继续训练下一个模型...`);
                        // 天气感知模型训练时间更长，增加等待时间
                        await new Promise(resolve => setTimeout(resolve, 8000));
                      }
                    } catch (error) {
                      console.error(`训练${modelType}天气感知模型时出错:`, error);
                    }
                  }
                  
                  console.log('所有天气感知模型训练任务已提交完成');
                };
                
                trainAllWeatherAwareModels();
              }}
              className="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded w-full md:w-auto"
            >
              {isLoading ? '训练进行中...' : '一键训练所有天气感知模型'}
            </button>
          </div>
          
          {/* 终极一键训练 */}
          <div className="flex space-x-4">
            <button
              type="button"
              disabled={isLoading}
              onClick={() => {
                // 依次训练所有类型的模型（传统+天气感知）
                const trainAllModelsComplete = async () => {
                  const modelTypes = ['load', 'pv', 'wind'];
                  const modelModes = [
                    { weather_aware: false, name: '传统' },
                    { weather_aware: true, name: '天气感知' }
                  ];
                  
                  for (let j = 0; j < modelModes.length; j++) {
                    const mode = modelModes[j];
                    console.log(`开始训练${mode.name}模型组...`);
                    
                    for (let i = 0; i < modelTypes.length; i++) {
                      const modelType = modelTypes[i];
                      try {
                        // 更新当前训练的模型类型和模式
                        const currentParams = { 
                          ...params,
                          forecast_type: modelType,
                          weather_aware: mode.weather_aware
                        };
                        
                        // 提交训练请求
                        console.log(`开始训练${modelType}${mode.name}模型...`);
                        const response = await onSubmit(currentParams);
                        
                        // 等待一段时间再训练下一个
                        const waitTime = mode.weather_aware ? 8000 : 5000;
                        if (!(j === modelModes.length - 1 && i === modelTypes.length - 1)) {
                          console.log(`${modelType}${mode.name}模型训练任务已提交，等待${waitTime/1000}秒后继续...`);
                          await new Promise(resolve => setTimeout(resolve, waitTime));
                        }
                      } catch (error) {
                        console.error(`训练${modelType}${mode.name}模型时出错:`, error);
                      }
                    }
                  }
                  
                  console.log('所有模型训练任务已提交完成（传统+天气感知）');
                };
                
                trainAllModelsComplete();
              }}
              className="bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded w-full md:w-auto"
            >
              {isLoading ? '训练进行中...' : '一键训练全部模型 (传统+天气感知)'}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}

export default TrainingSettings; 