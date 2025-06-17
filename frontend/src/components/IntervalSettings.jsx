import React, { useState, useEffect } from 'react';

function IntervalSettings({ availableProvinces, onSubmit, isLoading, initialParams, weatherScenario }) {
  const [params, setParams] = useState({
    forecast_type: initialParams?.forecast_type || 'load',
    province: initialParams?.province || (availableProvinces && availableProvinces.length > 0 ? availableProvinces[0] : ''),
    forecast_date: initialParams?.forecast_date || '2024-03-01',
    forecast_end_date: initialParams?.forecast_end_date || '2024-03-01',
    confidence_level: initialParams?.confidence_level || 0.9,
    model_type: initialParams?.model_type || 'peak_aware',
    historical_days: initialParams?.historical_days || 14,
    interval_minutes: initialParams?.interval || 15,
    force_refresh: initialParams?.force_refresh || false,
    weather_aware: initialParams?.weather_aware || false,
    renewable_enhanced: initialParams?.renewable_enhanced || false,
    enable_renewable_prediction: initialParams?.enable_renewable_prediction !== false, // 默认为true
  });

  useEffect(() => {
    if (initialParams) {
      setParams(prevParams => ({
        ...prevParams,
        ...(initialParams.forecast_type && { forecast_type: initialParams.forecast_type }),
        ...(initialParams.province && { province: initialParams.province }),
        ...(initialParams.forecast_date && { forecast_date: initialParams.forecast_date }),
        ...(initialParams.forecast_end_date && { forecast_end_date: initialParams.forecast_end_date }),
        ...(initialParams.confidence_level && { confidence_level: initialParams.confidence_level }),
        ...(initialParams.model_type && { model_type: initialParams.model_type }),
        ...(initialParams.historical_days && { historical_days: initialParams.historical_days }),
        ...(initialParams.interval && { interval_minutes: initialParams.interval }),
        ...(initialParams.force_refresh !== undefined && { force_refresh: initialParams.force_refresh }),
        ...(initialParams.weather_aware !== undefined && { weather_aware: initialParams.weather_aware }),
        ...(initialParams.renewable_enhanced !== undefined && { renewable_enhanced: initialParams.renewable_enhanced }),
        ...(initialParams.enable_renewable_prediction !== undefined && { enable_renewable_prediction: initialParams.enable_renewable_prediction }),
      }));
    }
  }, [initialParams]);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({
      predictionType: 'day-ahead',
      forecastType: params.forecast_type,
      province: params.province,
      forecastDate: params.forecast_date,
      forecastEndDate: params.forecast_end_date,
      confidenceLevel: params.confidence_level,
      modelType: params.model_type,
      historicalDays: params.historical_days,
      interval: params.interval_minutes,
      interval_forecast: true,
      force_refresh: params.force_refresh,
      weather_aware: params.weather_aware,
      renewable_enhanced: params.renewable_enhanced,
      enable_renewable_prediction: params.enable_renewable_prediction,
    });
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setParams({
      ...params,
      [name]: type === 'checkbox' ? checked : value,
    });
  };

  return (
    <div className="bg-neutral-800 rounded-lg border border-neutral-700 shadow-xl p-6">
      <h2 className="text-2xl text-neutral-100 mb-6">参数设置 - 区间预测</h2>
      
      {/* 天气场景信息展示 */}
      {weatherScenario && params.forecast_type === 'load' && (
        <div className="mb-6 p-4 bg-neutral-700 rounded-lg">
          <h3 className="text-lg font-medium text-neutral-100 mb-3">天气场景分析</h3>
          {weatherScenario.error ? (
            <p className="text-red-400">天气场景分析失败: {weatherScenario.error}</p>
          ) : (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-neutral-300">
                  <span className="font-medium">场景类型:</span> {weatherScenario.scenario_type}
                </p>
                <p className="text-neutral-300">
                  <span className="font-medium">平均温度:</span> {weatherScenario.temperature_mean?.toFixed(1)}°C
                </p>
                <p className="text-neutral-300">
                  <span className="font-medium">温度范围:</span> {weatherScenario.temperature_min?.toFixed(1)}°C - {weatherScenario.temperature_max?.toFixed(1)}°C
                </p>
              </div>
              <div>
                <p className="text-neutral-300">
                  <span className="font-medium">平均湿度:</span> {weatherScenario.humidity_mean?.toFixed(1)}%
                </p>
                <p className="text-neutral-300">
                  <span className="font-medium">平均风速:</span> {weatherScenario.wind_speed_mean?.toFixed(1)} m/s
                </p>
                <p className="text-neutral-300">
                  <span className="font-medium">累计降水:</span> {weatherScenario.precipitation_sum?.toFixed(1)} mm
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Row 1: Forecast Type & Province */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label htmlFor="interval-forecastType" className="block text-base font-medium text-neutral-300 mb-1">
              预测类型
            </label>
            <select
              id="interval-forecastType"
              name="forecast_type"
              value={params.forecast_type}
              onChange={handleChange}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            >
              <option value="load">电力负荷</option>
              <option value="pv">光伏发电</option>
              <option value="wind">风电出力</option>
              <option value="net_load">净负荷 (负荷-光伏-风电)</option>
            </select>
          </div>
          <div>
            <label htmlFor="interval-province" className="block text-base font-medium text-neutral-300 mb-1">
              省份
            </label>
            <select
              id="interval-province"
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

        {/* Row 2: Forecast Date & Forecast End Date */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label htmlFor="interval-startDate" className="block text-base font-medium text-neutral-300 mb-1">
              预测开始日期
            </label>
            <input
              type="date"
              id="interval-startDate"
              name="forecast_date"
              value={params.forecast_date}
              max={params.forecast_end_date}
              onChange={handleChange}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            />
          </div>
          <div>
            <label htmlFor="interval-endDate" className="block text-base font-medium text-neutral-300 mb-1">
              预测结束日期
            </label>
            <input
              type="date"
              id="interval-endDate"
              name="forecast_end_date"
              value={params.forecast_end_date}
              min={params.forecast_date}
              onChange={handleChange}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            />
          </div>
        </div>

        {/* Row 3: Confidence Level & Model Type */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label htmlFor="interval-confidence" className="block text-base font-medium text-neutral-300 mb-1">
              置信水平 (0.5-0.99)
            </label>
            <input
              type="number"
              id="interval-confidence"
              name="confidence_level"
              value={params.confidence_level}
              min="0.5"
              max="0.99"
              step="0.01"
              onChange={handleChange}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            />
          </div>
          <div>
            <label htmlFor="interval-modelType" className="block text-base font-medium text-neutral-300 mb-1">
              模型类型
            </label>
            <select
              id="interval-modelType"
              name="model_type"
              value={params.model_type}
              onChange={handleChange}
              title="区间预测推荐使用 peak_aware 模型，statistical 模型用于简单基准。"
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            >
              <option value="peak_aware">峰谷感知模型 (推荐)</option>
              <option value="statistical">统计模型 (基准)</option>
            </select>
          </div>
        </div>

        {/* Row 4: Historical Days & Interval */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label htmlFor="interval-historicalDays" className="block text-base font-medium text-neutral-300 mb-1">
              历史数据天数 (1-30)
            </label>
            <input
              type="number"
              id="interval-historicalDays"
              name="historical_days"
              value={params.historical_days}
              min="1"
              max="30"
              onChange={handleChange}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            />
          </div>
          <div>
            <label htmlFor="interval-intervalMinutes" className="block text-base font-medium text-neutral-300 mb-1">
              预测时间间隔 (分钟)
            </label>
            <input
              type="number"
              id="interval-intervalMinutes"
              name="interval_minutes"
              value={params.interval_minutes}
              min="15"
              step="15"
              onChange={handleChange}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            />
          </div>
        </div>

        {/* Row 5: Toggles */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
            <div className="flex items-center">
              <input
                type="checkbox"
                id="interval-weatherAware"
                name="weather_aware"
                checked={params.weather_aware}
                onChange={handleChange}
                disabled={params.forecast_type !== 'load'}
                className="h-4 w-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
              />
              <label 
                  htmlFor="interval-weatherAware" 
                  className="ml-2 block text-base text-neutral-300"
              >
                  启用天气感知
              </label>
              {params.forecast_type !== 'load' && <span className="text-xs text-neutral-500 ml-2">(仅支持负荷)</span>}
            </div>
            <div className="flex items-center">
              <input
                type="checkbox"
                id="interval-forceRefresh"
                name="force_refresh"
                checked={params.force_refresh}
                onChange={handleChange}
                className="h-4 w-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
              />
              <label 
                  htmlFor="interval-forceRefresh" 
                  className="ml-2 block text-base text-neutral-300"
              >
                  强制刷新
              </label>
            </div>
        </div>

        {/* Row 6: 可再生能源增强预测选项 */}
        <div className="border-t border-neutral-600 pt-6">
          <h3 className="text-lg font-medium text-neutral-200 mb-4">可再生能源增强预测 (实验性功能)</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="flex items-center">
              <input
                type="checkbox"
                id="interval-renewableEnhanced"
                name="renewable_enhanced"
                checked={params.renewable_enhanced}
                onChange={handleChange}
                disabled={params.forecast_type !== 'load'}
                className="h-4 w-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
              />
              <label 
                  htmlFor="interval-renewableEnhanced" 
                  className="ml-2 block text-base text-neutral-300"
              >
                  启用可再生能源增强区间预测
              </label>
              {params.forecast_type !== 'load' && <span className="text-xs text-neutral-500 ml-2">(仅支持负荷)</span>}
            </div>
            <div className="flex items-center">
              <input
                type="checkbox"
                id="interval-enableRenewablePrediction"
                name="enable_renewable_prediction"
                checked={params.enable_renewable_prediction}
                onChange={handleChange}
                disabled={!params.renewable_enhanced}
                className="h-4 w-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
              />
              <label 
                  htmlFor="interval-enableRenewablePrediction" 
                  className="ml-2 block text-base text-neutral-300"
              >
                  启用新能源出力预测
              </label>
              {!params.renewable_enhanced && <span className="text-xs text-neutral-500 ml-2">(需先启用增强预测)</span>}
            </div>
          </div>
          
          {/* 可再生能源增强预测说明 */}
          {params.renewable_enhanced && (
            <div className="mt-4 p-4 bg-neutral-700 rounded-lg">
              <h4 className="text-base font-medium text-neutral-200 mb-2">功能说明</h4>
              <ul className="text-sm text-neutral-300 space-y-1">
                <li>• 集成光伏和风电出力预测，识别新能源高出力时段</li>
                <li>• 基于天气条件和新能源状态进行增强场景识别</li>
                <li>• 动态调整预测区间以反映新能源不确定性影响</li>
                <li>• 提供综合风险等级评估和系统影响分析</li>
              </ul>
              <p className="text-xs text-amber-400 mt-2">
                注意：此功能需要训练好的光伏和风电模型，如模型不可用将自动降级为标准区间预测
              </p>
            </div>
          )}
        </div>

        {/* Submit Button */}
        <div>
          <button
            type="submit"
            disabled={isLoading}
            className="btn-primary w-full md:w-auto"
          >
            {isLoading ? '正在预测...' : '开始区间预测'}
          </button>
        </div>
      </form>
    </div>
  );
}

export default IntervalSettings; 