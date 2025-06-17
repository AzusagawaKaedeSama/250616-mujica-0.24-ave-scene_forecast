import React, { useState, useEffect } from 'react';

const allQuantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95];

function ProbabilisticSettings({ availableProvinces, onSubmit, isLoading, initialParams }) {
  const [forecastType, setForecastType] = useState(initialParams?.forecastType || 'load');
  const [province, setProvince] = useState(initialParams?.province || (availableProvinces && availableProvinces.length > 0 ? availableProvinces[0] : ''));
  const today = new Date().toISOString().split('T')[0];
  const [forecastDate, setForecastDate] = useState(initialParams?.forecastDate || '2024-03-01');
  const [selectedQuantiles, setSelectedQuantiles] = useState(initialParams?.quantiles || [0.1, 0.5, 0.9]);
  const [historicalDays, setHistoricalDays] = useState(initialParams?.historicalDays || 15);
  const [interval, setIntervalMinutes] = useState(initialParams?.interval || 15);
  const [modelType, setModelType] = useState('torch');

  useEffect(() => {
    if (initialParams) {
      if (initialParams.forecastType) setForecastType(initialParams.forecastType);
      if (initialParams.province) setProvince(initialParams.province);
      if (initialParams.forecastDate) setForecastDate(initialParams.forecastDate);
      if (initialParams.quantiles) setSelectedQuantiles(initialParams.quantiles);
      if (initialParams.historicalDays) setHistoricalDays(initialParams.historicalDays);
      if (initialParams.interval) setIntervalMinutes(initialParams.interval);
    }
  }, [initialParams]);

  const handleQuantileChange = (quantile) => {
    setSelectedQuantiles(prev => 
      prev.includes(quantile) 
        ? prev.filter(q => q !== quantile)
        : [...prev, quantile]
    );
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (selectedQuantiles.length === 0) {
      alert('请至少选择一个分位数。');
      return;
    }
    onSubmit({
      predictionType: 'day-ahead',
      forecastType,
      province,
      forecastDate,
      quantiles: selectedQuantiles.sort((a, b) => a - b),
      historicalDays,
      interval,
      probabilistic: true,
      modelType, 
    });
  };

  return (
    <div className="bg-neutral-800 rounded-lg border border-neutral-700 shadow-xl p-6">
      <h2 className="text-2xl text-neutral-100 mb-6">参数设置 - 概率预测</h2>
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Row 1: Forecast Type & Province */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label htmlFor="prob-forecastType" className="block text-base font-medium text-neutral-300 mb-1">
              预测类型
            </label>
            <select
              id="prob-forecastType"
              value={forecastType}
              onChange={(e) => setForecastType(e.target.value)}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            >
              <option value="load">电力负荷</option>
              <option value="pv">光伏发电</option>
              <option value="wind">风电出力</option>
            </select>
          </div>
          <div>
            <label htmlFor="prob-province" className="block text-base font-medium text-neutral-300 mb-1">
              省份
            </label>
            <select
              id="prob-province"
              value={province}
              onChange={(e) => setProvince(e.target.value)}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            >
              {availableProvinces && availableProvinces.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
              {(!availableProvinces || availableProvinces.length === 0) && <option value="上海">上海 (默认)</option>}
            </select>
          </div>
        </div>

        {/* Row 2: Forecast Date & Historical Days */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label htmlFor="prob-forecastDate" className="block text-base font-medium text-neutral-300 mb-1">
              预测日期
            </label>
            <input
              type="date"
              id="prob-forecastDate"
              value={forecastDate}
              onChange={(e) => setForecastDate(e.target.value)}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            />
          </div>
          <div>
            <label htmlFor="prob-historicalDays" className="block text-base font-medium text-neutral-300 mb-1">
              历史数据天数 (1-30)
            </label>
            <input
              type="number"
              id="prob-historicalDays"
              value={historicalDays}
              min="1"
              max="30"
              onChange={(e) => setHistoricalDays(parseInt(e.target.value, 10))}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            />
          </div>
        </div>

        {/* Row 3: Interval & Model Type */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label htmlFor="prob-interval" className="block text-base font-medium text-neutral-300 mb-1">
              预测时间间隔 (分钟)
            </label>
            <input
              type="number"
              id="prob-interval"
              value={interval}
              min="15"
              step="15"
              onChange={(e) => setIntervalMinutes(parseInt(e.target.value, 10))}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            />
          </div>
          <div>
            <label htmlFor="prob-modelType" className="block text-base font-medium text-neutral-300 mb-1">
              模型类型
            </label>
            <select
              id="prob-modelType"
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            >
              <option value="torch">Torch (推荐)</option>
              <option value="keras">Keras</option>
            </select>
          </div>
        </div>

        {/* Row 4: Quantiles Selection */}
        <div>
          <label className="block text-base font-medium text-neutral-300 mb-2">
            选择分位数 (Quantiles)
          </label>
          <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-7 gap-3">
            {allQuantiles.map(q => (
              <div key={q} className="flex items-center">
                <input
                  id={`quantile-${q}`}
                  type="checkbox"
                  checked={selectedQuantiles.includes(q)}
                  onChange={() => handleQuantileChange(q)}
                  className="h-4 w-4 text-red-500 border-neutral-600 rounded bg-neutral-700 focus:ring-red-600 focus:ring-offset-neutral-800"
                />
                <label 
                  htmlFor={`quantile-${q}`} 
                  className="ml-2 text-base text-neutral-300 relative"
                  style={{ bottom: '5px' }}
                >
                  {q.toFixed(2)} 
                </label>
              </div>
            ))}
          </div>
        </div>

        {/* Submit Button */}
        <div>
          <button
            type="submit"
            disabled={isLoading || selectedQuantiles.length === 0}
            className="btn-primary w-full md:w-auto"
            title={selectedQuantiles.length === 0 ? "请至少选择一个分位数" : "开始概率预测"}
          >
            {isLoading ? '正在预测...' : '开始概率预测'}
          </button>
        </div>
      </form>
    </div>
  );
}

export default ProbabilisticSettings; 