import React, { useState, useEffect } from 'react';

function HistoricalResultsSettings({ availableProvinces, onSubmit, isLoading, initialParams }) {
  const [params, setParams] = useState({
    forecast_type: initialParams?.forecast_type || 'load',
    province: initialParams?.province || (availableProvinces && availableProvinces.length > 0 ? availableProvinces[0] : ''),
    start_date: initialParams?.start_date || '2024-01-01',
    end_date: initialParams?.end_date || '2024-03-31',
    prediction_type: initialParams?.prediction_type || 'all',
    model_type: initialParams?.model_type || 'all',
    year: initialParams?.year || '',
  });

  // 更新参数当props变化时
  useEffect(() => {
    const updates = {};
    if (initialParams?.forecast_type) updates.forecast_type = initialParams.forecast_type;
    if (initialParams?.province) updates.province = initialParams.province;
    if (initialParams?.start_date) updates.start_date = initialParams.start_date;
    if (initialParams?.end_date) updates.end_date = initialParams.end_date;
    if (initialParams?.prediction_type) updates.prediction_type = initialParams.prediction_type;
    if (initialParams?.model_type) updates.model_type = initialParams.model_type;
    if (initialParams?.year) updates.year = initialParams.year;
    
    if (Object.keys(updates).length > 0) {
      setParams(prev => ({ ...prev, ...updates }));
    }
  }, [initialParams]);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({
      forecastType: params.forecast_type,
      province: params.province,
      startDate: params.year ? '' : params.start_date,
      endDate: params.year ? '' : params.end_date,
      predictionType: params.prediction_type,
      modelType: params.year ? 'convtrans_peak' : params.model_type,
      year: params.year,
    });
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setParams({
      ...params,
      [name]: value,
    });
  };

  return (
    <div className="bg-neutral-800 rounded-lg border border-neutral-700 shadow-xl p-6">
      <h2 className="text-2xl text-neutral-100 mb-6">历史结果查询</h2>
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Row 1: Forecast Type & Province */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label htmlFor="historical-forecastType" className="block text-base font-medium text-neutral-300 mb-1">
              预测类型 (Target Type)
            </label>
            <select
              id="historical-forecastType"
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
            <label htmlFor="historical-province" className="block text-base font-medium text-neutral-300 mb-1">
              省份 (Province)
            </label>
            <select
              id="historical-province"
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

        {/* Row 2: Year (for yearly CSV) OR Date Range (for JSON) */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label htmlFor="historical-year" className="block text-base font-medium text-neutral-300 mb-1">
              年份 (Year - for yearly CSV)
            </label>
            <input
              type="text"
              id="historical-year"
              name="year"
              value={params.year}
              onChange={handleChange}
              placeholder="例如: 2023 (优先于日期范围)"
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            />
          </div>
          <div>
            {/* Empty div for layout consistency, or add another relevant field if year is not used */}
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label htmlFor="historical-startDate" className="block text-base font-medium text-neutral-300 mb-1">
              开始日期 (Start Date - if Year not set)
            </label>
            <input
              type="date"
              id="historical-startDate"
              name="start_date"
              value={params.start_date}
              max={params.end_date}
              onChange={handleChange}
              disabled={!!params.year}
              className={`mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm ${params.year ? 'opacity-50 cursor-not-allowed' : ''}`}
            />
          </div>
          <div>
            <label htmlFor="historical-endDate" className="block text-base font-medium text-neutral-300 mb-1">
              结束日期 (End Date - if Year not set)
            </label>
            <input
              type="date"
              id="historical-endDate"
              name="end_date"
              value={params.end_date}
              min={params.start_date}
              onChange={handleChange}
              disabled={!!params.year}
              className={`mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm ${params.year ? 'opacity-50 cursor-not-allowed' : ''}`}
            />
          </div>
        </div>

        {/* Row 3: Prediction Type & Model Type */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label htmlFor="historical-predictionType" className="block text-base font-medium text-neutral-300 mb-1">
              预测方式 (Prediction Method)
            </label>
            <select
              id="historical-predictionType"
              name="prediction_type"
              value={params.prediction_type}
              onChange={handleChange}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            >
              <option value="all">全部</option>
              <option value="day-ahead">日前预测</option>
              <option value="rolling">滚动预测</option>
              <option value="interval">区间预测</option>
              <option value="probabilistic">概率预测</option>
            </select>
          </div>
          <div>
            <label htmlFor="historical-modelType" className="block text-base font-medium text-neutral-300 mb-1">
              模型类型 (Model Type - if Year not set)
            </label>
            <select
              id="historical-modelType"
              name="model_type"
              value={params.model_type}
              onChange={handleChange}
              disabled={!!params.year}
              className={`mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm ${params.year ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              <option value="all">全部</option>
              <option value="torch">PyTorch</option>
              <option value="peak_aware">峰值感知</option>
              <option value="statistical">统计模型</option>
            </select>
            {params.year && (
              <p className="mt-1 text-xs text-neutral-400">
                (年度CSV查询时固定为 convtrans_peak)
              </p>
            )}
          </div>
        </div>

        {/* Submit Button */}
        <div>
          <button
            type="submit"
            disabled={isLoading}
            className="btn-primary w-full md:w-auto"
          >
            {isLoading ? '正在查询...' : '查询历史结果'}
          </button>
        </div>
      </form>
    </div>
  );
}

export default HistoricalResultsSettings; 