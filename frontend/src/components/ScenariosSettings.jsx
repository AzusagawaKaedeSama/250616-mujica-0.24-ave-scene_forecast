import React, { useState, useEffect } from 'react';

const ScenariosSettings = ({ availableProvinces, onSubmit, isLoading, initialParams }) => {
  const [province, setProvince] = useState(initialParams?.province || (availableProvinces && availableProvinces.length > 0 ? availableProvinces[0] : ''));
  const [forecastDate, setForecastDate] = useState(initialParams?.forecastDate || '2024-03-01');
  const [forceRefresh, setForceRefresh] = useState(initialParams?.force_refresh || false);

  useEffect(() => {
    if (initialParams?.province) setProvince(initialParams.province);
    if (initialParams?.forecastDate) setForecastDate(initialParams.forecastDate);
    if (initialParams?.force_refresh !== undefined) setForceRefresh(initialParams.force_refresh);
  }, [initialParams]);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({
      province,
      date: forecastDate, // API expects 'date'
    });
  };

  return (
    <div className="bg-neutral-800 rounded-lg border border-neutral-700 shadow-xl p-6">
      <h2 className="text-2xl text-neutral-100 mb-6">参数设置 - 场景识别</h2>
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label htmlFor="scenario-province" className="block text-base font-medium text-neutral-300 mb-1">
              省份
            </label>
            <select
              id="scenario-province"
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
          <div>
            <label htmlFor="scenario-forecastDate" className="block text-base font-medium text-neutral-300 mb-1">
              预测日期 (用于场景识别)
            </label>
            <input
              type="date"
              id="scenario-forecastDate"
              value={forecastDate}
              onChange={(e) => setForecastDate(e.target.value)}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            />
          </div>
        </div>
        
        <div className="pt-4">
          <button
            type="submit"
            disabled={isLoading}
            className="btn-primary w-full md:w-auto"
          >
            {isLoading ? '正在识别...' : '识别场景'}
          </button>
        </div>
      </form>
    </div>
  );
}

export default ScenariosSettings; 