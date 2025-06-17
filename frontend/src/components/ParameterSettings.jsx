import React, { useState, useEffect } from 'react';

// Helper for TABS if not passed as prop, or define globally in App.jsx and import
const TABS_LOC = { dayAhead: '日前预测' }; // Simplified for this example
const defaultProvinceList = ['上海', '福建', '江苏', '浙江', '安徽'];

const ParameterSettings = ({ title, onSubmit, isLoading, provinces, initialParams = {}, className, activeTabKey = 'dayAhead' }) => {
  const getInitialProvince = () => {
    const effectiveProvinces = (provinces && provinces.length > 0) ? provinces : defaultProvinceList;
    if (initialParams && initialParams.province && effectiveProvinces.includes(initialParams.province)) {
      return initialParams.province;
    }
    return effectiveProvinces.length > 0 ? effectiveProvinces[0] : '';
  };
  
  const [localParams, setLocalParams] = useState({
    forecastType: 'load',
    province: getInitialProvince(),
    forecastDate: new Date(new Date().setDate(new Date().getDate() + 1)).toISOString().split('T')[0],
    forecastEndDate: new Date(new Date().setDate(new Date().getDate() + 1)).toISOString().split('T')[0],
    historicalDays: 15,
    interval: 15,
    weatherAware: false,
    weatherFeatures: ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction', 'precipitation', 'solar_radiation'],
    weatherDataPath: '',
    weatherModelDir: '',
    ...initialParams,
    province: (initialParams && initialParams.province) ? initialParams.province : getInitialProvince(),
    forecastType: initialParams?.forecastType || 'load',
  });

  useEffect(() => {
    const effectiveProvinces = (provinces && provinces.length > 0) ? provinces : defaultProvinceList;
    
    setLocalParams(prev => {
      let newProvince = prev.province;
      if (initialParams && initialParams.province && effectiveProvinces.includes(initialParams.province)) {
        newProvince = initialParams.province;
      } else if (!effectiveProvinces.includes(prev.province)) {
        newProvince = effectiveProvinces.length > 0 ? effectiveProvinces[0] : '';
      }

      return {
        ...prev,
        ...initialParams,
        province: newProvince,
        interval: initialParams?.interval || prev.interval || 15,
        forecastType: initialParams?.forecastType || prev.forecastType || 'load',
      };
    });
  }, [initialParams, provinces]);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setLocalParams(prevParams => ({
      ...prevParams,
      [name]: type === 'checkbox' ? checked : type === 'number' ? parseInt(value, 10) : value,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const isNetLoad = localParams.forecastType === 'net_load';
    const actualForecastType = isNetLoad ? 'load' : localParams.forecastType;
    
    onSubmit({
      ...localParams,
      predictionType: 'day-ahead',
      forecastType: actualForecastType,
      calculateNetLoad: isNetLoad,
    });
  };

  const currentProvinceOptions = (provinces && provinces.length > 0) ? provinces : defaultProvinceList;

  return (
    <div className="bg-neutral-800 rounded-lg border border-neutral-700 shadow-xl p-6">
      <h2 className="text-2xl text-neutral-100 mb-6">
        {title || `参数设置 - ${TABS_LOC[activeTabKey]}`}
      </h2>
      <form onSubmit={handleSubmit} className="space-y-6">
        
        {/* Row 1: Forecast Type & Province */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4">
          <div>
            <label htmlFor={`forecastType-${activeTabKey}`} className="block text-base font-medium text-neutral-300 mb-1">
              预测类型
            </label>
            <select 
              id={`forecastType-${activeTabKey}`}
              name="forecastType" 
              value={localParams.forecastType}
              onChange={handleChange}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            >
              <option value="load">电力负荷</option>
              <option value="pv">光伏</option>
              <option value="wind">风电</option>
              <option value="net_load">净负荷 (负荷-光伏-风电)</option>
            </select>
          </div>
          <div>
            <label htmlFor={`province-${activeTabKey}`} className="block text-base font-medium text-neutral-300 mb-1">
              省份
            </label>
            <select 
              id={`province-${activeTabKey}`}
              name="province" 
              value={localParams.province}
              onChange={handleChange} 
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            >
              {currentProvinceOptions.map(p => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>
        </div>

        {/* Row 2: Forecast Date & Forecast End Date */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4">
          <div>
            <label htmlFor={`forecastDate-${activeTabKey}`} className="block text-base font-medium text-neutral-300 mb-1">
              预测开始日期
            </label>
            <input 
              type="date" 
              id={`forecastDate-${activeTabKey}`}
              name="forecastDate" 
              value={localParams.forecastDate}
              onChange={handleChange}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            />
          </div>
          <div>
            <label htmlFor={`forecastEndDate-${activeTabKey}`} className="block text-base font-medium text-neutral-300 mb-1">
              预测结束日期
            </label>
            <input 
              type="date" 
              id={`forecastEndDate-${activeTabKey}`}
              name="forecastEndDate" 
              value={localParams.forecastEndDate}
              min={localParams.forecastDate}
              onChange={handleChange}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            />
          </div>
        </div>

        {/* Row 3: Historical Days & Interval */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4">
          <div>
            <label htmlFor={`historicalDays-${activeTabKey}`} className="block text-base font-medium text-neutral-300 mb-1">
              历史数据天数 (1-30)
            </label>
            <input 
              type="number" 
              id={`historicalDays-${activeTabKey}`}
              name="historicalDays" 
              value={localParams.historicalDays}
              onChange={handleChange}
              min="1" 
              max="30"
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            />
          </div>
          <div>
            <label htmlFor={`interval-${activeTabKey}`} className="block text-base font-medium text-neutral-300 mb-1">
              预测时间间隔 (分钟)
            </label>
            <input
              type="number"
              id={`interval-${activeTabKey}`}
              name="interval"
              value={localParams.interval}
              min="15"
              step="15"
              onChange={handleChange}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            />
          </div>
        </div>

        {/* 天气感知预测设置 */}
        {localParams.forecastType === 'load' && (
          <div className="bg-neutral-900 rounded-lg p-4 space-y-4">
            <div className="flex items-center space-x-3">
              <input
                type="checkbox"
                id={`weatherAware-${activeTabKey}`}
                name="weatherAware"
                checked={localParams.weatherAware}
                onChange={handleChange}
                className="h-4 w-4 text-red-500 border-neutral-600 rounded bg-neutral-700 focus:ring-red-600 focus:ring-offset-neutral-800"
              />
              <label htmlFor={`weatherAware-${activeTabKey}`} className="text-base font-medium text-neutral-300">
                启用天气感知预测
              </label>
            </div>
            
            {localParams.weatherAware && (
              <div className="space-y-4 pl-7">
                <div>
                  <label className="block text-sm font-medium text-neutral-300 mb-2">
                    天气特征选择
                  </label>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                    {[
                      { key: 'temperature', label: '温度' },
                      { key: 'humidity', label: '湿度' },
                      { key: 'pressure', label: '气压' },
                      { key: 'wind_speed', label: '风速' },
                      { key: 'wind_direction', label: '风向' },
                      { key: 'precipitation', label: '降水' },
                      { key: 'solar_radiation', label: '太阳辐射' }
                    ].map(feature => (
                      <label key={feature.key} className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          checked={localParams.weatherFeatures.includes(feature.key)}
                          onChange={(e) => {
                            const isChecked = e.target.checked;
                            setLocalParams(prev => ({
                              ...prev,
                              weatherFeatures: isChecked 
                                ? [...prev.weatherFeatures, feature.key]
                                : prev.weatherFeatures.filter(f => f !== feature.key)
                            }));
                          }}
                          className="h-3 w-3 text-red-500 border-neutral-600 rounded bg-neutral-700 focus:ring-red-600"
                        />
                        <span className="text-xs text-neutral-300">{feature.label}</span>
                      </label>
                    ))}
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label htmlFor={`weatherDataPath-${activeTabKey}`} className="block text-sm font-medium text-neutral-300 mb-1">
                      天气数据路径 (可选)
                    </label>
                    <input
                      type="text"
                      id={`weatherDataPath-${activeTabKey}`}
                      name="weatherDataPath"
                      value={localParams.weatherDataPath}
                      onChange={handleChange}
                      placeholder="留空则自动构建"
                      className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm text-sm"
                    />
                  </div>
                  <div>
                    <label htmlFor={`weatherModelDir-${activeTabKey}`} className="block text-sm font-medium text-neutral-300 mb-1">
                      天气模型目录 (可选)
                    </label>
                    <input
                      type="text"
                      id={`weatherModelDir-${activeTabKey}`}
                      name="weatherModelDir"
                      value={localParams.weatherModelDir}
                      onChange={handleChange}
                      placeholder="留空则自动构建"
                      className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm text-sm"
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        <div className="pt-4">
          <button 
            type="submit" 
            className="btn-primary w-full md:w-auto" 
            disabled={isLoading}
          >
            {isLoading ? '预测中...' : `开始${TABS_LOC[activeTabKey] || '预测'}`}
          </button>
        </div>
      </form>
    </div>
  );
};

export default ParameterSettings; 