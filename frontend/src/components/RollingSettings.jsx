import React, { useState, useEffect } from 'react';

const RollingSettings = ({ availableProvinces, onSubmit, isLoading, initialParams }) => {
  const [forecastType, setForecastType] = useState(initialParams?.forecastType || 'load');
  const [province, setProvince] = useState(initialParams?.province || (availableProvinces && availableProvinces.length > 0 ? availableProvinces[0] : ''));
  const [startDate, setStartDate] = useState(initialParams?.startDate || '2024-03-01');
  const [endDate, setEndDate] = useState(initialParams?.endDate || '2024-03-03');
  const [interval, setIntervalMinutes] = useState(initialParams?.interval || 15);
  const [historicalDays, setHistoricalDays] = useState(initialParams?.historicalDays || 15);
  const [realTimeAdjustment, setRealTimeAdjustment] = useState(initialParams?.realTimeAdjustment || false);
  
  // 天气感知相关状态
  const [weatherAware, setWeatherAware] = useState(initialParams?.weatherAware || false);
  const [weatherFeatures, setWeatherFeatures] = useState(
    initialParams?.weatherFeatures || ['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction', 'precipitation', 'solar_radiation']
  );
  const [weatherDataPath, setWeatherDataPath] = useState(initialParams?.weatherDataPath || '');
  const [weatherModelDir, setWeatherModelDir] = useState(initialParams?.weatherModelDir || '');
  
  // PID修正相关状态
  const [enablePidCorrection, setEnablePidCorrection] = useState(initialParams?.enablePidCorrection || false);
  const [pidCorrectionParams, setPidCorrectionParams] = useState({
    pretrainDays: initialParams?.pretrainDays || 3,
    windowSizeHours: initialParams?.windowSizeHours || 72,
    enableAdaptation: initialParams?.enableAdaptation !== false,
    // Store Kp, Ki, Kd as strings for better input experience
    initialKp: initialParams?.initialKp?.toString() || '0.7', 
    initialKi: initialParams?.initialKi?.toString() || '0.05',
    initialKd: initialParams?.initialKd?.toString() || '0.1',
  });

  useEffect(() => {
    if (initialParams?.forecastType) setForecastType(initialParams.forecastType);
    if (initialParams?.province) setProvince(initialParams.province);
    if (initialParams?.startDate) setStartDate(initialParams.startDate);
    if (initialParams?.endDate) setEndDate(initialParams.endDate);
    if (initialParams?.interval) setIntervalMinutes(initialParams.interval);
    if (initialParams?.historicalDays) setHistoricalDays(initialParams.historicalDays);
    if (initialParams?.realTimeAdjustment !== undefined) setRealTimeAdjustment(initialParams.realTimeAdjustment);
    if (initialParams?.enablePidCorrection !== undefined) setEnablePidCorrection(initialParams.enablePidCorrection);
    if (initialParams?.weatherAware !== undefined) setWeatherAware(initialParams.weatherAware);
    if (initialParams?.weatherFeatures !== undefined) setWeatherFeatures(initialParams.weatherFeatures);
    if (initialParams?.weatherDataPath !== undefined) setWeatherDataPath(initialParams.weatherDataPath);
    if (initialParams?.weatherModelDir !== undefined) setWeatherModelDir(initialParams.weatherModelDir);
  }, [initialParams]);

  const handleSubmit = () => {
    
    const isNetLoad = forecastType === 'net_load';
    const actualForecastType = isNetLoad ? 'load' : forecastType;

    const params = {
      predictionType: 'rolling',
      forecastType: actualForecastType,
      province,
      startDate,
      endDate,
      interval,
      historicalDays,
      realTimeAdjustment: realTimeAdjustment && actualForecastType === 'load',
      calculateNetLoad: isNetLoad,
    };

    // 添加天气感知参数
    if (weatherAware && actualForecastType === 'load') {
      params.weatherAware = true;
      params.weatherFeatures = weatherFeatures;
      if (weatherDataPath) params.weatherDataPath = weatherDataPath;
      if (weatherModelDir) params.weatherModelDir = weatherModelDir;
    }

    // 添加PID修正参数
    if (enablePidCorrection && actualForecastType === 'load') {
      params.enablePidCorrection = true;
      params.pretrainDays = pidCorrectionParams.pretrainDays;
      params.windowSizeHours = pidCorrectionParams.windowSizeHours;
      params.enableAdaptation = pidCorrectionParams.enableAdaptation;
      
      // Parse Kp, Ki, Kd from strings to numbers before sending
      const kp = parseFloat(pidCorrectionParams.initialKp);
      const ki = parseFloat(pidCorrectionParams.initialKi);
      const kd = parseFloat(pidCorrectionParams.initialKd);

      // Send null if parsing fails, so backend can use defaults
      params.initialKp = isNaN(kp) ? null : kp;
      params.initialKi = isNaN(ki) ? null : ki;
      params.initialKd = isNaN(kd) ? null : kd;
    }

    onSubmit(params);
  };

  // PID参数组件
  const PIDCorrectionSettings = () => {
    // const updatePidParam = (period, param, value) => { // Old function for period-specific PID
    //   setPidCorrectionParams(prev => ({
    //     ...prev,
    //     pidParams: {
    //       ...prev.pidParams,
    //       [period]: {
    //         ...prev.pidParams[period],
    //         [param]: parseFloat(value) || 0
    //       }
    //     }
    //   }));
    // };

    const updateInitialPidParam = (param, value) => {
      setPidCorrectionParams(prev => ({
        ...prev,
        [param]: value // Store the raw string value from input
      }));
    };

    return (
      <div className="bg-neutral-900 rounded-lg p-4 space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-neutral-300 mb-1">
              预训练天数
            </label>
            <input
              type="number"
              value={pidCorrectionParams.pretrainDays}
              onChange={(e) => setPidCorrectionParams(prev => ({
                ...prev,
                pretrainDays: parseInt(e.target.value) || 1
              }))}
              min="1"
              max="7"
              className="w-full bg-neutral-700 border-neutral-600 text-neutral-100 rounded-md shadow-sm focus:ring-red-500 focus:border-red-500"
            />
            <p className="text-xs text-neutral-400 mt-1">用于收集初始误差数据</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-neutral-300 mb-1">
              滑动窗口大小 (小时)
            </label>
            <input
              type="number"
              value={pidCorrectionParams.windowSizeHours}
              onChange={(e) => setPidCorrectionParams(prev => ({
                ...prev,
                windowSizeHours: parseInt(e.target.value) || 24
              }))}
              min="24"
              max="168"
              step="24"
              className="w-full bg-neutral-700 border-neutral-600 text-neutral-100 rounded-md shadow-sm focus:ring-red-500 focus:border-red-500"
            />
            <p className="text-xs text-neutral-400 mt-1">误差分析的时间窗口</p>
          </div>

          <div className="flex items-end">
            <div className="flex items-center">
              <input
                id="enable-adaptation"
                type="checkbox"
                checked={pidCorrectionParams.enableAdaptation}
                onChange={(e) => setPidCorrectionParams(prev => ({
                  ...prev,
                  enableAdaptation: e.target.checked
                }))}
                className="h-4 w-4 text-red-500 border-neutral-600 rounded bg-neutral-700 focus:ring-red-600 focus:ring-offset-neutral-800"
              />
              <label 
                htmlFor="enable-adaptation" 
                className="ml-2 text-sm text-neutral-300"
              >
                启用自适应调整
              </label>
            </div>
          </div>
        </div>

        {/* PID参数设置 */}
        <div className="border-t border-neutral-600 pt-4">
          <h4 className="text-sm font-medium text-neutral-300 mb-3">初始PID参数设置</h4>
          <p className="text-xs text-neutral-400 mb-2">这些值将作为PID控制器自适应调整的起点。</p>
          
          {/* {['peak', 'valley', 'normal'].map(period => ( // Old rendering for period-specific PID */}
          <div className="mb-3">
            {/* <h5 className="text-xs font-medium text-neutral-400 mb-2 capitalize">
              {period === 'peak' ? '高峰时段' : period === 'valley' ? '低谷时段' : '正常时段'}
            </h5> */}
            <div className="grid grid-cols-3 gap-2">
              <div>
                <label className="block text-xs text-neutral-400">初始 Kp (比例)</label>
                <input
                  type="number"
                  value={pidCorrectionParams.initialKp} // Use direct initialKp
                  onChange={(e) => updateInitialPidParam('initialKp', e.target.value)} // Updated handler
                  min="0.1"
                  max="2.0"
                  step="0.1"
                  className="w-full bg-neutral-700 border-neutral-600 text-neutral-100 text-sm rounded px-2 py-1"
                />
              </div>
              <div>
                <label className="block text-xs text-neutral-400">初始 Ki (积分)</label>
                <input
                  type="number"
                  value={pidCorrectionParams.initialKi} // Use direct initialKi
                  onChange={(e) => updateInitialPidParam('initialKi', e.target.value)} // Updated handler
                  min="0.01"
                  max="0.5"
                  step="0.01"
                  className="w-full bg-neutral-700 border-neutral-600 text-neutral-100 text-sm rounded px-2 py-1"
                />
              </div>
              <div>
                <label className="block text-xs text-neutral-400">初始 Kd (微分)</label>
                <input
                  type="number"
                  value={pidCorrectionParams.initialKd} // Use direct initialKd
                  onChange={(e) => updateInitialPidParam('initialKd', e.target.value)} // Updated handler
                  min="0.01"
                  max="0.5"
                  step="0.01"
                  className="w-full bg-neutral-700 border-neutral-600 text-neutral-100 text-sm rounded px-2 py-1"
                />
              </div>
            </div>
          </div>
          {/* ))} */}
        </div>
      </div>
    );
  };

  return (
    <div className="bg-neutral-800 rounded-lg border border-neutral-700 shadow-xl p-6">
      <h2 className="text-2xl text-neutral-100 mb-6">滚动预测参数</h2>
      <div className="space-y-6">
        {/* Row 1: Forecast Type & Province */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label htmlFor="roll-forecastType" className="block text-base font-medium text-neutral-300 mb-1">
              预测类型
            </label>
            <select
              id="roll-forecastType"
              value={forecastType}
              onChange={(e) => {
                setForecastType(e.target.value);
                if (e.target.value !== 'load' && e.target.value !== 'net_load') {
                  setRealTimeAdjustment(false);
                  setEnablePidCorrection(false);
                }
              }}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            >
              <option value="load">电力负荷</option>
              <option value="pv">光伏发电</option>
              <option value="wind">风电出力</option>
              <option value="net_load">净负荷 (负荷-光伏-风电)</option>
            </select>
          </div>
          <div>
            <label htmlFor="roll-province" className="block text-base font-medium text-neutral-300 mb-1">
              省份
            </label>
            <select
              id="roll-province"
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

        {/* Row 2: Start Date & End Date */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label htmlFor="roll-startDate" className="block text-base font-medium text-neutral-300 mb-1">
              开始日期时间
            </label>
            <input
              type="date"
              id="roll-startDate"
              value={startDate}
              max={endDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            />
          </div>
          <div>
            <label htmlFor="roll-endDate" className="block text-base font-medium text-neutral-300 mb-1">
              结束日期时间
            </label>
            <input
              type="date"
              id="roll-endDate"
              value={endDate}
              min={startDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            />
          </div>
        </div>

        {/* Row 3: Interval & Historical Days */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label htmlFor="roll-interval" className="block text-base font-medium text-neutral-300 mb-1">
              滚动间隔 (分钟)
            </label>
            <input
              type="number"
              id="roll-interval"
              value={interval}
              min="15"
              step="15"
              onChange={(e) => setIntervalMinutes(parseInt(e.target.value, 10))}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            />
          </div>
          <div>
            <label htmlFor="roll-historicalDays" className="block text-base font-medium text-neutral-300 mb-1">
              历史数据天数 (1-30)
            </label>
            <input
              type="number"
              id="roll-historicalDays"
              value={historicalDays}
              min="1"
              max="30"
              onChange={(e) => setHistoricalDays(parseInt(e.target.value, 10))}
              className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm"
            />
          </div>
        </div>

        {/* Row 4: 负荷预测特有选项 */}
        {(forecastType === 'load' || forecastType === 'net_load') && (
          <div className="space-y-4">
            {/* 天气感知预测选项 */}
            <div className="flex items-center">
              <input
                id="roll-weatherAware"
                type="checkbox"
                checked={weatherAware}
                onChange={(e) => setWeatherAware(e.target.checked)}
                disabled={forecastType !== 'load'}
                className="h-4 w-4 text-red-500 border-neutral-600 rounded bg-neutral-700 focus:ring-red-600 focus:ring-offset-neutral-800"
              />
              <label 
                htmlFor="roll-weatherAware" 
                className={`ml-2 block text-base text-neutral-300 ${forecastType !== 'load' ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                启用天气感知预测
              </label>
            </div>

            {/* 天气感知参数设置面板 */}
            {weatherAware && forecastType === 'load' && (
              <div className="bg-neutral-900 rounded-lg p-4 space-y-4">
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
                          checked={weatherFeatures.includes(feature.key)}
                          onChange={(e) => {
                            const isChecked = e.target.checked;
                            setWeatherFeatures(prev => 
                              isChecked 
                                ? [...prev, feature.key]
                                : prev.filter(f => f !== feature.key)
                            );
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
                    <label htmlFor="roll-weatherDataPath" className="block text-sm font-medium text-neutral-300 mb-1">
                      天气数据路径 (可选)
                    </label>
                    <input
                      type="text"
                      id="roll-weatherDataPath"
                      value={weatherDataPath}
                      onChange={(e) => setWeatherDataPath(e.target.value)}
                      placeholder="留空则自动构建"
                      className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm text-sm"
                    />
                  </div>
                  <div>
                    <label htmlFor="roll-weatherModelDir" className="block text-sm font-medium text-neutral-300 mb-1">
                      天气模型目录 (可选)
                    </label>
                    <input
                      type="text"
                      id="roll-weatherModelDir"
                      value={weatherModelDir}
                      onChange={(e) => setWeatherModelDir(e.target.value)}
                      placeholder="留空则自动构建"
                      className="mt-1 block w-full bg-neutral-700 border-neutral-600 text-neutral-100 focus:ring-red-500 focus:border-red-500 rounded-md shadow-sm text-sm"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* 实时校正选项 */}
            <div className="flex items-center">
              <input
                id="roll-realTimeAdjustment"
                type="checkbox"
                checked={realTimeAdjustment}
                onChange={(e) => setRealTimeAdjustment(e.target.checked)}
                disabled={forecastType !== 'load' && forecastType !== 'net_load'}
                className="h-4 w-4 text-red-500 border-neutral-600 rounded bg-neutral-700 focus:ring-red-600 focus:ring-offset-neutral-800"
              />
              <label 
                htmlFor="roll-realTimeAdjustment" 
                className={`ml-2 block text-base text-neutral-300 ${(forecastType !== 'load' && forecastType !== 'net_load') ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                启用实时校正 (简单)
              </label>
            </div>

            {/* PID误差修正选项 */}
            <div className="flex items-center">
              <input
                id="roll-enablePidCorrection"
                type="checkbox"
                checked={enablePidCorrection}
                onChange={(e) => setEnablePidCorrection(e.target.checked)}
                disabled={forecastType !== 'load'}
                className="h-4 w-4 text-red-500 border-neutral-600 rounded bg-neutral-700 focus:ring-red-600 focus:ring-offset-neutral-800"
              />
              <label 
                htmlFor="roll-enablePidCorrection" 
                className={`ml-2 block text-base text-neutral-300 ${forecastType !== 'load' ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                启用PID误差修正 (高级)
              </label>
            </div>

            {/* PID参数设置面板 */}
            {enablePidCorrection && forecastType === 'load' && (
              <PIDCorrectionSettings />
            )}
          </div>
        )}

        {/* Submit Button */}
        <div>
          <button
            onClick={handleSubmit}
            disabled={isLoading}
            className="btn-primary w-full md:w-auto"
          >
            {isLoading ? '正在滚动预测...' : '开始滚动预测'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default RollingSettings;