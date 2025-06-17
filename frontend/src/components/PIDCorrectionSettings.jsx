import React, { useState, useEffect } from 'react';

const PIDCorrectionSettings = ({ initialParams = {}, onParamsChange }) => {
  // PID修正相关参数
  const [enablePidCorrection, setEnablePidCorrection] = useState(
    initialParams.enablePidCorrection || false
  );
  const [pretrainDays, setPretrainDays] = useState(
    initialParams.pretrainDays || 3
  );
  const [windowSizeHours, setWindowSizeHours] = useState(
    initialParams.windowSizeHours || 72
  );
  const [enableAdaptation, setEnableAdaptation] = useState(
    initialParams.enableAdaptation !== false
  );
  
  // PID参数
  const [pidParams, setPidParams] = useState(
    initialParams.pidParams || {
      peak: { kp: 0.8, ki: 0.1, kd: 0.2 },
      valley: { kp: 0.6, ki: 0.05, kd: 0.1 },
      normal: { kp: 0.7, ki: 0.08, kd: 0.15 }
    }
  );

  // 当参数变化时通知父组件
  useEffect(() => {
    if (onParamsChange) {
      onParamsChange({
        enablePidCorrection,
        pretrainDays,
        windowSizeHours,
        enableAdaptation,
        pidParams
      });
    }
  }, [enablePidCorrection, pretrainDays, windowSizeHours, enableAdaptation, pidParams]);

  // 更新特定时段的PID参数
  const updatePidParam = (period, param, value) => {
    setPidParams(prev => ({
      ...prev,
      [period]: {
        ...prev[period],
        [param]: parseFloat(value) || 0
      }
    }));
  };

  return (
    <div className="bg-neutral-800 rounded-lg border border-neutral-700 p-4 mt-4">
      <div className="flex items-center mb-4">
        <input
          id="enable-pid-correction"
          type="checkbox"
          checked={enablePidCorrection}
          onChange={(e) => setEnablePidCorrection(e.target.checked)}
          className="h-4 w-4 text-red-500 border-neutral-600 rounded bg-neutral-700 focus:ring-red-600 focus:ring-offset-neutral-800"
        />
        <label 
          htmlFor="enable-pid-correction" 
          className="ml-2 text-base font-medium text-neutral-100"
        >
          启用PID误差修正
        </label>
      </div>

      {enablePidCorrection && (
        <div className="space-y-4 pl-6">
          {/* 基本参数 */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-neutral-300 mb-1">
                预训练天数
              </label>
              <input
                type="number"
                value={pretrainDays}
                onChange={(e) => setPretrainDays(parseInt(e.target.value) || 1)}
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
                value={windowSizeHours}
                onChange={(e) => setWindowSizeHours(parseInt(e.target.value) || 24)}
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
                  checked={enableAdaptation}
                  onChange={(e) => setEnableAdaptation(e.target.checked)}
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
            <h4 className="text-sm font-medium text-neutral-300 mb-3">PID参数设置</h4>
            
            {['peak', 'valley', 'normal'].map(period => (
              <div key={period} className="mb-3">
                <h5 className="text-xs font-medium text-neutral-400 mb-2 capitalize">
                  {period === 'peak' ? '高峰时段' : period === 'valley' ? '低谷时段' : '正常时段'}
                </h5>
                <div className="grid grid-cols-3 gap-2">
                  <div>
                    <label className="block text-xs text-neutral-400">Kp (比例)</label>
                    <input
                      type="number"
                      value={pidParams[period].kp}
                      onChange={(e) => updatePidParam(period, 'kp', e.target.value)}
                      min="0.1"
                      max="2.0"
                      step="0.1"
                      className="w-full bg-neutral-700 border-neutral-600 text-neutral-100 text-sm rounded px-2 py-1"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-neutral-400">Ki (积分)</label>
                    <input
                      type="number"
                      value={pidParams[period].ki}
                      onChange={(e) => updatePidParam(period, 'ki', e.target.value)}
                      min="0.01"
                      max="0.5"
                      step="0.01"
                      className="w-full bg-neutral-700 border-neutral-600 text-neutral-100 text-sm rounded px-2 py-1"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-neutral-400">Kd (微分)</label>
                    <input
                      type="number"
                      value={pidParams[period].kd}
                      onChange={(e) => updatePidParam(period, 'kd', e.target.value)}
                      min="0.01"
                      max="0.5"
                      step="0.01"
                      className="w-full bg-neutral-700 border-neutral-600 text-neutral-100 text-sm rounded px-2 py-1"
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* 参数说明 */}
          <div className="text-xs text-neutral-400 border-t border-neutral-600 pt-3">
            <p className="mb-1">• Kp: 控制响应速度，值越大响应越快但可能振荡</p>
            <p className="mb-1">• Ki: 消除稳态误差，值越大消除越快但可能超调</p>
            <p>• Kd: 抑制振荡，提高稳定性</p>
          </div>
        </div>
      )}
    </div>
  );
};