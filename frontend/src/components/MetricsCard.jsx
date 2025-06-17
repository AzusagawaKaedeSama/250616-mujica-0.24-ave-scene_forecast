import React from 'react';

function MetricsCard({ metrics }) {
  console.log('MetricsCard 接收到的指标：', metrics);
  
  if (!metrics || Object.keys(metrics).length === 0) {
    return null;
  }

  // 定义指标显示配置
  const metricsConfig = [
    {
      key: 'mape',
      label: 'MAPE',
      unit: '%',
      description: '平均绝对百分比误差',
      formatter: (value) => `${value.toFixed(2)}%`
    },
    {
      key: 'mae',
      label: 'MAE',
      unit: 'MW',
      description: '平均绝对误差',
      formatter: (value) => `${value.toFixed(2)} MW`
    },
    {
      key: 'rmse',
      label: 'RMSE',
      unit: 'MW',
      description: '均方根误差',
      formatter: (value) => `${value.toFixed(2)} MW`
    },
    {
      key: 'peak_mape',
      label: '峰时MAPE',
      unit: '%',
      description: '高峰时段平均绝对百分比误差',
      formatter: (value) => `${value.toFixed(2)}%`
    },
    {
      key: 'valley_mape',
      label: '谷时MAPE',
      unit: '%',
      description: '低谷时段平均绝对百分比误差',
      formatter: (value) => `${value.toFixed(2)}%`
    },
    {
      key: 'max_error',
      label: '最大误差',
      unit: '%',
      description: '最大绝对百分比误差',
      formatter: (value) => `${value.toFixed(2)}%`
    },
    // 区间预测特有指标
    {
      key: 'hit_rate',
      label: '命中率',
      unit: '%',
      description: '预测区间命中率',
      formatter: (value) => `${value.toFixed(2)}%`
    },
    {
      key: 'avg_interval_width',
      label: '平均区间宽度',
      unit: 'MW',
      description: '预测区间平均宽度',
      formatter: (value) => `${value.toFixed(2)} MW`
    },
    {
      key: 'average_interval_width',
      label: '平均区间宽度',
      unit: 'MW',
      description: '预测区间平均宽度',
      formatter: (value) => `${value.toFixed(2)} MW`
    },
    {
      key: 'confidence_level',
      label: '置信水平',
      unit: '',
      description: '预测置信水平',
      formatter: (value) => `${(value * 100).toFixed(0)}%`
    },
    {
      key: 'total_predictions',
      label: '预测点数',
      unit: '',
      description: '总预测点数',
      formatter: (value) => `${value}`
    }
  ];

  // 过滤出有效的指标
  const validMetrics = metricsConfig.filter(config => 
    metrics[config.key] !== undefined && 
    metrics[config.key] !== null &&
    !isNaN(metrics[config.key])
  );

  if (validMetrics.length === 0) {
    console.log('没有有效的指标可显示');
    return null;
  }

  return (
    <div className="mt-6 p-4 bg-neutral-800 rounded-lg border border-neutral-700 shadow-xl">
      <h3 className="text-xl font-medium text-neutral-100 mb-4">预测指标</h3>
      
      {/* 可再生能源增强预测结果展示 */}
      {metrics.renewable_predictions && (
        <div className="mb-6 p-4 bg-gradient-to-r from-green-900/20 to-blue-900/20 rounded-lg border border-green-600/30">
          <h4 className="text-lg font-medium text-green-400 mb-3 flex items-center">
            <span className="mr-2">🌱</span>
            可再生能源预测分析
          </h4>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* 光伏预测结果 */}
            {metrics.renewable_predictions.pv && (
              <div className="bg-neutral-700 rounded-lg p-4 border border-orange-600/30">
                <h5 className="text-md font-medium text-orange-300 mb-3 flex items-center">
                  <span className="mr-2">☀️</span>
                  光伏发电预测
                </h5>
                
                <div className="space-y-3">
                  {/* 预测状态 */}
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-neutral-400">预测状态</span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      metrics.renewable_predictions.pv.status === 'success' 
                        ? 'bg-green-600 text-white' 
                        : metrics.renewable_predictions.pv.status === 'no_data'
                        ? 'bg-yellow-600 text-white'
                        : metrics.renewable_predictions.pv.status === 'missing_files'
                        ? 'bg-orange-600 text-white'
                        : 'bg-red-600 text-white'
                    }`}>
                      {metrics.renewable_predictions.pv.status === 'success' ? '成功' : 
                       metrics.renewable_predictions.pv.status === 'no_data' ? '数据缺失' :
                       metrics.renewable_predictions.pv.status === 'missing_files' ? '模型缺失' : '失败'}
                    </span>
                  </div>
                  
                  {/* 错误信息 */}
                  {metrics.renewable_predictions.pv.error && (
                    <div className="p-2 bg-red-900/20 border border-red-600/30 rounded text-sm text-red-300">
                      错误: {metrics.renewable_predictions.pv.error}
                    </div>
                  )}
                  
                  {/* 质量评估 */}
                  {metrics.renewable_predictions.pv.quality_assessment && (
                    <div className="grid grid-cols-2 gap-2">
                      <div className="bg-neutral-800 rounded p-2">
                        <div className="text-xs text-neutral-400">夜间零值率</div>
                        <div className="text-sm font-semibold text-orange-300">
                          {(metrics.renewable_predictions.pv.quality_assessment.night_zero_rate * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="bg-neutral-800 rounded p-2">
                        <div className="text-xs text-neutral-400">日间有效率</div>
                        <div className="text-sm font-semibold text-orange-300">
                          {(metrics.renewable_predictions.pv.quality_assessment.day_valid_rate * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {/* 高发时段 */}
                  {metrics.renewable_predictions.pv.high_output_periods && 
                   metrics.renewable_predictions.pv.high_output_periods.length > 0 && (
                    <div className="space-y-2">
                      <div className="text-sm text-neutral-400">高发时段 ({metrics.renewable_predictions.pv.high_output_periods.length}个)</div>
                      {metrics.renewable_predictions.pv.high_output_periods.slice(0, 2).map((period, idx) => (
                        <div key={idx} className="bg-neutral-800 rounded p-2 text-xs">
                          <div className="text-orange-300 font-medium">
                            {new Date(period.start_time).toLocaleTimeString('zh-CN', {hour: '2-digit', minute: '2-digit'})} - 
                            {new Date(period.end_time).toLocaleTimeString('zh-CN', {hour: '2-digit', minute: '2-digit'})}
                          </div>
                          <div className="text-neutral-300">
                            平均: {period.avg_output?.toFixed(1)}MW, 峰值: {period.max_output?.toFixed(1)}MW
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
            
            {/* 风电预测结果 */}
            {metrics.renewable_predictions.wind && (
              <div className="bg-neutral-700 rounded-lg p-4 border border-cyan-600/30">
                <h5 className="text-md font-medium text-cyan-300 mb-3 flex items-center">
                  <span className="mr-2">💨</span>
                  风力发电预测
                </h5>
                
                <div className="space-y-3">
                  {/* 预测状态 */}
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-neutral-400">预测状态</span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      metrics.renewable_predictions.wind.status === 'success' 
                        ? 'bg-green-600 text-white' 
                        : metrics.renewable_predictions.wind.status === 'no_data'
                        ? 'bg-yellow-600 text-white'
                        : metrics.renewable_predictions.wind.status === 'missing_files'
                        ? 'bg-orange-600 text-white'
                        : 'bg-red-600 text-white'
                    }`}>
                      {metrics.renewable_predictions.wind.status === 'success' ? '成功' : 
                       metrics.renewable_predictions.wind.status === 'no_data' ? '数据缺失' :
                       metrics.renewable_predictions.wind.status === 'missing_files' ? '模型缺失' : '失败'}
                    </span>
                  </div>
                  
                  {/* 错误信息 */}
                  {metrics.renewable_predictions.wind.error && (
                    <div className="p-2 bg-red-900/20 border border-red-600/30 rounded text-sm text-red-300">
                      错误: {metrics.renewable_predictions.wind.error}
                    </div>
                  )}
                  
                  {/* 质量评估 */}
                  {metrics.renewable_predictions.wind.quality_assessment && (
                    <div className="grid grid-cols-2 gap-2">
                      <div className="bg-neutral-800 rounded p-2">
                        <div className="text-xs text-neutral-400">有效预测率</div>
                        <div className="text-sm font-semibold text-cyan-300">
                          {(metrics.renewable_predictions.wind.quality_assessment.valid_rate * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="bg-neutral-800 rounded p-2">
                        <div className="text-xs text-neutral-400">平均出力</div>
                        <div className="text-sm font-semibold text-cyan-300">
                          {metrics.renewable_predictions.wind.quality_assessment.avg_output?.toFixed(1)}MW
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {/* 高发时段 */}
                  {metrics.renewable_predictions.wind.high_output_periods && 
                   metrics.renewable_predictions.wind.high_output_periods.length > 0 && (
                    <div className="space-y-2">
                      <div className="text-sm text-neutral-400">高发时段 ({metrics.renewable_predictions.wind.high_output_periods.length}个)</div>
                      {metrics.renewable_predictions.wind.high_output_periods.slice(0, 2).map((period, idx) => (
                        <div key={idx} className="bg-neutral-800 rounded p-2 text-xs">
                          <div className="text-cyan-300 font-medium">
                            {new Date(period.start_time).toLocaleTimeString('zh-CN', {hour: '2-digit', minute: '2-digit'})} - 
                            {new Date(period.end_time).toLocaleTimeString('zh-CN', {hour: '2-digit', minute: '2-digit'})}
                          </div>
                          <div className="text-neutral-300">
                            平均: {period.avg_output?.toFixed(1)}MW, 峰值: {period.max_output?.toFixed(1)}MW
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
          
          {/* 联合高发时段分析 */}
          {metrics.renewable_predictions.combined_high_output && (
            <div className="mt-4 p-4 bg-gradient-to-r from-purple-900/20 to-pink-900/20 rounded-lg border border-purple-600/30">
              <h5 className="text-md font-medium text-purple-300 mb-3 flex items-center">
                <span className="mr-2">⚡</span>
                联合高发时段分析
              </h5>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
                <div className="bg-neutral-800 rounded p-2 text-center">
                  <div className="text-xs text-neutral-400">总高发时段</div>
                  <div className="text-lg font-bold text-purple-300">
                    {metrics.renewable_predictions.combined_high_output.analysis?.total_renewable_high_periods || 0}
                  </div>
                </div>
                <div className="bg-neutral-800 rounded p-2 text-center">
                  <div className="text-xs text-neutral-400">光伏独有</div>
                  <div className="text-lg font-bold text-orange-300">
                    {metrics.renewable_predictions.combined_high_output.analysis?.pv_only_periods || 0}
                  </div>
                </div>
                <div className="bg-neutral-800 rounded p-2 text-center">
                  <div className="text-xs text-neutral-400">风电独有</div>
                  <div className="text-lg font-bold text-cyan-300">
                    {metrics.renewable_predictions.combined_high_output.analysis?.wind_only_periods || 0}
                  </div>
                </div>
                <div className="bg-neutral-800 rounded p-2 text-center">
                  <div className="text-xs text-neutral-400">同时高发</div>
                  <div className="text-lg font-bold text-pink-300">
                    {metrics.renewable_predictions.combined_high_output.analysis?.both_high_periods || 0}
                  </div>
                </div>
              </div>
              
              {/* 渗透率等级 */}
              {metrics.renewable_predictions.combined_high_output.analysis?.renewable_penetration_level && (
                <div className="text-center">
                  <span className="text-sm text-neutral-400">新能源渗透率等级: </span>
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                    metrics.renewable_predictions.combined_high_output.analysis.renewable_penetration_level === 'high' 
                      ? 'bg-red-600 text-white'
                      : metrics.renewable_predictions.combined_high_output.analysis.renewable_penetration_level === 'medium'
                      ? 'bg-yellow-600 text-white'
                      : 'bg-green-600 text-white'
                  }`}>
                    {metrics.renewable_predictions.combined_high_output.analysis.renewable_penetration_level === 'high' ? '高' :
                     metrics.renewable_predictions.combined_high_output.analysis.renewable_penetration_level === 'medium' ? '中' : '低'}
                  </span>
                </div>
              )}
            </div>
          )}
        </div>
      )}
      
      {/* 增强场景信息展示 */}
      {metrics.enhanced_scenario && (
        <div className="mb-6 p-4 bg-gradient-to-r from-indigo-900/20 to-purple-900/20 rounded-lg border border-indigo-600/30">
          <h4 className="text-lg font-medium text-indigo-400 mb-3 flex items-center">
            <span className="mr-2">🎭</span>
            增强场景分析
          </h4>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* 增强场景信息 */}
            {metrics.enhanced_scenario.enhanced_scenario && (
              <div className="bg-neutral-700 rounded-lg p-3">
                <div className="text-sm text-neutral-400 mb-2">识别场景</div>
                <div className="text-base font-semibold text-indigo-300 mb-1">
                  {metrics.enhanced_scenario.enhanced_scenario.name}
                </div>
                <div className="text-xs text-neutral-300">
                  风险等级: {metrics.enhanced_scenario.enhanced_scenario.risk_level}
                </div>
                <div className="text-xs text-neutral-300">
                  负荷影响: {metrics.enhanced_scenario.enhanced_scenario.load_impact}
                </div>
              </div>
            )}
            
            {/* 新能源状态 */}
            {metrics.enhanced_scenario.renewable_status && (
              <div className="bg-neutral-700 rounded-lg p-3">
                <div className="text-sm text-neutral-400 mb-2">新能源状态</div>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-neutral-300">光伏:</span>
                    <span className={`px-2 py-0.5 rounded text-xs ${
                      metrics.enhanced_scenario.renewable_status.pv_status === 'high_output' ? 'bg-orange-600 text-white' :
                      metrics.enhanced_scenario.renewable_status.pv_status === 'moderate_output' ? 'bg-yellow-600 text-white' :
                      'bg-green-600 text-white'
                    }`}>
                      {metrics.enhanced_scenario.renewable_status.pv_status === 'high_output' ? '高发' :
                       metrics.enhanced_scenario.renewable_status.pv_status === 'moderate_output' ? '中发' :
                       metrics.enhanced_scenario.renewable_status.pv_status === 'low_output' ? '低发' : '正常'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-neutral-300">风电:</span>
                    <span className={`px-2 py-0.5 rounded text-xs ${
                      metrics.enhanced_scenario.renewable_status.wind_status === 'high_output' ? 'bg-cyan-600 text-white' :
                      metrics.enhanced_scenario.renewable_status.wind_status === 'moderate_output' ? 'bg-blue-600 text-white' :
                      'bg-green-600 text-white'
                    }`}>
                      {metrics.enhanced_scenario.renewable_status.wind_status === 'high_output' ? '高发' :
                       metrics.enhanced_scenario.renewable_status.wind_status === 'moderate_output' ? '中发' :
                       metrics.enhanced_scenario.renewable_status.wind_status === 'low_output' ? '低发' : '正常'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-neutral-300">渗透率:</span>
                    <span className="text-neutral-200">
                      {metrics.enhanced_scenario.renewable_status.renewable_penetration}
                    </span>
                  </div>
                </div>
              </div>
            )}
            
            {/* 综合风险等级 */}
            <div className="bg-neutral-700 rounded-lg p-3">
              <div className="text-sm text-neutral-400 mb-2">综合评估</div>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-neutral-300">风险等级:</span>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                    metrics.enhanced_scenario.composite_risk_level === 'high' ? 'bg-red-600 text-white' :
                    metrics.enhanced_scenario.composite_risk_level === 'medium' || 
                    metrics.enhanced_scenario.composite_risk_level === 'low-medium' ? 'bg-yellow-600 text-white' :
                    'bg-green-600 text-white'
                  }`}>
                    {metrics.enhanced_scenario.composite_risk_level}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-neutral-300">不确定性调整:</span>
                  <span className="text-sm font-semibold text-neutral-200">
                    ×{metrics.enhanced_scenario.uncertainty_adjustment?.toFixed(2)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 天气场景信息展示 */}
      {metrics.weather_scenario && !metrics.weather_scenario.error && (
        <div className="mb-6 p-4 bg-neutral-700 rounded-lg border border-neutral-600">
          <h4 className="text-lg font-medium text-blue-400 mb-3">天气场景分析</h4>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div className="bg-neutral-800 rounded-lg p-3">
              <div className="text-sm text-neutral-400 mb-1">场景类型</div>
              <div className="text-base font-semibold text-white">
                {metrics.weather_scenario.scenario_type}
              </div>
            </div>
            <div className="bg-neutral-800 rounded-lg p-3">
              <div className="text-sm text-neutral-400 mb-1">平均温度</div>
              <div className="text-base font-semibold text-white">
                {metrics.weather_scenario.temperature_mean?.toFixed(1) || 'N/A'}°C
              </div>
            </div>
            <div className="bg-neutral-800 rounded-lg p-3">
              <div className="text-sm text-neutral-400 mb-1">温度范围</div>
              <div className="text-base font-semibold text-white">
                {metrics.weather_scenario.temperature_min?.toFixed(1) || 'N/A'}°C - 
                {metrics.weather_scenario.temperature_max?.toFixed(1) || 'N/A'}°C
              </div>
            </div>
            <div className="bg-neutral-800 rounded-lg p-3">
              <div className="text-sm text-neutral-400 mb-1">平均湿度</div>
              <div className="text-base font-semibold text-white">
                {metrics.weather_scenario.humidity_mean?.toFixed(1) || 'N/A'}%
              </div>
            </div>
            <div className="bg-neutral-800 rounded-lg p-3">
              <div className="text-sm text-neutral-400 mb-1">平均风速</div>
              <div className="text-base font-semibold text-white">
                {metrics.weather_scenario.wind_speed_mean?.toFixed(1) || 'N/A'} m/s
              </div>
            </div>
            <div className="bg-neutral-800 rounded-lg p-3">
              <div className="text-sm text-neutral-400 mb-1">累计降水</div>
              <div className="text-base font-semibold text-white">
                {metrics.weather_scenario.precipitation_sum?.toFixed(1) || 'N/A'} mm
              </div>
            </div>
          </div>
          
          {/* 新增：典型场景匹配结果 */}
          {metrics.weather_scenario.scenario_match && (
            <div className="mt-4 p-4 bg-gradient-to-r from-purple-900/20 to-blue-900/20 rounded-lg border border-purple-600/30">
              <h5 className="text-md font-medium text-purple-300 mb-3 flex items-center">
                <span className="mr-2">🎯</span>
                典型场景匹配
              </h5>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* 最佳匹配场景 */}
                <div className="bg-neutral-800 rounded-lg p-3">
                  <div className="text-sm text-neutral-400 mb-2">最佳匹配场景</div>
                  <div className="text-base font-semibold text-purple-300 mb-1">
                    {metrics.weather_scenario.scenario_match.matched_scenario}
                  </div>
                  <div className="text-sm text-neutral-300">
                    相似度: {metrics.weather_scenario.scenario_match.similarity_percentage?.toFixed(1) || 'N/A'}%
                  </div>
                  <div className="text-xs text-neutral-400 mt-1">
                    置信度: {(metrics.weather_scenario.scenario_match.confidence_level * 100)?.toFixed(0) || 'N/A'}%
                  </div>
                </div>
                
                {/* 场景描述 */}
                <div className="bg-neutral-800 rounded-lg p-3">
                  <div className="text-sm text-neutral-400 mb-2">场景特点</div>
                  <div className="text-sm text-neutral-300 leading-relaxed">
                    {metrics.weather_scenario.scenario_match.description || '无描述'}
                  </div>
                  <div className="text-xs text-neutral-400 mt-1">
                    历史占比: {metrics.weather_scenario.scenario_match.typical_percentage?.toFixed(1) || 'N/A'}%
                  </div>
                </div>
              </div>
              
              {/* 相似场景排名 */}
              {metrics.weather_scenario.scenario_match.top_scenarios && metrics.weather_scenario.scenario_match.top_scenarios.length > 0 && (
                <div className="mt-3">
                  <div className="text-sm text-neutral-400 mb-2">相似场景排名</div>
                  <div className="flex flex-wrap gap-2">
                    {metrics.weather_scenario.scenario_match.top_scenarios.map((scenario, index) => (
                      <div 
                        key={index}
                        className={`px-3 py-1 rounded-full text-xs ${
                          index === 0 
                            ? 'bg-purple-600 text-white' 
                            : index === 1
                            ? 'bg-blue-600 text-white'
                            : 'bg-neutral-600 text-neutral-300'
                        }`}
                      >
                        {scenario.rank}. {scenario.name} ({scenario.similarity_percentage?.toFixed(1)}%)
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {/* 负荷特征 */}
              {(metrics.weather_scenario.daily_load_mean || metrics.weather_scenario.daily_load_volatility) && (
                <div className="mt-3 grid grid-cols-2 gap-3">
                  <div className="bg-neutral-800 rounded-lg p-2">
                    <div className="text-xs text-neutral-400">平均负荷</div>
                    <div className="text-sm font-semibold text-white">
                      {metrics.weather_scenario.daily_load_mean?.toFixed(0) || 'N/A'} MW
                    </div>
                  </div>
                  <div className="bg-neutral-800 rounded-lg p-2">
                    <div className="text-xs text-neutral-400">负荷波动率</div>
                    <div className="text-sm font-semibold text-white">
                      {(metrics.weather_scenario.daily_load_volatility * 100)?.toFixed(1) || 'N/A'}%
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
          
          {/* 场景匹配错误显示 */}
          {metrics.weather_scenario.scenario_match_error && (
            <div className="mt-4 p-3 bg-yellow-900/20 border border-yellow-600/30 rounded-lg">
              <div className="text-sm text-yellow-300">
                ⚠️ 场景匹配暂不可用: {metrics.weather_scenario.scenario_match_error}
              </div>
            </div>
          )}
        </div>
      )}
      
      {/* 显示天气场景错误信息 */}
      {metrics.weather_scenario && metrics.weather_scenario.error && (
        <div className="mb-6 p-4 bg-red-900 bg-opacity-20 border border-red-600 rounded-lg">
          <h4 className="text-lg font-medium text-red-400 mb-2">天气场景分析</h4>
          <p className="text-red-300">分析失败: {metrics.weather_scenario.error}</p>
        </div>
      )}
      
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {validMetrics.map((config) => {
          const value = metrics[config.key];
          const displayValue = config.formatter ? config.formatter(value) : `${value} ${config.unit}`;
          
          return (
            <div 
              key={config.key}
              className="bg-neutral-700 rounded-lg p-3 border border-neutral-600 hover:border-neutral-500 transition-colors"
              title={config.description}
            >
              <div className="text-sm text-neutral-400 mb-1">{config.label}</div>
              <div className="text-lg font-semibold text-neutral-100">{displayValue}</div>
            </div>
          );
        })}
      </div>
      
      {/* 如果是区间预测，显示额外信息 */}
      {(metrics.hit_rate !== undefined || metrics.avg_interval_width !== undefined) && (
        <div className="mt-4 p-3 bg-neutral-700 rounded-lg border border-neutral-600">
          <h4 className="text-md font-medium text-neutral-200 mb-2">区间预测性能</h4>
          <div className="text-sm text-neutral-300">
            {metrics.hit_rate !== undefined && (
              <div>实际值落在预测区间内的比例：{metrics.hit_rate.toFixed(2)}%</div>
            )}
            {(metrics.avg_interval_width !== undefined || metrics.average_interval_width !== undefined) && (
              <div>
                平均预测区间宽度：{(metrics.avg_interval_width || metrics.average_interval_width).toFixed(2)} MW
              </div>
            )}
            {metrics.confidence_level !== undefined && (
              <div>置信水平：{(metrics.confidence_level * 100).toFixed(0)}%</div>
            )}
          </div>
        </div>
      )}
      
      {/* 调试信息（仅在开发环境显示） */}
      {process.env.NODE_ENV === 'development' && (
        <details className="mt-4">
          <summary className="text-sm text-neutral-400 cursor-pointer">调试信息</summary>
          <pre className="mt-2 p-2 bg-black text-green-300 text-xs rounded overflow-auto max-h-32">
            {JSON.stringify(metrics, null, 2)}
          </pre>
        </details>
      )}
    </div>
  );
}

export default MetricsCard;