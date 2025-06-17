import React, { useState, useEffect } from 'react';
import { renderForecastChart, renderIntervalForecastChart } from '../utils/chartRenderers';

const OptimizedHistoricalResultsTab = () => {
  // 状态管理
  const [loading, setLoading] = useState(false);
  const [resultsList, setResultsList] = useState([]);
  const [groupedResults, setGroupedResults] = useState({});
  const [selectedResults, setSelectedResults] = useState(new Set());
  const [resultData, setResultData] = useState({});
  const [error, setError] = useState(null);
  const [searchParams, setSearchParams] = useState({
    forecastType: 'load',
    province: '上海',
    startDate: '2024-01-01',
    endDate: '2024-01-31',
    predictionType: 'all',
    modelType: 'all'
  });

  // 初始加载
  useEffect(() => {
    handleSearch();
  }, []);

  // 搜索历史结果
  const handleSearch = async () => {
    setLoading(true);
    setError(null);
    setResultData({});
    setSelectedResults(new Set());
    
    try {
      const url = new URL('/api/historical-results', window.location.origin);
      Object.entries(searchParams).forEach(([key, value]) => {
        if (value) url.searchParams.append(key, value);
      });
      
      console.log('查询历史结果:', url.toString());
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`HTTP错误 ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.status === 'success') {
        setResultsList(data.results || []);
        setGroupedResults(data.grouped_results || {});
        
        // 如果只有一个结果，自动查看
        if (data.results && data.results.length === 1) {
          await handleViewResult(data.results[0].path, data.results[0]);
        }
      } else {
        setError(`查询失败: ${data.message || '未知错误'}`);
      }
    } catch (error) {
      console.error('获取历史结果失败:', error);
      setError(`获取历史结果失败: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // 查看单个结果
  const handleViewResult = async (resultPath, resultInfo) => {
    const resultKey = resultPath;
    
    // 如果已经加载过，直接使用缓存
    if (resultData[resultKey]) {
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const url = new URL('/api/historical-results/view', window.location.origin);
      url.searchParams.append('path', resultPath);
      
      // 添加日期过滤参数
      if (searchParams.startDate) {
        url.searchParams.append('startDate', searchParams.startDate);
      }
      if (searchParams.endDate) {
        url.searchParams.append('endDate', searchParams.endDate);
      }
      
      console.log('查看历史结果:', url.toString());
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`HTTP错误 ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.status === 'success') {
        setResultData(prev => ({
          ...prev,
          [resultKey]: { ...data, resultInfo }
        }));
        
        // 渲染图表
        setTimeout(() => {
          renderResultChart(data, resultKey, resultInfo);
        }, 100);
      } else {
        setError(`查询失败: ${data.message || '未知错误'}`);
      }
    } catch (error) {
      console.error('查看历史结果失败:', error);
      setError(`查看历史结果失败: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // 渲染图表
  const renderResultChart = (data, resultKey, resultInfo) => {
    const containerId = `chart-${resultKey.replace(/[^a-zA-Z0-9]/g, '-')}`;
    const forecastType = data.metadata?.forecast_type || resultInfo?.forecast_type || 'load';
    const predictionType = data.metadata?.prediction_type || resultInfo?.prediction_type || 'day-ahead';
    
    setTimeout(() => {
      if (predictionType === 'interval') {
        renderIntervalForecastChart(data, containerId, forecastType);
      } else {
        renderForecastChart(data, containerId, forecastType);
      }
    }, 100);
  };

  // 处理结果选择
  const handleResultSelection = (resultPath, isSelected) => {
    const newSelected = new Set(selectedResults);
    if (isSelected) {
      newSelected.add(resultPath);
    } else {
      newSelected.delete(resultPath);
    }
    setSelectedResults(newSelected);
  };

  // 查看选中的结果
  const handleViewSelectedResults = async () => {
    for (const resultPath of selectedResults) {
      const resultInfo = resultsList.find(r => r.path === resultPath);
      if (resultInfo && !resultData[resultPath]) {
        await handleViewResult(resultPath, resultInfo);
      }
    }
  };

  // 参数变更处理
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setSearchParams({ ...searchParams, [name]: value });
  };

  // 渲染分组结果
  const renderGroupedResults = () => {
    if (Object.keys(groupedResults).length === 0) {
      return (
        <div className="text-center py-8">
          <p className="text-neutral-400">没有找到符合条件的历史结果</p>
          <p className="text-neutral-500 text-sm mt-2">
            请尝试调整查询条件：扩大日期范围、选择其他预测类型或模型类型
          </p>
        </div>
      );
    }

    return (
      <div className="space-y-6">
        {Object.entries(groupedResults).map(([modelType, predictions]) => (
          <div key={modelType} className="bg-neutral-700 rounded-lg p-4">
            <h4 className="text-lg font-medium text-neutral-100 mb-3">
              {modelType === 'convtrans_peak' ? '峰值感知模型' : modelType}
            </h4>
            
            <div className="space-y-2">
              {Object.entries(predictions).map(([predictionType, results]) => (
                <div key={predictionType} className="bg-neutral-800 rounded p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-neutral-200 font-medium">
                      {getPredictionTypeLabel(predictionType)}
                    </span>
                    <span className="text-neutral-400 text-sm">
                      {results.length} 个结果
                    </span>
                  </div>
                  
                  <div className="space-y-1">
                    {results.map((result, index) => (
                      <div key={result.path} className="flex items-center space-x-3 py-1">
                        <input
                          type="checkbox"
                          id={`result-${result.path}`}
                          checked={selectedResults.has(result.path)}
                          onChange={(e) => handleResultSelection(result.path, e.target.checked)}
                          className="rounded border-neutral-600 text-red-600 focus:ring-red-500"
                        />
                        <label
                          htmlFor={`result-${result.path}`}
                          className="flex-1 text-neutral-300 text-sm cursor-pointer"
                        >
                          {result.forecast_type} | {result.province} | 
                          {result.actual_start_date && result.actual_end_date
                            ? ` ${result.actual_start_date} 至 ${result.actual_end_date}`
                            : ` ${result.year}年`}
                        </label>
                        <button
                          onClick={() => handleViewResult(result.path, result)}
                          className="text-red-400 hover:text-red-300 text-sm"
                        >
                          单独查看
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
        
        {selectedResults.size > 0 && (
          <div className="bg-neutral-800 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <span className="text-neutral-200">
                已选择 {selectedResults.size} 个结果
              </span>
              <button
                onClick={handleViewSelectedResults}
                className="btn-primary"
              >
                查看选中结果
              </button>
            </div>
          </div>
        )}
      </div>
    );
  };

  // 获取预测类型标签
  const getPredictionTypeLabel = (predictionType) => {
    const labels = {
      'day-ahead': '日前预测',
      'rolling': '滚动预测',
      'interval': '区间预测',
      'probabilistic': '概率预测'
    };
    return labels[predictionType] || predictionType;
  };

  // 渲染结果图表
  const renderResultCharts = () => {
    const selectedData = Object.entries(resultData).filter(([key]) => 
      selectedResults.size === 0 || selectedResults.has(key) || selectedResults.size === 0
    );

    if (selectedData.length === 0) {
      return null;
    }

    return (
      <div className="space-y-6">
        {selectedData.map(([resultKey, data]) => {
          const containerId = `chart-${resultKey.replace(/[^a-zA-Z0-9]/g, '-')}`;
          const resultInfo = data.resultInfo || {};
          
          return (
            <div key={resultKey} className="bg-neutral-800 rounded-lg p-4">
              <div className="mb-4">
                <h4 className="text-lg font-medium text-neutral-100">
                  {data.metadata?.forecast_type || '负荷'} - {data.metadata?.province || '未知'} - {getPredictionTypeLabel(data.metadata?.prediction_type)}
                </h4>
                <p className="text-neutral-400 text-sm">
                  {data.start_time} 至 {data.end_time} | 
                  数据点: {data.predictions?.length || 0}
                  {data.metadata?.filtered_rows && (
                    <span> (过滤后: {data.metadata.filtered_rows})</span>
                  )}
                </p>
              </div>
              
              <div
                id={containerId}
                className="w-full h-96 bg-neutral-900 rounded border border-neutral-700"
              ></div>
              
              {/* 显示指标 */}
              {data.metrics && Object.keys(data.metrics).length > 0 && (
                <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
                  {data.metrics.mae && (
                    <div className="text-center">
                      <div className="text-2xl font-bold text-neutral-100">
                        {data.metrics.mae.toFixed(2)}
                      </div>
                      <div className="text-neutral-400 text-sm">MAE</div>
                    </div>
                  )}
                  {data.metrics.mape && (
                    <div className="text-center">
                      <div className="text-2xl font-bold text-neutral-100">
                        {data.metrics.mape.toFixed(2)}%
                      </div>
                      <div className="text-neutral-400 text-sm">MAPE</div>
                    </div>
                  )}
                  {data.metrics.rmse && (
                    <div className="text-center">
                      <div className="text-2xl font-bold text-neutral-100">
                        {data.metrics.rmse.toFixed(2)}
                      </div>
                      <div className="text-neutral-400 text-sm">RMSE</div>
                    </div>
                  )}
                  {data.interval_statistics?.hit_rate && (
                    <div className="text-center">
                      <div className="text-2xl font-bold text-neutral-100">
                        {data.interval_statistics.hit_rate.toFixed(1)}%
                      </div>
                      <div className="text-neutral-400 text-sm">命中率</div>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="p-4">
      <h2 className="text-xl font-semibold mb-4 text-neutral-100">历史结果查询</h2>
      
      {/* 搜索表单 */}
      <div className="bg-neutral-800 rounded-lg p-4 mb-6 border border-neutral-700">
        <h3 className="text-lg font-semibold mb-3 text-neutral-100">查询条件</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1 text-neutral-300">预测类型</label>
            <select 
              name="forecastType" 
              className="w-full p-2 border rounded bg-neutral-700 border-neutral-600 text-neutral-100"
              value={searchParams.forecastType}
              onChange={handleInputChange}
            >
              <option value="load">电力负荷</option>
              <option value="pv">光伏发电</option>
              <option value="wind">风电出力</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-1 text-neutral-300">省份</label>
            <select 
              name="province" 
              className="w-full p-2 border rounded bg-neutral-700 border-neutral-600 text-neutral-100"
              value={searchParams.province}
              onChange={handleInputChange}
            >
              <option value="上海">上海</option>
              <option value="江苏">江苏</option>
              <option value="浙江">浙江</option>
              <option value="安徽">安徽</option>
              <option value="福建">福建</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-1 text-neutral-300">预测方式</label>
            <select 
              name="predictionType" 
              className="w-full p-2 border rounded bg-neutral-700 border-neutral-600 text-neutral-100"
              value={searchParams.predictionType}
              onChange={handleInputChange}
            >
              <option value="all">全部</option>
              <option value="day-ahead">日前预测</option>
              <option value="rolling">滚动预测</option>
              <option value="interval">区间预测</option>
              <option value="probabilistic">概率预测</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-1 text-neutral-300">模型类型</label>
            <select 
              name="modelType" 
              className="w-full p-2 border rounded bg-neutral-700 border-neutral-600 text-neutral-100"
              value={searchParams.modelType}
              onChange={handleInputChange}
            >
              <option value="all">全部</option>
              <option value="convtrans_peak">峰值感知模型</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-1 text-neutral-300">开始日期</label>
            <input 
              type="date" 
              name="startDate" 
              className="w-full p-2 border rounded bg-neutral-700 border-neutral-600 text-neutral-100"
              value={searchParams.startDate}
              onChange={handleInputChange}
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-1 text-neutral-300">结束日期</label>
            <input 
              type="date" 
              name="endDate" 
              className="w-full p-2 border rounded bg-neutral-700 border-neutral-600 text-neutral-100"
              value={searchParams.endDate}
              onChange={handleInputChange}
            />
          </div>
        </div>
        
        <div className="mt-4 flex justify-end">
          <button
            className="btn-primary"
            onClick={handleSearch}
            disabled={loading}
          >
            {loading ? '查询中...' : '查询'}
          </button>
        </div>
      </div>
      
      {/* 错误信息 */}
      {error && (
        <div className="bg-red-900 border border-red-700 text-red-100 p-4 rounded-lg mb-4">
          <h4 className="font-medium">查询出错</h4>
          <p className="text-sm mt-1">{error}</p>
        </div>
      )}
      
      {/* 结果列表 */}
      {!loading && !error && (
        <div className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-neutral-100">查询结果</h3>
            {resultsList.length > 0 && (
              <span className="text-neutral-400 text-sm">
                找到 {resultsList.length} 个匹配结果
              </span>
            )}
          </div>
          
          {renderGroupedResults()}
        </div>
      )}
      
      {/* 结果图表展示 */}
      {Object.keys(resultData).length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-4 text-neutral-100">预测结果图表</h3>
          {renderResultCharts()}
        </div>
      )}
    </div>
  );
};

export default OptimizedHistoricalResultsTab;