import React, { useState, useEffect, useCallback, useRef } from 'react';
import Header from './components/Header';
import ParameterSettings from './components/ParameterSettings';
import RollingSettings from './components/RollingSettings';
import IntervalSettings from './components/IntervalSettings';
import ProbabilisticSettings from './components/ProbabilisticSettings';
import ScenariosSettings from './components/ScenariosSettings';
import TrainingSettings from './components/TrainingSettings';
import HistoricalResultsSettings from './components/HistoricalResultsSettings';
import ScenarioAwareUncertaintyForecast from './components/ScenarioAwareUncertaintyForecast';
import MetricsCard from './components/MetricsCard';
// Import the new DayAheadTab component
import DayAheadTab from './features/DayAhead/DayAheadTab';
// Import the new IntervalForecastTab component
import IntervalForecastTab from './features/IntervalForecast/IntervalForecastTab';
// Import the new TrainingTab component
import TrainingTab from './features/Training/TrainingTab';
// 导入 API 方法
import { 
  getProvinces, 
  runPrediction, 
  runIntervalForecast, 
  trainModel, 
  recognizeScenarios,
  clearCache,
  getTrainingStatus,
  getHistoricalResults 
} from './utils/api';
// 导入图表渲染函数
import { renderForecastChart, renderIntervalForecastChart } from './utils/chartRenderers';

const TABS = {
  dayAhead: '日前预测',
  rolling: '滚动预测',
  interval: '区间预测',
  probabilistic: '概率预测',
  scenarios: '场景识别',
  scenarioUncertainty: '场景感知不确定性预测',
  training: '模型训练',
  historical: '历史结果查询',
};

// 默认日期范围
const DEFAULT_TRAIN_START_DATE = '2024-01-01';
const DEFAULT_TRAIN_END_DATE = '2024-02-28';
const DEFAULT_FORECAST_START_DATE = '2024-03-01';
const DEFAULT_FORECAST_END_DATE = '2024-03-02';

// 默认省份列表
const DEFAULT_PROVINCES = ['上海', '安徽', '浙江', '江苏', '福建'];

const TabButton = ({ label, isActive, onClick }) => {
  return (
    <button
      onClick={onClick}
      className={`relative px-4 py-3 font-medium text-sm focus:outline-none group whitespace-nowrap 
                  hover:text-red-600 transition-colors duration-150
                  ${isActive ? 'text-red-500 bg-neutral-800 shadow-[0_0_8px_rgba(220,38,38,0.4)] rounded-t-md' : 'text-neutral-700'}`}
    >
      {label}
      <span
        className={`absolute bottom-0 left-0 right-0 h-0.5 bg-red-600 transform transition-transform duration-200 ease-out
                    ${isActive ? 'scale-x-100' : 'scale-x-0 group-hover:scale-x-100'}`}
        style={{ transformOrigin: 'center' }} // Center origin for expand/contract from center if preferred, or left
      ></span>
    </button>
  );
};

function App() {
  const [activeTab, setActiveTab] = useState('dayAhead'); // Default to dayAhead
  const [loading, setLoading] = useState({}); // Changed to object for per-tab loading
  const [initialDataLoaded, setInitialDataLoaded] = useState({}); // 记录每个标签页是否已经加载过初始数据
  const chartContainerRef = useRef(null);
  
  // 设置默认参数
  const [params, setParams] = useState({
    rolling: {
      forecastType: 'load',
      startDate: DEFAULT_FORECAST_START_DATE,
      endDate: DEFAULT_FORECAST_END_DATE,
      province: DEFAULT_PROVINCES[0],
      predictionType: 'rolling',
      interval: 15, // 默认滚动间隔为15分钟
      historicalDays: 15,
      modelType: 'torch'
    },
    probabilistic: {
      forecastType: 'load',
      forecastDate: DEFAULT_FORECAST_START_DATE,
      forecastEndDate: DEFAULT_FORECAST_END_DATE,
      province: DEFAULT_PROVINCES[0],
      predictionType: 'day-ahead',
      probabilistic: true,
      quantiles: [0.1, 0.2, 0.5, 0.8, 0.9],
      historicalDays: 15,
      modelType: 'torch'
    },
    scenarios: {
      date: DEFAULT_FORECAST_START_DATE,
      province: DEFAULT_PROVINCES[0]
    },
    scenarioUncertainty: {
      province: DEFAULT_PROVINCES[0],
      forecastType: 'load',
      forecastDate: DEFAULT_FORECAST_START_DATE,
      forecastEndDate: '',
      confidenceLevel: 0.9,
      historicalDays: 14,
      includeExplanations: true
    },
    historical: {
      forecast_type: 'load',
      start_date: '2024-01-01',
      end_date: '2024-12-31',
      province: DEFAULT_PROVINCES[0],
      prediction_type: 'all',
      model_type: 'all',
      year: '',
    }
  }); 
  
  const [results, setResults] = useState({}); // 存储 API 返回的结果
  const [error, setError] = useState({}); // 存储 API 调用时的错误
  
  const [availableProvinces, setAvailableProvinces] = useState(DEFAULT_PROVINCES);
  const [predictionMetrics, setPredictionMetrics] = useState(null); // State for overall metrics

  // 获取可用省份列表，但默认值优先
  useEffect(() => {
    const fetchProvinces = async () => {
      try {
        setLoading(prev => ({ ...prev, provinces: true }));
        // 调用 API 获取省份列表
        const provinces = await getProvinces();
        // 合并API返回的省份和默认省份，确保默认省份在列表中且排在前面
        const mergedProvinces = [...DEFAULT_PROVINCES];
        provinces.forEach(province => {
          if (!mergedProvinces.includes(province)) {
            mergedProvinces.push(province);
          }
        });
        setAvailableProvinces(mergedProvinces);
      } catch (err) {
        console.error('获取省份列表失败:', err);
        setError(prev => ({ ...prev, provinces: err.message }));
        // 使用默认省份列表
        setAvailableProvinces(DEFAULT_PROVINCES);
      } finally {
        setLoading(prev => ({ ...prev, provinces: false }));
      }
    };

    fetchProvinces();
  }, []);

  const handleClearCacheApi = async () => {
    try {
      console.log("Clear Cache API Call Initiated.");
      setLoading(prev => ({ ...prev, cache: true }));
      const response = await clearCache();
      alert(`缓存已清除: ${response.message}`);
    } catch (err) {
      console.error('清除缓存失败:', err);
      alert(`清除缓存失败: ${err.message}`);
    } finally {
      setLoading(prev => ({ ...prev, cache: false }));
    }
  };

  // Generic API handler
  const handleApiSubmit = useCallback(async (tabKey, submittedParams) => {
    console.log(`${TABS[tabKey]} Parameters Submitted:`, submittedParams);
    
    // 更新参数状态
    setParams(prev => ({ ...prev, [tabKey]: submittedParams }));
    setLoading(prev => ({ ...prev, [tabKey]: true }));
    setError(prev => ({ ...prev, [tabKey]: null }));
    
    let apiResponse = null;
    
    try {
      // Format parameters as required by API
      let formattedParams = { ...submittedParams };
  
      // For some tabs, convert 'forecastType' to 'forecast_type' for API
      if (tabKey === 'interval') {
        formattedParams.forecast_type = submittedParams.forecastType;
      }
      
      switch(tabKey) {
        case 'rolling':
        case 'probabilistic':
          // 检查是否启用了天气感知预测
          if (submittedParams.weatherAware && submittedParams.forecastType === 'load') {
            console.log('使用天气感知预测API');
            // 调用天气感知预测API
            const response = await fetch('/api/weather-forecast', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify(formattedParams),
            });
            
            if (!response.ok) {
              throw new Error(`天气感知预测API响应错误: ${response.status}`);
            }
            
            const result = await response.json();
            if (!result.success) {
              throw new Error(result.error || '天气感知预测失败');
            }
            
            apiResponse = result.data;
          } else {
            // 使用普通预测API
            apiResponse = await runPrediction(formattedParams);
          }
          break;
        
        case 'scenarios':
          apiResponse = await recognizeScenarios(formattedParams);
          break;
        
        case 'historical':
          apiResponse = await getHistoricalResults(formattedParams);
          break;
          
        default:
          throw new Error(`未知的标签类型: ${tabKey}`);
      }
      
      console.log(`${TABS[tabKey]} API Call Completed:`, apiResponse);
      setResults(prev => ({ ...prev, [tabKey]: apiResponse }));
      
      // 修复图表渲染调用
      setTimeout(() => {
        if (tabKey === 'interval') {
          console.log('准备渲染区间预测图表，数据：', apiResponse);
          
          // 区间预测 API 直接返回 predictions，不包装在 data 中
          if (apiResponse && apiResponse.predictions) {
            const predictions = apiResponse.predictions;
            console.log('预测数据示例：', predictions[0]);
            
            // 检查是否需要字段映射
            const needsMapping = predictions.some(p => 
              p.point_forecast === undefined && p.predicted !== undefined
            );
            
            if (needsMapping) {
              console.log('执行字段映射：predicted -> point_forecast');
              predictions.forEach(p => {
                if (p.point_forecast === undefined && p.predicted !== undefined) {
                  p.point_forecast = p.predicted;
                }
              });
            }
            
            // 传递整个 apiResponse 给渲染函数
            renderIntervalForecastChart(apiResponse, `chart-${tabKey}`, formattedParams.forecast_type);
          } else {
            console.error('区间预测数据结构不正确：', apiResponse);
          }
        } else if (tabKey !== 'scenarios' && tabKey !== 'historical') {
          console.log('准备渲染普通预测图表，数据：', apiResponse);
          
          // 普通预测需要检查是否有 data 包装
          let chartData = apiResponse.data || apiResponse;
          
          if (chartData && chartData.predictions) {
            const predictions = chartData.predictions;
            const needsMapping = predictions.some(p => 
              p.predicted === undefined && p.point_forecast !== undefined
            );
            
            if (needsMapping) {
              console.log('执行字段映射：point_forecast -> predicted');
              predictions.forEach(p => {
                if (p.predicted === undefined && p.point_forecast !== undefined) {
                  p.predicted = p.point_forecast;
                }
              });
            }
          }
          
          renderForecastChart(chartData, `chart-${tabKey}`, formattedParams.forecast_type || formattedParams.forecastType);
        }
      }, 100);
  
      // 修复指标提取逻辑
      if (apiResponse) {
        let metrics = null;
        
        if (tabKey === 'interval') {
          // 区间预测直接从根级别获取指标
          metrics = apiResponse.metrics || apiResponse.interval_statistics || null;
        } else {
          // 其他预测类型优先从 data.metrics 获取指标
          if (apiResponse.data && apiResponse.data.metrics) {
            metrics = apiResponse.data.metrics;
          } else if (apiResponse.metrics) {
            metrics = apiResponse.metrics;
          }
          
          // 如果还是没有，尝试从其他可能的位置获取
          if (!metrics || Object.keys(metrics).length === 0) {
            if (apiResponse.data && apiResponse.data.interval_statistics) {
              metrics = {
                ...apiResponse.data.interval_statistics,
                ...metrics
              };
            }
          }
        }
        
        console.log('提取到的指标：', metrics);
        setPredictionMetrics(metrics || null);
      }
      
    } catch (err) {
      console.error(`${TABS[tabKey]} API Call Failed:`, err);
      setError(prev => ({ ...prev, [tabKey]: err.message }));
      alert(`${TABS[tabKey]}失败: ${err.message}`);
    } finally {
      setLoading(prev => ({ ...prev, [tabKey]: false }));
    }
  }, []);

  // 查看历史结果
  const handleViewHistoricalResult = async (filePath) => {
    try {
      setLoading(prev => ({ ...prev, historicalFile: true }));
      const response = await fetch(`/api/historical-results/view?path=${encodeURIComponent(filePath)}`);
      if (!response.ok) {
        throw new Error(`API响应错误: ${response.status}`);
      }
      const resultData = await response.json();
      
      // 更新结果状态，添加选中的结果
      setResults(prev => ({
        ...prev,
        historical: {
          ...prev.historical,
          selectedResult: resultData
        }
      }));
      
      // 渲染图表
      setTimeout(() => {
        // 根据结果类型选择合适的图表渲染函数
        const isProbabilistic = filePath.includes('probabilistic') || resultData.quantiles;
        const isInterval = filePath.includes('interval') || (resultData.lower_bound && resultData.upper_bound);
        
        if (isInterval) {
          renderIntervalForecastChart(resultData, 'historical-result-chart', params.historical.forecast_type);
        } else {
          renderForecastChart(resultData, 'historical-result-chart', params.historical.forecast_type);
        }
      }, 100);
      
    } catch (err) {
      console.error('获取历史结果文件失败:', err);
      setError(prev => ({ ...prev, historicalFile: err.message }));
    } finally {
      setLoading(prev => ({ ...prev, historicalFile: false }));
    }
  };

  const renderTabContent = () => {
    const commonProps = {
      availableProvinces, // 使用从 API 获取的省份列表
      onSubmit: (p) => handleApiSubmit(activeTab, p),
      isLoading: loading[activeTab] || false,
    };

    const cardClassName = "content-card";

    // Helper to render chart and metrics card for relevant tabs
    const renderChartAndMetrics = (tabKey, apiResponse, chartParams) => {
      if (!apiResponse) return null;
      const chartType = tabKey === 'interval' ? 'interval' : 'forecast';
      const forecastTypeKey = chartType === 'interval' ? 'forecast_type' : 'forecastType';

      return (
        <>
          <div
            id={`chart-${tabKey}`}
            ref={chartContainerRef}
            className="w-full h-96 bg-neutral-800 rounded-lg border border-neutral-700 mt-6 shadow-xl"
          ></div>
          <MetricsCard metrics={predictionMetrics} />
        </>
      );
    };

    // Determine current results and parameters for the active tab
    const currentResults = results[activeTab];
    const currentApiParams = params[activeTab] || {};

    return (
      <>
        {/* Parameter Settings Components */}
        {activeTab === 'dayAhead' && <DayAheadTab availableProvinces={availableProvinces} />}
        {activeTab === 'rolling' && <RollingSettings {...commonProps} initialParams={params.rolling} title="参数设置 - 滚动预测" className={cardClassName} activeTabKey={activeTab}/>}
        {activeTab === 'interval' && <IntervalForecastTab availableProvinces={availableProvinces} />}
        {activeTab === 'probabilistic' && <ProbabilisticSettings {...commonProps} initialParams={params.probabilistic} title="参数设置 - 概率预测" className={cardClassName} activeTabKey={activeTab}/>}
        {activeTab === 'scenarios' && <ScenariosSettings {...commonProps} initialParams={params.scenarios} title="参数设置 - 场景识别" className={cardClassName} activeTabKey={activeTab}/>}
        {activeTab === 'scenarioUncertainty' && <ScenarioAwareUncertaintyForecast />}
        {activeTab === 'training' && <TrainingTab availableProvinces={availableProvinces} />}
        {activeTab === 'historical' && <HistoricalResultsSettings {...commonProps} initialParams={params.historical} title="历史结果查询" className={cardClassName} activeTabKey={activeTab}/>}

        {/* Loading Spinner */}
        {loading[activeTab] && (
          <div className="flex justify-center items-center mt-6">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-red-600"></div>
          </div>
        )}

        {/* Error Display */}
        {error[activeTab] && (
          <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-md shadow">
            <h3 className="text-red-700">错误 ({TABS[activeTab]}):</h3>
            <p className="text-red-600">{error[activeTab]}</p>
          </div>
        )}

        {/* Results Display Area (Chart and Metrics) */}
        {!loading[activeTab] && currentResults && 
          (activeTab === 'rolling' || activeTab === 'probabilistic') && (
          <div className="mt-6 p-4 bg-neutral-800 rounded-lg border border-neutral-700 shadow-xl">
            {renderChartAndMetrics(activeTab, currentResults, currentApiParams)}
          </div>
        )}
        
        {/* Historical Results Display */}
        {!loading[activeTab] && currentResults && activeTab === 'historical' && (
          <div className="mt-6 p-4 bg-neutral-800 rounded-lg border border-neutral-700 shadow-xl">
            <h3 className="text-xl font-medium text-neutral-100 mb-4">历史预测结果</h3>
            
            {currentResults.results && currentResults.results.length > 0 ? (
              <>
                {/* 文件列表 */}
                <div className="overflow-x-auto mb-6">
                  <table className="min-w-full divide-y divide-neutral-700">
                    <thead className="bg-neutral-800">
                      <tr>
                        <th className="px-4 py-2 text-left text-xs font-medium text-neutral-400 uppercase tracking-wider">预测日期</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-neutral-400 uppercase tracking-wider">年份</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-neutral-400 uppercase tracking-wider">预测方式</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-neutral-400 uppercase tracking-wider">模型类型</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-neutral-400 uppercase tracking-wider">文件名</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-neutral-400 uppercase tracking-wider">操作</th>
                      </tr>
                    </thead>
                    <tbody className="bg-neutral-800 divide-y divide-neutral-700">
                      {currentResults.results.map((file, index) => (
                        <tr key={index} className="hover:bg-neutral-700">
                          <td className="px-4 py-2 whitespace-nowrap text-sm text-neutral-300">{file.date}</td>
                          <td className="px-4 py-2 whitespace-nowrap text-sm text-neutral-300">{file.year || 'N/A'}</td>
                          <td className="px-4 py-2 whitespace-nowrap text-sm text-neutral-300">{file.prediction_type}</td>
                          <td className="px-4 py-2 whitespace-nowrap text-sm text-neutral-300">{file.model_type}</td>
                          <td className="px-4 py-2 whitespace-nowrap text-sm text-neutral-300">{file.filename}</td>
                          <td className="px-4 py-2 whitespace-nowrap text-sm">
                            <button 
                              onClick={() => handleViewHistoricalResult(file.path)}
                              className="text-primary hover:text-primary-light font-medium"
                            >
                              查看
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                
                {/* 如果有选中的结果，显示预测图表 */}
                {currentResults.selectedResult && (
                  <div className="mt-4">
                    <h4 className="text-lg font-medium text-neutral-100 mb-2">预测结果图表</h4>
                    <div
                      id="historical-result-chart"
                      className="w-full h-96 bg-neutral-800 rounded-lg border border-neutral-700 shadow-xl"
                    ></div>
                  </div>
                )}
              </>
            ) : (
              <div className="text-center py-8">
                <p className="text-neutral-400">没有找到符合条件的历史结果</p>
              </div>
            )}
          </div>
        )}

        {/* Scenario Results Table */}
        {!loading[activeTab] && currentResults && activeTab === 'scenarios' && (
          <div className="mt-6 p-4 bg-neutral-800 rounded-lg border border-neutral-700 shadow-xl">
            <h3 className="text-xl font-medium text-neutral-100 mb-4">场景识别结果</h3>
            <div className="bg-neutral-800 rounded-md p-4">
              <h4 className="text-lg font-medium text-neutral-100 mb-2">识别的场景:</h4>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-neutral-700">
                  <thead className="bg-neutral-700">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-neutral-300 uppercase tracking-wider">时间</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-neutral-300 uppercase tracking-wider">场景类型</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-neutral-300 uppercase tracking-wider">净负荷</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-neutral-300 uppercase tracking-wider">可再生占比</th>
                    </tr>
                  </thead>
                  <tbody className="bg-neutral-800 divide-y divide-neutral-700">
                    {currentResults.data?.scenarios?.map((scenario, index) => (
                      <tr key={index}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-neutral-300">{scenario.datetime}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-neutral-300">{scenario.scenario_type}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-neutral-300">{scenario.net_load?.toFixed(2)} MW</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-neutral-300">{(scenario.renewable_ratio * 100)?.toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </>
    );
  };

  return (
    <div className="flex flex-col min-h-screen bg-black">
      <Header onClearCache={handleClearCacheApi} />
      <div className="container mx-auto px-4 py-6 flex-grow">
        {/* 导航栏 */}
        <div className="flex flex-wrap justify-center border-b border-neutral-700 mb-6">
          {Object.entries(TABS).map(([key, label]) => (
            <TabButton
              key={key}
              label={label}
              isActive={activeTab === key}
              onClick={() => setActiveTab(key)}
            />
          ))}
        </div>
        {renderTabContent()}
      </div>
    </div>
  );
}

export default App;