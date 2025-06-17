import React, { useState } from 'react';

const ScenarioAwareUncertaintyForecast = () => {
    const [params, setParams] = useState({
        province: '上海',
        forecastType: 'load',
        forecastDate: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString().split('T')[0], // 明天
        forecastEndDate: '',
        confidenceLevel: 0.9,
        historicalDays: 14,
        includeExplanations: true
    });

    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);

    const availableProvinces = ['上海', '安徽', '浙江', '江苏', '福建'];
    const forecastTypes = [
        { value: 'load', label: '负荷' },
        { value: 'pv', label: '光伏' },
        { value: 'wind', label: '风电' },
        { value: 'net_load', label: '净负荷' }
    ];

    const handleInputChange = (e) => {
        const { name, value, type, checked } = e.target;
        setParams({
            ...params,
            [name]: type === 'checkbox' ? checked : value
        });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setResults(null);

        try {
            const response = await fetch('/api/scenario-aware-uncertainty-forecast', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(params),
            });

            if (!response.ok) {
                throw new Error(`HTTP错误: ${response.status}`);
            }

            const data = await response.json();
            
            if (!data.success) {
                throw new Error(data.error || '预测失败');
            }

            setResults(data.data);
        } catch (err) {
            console.error('场景感知不确定性预测失败:', err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const formatDate = (dateString) => {
        if (!dateString) return '';
        const date = new Date(dateString);
        return date.toLocaleString('zh-CN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    const getRiskLevelClass = (riskLevel) => {
        switch (riskLevel) {
            case 'high':
                return 'bg-red-700 text-white';
            case 'medium':
                return 'bg-yellow-600 text-white';
            case 'low':
                return 'bg-green-700 text-white';
            default:
                return 'bg-gray-600 text-white';
        }
    };

    return (
        <div className="container mx-auto px-4 py-6">
            <div className="bg-neutral-800 rounded-lg border border-neutral-700 shadow-xl p-6 mb-6">
                <h2 className="text-xl font-medium text-neutral-100 mb-4">场景感知不确定性预测参数设置</h2>
                <form onSubmit={handleSubmit}>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                            <label className="block text-sm font-medium text-neutral-300 mb-1">省份</label>
                            <select
                                name="province"
                                value={params.province}
                                onChange={handleInputChange}
                                className="w-full bg-neutral-700 border border-neutral-600 rounded-md py-2 px-3 text-neutral-100 focus:outline-none focus:ring-2 focus:ring-red-500"
                            >
                                {availableProvinces.map(province => (
                                    <option key={province} value={province}>{province}</option>
                                ))}
                            </select>
                        </div>
                        
                        <div>
                            <label className="block text-sm font-medium text-neutral-300 mb-1">预测类型</label>
                            <select
                                name="forecastType"
                                value={params.forecastType}
                                onChange={handleInputChange}
                                className="w-full bg-neutral-700 border border-neutral-600 rounded-md py-2 px-3 text-neutral-100 focus:outline-none focus:ring-2 focus:ring-red-500"
                            >
                                {forecastTypes.map(type => (
                                    <option key={type.value} value={type.value}>{type.label}</option>
                                ))}
                            </select>
                        </div>
                        
                        <div>
                            <label className="block text-sm font-medium text-neutral-300 mb-1">预测日期</label>
                            <input
                                type="date"
                                name="forecastDate"
                                value={params.forecastDate}
                                onChange={handleInputChange}
                                className="w-full bg-neutral-700 border border-neutral-600 rounded-md py-2 px-3 text-neutral-100 focus:outline-none focus:ring-2 focus:ring-red-500"
                            />
                        </div>
                        
                        <div>
                            <label className="block text-sm font-medium text-neutral-300 mb-1">结束日期 (可选)</label>
                            <input
                                type="date"
                                name="forecastEndDate"
                                value={params.forecastEndDate}
                                onChange={handleInputChange}
                                className="w-full bg-neutral-700 border border-neutral-600 rounded-md py-2 px-3 text-neutral-100 focus:outline-none focus:ring-2 focus:ring-red-500"
                            />
                        </div>
                        
                        <div>
                            <label className="block text-sm font-medium text-neutral-300 mb-1">置信水平</label>
                            <select
                                name="confidenceLevel"
                                value={params.confidenceLevel}
                                onChange={handleInputChange}
                                className="w-full bg-neutral-700 border border-neutral-600 rounded-md py-2 px-3 text-neutral-100 focus:outline-none focus:ring-2 focus:ring-red-500"
                            >
                                <option value="0.8">80%</option>
                                <option value="0.9">90%</option>
                                <option value="0.95">95%</option>
                                <option value="0.99">99%</option>
                            </select>
                        </div>
                        
                        <div>
                            <label className="block text-sm font-medium text-neutral-300 mb-1">历史数据天数</label>
                            <input
                                type="number"
                                name="historicalDays"
                                value={params.historicalDays}
                                onChange={handleInputChange}
                                min="1"
                                max="30"
                                className="w-full bg-neutral-700 border border-neutral-600 rounded-md py-2 px-3 text-neutral-100 focus:outline-none focus:ring-2 focus:ring-red-500"
                            />
                        </div>
                        
                        <div className="flex items-center">
                            <input
                                type="checkbox"
                                name="includeExplanations"
                                checked={params.includeExplanations}
                                onChange={handleInputChange}
                                id="includeExplanations"
                                className="h-4 w-4 text-red-600 focus:ring-red-500 border-neutral-500 rounded"
                            />
                            <label htmlFor="includeExplanations" className="ml-2 block text-sm text-neutral-300">
                                包含详细解释
                            </label>
                        </div>
                    </div>
                    
                    <div className="mt-6">
                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full bg-red-600 hover:bg-red-700 text-white font-medium py-2 px-4 rounded-md focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 focus:ring-offset-neutral-800 disabled:opacity-50"
                        >
                            {loading ? '预测中...' : '执行预测'}
                        </button>
                    </div>
                </form>
            </div>
            
            {error && (
                <div className="bg-red-900 bg-opacity-30 border border-red-800 rounded-lg p-4 mb-6">
                    <h3 className="text-lg font-medium text-red-300">预测失败</h3>
                    <p className="text-red-200">{error}</p>
                </div>
            )}
            
            {results && (
                <div className="space-y-6">
                    {/* 天气场景分析 */}
                    <div className="bg-neutral-800 rounded-lg border border-neutral-700 shadow-xl p-6">
                        <h2 className="text-2xl font-bold text-white mb-4">天气场景分析</h2>
                        
                        {results.scenarios && results.scenarios.length > 0 && (
                            <div className="bg-neutral-900 rounded-lg p-4 mb-4">
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div>
                                        <h3 className="text-xl font-medium text-blue-400 mb-2">主导天气场景</h3>
                                        <div className="bg-neutral-800 rounded-lg p-4">
                                            <p className="text-lg font-medium text-white">{results.scenarios[0].scenario}</p>
                                            <p className="text-neutral-300 mt-2">{results.scenarios[0].description}</p>
                                        </div>
                                    </div>
                                    
                                    <div>
                                        <h3 className="text-xl font-medium text-blue-400 mb-2">不确定性倍数</h3>
                                        <div className="bg-neutral-800 rounded-lg p-4">
                                            <p className="text-2xl font-bold text-white">{results.scenarios[0].uncertainty_multiplier}X</p>
                                        </div>
                                    </div>
                                </div>
                                
                                <div className="mt-4">
                                    <h3 className="text-xl font-medium text-blue-400 mb-2">场景描述</h3>
                                    <p className="text-neutral-300">{results.scenarios[0].description}</p>
                                </div>
                                
                                <div className="mt-4">
                                    <h3 className="text-xl font-medium text-blue-400 mb-2">系统风险等级</h3>
                                    <div className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${getRiskLevelClass(results.scenarios[0].risk_level)}`}>
                                        {results.scenarios[0].risk_level === 'high' ? '高风险' : 
                                         results.scenarios[0].risk_level === 'medium' ? '中风险' : '低风险'}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                    
                    {/* 不确定性分析 */}
                    <div className="bg-neutral-800 rounded-lg border border-neutral-700 shadow-xl p-6">
                        <h2 className="text-2xl font-bold text-white mb-4">不确定性分析</h2>
                        
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div className="bg-neutral-700 rounded-lg p-4">
                                <h3 className="text-lg font-medium text-neutral-300 mb-2">建模方法</h3>
                                <p className="text-xl font-medium text-white">天气场景感知</p>
                            </div>
                            
                            <div className="bg-neutral-700 rounded-lg p-4">
                                <h3 className="text-lg font-medium text-neutral-300 mb-2">基础不确定性</h3>
                                <p className="text-xl font-medium text-white">
                                    {results.predictions && results.predictions.length > 0 ? 
                                        `${(results.predictions[0].uncertainty / results.scenarios[0].uncertainty_multiplier).toFixed(1)}%` : '5.0%'}
                                </p>
                            </div>
                            
                            <div className="bg-neutral-700 rounded-lg p-4">
                                <h3 className="text-lg font-medium text-neutral-300 mb-2">场景调整倍数</h3>
                                <p className="text-xl font-medium text-white">
                                    {results.scenarios && results.scenarios.length > 0 ? 
                                        `${results.scenarios[0].uncertainty_multiplier.toFixed(2)}X` : '1.00X'}
                                </p>
                            </div>
                        </div>
                    </div>
                    
                    {/* 不确定性来源分析 */}
                    {results.explanations && results.explanations.length > 0 && (
                        <div className="bg-neutral-800 rounded-lg border border-neutral-700 shadow-xl p-6">
                            <h2 className="text-2xl font-bold text-white mb-4">不确定性来源分析</h2>
                            
                            <div className="bg-amber-900 bg-opacity-30 border border-amber-800 rounded-lg p-4">
                                <h3 className="text-xl font-medium text-amber-300 mb-2">场景影响分析</h3>
                                <p className="text-neutral-200">{results.explanations[0].explanation.scenario_impact}</p>
                            </div>
                            
                            {results.explanations[0].explanation.uncertainty_sources && (
                                <div className="mt-4">
                                    <h3 className="text-xl font-medium text-amber-300 mb-2">不确定性来源</h3>
                                    <ul className="list-disc pl-5 text-neutral-200 space-y-1">
                                        {results.explanations[0].explanation.uncertainty_sources.map((source, index) => (
                                            <li key={index}>{source}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                            
                            {results.explanations[0].explanation.calculation_method && (
                                <div className="mt-4">
                                    <h3 className="text-xl font-medium text-amber-300 mb-2">计算方法</h3>
                                    <p className="text-neutral-200">{results.explanations[0].explanation.calculation_method}</p>
                                </div>
                            )}
                            
                            {results.explanations[0].explanation.operation_suggestions && (
                                <div className="mt-4">
                                    <h3 className="text-xl font-medium text-amber-300 mb-2">运行建议</h3>
                                    <ul className="list-disc pl-5 text-neutral-200 space-y-1">
                                        {results.explanations[0].explanation.operation_suggestions.map((suggestion, index) => (
                                            <li key={index}>{suggestion}</li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                        </div>
                    )}
                    
                    {/* 预测结果表格 */}
                    <div className="bg-neutral-800 rounded-lg border border-neutral-700 shadow-xl p-6">
                        <h2 className="text-2xl font-bold text-white mb-4">预测结果</h2>
                        
                        <div className="overflow-x-auto">
                            <table className="min-w-full divide-y divide-neutral-700">
                                <thead className="bg-neutral-900">
                                    <tr>
                                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-neutral-400 uppercase tracking-wider">时间</th>
                                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-neutral-400 uppercase tracking-wider">预测值</th>
                                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-neutral-400 uppercase tracking-wider">下限</th>
                                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-neutral-400 uppercase tracking-wider">上限</th>
                                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-neutral-400 uppercase tracking-wider">不确定性</th>
                                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-neutral-400 uppercase tracking-wider">场景</th>
                                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-neutral-400 uppercase tracking-wider">风险等级</th>
                                    </tr>
                                </thead>
                                <tbody className="bg-neutral-800 divide-y divide-neutral-700">
                                    {results.predictions && results.predictions.slice(0, 24).map((prediction, index) => (
                                        <tr key={index} className={index % 2 === 0 ? 'bg-neutral-800' : 'bg-neutral-750'}>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-neutral-300">{formatDate(prediction.datetime)}</td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-neutral-300">{prediction.predicted.toFixed(2)}</td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-neutral-300">{prediction.lower_bound.toFixed(2)}</td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-neutral-300">{prediction.upper_bound.toFixed(2)}</td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-neutral-300">{prediction.uncertainty.toFixed(1)}%</td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-neutral-300">{prediction.scenario}</td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm">
                                                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getRiskLevelClass(prediction.risk_level)}`}>
                                                    {prediction.risk_level === 'high' ? '高' : 
                                                     prediction.risk_level === 'medium' ? '中' : '低'}
                                                </span>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                            
                            {results.predictions && results.predictions.length > 24 && (
                                <div className="mt-4 text-center text-neutral-400">
                                    <p>显示前24条记录，共{results.predictions.length}条</p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default ScenarioAwareUncertaintyForecast; 