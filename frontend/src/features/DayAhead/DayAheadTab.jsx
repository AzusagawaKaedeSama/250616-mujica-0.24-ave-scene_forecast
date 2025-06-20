import React from 'react';
import useDayAheadForecast from './useDayAheadForecast';
import ParameterSettings from '../../components/ParameterSettings';
import ForecastChart from '../../components/charts/ForecastChart';
import MetricsCard from '../../components/MetricsCard';

const DayAheadTab = ({ availableProvinces }) => {
  const {
    params,
    result,
    isLoading,
    error,
    runPrediction,
  } = useDayAheadForecast();

  const cardClassName = "bg-neutral-900 border border-neutral-800 rounded-lg shadow-lg p-6";

  return (
    <div className="space-y-6">
      <ParameterSettings
        availableProvinces={availableProvinces}
        initialParams={params}
        onSubmit={runPrediction}
        isLoading={isLoading}
        title="参数设置 - 日前预测"
        className={cardClassName}
        activeTabKey="dayAhead"
      />

      {isLoading && (
        <div className="flex justify-center items-center mt-6">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-red-600"></div>
        </div>
      )}

      {error && (
        <div className="mt-6 p-4 bg-red-900 bg-opacity-30 border border-red-800 rounded-lg shadow-lg">
          <h3 className="text-red-400 font-semibold">错误:</h3>
          <p className="text-red-300 mt-2">{error}</p>
        </div>
      )}

      {result && !isLoading && (
        <div className={`mt-6 ${cardClassName}`}>
            <div
                className="w-full h-96 bg-neutral-800 rounded-lg border border-neutral-700 shadow-xl"
            >
                <ForecastChart chartData={result} forecastType={params.forecastType} />
            </div>
            <MetricsCard metrics={result.metrics} />
        </div>
      )}
    </div>
  );
};

export default DayAheadTab; 