import React from 'react';
import useTraining from './useTraining';
import TrainingSettings from '../../components/TrainingSettings';
import TrainingStatus from '../../components/TrainingStatus';

const TrainingTab = ({ availableProvinces }) => {
  const {
    params,
    result,
    isLoading,
    error,
    startTraining,
  } = useTraining();

  const cardClassName = "bg-neutral-900 border border-neutral-800 rounded-lg shadow-lg p-6";

  return (
    <div className="space-y-6">
      <TrainingSettings
        availableProvinces={availableProvinces}
        initialParams={params}
        onSubmit={startTraining}
        isLoading={isLoading}
        title="参数设置 - 模型训练"
        className={cardClassName}
        activeTabKey="training"
      />

      {/* The loading spinner is handled implicitly by the TrainingStatus component's initial state */}
      
      <TrainingStatus trainingResult={result} error={error} />
    </div>
  );
};

export default TrainingTab; 