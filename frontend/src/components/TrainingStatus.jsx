import React from 'react';

const TrainingStatus = ({ trainingResult, error }) => {
  if (error) {
    return (
      <div className="mt-6 p-4 bg-red-900 bg-opacity-30 border border-red-800 rounded-lg shadow-lg">
        <h3 className="text-red-400 font-semibold">训练出错:</h3>
        <p className="text-red-300 mt-2">{error}</p>
      </div>
    );
  }

  if (!trainingResult || !trainingResult.task_id) {
    return null; // Don't render anything if training hasn't started
  }

  const {
    progress = 0,
    currentEpoch = 0,
    totalEpochs = 100,
    eta = '计算中...',
    status = 'pending',
    error: taskError,
    task_id,
  } = trainingResult;

  const getStatusPill = () => {
    switch (status) {
      case 'completed':
        return <span className="px-3 py-1 text-sm font-medium rounded-full bg-green-800 text-green-100">已完成</span>;
      case 'failed':
        return <span className="px-3 py-1 text-sm font-medium rounded-full bg-red-800 text-red-100">失败</span>;
      case 'running':
        return <span className="px-3 py-1 text-sm font-medium rounded-full bg-blue-800 text-blue-100">训练中</span>;
      default:
        return <span className="px-3 py-1 text-sm font-medium rounded-full bg-yellow-800 text-yellow-100">等待中</span>;
    }
  };

  return (
    <div className="mt-6 p-6 bg-neutral-900 border border-neutral-800 rounded-lg shadow-lg">
      <h3 className="text-xl font-medium text-neutral-100 mb-4">训练状态 (任务 ID: {task_id})</h3>
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <h4 className="text-lg font-medium text-neutral-300">状态:</h4>
          {getStatusPill()}
        </div>
        <div className="flex justify-between items-center">
          <h4 className="text-lg font-medium text-neutral-300">进度:</h4>
          <span className="text-neutral-100 font-mono">{progress}%</span>
        </div>
        <div className="w-full bg-neutral-700 rounded-full h-2.5">
          <div
            className="bg-red-600 h-2.5 rounded-full transition-all duration-500 ease-out"
            style={{ width: `${progress}%` }}
          ></div>
        </div>
        <div className="flex justify-between items-center text-sm">
          <div>
            <span className="text-neutral-400">轮次: </span>
            <span className="text-neutral-200 font-mono">{currentEpoch}/{totalEpochs}</span>
          </div>
          <div>
            <span className="text-neutral-400">预计剩余时间: </span>
            <span className="text-neutral-200">{eta}</span>
          </div>
        </div>
        {taskError && (
          <div className="mt-4 p-3 bg-red-900 bg-opacity-30 border border-red-800 rounded">
            <h4 className="text-red-400 font-medium">错误信息:</h4>
            <p className="text-red-300 text-sm mt-1 font-mono">{taskError}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default TrainingStatus; 