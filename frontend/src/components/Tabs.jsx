import React from 'react';

const TabButton = ({ label, tabName, activeTab, setActiveTab }) => {
  const isActive = activeTab === tabName;
  return (
    <button
      onClick={() => setActiveTab(tabName)}
      className={`px-4 py-2.5 font-medium text-sm leading-5 rounded-t-lg 
                  focus:outline-none transition-colors duration-150 ease-in-out 
                  ${isActive 
                    ? 'text-primary-darker bg-white dark:bg-gray-800 border-b-2 border-primary dark:text-primary-light' 
                    : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700'}
                  `}
    >
      {label}
    </button>
  );
};

function Tabs({ activeTab, setActiveTab }) {
  const tabs = [
    { name: 'dayAhead', label: '日前预测' },
    { name: 'rolling', label: '滚动预测' },
    { name: 'interval', label: '区间预测' },
    { name: 'probabilistic', label: '概率预测' },
    { name: 'scenarios', label: '场景识别' },
    { name: 'scenarioUncertainty', label: '场景感知不确定性预测' },
    { name: 'training', label: '模型训练' },
    { name: 'historical', label: '历史结果查询' },
  ];

  return (
    <div className="bg-secondary-DEFAULT dark:bg-gray-850 shadow-sm sticky top-0 z-10">
      <div className="container mx-auto px-4">
        <div className="flex border-b border-gray-200 dark:border-gray-700">
          {tabs.map((tab) => (
            <TabButton
              key={tab.name}
              label={tab.label}
              tabName={tab.name}
              activeTab={activeTab}
              setActiveTab={setActiveTab}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

export default Tabs; 