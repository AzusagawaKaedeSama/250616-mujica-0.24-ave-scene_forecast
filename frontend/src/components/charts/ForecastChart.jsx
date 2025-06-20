import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  Filler
} from 'chart.js';
import 'chartjs-adapter-date-fns';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  Filler
);

const ForecastChart = ({ chartData, forecastType }) => {
  if (!chartData || !chartData.predictions || chartData.predictions.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-neutral-400">没有可供显示的预测数据。</p>
      </div>
    );
  }

  const { predictions, actuals } = chartData;

  // 转换时间戳格式 - 将RFC格式字符串转换为Date对象
  const processedPredictions = predictions.map(p => ({
    ...p,
    timestamp: new Date(p.timestamp)
  }));

  const labels = processedPredictions.map(p => p.timestamp);
  
  const data = {
    labels: labels,
    datasets: [
      {
        label: `${forecastType} 预测值`,
        data: processedPredictions.map(p => ({ x: p.timestamp, y: p.value })),
        borderColor: '#DC2626', // red-600
        backgroundColor: 'rgba(220, 38, 38, 0.1)',
        borderWidth: 2,
        pointRadius: 1,
        tension: 0.4,
        fill: false,
      },
      // Optional: Actual values
      ...(actuals && actuals.length > 0
        ? [
            {
              label: `${forecastType} 实际值`,
              data: actuals.map(a => ({ x: new Date(a.timestamp), y: a.actual })),
              borderColor: '#16A34A', // green-600
              backgroundColor: 'rgba(22, 163, 74, 0.1)',
              borderWidth: 2,
              pointRadius: 1,
              tension: 0.4,
              fill: false,
            },
          ]
        : []),
      // Optional: Interval forecast
      ...(predictions[0]?.lower_bound !== undefined && predictions[0]?.upper_bound !== undefined ? [
        {
            label: '预测下界',
            data: processedPredictions.map(p => ({ x: p.timestamp, y: p.lower_bound })),
            borderColor: 'rgba(255, 255, 255, 0.2)',
            borderWidth: 1,
            pointRadius: 0,
            fill: '+1', // Fill to the next dataset (upper bound)
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
            tension: 0.4,
        },
        {
            label: '预测上界',
            data: processedPredictions.map(p => ({ x: p.timestamp, y: p.upper_bound })),
            borderColor: 'rgba(255, 255, 255, 0.2)',
            borderWidth: 1,
            pointRadius: 0,
            fill: false, // Don't fill
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
            tension: 0.4,
        },
      ] : [])
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
            color: '#d4d4d4' // text-neutral-300
        }
      },
      title: {
        display: true,
        text: `${forecastType} 预测结果`,
        color: '#f5f5f5', // text-neutral-100
        font: {
            size: 18
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
      }
    },
    scales: {
      x: {
        type: 'time',
        time: {
          unit: 'hour',
          tooltipFormat: 'yyyy-MM-dd HH:mm',
          displayFormats: {
            hour: 'HH:mm',
            day: 'MM-dd'
          }
        },
        ticks: {
          color: '#a3a3a3', // text-neutral-400
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)',
        }
      },
      y: {
        ticks: {
          color: '#a3a3a3', // text-neutral-400
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.1)',
        },
        title: {
            display: true,
            text: '值',
            color: '#d4d4d4'
        }
      },
    },
    interaction: {
        mode: 'index',
        intersect: false,
    },
  };

  return <Line options={options} data={data} />;
};

export default ForecastChart; 