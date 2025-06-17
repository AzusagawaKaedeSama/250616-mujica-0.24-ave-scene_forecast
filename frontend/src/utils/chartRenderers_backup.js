/**
 * 图表渲染工具 - 用于将API返回的预测数据渲染为图表
 * 修复版本：解决区间预测中点预测显示和置信区间阴影问题
 */

/**
 * 渲染预测结果图表
 * @param {Object} data - 预测结果数据
 * @param {string} containerId - 图表容器元素的ID
 * @param {string} forecastType - 预测类型 (load/pv/wind)
 */
export function renderForecastChart(data, containerId, forecastType) {
  if (!data || !data.predictions || !Array.isArray(data.predictions)) {
    console.error('没有可用的预测数据进行渲染', data);
    return;
  }

  // 准备图表数据
  const times = data.predictions.map(p => p.datetime.substring(11, 16)); // 提取时间部分，HH:MM格式
  const dates = data.predictions.map(p => p.datetime.substring(0, 10)); // 提取日期部分，YYYY-MM-DD格式
  
  // 根据不同预测类型设置不同的Y轴标签
  const yAxisLabel = getYAxisLabel(forecastType);
  
  // 准备序列数据
  const seriesData = [];
  
  // 添加预测值序列
  seriesData.push({
    name: '预测值',
    type: 'line',
    data: data.predictions.map(p => p.predicted),
    smooth: true,
    symbol: 'none',
    lineStyle: {
      width: 3
    }
  });
  
  // 如果有实际值，添加实际值序列
  if (data.predictions.some(p => p.actual !== undefined && p.actual !== null)) {
    seriesData.push({
      name: '实际值',
      type: 'line',
      data: data.predictions.map(p => p.actual),
      smooth: true,
      symbol: 'none',
      lineStyle: {
        width: 3
      }
    });
  }
  
  // 如果是净负荷预测，添加光伏和风电预测值
  if (data.is_net_load) {
    if (data.predictions.some(p => p.pv_predicted !== undefined)) {
      seriesData.push({
        name: '光伏出力',
        type: 'line',
        data: data.predictions.map(p => p.pv_predicted),
        smooth: true,
        symbol: 'none',
        lineStyle: {
          width: 2,
          type: 'dashed'
        }
      });
    }
    
    if (data.predictions.some(p => p.wind_predicted !== undefined)) {
      seriesData.push({
        name: '风电出力',
        type: 'line',
        data: data.predictions.map(p => p.wind_predicted),
        smooth: true,
        symbol: 'none',
        lineStyle: {
          width: 2,
          type: 'dashed'
        }
      });
    }
    
    if (data.predictions.some(p => p.predicted_net !== undefined)) {
      seriesData.push({
        name: '净负荷',
        type: 'line',
        data: data.predictions.map(p => p.predicted_net),
        smooth: true,
        symbol: 'none',
        lineStyle: {
          width: 3
        }
      });
    }
  }
  
  // 创建图表配置
  const option = {
    title: {
      text: `${getForecastTypeLabel(forecastType)}预测结果`,
      left: 'center'
    },
    tooltip: {
      trigger: 'axis',
      formatter: function(params) {
        let tooltip = `${dates[params[0].dataIndex]} ${times[params[0].dataIndex]}<br/>`;
        params.forEach(param => {
          const seriesName = param.seriesName;
          const value = param.value;
          tooltip += `${param.marker}${seriesName}: ${value?.toFixed(2) || 'N/A'} ${getUnitLabel(forecastType)}`;
          
          // Access error_pct from the original data.predictions array
          if (seriesName === '预测值' && data.predictions && data.predictions[params[0].dataIndex]) {
            const errorPct = data.predictions[params[0].dataIndex].error_pct;
            if (errorPct !== null && errorPct !== undefined) {
              tooltip += ` (误差: ${errorPct.toFixed(2)}%)`;
            }
          }
          tooltip += '<br/>';
        });
        return tooltip;
      }
    },
    legend: {
      data: seriesData.map(s => s.name),
      bottom: 0
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '10%',
      top: '10%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: times,
      axisLabel: {
        formatter: function(value, index) {
          // 每隔4个点(每小时)显示一次时间标签
          return index % 4 === 0 ? value : '';
        }
      }
    },
    yAxis: {
      type: 'value',
      name: yAxisLabel,
      nameLocation: 'middle',
      nameGap: 40
    },
    series: seriesData,
    // color: ['#1890ff', '#ff4d4f', '#52c41a', '#faad14', '#722ed1'] // 使用暗色主题的默认颜色
  };
  
  // 初始化图表
  const container = document.getElementById(containerId);
  if (!container) {
    console.error(`找不到ID为${containerId}的图表容器元素`);
    return;
  }
  
  // 确保echarts已全局加载
  if (typeof echarts === 'undefined') {
    console.error('找不到echarts对象，请确保已加载echarts库');
    return;
  }
  
  const chart = echarts.init(container, 'dark'); // 使用 'dark' 主题
  chart.setOption({...option, backgroundColor: 'transparent' });
  
  // 响应窗口大小变化
  window.addEventListener('resize', () => {
    chart.resize();
  });
  
  return chart;
}

/**
 * 渲染区间预测结果图表 - 完整修复版本
 * @param {Object} data - 区间预测结果数据
 * @param {string} containerId - 图表容器元素的ID
 * @param {string} forecastType - 预测类型 (load/pv/wind)
 */
export function renderIntervalForecastChart(data, containerId, forecastType) {
  console.log('渲染区间预测图表，接收到的数据：', data);
  
  if (!data || !data.predictions || !Array.isArray(data.predictions)) {
    console.error('没有可用的区间预测数据进行渲染', data);
    
    // 显示错误信息到容器中
    const container = document.getElementById(containerId);
    if (container) {
      container.innerHTML = '<div style="padding: 20px; text-align: center; color: #ff6b6b;">没有可用的区间预测数据</div>';
    }
    return;
  }

  // 准备图表数据
  const times = data.predictions.map(p => p.datetime.substring(11, 16)); // 提取时间部分，HH:MM格式
  const dates = data.predictions.map(p => p.datetime.substring(0, 10)); // 提取日期部分，YYYY-MM-DD格式
  
  console.log('图表时间点数量：', times.length);
  console.log('前5个时间点：', times.slice(0, 5));
  
  // 检查点预测数据
  const predictedValues = data.predictions.map(p => p.predicted);
  console.log('点预测数据前5个值：', predictedValues.slice(0, 5));
  console.log('点预测数据统计：', {
    总数: predictedValues.length,
    有效值数量: predictedValues.filter(v => v !== null && v !== undefined && !isNaN(v)).length,
    最小值: Math.min(...predictedValues.filter(v => v !== null && v !== undefined && !isNaN(v))),
    最大值: Math.max(...predictedValues.filter(v => v !== null && v !== undefined && !isNaN(v)))
  });
  
  // 根据不同预测类型设置不同的Y轴标签
  const yAxisLabel = getYAxisLabel(forecastType);
  
  // 定义统一的颜色方案
  const colors = {
    pointForecast: '#1890ff',    // 蓝色 - 点预测
    upperBound: '#52c41a',       // 绿色 - 上边界
    lowerBound: '#52c41a',       // 绿色 - 下边界
    actual: '#ff4d4f',           // 红色 - 实际值
    confidenceArea: 'rgba(82, 196, 26, 0.2)'  // 半透明绿色 - 置信区间
  };
  
  // 准备序列数据
  const seriesData = [];
  
  // 修复：使用正确的方式添加置信区间阴影（仅在上下边界之间填充）
  if (data.predictions.some(p => p.upper_bound !== undefined && p.lower_bound !== undefined)) {
    console.log('添加置信区间阴影 - 使用修复后的方法');
    
    // 方法1：使用堆叠但设置正确的基准值
    // 先添加下边界作为基准
    seriesData.push({
      name: '_confidence_base',
      type: 'line',
      data: data.predictions.map((p, index) => [index, Number(p.lower_bound)]),
      lineStyle: { opacity: 0 },
      symbol: 'none',
      smooth: true,
      showInLegend: false,
      stack: 'confidence'
    });
    
    // 再添加区间宽度作为填充区域
    seriesData.push({
      name: '置信区间',
      type: 'line',
      data: data.predictions.map((p, index) => [index, Number(p.upper_bound) - Number(p.lower_bound)]),
      lineStyle: { opacity: 0 },
      symbol: 'none',
      areaStyle: {
        color: colors.confidenceArea,
        opacity: 0.4
      },
      smooth: true,
      stack: 'confidence', // 堆叠在基准之上
      z: 1,
      showInLegend: true
    });
  }
  
  // 修复：确保点预测序列数据格式正确且值不为0
  if (data.predictions.some(p => p.predicted !== undefined && p.predicted !== null && !isNaN(p.predicted))) {
    console.log('添加点预测序列 - 使用修复后的数据格式');
    
    const pointForecastData = data.predictions.map((p, index) => {
      // 确保数据格式正确，并过滤无效值
      if (p.predicted !== undefined && p.predicted !== null && !isNaN(p.predicted)) {
        return [index, Number(p.predicted)]; // 明确转换为数字
      }
      return [index, null]; // 无效数据点
    });
    
    console.log('点预测序列数据前5个点：', pointForecastData.slice(0, 5));
    
    seriesData.push({
      name: '点预测',
      type: 'line',
      data: pointForecastData,
      smooth: true,
      symbol: 'none',
      lineStyle: {
        width: 3,
        color: colors.pointForecast
      },
      itemStyle: {
        color: colors.pointForecast
      },
      z: 3, // 确保在置信区间之上
      connectNulls: false // 不连接空值点
    });
  } else {
    console.warn('警告：没有找到有效的点预测数据');
  }
  
  // 添加上下边界线
  if (data.predictions.some(p => p.upper_bound !== undefined && p.lower_bound !== undefined)) {
    console.log('添加上下边界线');
    
    const upperBoundData = data.predictions.map((p, index) => [index, Number(p.upper_bound)]);
    const lowerBoundData = data.predictions.map((p, index) => [index, Number(p.lower_bound)]);
    
    seriesData.push({
      name: '上边界',
      type: 'line',
      data: upperBoundData,
      smooth: true,
      symbol: 'none',
      lineStyle: {
        width: 2,
        type: 'dashed',
        color: colors.upperBound
      },
      itemStyle: {
        color: colors.upperBound
      },
      z: 2
    });
    
    seriesData.push({
      name: '下边界',
      type: 'line',
      data: lowerBoundData,
      smooth: true,
      symbol: 'none',
      lineStyle: {
        width: 2,
        type: 'dashed',
        color: colors.lowerBound
      },
      itemStyle: {
        color: colors.lowerBound
      },
      z: 2
    });
  }
  
  // 添加实际值序列（如果存在）
  if (data.predictions.some(p => p.actual !== undefined && p.actual !== null && !isNaN(p.actual))) {
    console.log('添加实际值序列');
    
    const actualData = data.predictions.map((p, index) => {
      if (p.actual !== undefined && p.actual !== null && !isNaN(p.actual)) {
        return [index, Number(p.actual)];
      }
      return [index, null];
    });
    
    seriesData.push({
      name: '实际值',
      type: 'line',
      data: actualData,
      smooth: true,
      symbol: 'circle',
      symbolSize: 4,
      lineStyle: {
        width: 3,
        color: colors.actual
      },
      itemStyle: {
        color: colors.actual
      },
      z: 4, // 确保在最上层
      connectNulls: false
    });
  }
  
  console.log('序列数量：', seriesData.length);
  
  // 获取置信水平信息
  const confidenceLevel = data.confidence_level || 0.9;
  
  // 创建图表配置
  const option = {
    title: {
      text: `${getForecastTypeLabel(forecastType)}区间预测结果 (${(confidenceLevel * 100).toFixed(0)}% 置信水平)`,
      left: 'center',
      textStyle: {
        color: '#ffffff',
        fontSize: 16,
        fontWeight: 'bold'
      }
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
      borderColor: '#333',
      textStyle: {
        color: '#fff'
      },
      formatter: function(params) {
        const dataIndex = params[0].dataIndex;
        let tooltip = `<div style="margin-bottom: 8px; font-weight: bold;">${dates[dataIndex]} ${times[dataIndex]}</div>`;
        
        // 按特定顺序显示数据 - 修复数据获取方式
        const actualValue = data.predictions[dataIndex].actual;
        const pointValue = data.predictions[dataIndex].predicted;
        const upperValue = data.predictions[dataIndex].upper_bound;
        const lowerValue = data.predictions[dataIndex].lower_bound;
        
        if (actualValue !== null && actualValue !== undefined && !isNaN(actualValue)) {
          tooltip += `<div><span style="color: ${colors.actual};">●</span> 实际值: <span style="font-weight: bold;">${Number(actualValue).toFixed(2)} ${getUnitLabel(forecastType)}</span></div>`;
        }
        
        if (pointValue !== null && pointValue !== undefined && !isNaN(pointValue)) {
          tooltip += `<div><span style="color: ${colors.pointForecast};">●</span> 点预测: <span style="font-weight: bold;">${Number(pointValue).toFixed(2)} ${getUnitLabel(forecastType)}</span></div>`;
          
          // 添加误差信息
          const prediction = data.predictions[dataIndex];
          if (prediction.error_pct !== null && prediction.error_pct !== undefined) {
            const errorColor = Math.abs(prediction.error_pct) > 2 ? '#ff4d4f' : '#52c41a';
            tooltip += `<div style="color: ${errorColor}; margin-left: 20px;">误差: ${Number(prediction.error_pct).toFixed(2)}%</div>`;
          }
        }
        
        if (upperValue !== null && upperValue !== undefined && lowerValue !== null && lowerValue !== undefined) {
          tooltip += `<div style="margin-top: 4px; border-top: 1px solid #555; padding-top: 4px;">`;
          tooltip += `<div><span style="color: ${colors.upperBound};">--</span> 上边界: ${Number(upperValue).toFixed(2)} ${getUnitLabel(forecastType)}</div>`;
          tooltip += `<div><span style="color: ${colors.lowerBound};">--</span> 下边界: ${Number(lowerValue).toFixed(2)} ${getUnitLabel(forecastType)}</div>`;
          const intervalWidth = Number(upperValue) - Number(lowerValue);
          tooltip += `<div style="color: #999; font-size: 12px;">区间宽度: ${intervalWidth.toFixed(2)} ${getUnitLabel(forecastType)}</div>`;
          tooltip += `</div>`;
        }
        
        return tooltip;
      }
    },
    legend: {
      data: seriesData.filter(s => s.showInLegend !== false).map(s => s.name),
      bottom: 10,
      textStyle: {
        color: '#ffffff'
      },
      itemGap: 20
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '15%',
      top: '15%',
      containLabel: true
    },
    dataZoom: [{
      type: 'slider',
      show: true,
      xAxisIndex: 0,
      start: 0,
      end: 100,
      bottom: '5%'
    }],
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: times,
      axisLabel: {
        formatter: function(value, index) {
          // 每隔4个点(每小时)显示一次时间标签
          return index % 4 === 0 ? value : '';
        },
        color: '#ffffff',
        fontSize: 12
      },
      axisLine: {
        lineStyle: {
          color: '#666666'
        }
      },
      axisTick: {
        lineStyle: {
          color: '#666666'
        }
      }
    },
    yAxis: {
      type: 'value',
      name: yAxisLabel,
      nameLocation: 'middle',
      nameGap: 50,
      nameTextStyle: {
        color: '#ffffff',
        fontSize: 14,
        fontWeight: 'bold'
      },
      axisLabel: {
        color: '#ffffff',
        fontSize: 12,
        formatter: function(value) {
          return value.toFixed(0);
        }
      },
      axisLine: {
        lineStyle: {
          color: '#666666'
        }
      },
      splitLine: {
        lineStyle: {
          color: '#333333',
          type: 'dashed'
        }
      }
    },
    series: seriesData,
    color: [colors.confidenceArea, colors.pointForecast, colors.upperBound, colors.lowerBound, colors.actual],
    animation: true,
    animationDuration: 1000,
    animationEasing: 'cubicOut'
  };
  
  // 初始化图表
  const container = document.getElementById(containerId);
  if (!container) {
    console.error(`找不到ID为${containerId}的图表容器元素`);
    return;
  }
  
  // 确保echarts已全局加载
  if (typeof echarts === 'undefined') {
    console.error('找不到echarts对象，请确保已加载echarts库');
    return;
  }
  
  console.log('初始化ECharts图表');
  const chart = echarts.init(container, 'dark'); // 使用 'dark' 主题
  chart.setOption({...option, backgroundColor: 'transparent' });
  
  // 响应窗口大小变化
  const resizeHandler = () => {
    chart.resize();
  };
  
  // 移除之前的监听器（如果存在）
  window.removeEventListener('resize', resizeHandler);
  window.addEventListener('resize', resizeHandler);
  
  console.log('区间预测图表渲染完成');
  return chart;
}

/**
 * 获取预测类型标签
 * @param {string} forecastType - 预测类型 (load/pv/wind)
 * @returns {string} 预测类型的中文标签
 */
function getForecastTypeLabel(forecastType) {
  switch (forecastType) {
    case 'load':
      return '电力负荷';
    case 'pv':
      return '光伏出力';
    case 'wind':
      return '风电出力';
    case 'net_load':
      return '净负荷';
    default:
      return forecastType;
  }
}

/**
 * 获取Y轴标签
 * @param {string} forecastType - 预测类型 (load/pv/wind)
 * @returns {string} Y轴标签
 */
function getYAxisLabel(forecastType) {
  switch (forecastType) {
    case 'load':
      return '负荷 (MW)';
    case 'pv':
      return '光伏出力 (MW)';
    case 'wind':
      return '风电出力 (MW)';
    default:
      return '出力 (MW)';
  }
}

/**
 * 获取单位标签
 * @param {string} forecastType - 预测类型 (load/pv/wind)
 * @returns {string} 单位标签
 */
function getUnitLabel(forecastType) {
  return 'MW';
}

/**
 * 渲染年度预测数据的图表
 * 针对大数据量的年度预测CSV文件进行了优化
 */
export function renderYearlyForecastChart(data, containerId, forecastType) {
  // 确保容器存在
  const container = document.getElementById(containerId);
  if (!container) {
    console.error(`图表容器 #${containerId} 不存在`);
    return;
  }
  
  // 检查数据是否有效
  if (!data || !data.predictions || !Array.isArray(data.predictions)) {
    console.error('无效的预测数据格式', data);
    return;
  }
  
  const predictions = data.predictions;
  
  // 检查数组是否为空
  if (!predictions || predictions.length === 0) {
    container.innerHTML = '<div style="padding: 20px; text-align: center;">没有可用的预测数据</div>';
    return;
  }
  
  console.log(`渲染年度预测图表，数据点数量: ${predictions.length}`);
  
  // 准备ECharts数据
  const timeData = [];
  const predictedData = [];
  const actualData = [];
  const errorData = [];
  
  // 处理数据
  predictions.forEach(item => {
    // 检查时间戳是否有效
    if (!item.datetime) {
      return; // 跳过无效行
    }
    
    // 解析时间戳
    let timestamp;
    try {
      timestamp = new Date(item.datetime);
    } catch (e) {
      console.warn(`无效的时间格式: ${item.datetime}`);
      return; // 跳过此行
    }
    
    timeData.push(timestamp);
    
    // 处理预测值
    if (item.predicted !== undefined && item.predicted !== null) {
      predictedData.push([timestamp, parseFloat(item.predicted)]);
    }
    
    // 处理实际值（可能不存在）
    if (item.actual !== undefined && item.actual !== null) {
      actualData.push([timestamp, parseFloat(item.actual)]);
    }
    
    // 处理误差（可能需要计算）
    if (item.error !== undefined && item.error !== null) {
      errorData.push([timestamp, parseFloat(item.error)]);
    } else if (item.actual !== undefined && item.actual !== null && item.predicted !== undefined && item.predicted !== null) {
      // 计算误差百分比
      const actual = parseFloat(item.actual);
      const predicted = parseFloat(item.predicted);
      let error = actual !== 0 ? ((predicted - actual) / actual * 100) : null;
      // 限制误差范围以避免极端值影响图表
      if (error !== null) {
        error = Math.max(-50, Math.min(50, error));
        errorData.push([timestamp, error]);
      }
    }
  });
  
  // 确保数据不为空
  if (predictedData.length === 0) {
    container.innerHTML = '<div style="padding: 20px; text-align: center;">处理后没有有效的预测数据点</div>';
    return;
  }
  
  // 初始化ECharts实例
  const chart = echarts.init(container);
  
  // 获取预测类型标签
  const typeLabel = getForecastTypeLabel(forecastType);
  const yAxisLabel = getYAxisLabel(forecastType);
  const unitLabel = getUnitLabel(forecastType);
  
  // 准备图表配置
  const option = {
    title: {
      text: `${typeLabel}年度预测结果`,
      left: 'center'
    },
    tooltip: {
      trigger: 'axis',
      formatter: function(params) {
        const date = new Date(params[0].value[0]);
        const dateStr = date.toLocaleDateString();
        const timeStr = date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        let result = `${dateStr} ${timeStr}<br/>`;
        
        params.forEach(param => {
          let value = param.value[1];
          if (value !== null && !isNaN(value)) {
            if (param.seriesName.includes('误差')) {
              result += `${param.marker} ${param.seriesName}: ${value.toFixed(2)}%<br/>`;
            } else {
              result += `${param.marker} ${param.seriesName}: ${value.toFixed(2)} ${unitLabel}<br/>`;
            }
          }
        });
        
        return result;
      }
    },
    legend: {
      data: ['预测值', '实际值', '误差 (%)'],
      top: 30
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '10%',
      containLabel: true
    },
    toolbox: {
      feature: {
        saveAsImage: {}
      }
    },
    dataZoom: [
      {
        type: 'slider',
        start: 0,
        end: 100
      },
      {
        type: 'inside',
        start: 0,
        end: 100
      }
    ],
    xAxis: {
      type: 'time',
      boundaryGap: false,
      axisLabel: {
        formatter: '{MM}-{dd} {HH}:{mm}'
      }
    },
    yAxis: [
      {
        type: 'value',
        name: `${yAxisLabel} (${unitLabel})`,
        position: 'left'
      },
      {
        type: 'value',
        name: '误差 (%)',
        position: 'right',
        min: -50,
        max: 50,
        splitLine: {
          show: false
        }
      }
    ],
    series: [
      {
        name: '预测值',
        type: 'line',
        symbol: 'none',
        sampling: 'lttb', // 大数据量下使用LTTB采样
        data: predictedData,
        lineStyle: { 
          width: 2 
        }
      },
      {
        name: '实际值',
        type: 'line',
        symbol: 'none',
        sampling: 'lttb', // 大数据量下使用LTTB采样
        data: actualData,
        lineStyle: { 
          width: 2 
        }
      },
      {
        name: '误差 (%)',
        type: 'line',
        yAxisIndex: 1,
        symbol: 'none',
        sampling: 'lttb', // 大数据量下使用LTTB采样
        data: errorData,
        lineStyle: { 
          width: 1, 
          opacity: 0.5 
        }
      }
    ]
  };
  
  // 渲染图表
  chart.setOption(option);
  
  // 响应窗口调整大小
  window.addEventListener('resize', () => {
    chart.resize();
  });
  
  return chart;
}