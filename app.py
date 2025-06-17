from flask import Flask, request, jsonify, send_from_directory
import subprocess
import json
import os
import pandas as pd
from datetime import datetime, timedelta
import time
from flask_cors import CORS  # 用于处理跨域请求
import re
from flask_caching import Cache  # 添加缓存支持
import hashlib  # 用于生成缓存键
import signal  # 用于处理子进程超时
import numpy as np
import sys
from scripts.forecast.interval_forecast_fixed import perform_interval_forecast
import glob
import logging
import torch
import sys
sys.stdout.reconfigure(encoding='utf-8')


# 强制设置环境编码为UTF-8
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

# 设置Windows代码页为UTF-8 (仅Windows)
if sys.platform.startswith('win'):
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleCP(65001)
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except:
        pass

# 确保标准输出编码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log", encoding='utf-8'), # 为文件处理器指定UTF-8编码
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
# 更完善的CORS配置，确保OPTIONS请求正确响应
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# 配置缓存
cache_config = {
    "DEBUG": True,
    "CACHE_TYPE": "SimpleCache",  # 使用内存缓存，生产环境可考虑使用Redis等
    "CACHE_DEFAULT_TIMEOUT": 300,  # 默认缓存时间为5分钟
    "CACHE_THRESHOLD": 500,       # 最大缓存项数量
    "CACHE_KEY_PREFIX": "vpp_"    # 缓存键前缀，便于识别
}
cache = Cache(app, config=cache_config)

# 定义不同类型数据的缓存时间
CACHE_TIMES = {
    'predict': 300,        # 预测结果缓存5分钟
    'provinces': 3600,     # 省份列表缓存1小时
    'demo_data': 3600,     # 演示数据缓存1小时
    'integrated': 600,     # 集成预测缓存10分钟
    'scenarios': 600,      # 场景识别缓存10分钟
    'interval': 900        # 区间预测缓存15分钟
}

# 缓存命中计数器
cache_hits = {
    'predict': 0,
    'provinces': 0,
    'demo_data': 0,
    'integrated': 0,
    'scenarios': 0,
    'interval': 0
}

# 缓存管理函数
def increment_cache_hit(cache_type):
    """增加缓存命中计数"""
    if cache_type in cache_hits:
        cache_hits[cache_type] += 1
    return cache_hits[cache_type]

def get_cache_stats():
    """获取缓存统计信息"""
    total_hits = sum(cache_hits.values())
    return {
        'hits': cache_hits, 
        'total': total_hits,
        'config': {
            'default_timeout': cache_config['CACHE_DEFAULT_TIMEOUT'],
            'specific_timeouts': CACHE_TIMES
        }
    }

def clear_type_cache(cache_type):
    """清除特定类型的缓存"""
    if not cache_type:
        return False
    
    # 尝试清除特定类型缓存
    try:
        # 对于SimpleCache，我们不能选择性清除，所以这里只重置命中计数
        if cache_type in cache_hits:
            cache_hits[cache_type] = 0
        return True
    except:
        return False

# 创建一个字典用于存储训练任务状态
training_tasks = {}

# 全局变量，用于存储最新一次预测的结果路径
latest_prediction_result = None

# 创建缓存键的辅助函数
def make_cache_key(*args, **kwargs):
    """根据请求内容和类型创建唯一的缓存键"""
    json_data = request.get_json(silent=True) or {}
    
    force_refresh = json_data.get('force_refresh')
    
    # 基础参数用于生成键
    key_dict = {
        "endpoint": request.endpoint,
        "predictionType": json_data.get('predictionType'),
        "forecastType": json_data.get('forecastType'),
        "province": json_data.get('province'),
        "historicalDays": json_data.get('historicalDays')
        # 可以添加更多你认为重要的、影响结果的参数
    }
    
    # 根据预测类型添加不同的日期参数
    prediction_type = json_data.get('predictionType')
    if prediction_type == 'day-ahead':
        key_dict['forecastDate'] = json_data.get('forecastDate')
        key_dict['forecastEndDate'] = json_data.get('forecastEndDate')
    elif prediction_type == 'rolling':
        key_dict['startDate'] = json_data.get('startDate')
        key_dict['endDate'] = json_data.get('endDate')
        key_dict['interval'] = json_data.get('interval')
        key_dict['realTimeAdjustment'] = json_data.get('realTimeAdjustment')
        
    # 如果是强制刷新，添加时间戳使键唯一
    if force_refresh:
        key_dict['_timestamp'] = datetime.now().timestamp()
    
    # 序列化并哈希
    try:
        key_str = json.dumps(key_dict, sort_keys=True)
        cache_key = hashlib.md5(key_str.encode('utf-8')).hexdigest()
        print(f"生成的缓存键: {cache_key} (基于: {key_str})") # 增加日志
        return cache_key
    except Exception as e:
        print(f"生成缓存键时出错: {e}")
        # Fallback to a generic key if serialization fails
        return f"fallback_{request.endpoint}_{hashlib.md5(str(json_data).encode()).hexdigest()}"

# 添加进程超时处理函数
# 2. 修复 run_process_with_timeout 函数
def run_process_with_timeout(cmd, timeout=300):
    """
    运行命令并设置超时时间（秒）
    修复编码问题
    """
    try:
        logger.debug(f"Executing command: {' '.join(cmd)}")
        
        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        env['PYTHONPATH'] = os.pathsep.join(sys.path)
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 将 stderr 合并到 stdout
            text=True,
            encoding='utf-8',  # 明确指定UTF-8编码
            errors='replace',  # 遇到无法解码的字符时替换为?
            bufsize=1,  # 行缓冲
            env=env,  # 传递环境变量
            universal_newlines=True
        )
        
        stdout_lines = []
        start_time = time.time()
        
        logger.debug(f"Starting to read output for command: {' '.join(cmd)}")
        while True:
            # 检查超时
            if timeout and (time.time() - start_time > timeout):
                logger.warning(f"Command {' '.join(cmd)} timed out after {timeout} seconds. Attempting to kill.")
                process.kill()
                try:
                    remaining_output = process.stdout.read()
                    if remaining_output:
                        logger.debug(f"Remaining output after kill: {remaining_output.strip()}")
                        stdout_lines.append(remaining_output)
                except Exception as read_ex:
                    logger.debug(f"Error reading remaining output after kill: {read_ex}")
                process.wait()
                logger.warning(f"Command {' '.join(cmd)} killed due to timeout.")
                return -1, "".join(stdout_lines), "命令执行超时，可能是参数配置导致了长时间运行或脚本卡住。请考虑减少历史数据天数或其他参数。"

            line = process.stdout.readline()
            if line:
                # 确保输出是正确编码的
                try:
                    # 如果line包含乱码，尝试重新编码
                    if '�' in line:
                        # 尝试从GBK解码然后编码为UTF-8
                        try:
                            line_bytes = line.encode('latin1')
                            line = line_bytes.decode('gbk', errors='replace')
                        except:
                            line = line.encode('utf-8', errors='replace').decode('utf-8')
                    
                    stdout_lines.append(line)
                    clean_line = line.strip()
                    if clean_line:
                        logger.debug(f"[PID:{process.pid}] Read line: {clean_line}")
                except UnicodeError as ue:
                    logger.warning(f"Unicode error in line: {ue}")
                    stdout_lines.append(line.encode('utf-8', errors='replace').decode('utf-8'))
                    
            elif process.poll() is not None:
                logger.debug(f"Process {process.pid} finished with poll status: {process.poll()}")
                break
            else:
                time.sleep(0.005)

        logger.debug(f"Finished reading output for command: {' '.join(cmd)}. Waiting for final process completion.")
        
        if process.returncode is None:
             process.wait(timeout=10)
        
        return_code = process.returncode if process.returncode is not None else -99
        stdout_data = "".join(stdout_lines)
        
        logger.info(f"Command {' '.join(cmd)} completed with return code: {return_code}. Total output lines: {len(stdout_lines)}.")

        if return_code != 0 and not (timeout and (time.time() - start_time > timeout) and return_code == -1):
             logger.error(f"Command {' '.join(cmd)} failed with return code: {return_code}. Output:\n{stdout_data}")

        return return_code, stdout_data, ""

    except Exception as e:
        logger.error(f"Exception in run_process_with_timeout for command {' '.join(cmd if 'cmd' in locals() else '[unknown cmd]')}: {str(e)}", exc_info=True)
        return -2, "", f"启动或运行进程时发生严重错误: {str(e)}"

# --- Helper to build command list ---
def build_python_command(script_path, args):
    """Builds the command list for running a python script, including the -u flag."""
    # Always start with python executable and -u flag for unbuffered output
    cmd = [sys.executable, '-u', script_path]
    # Extend with other arguments passed as a list
    cmd.extend(args)
    return cmd

# CORS 预检请求处理函数
def _build_cors_preflight_response():
    response = app.make_default_options_response()
    headers = response.headers
    # Allow specific headers and methods needed for your frontend
    headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', '*')
    headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    headers['Access-Control-Allow-Credentials'] = 'true'
    return response

# 添加一个健康检查端点
@app.route('/api/health', methods=['GET'])
def health_check():
    """API服务健康检查"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0"
    })

# 在app.py中添加
@app.route('/dashboard')
def dashboard():
    return send_from_directory('./', 'dashboard.html')

# @app.route('/')
# def index():
#     return send_from_directory('./', 'dashboard.html')

@app.route('/api/predict', methods=['POST'])
@cache.cached(timeout=CACHE_TIMES['predict'], make_cache_key=make_cache_key)
def run_prediction():
    """执行负荷、光伏或风电预测并返回结果"""
    global latest_prediction_result

    try:
        # 获取前端传过来的参数
        original_params = request.get_json()
        logger.info(f"收到预测请求，原始参数: {original_params}")
        
        # === 调试代码：记录天气感知参数 ===
        weather_aware_param = original_params.get('weatherAware', 'NOT_SET')
        forecast_type_param = original_params.get('forecastType', 'NOT_SET')
        prediction_type_param = original_params.get('predictionType', 'NOT_SET')
        logger.warning(f"[DEBUG] 普通预测API接收参数 - weatherAware: {weather_aware_param}, forecastType: {forecast_type_param}, predictionType: {prediction_type_param}")
        
        # 写入调试文件
        debug_entry = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "api": "普通预测API (/api/predict)",
            "weatherAware": weather_aware_param,
            "forecastType": forecast_type_param,
            "predictionType": prediction_type_param,
            "all_params": original_params
        }
        try:
            with open('debug_api_params.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps(debug_entry, ensure_ascii=False, indent=2) + '\n\n')
        except Exception as debug_err:
            logger.warning(f"写入调试日志失败: {debug_err}")
        # === 调试代码结束 ===

        # --- 参数提取和验证 ---
        prediction_type = original_params.get('predictionType', 'day-ahead') # day-ahead or rolling
        base_forecast_type = original_params.get('forecastType', 'load') # load, pv, wind
        province = original_params.get('province')
        calculate_net_load = original_params.get('calculate_net_load', False) and base_forecast_type == 'load'
        
        if not province:
            cache.delete_memoized(run_prediction)
            return jsonify({"success": False, "error": "缺少省份参数(province)"}), 400

        # Function to execute a single forecast and get results path
        def execute_single_forecast(params_override):
            current_params = {**original_params, **params_override}
            current_forecast_type = current_params.get('forecastType')
            
            # --- 参数提取和验证 (部分重复，但为了独立调用而保留) ---
            historical_days_str = current_params.get('historicalDays')
            if historical_days_str is not None:
                try:
                    historical_days_val = int(historical_days_str)
                    if not (1 <= historical_days_val <= 30): # Adjusted range as per existing validation
                        raise ValueError("历史数据天数必须在1-30之间")
                except (ValueError, TypeError):
                    raise ValueError("历史数据天数必须是有效整数")

        # --- 构造命令行参数 ---
            script_path = 'scripts/scene_forecasting.py'
            cmd_args = ['--mode', 'forecast']
            cmd_args.extend(['--forecast_type', current_forecast_type])
            cmd_args.extend(['--province', province])
            
            model_type_script = current_params.get('modelType', 'torch')
            # cmd_args.extend(['--model', model_type_script]) # modelType is not used by scene_forecasting.py yet

            is_probabilistic_script = current_params.get('probabilistic', False)
            if is_probabilistic_script:
                cmd_args.extend(['--probabilistic'])
                if current_params.get('quantiles'):
                    quantiles_str_script = ",".join(map(str, current_params['quantiles'])) if isinstance(current_params['quantiles'], list) else str(current_params['quantiles'])
                    cmd_args.extend(['--quantiles', quantiles_str_script])
            
            # --- 添加天气感知预测参数处理 ---
            weather_aware = current_params.get('weatherAware', False)
            if weather_aware and current_forecast_type in ['load', 'pv', 'wind']:
                cmd_args.extend(['--weather_aware'])
                
                # 处理天气特征参数
                weather_features = current_params.get('weatherFeatures')
                if weather_features:
                    if isinstance(weather_features, list):
                        weather_features_str = ','.join(weather_features)
                    else:
                        weather_features_str = str(weather_features)
                    cmd_args.extend(['--weather_features', weather_features_str])
                
                # 处理天气数据路径（通常由脚本自动构建，但也可以手动指定）
                weather_data_path = current_params.get('weatherDataPath')
                if weather_data_path:
                    cmd_args.extend(['--weather_data_path', weather_data_path])
                
                # 处理天气模型目录路径
                weather_model_dir = current_params.get('weatherModelDir')
                if weather_model_dir:
                    cmd_args.extend(['--weather_model_dir', weather_model_dir])
            
            current_prediction_type = current_params.get('predictionType', 'day-ahead')
            if current_prediction_type == 'day-ahead':
                cmd_args.extend(['--day_ahead'])
                if current_params.get('forecastDate'):
                    cmd_args.extend(['--forecast_date', current_params['forecastDate']])
                if current_params.get('forecastEndDate'):
                    cmd_args.extend(['--forecast_end_date', current_params['forecastEndDate']])
                if not is_probabilistic_script and current_params.get('enhancedSmoothing'):
                    cmd_args.extend(['--enhanced_smoothing'])
                    if current_params.get('maxDiffPct') is not None: cmd_args.extend(['--max_diff_pct', str(current_params['maxDiffPct'])])
                    if current_params.get('smoothingWindow') is not None: cmd_args.extend(['--smoothing_window', str(current_params['smoothingWindow'])])
            elif current_prediction_type == 'rolling': # 滚动预测
                if is_probabilistic_script: raise ValueError("滚动概率预测当前不支持") # 保持这个检查
                if current_params.get('startDate'): cmd_args.extend(['--forecast_start', current_params['startDate']])
                if current_params.get('endDate'): cmd_args.extend(['--forecast_end', current_params['endDate']])
                if current_params.get('interval'): cmd_args.extend(['--interval', str(current_params['interval'])])
                
                # 仅当预测类型为负荷时，才处理实时校正和PID修正参数
                if current_forecast_type == 'load':
                    if current_params.get('realTimeAdjustment'): 
                        cmd_args.extend(['--real_time_adjustment'])
                    
                    # 处理PID修正参数
                    if current_params.get('enablePidCorrection'):
                        cmd_args.append('--enable_pid_correction')
                        if current_params.get('pretrainDays') is not None:
                            cmd_args.extend(['--pretrain_days', str(current_params['pretrainDays'])])
                        if current_params.get('windowSizeHours') is not None:
                            cmd_args.extend(['--window_size_hours', str(current_params['windowSizeHours'])])
                        if current_params.get('enableAdaptation') is not None: # 布尔值，存在即代表True，脚本侧用 action='store_true'
                            if current_params.get('enableAdaptation'): # 只有为True时才添加
                                cmd_args.append('--enable_adaptation')
                        # Add initial PID parameters if they exist in current_params
                        if current_params.get('initialKp') is not None:
                            cmd_args.extend(['--initial_kp', str(current_params['initialKp'])])
                        if current_params.get('initialKi') is not None:
                            cmd_args.extend(['--initial_ki', str(current_params['initialKi'])])
                        if current_params.get('initialKd') is not None:
                            cmd_args.extend(['--initial_kd', str(current_params['initialKd'])])
            else: # 如果 predictionType 不是 'day-ahead' 或 'rolling'
                pass # 或者抛出错误

            if current_params.get('historicalDays') is not None:
                cmd_args.extend(['--historical_days', str(current_params['historicalDays'])])

            timestamp_script = datetime.now().strftime('%Y%m%d_%H%M%S%f') # Added microseconds for uniqueness
            results_base_dir_script = f"results/{'probabilistic' if is_probabilistic_script else ('day_ahead' if current_prediction_type == 'day-ahead' else 'rolling')}/{current_forecast_type}"
            results_province_dir_script = f"{results_base_dir_script}/{province}"
            os.makedirs(results_province_dir_script, exist_ok=True)
            # Ensure unique filenames for concurrent calls if any in future (though not strictly needed here as calls are sequential for net load)
            results_json_path = f"{results_province_dir_script}/forecast_{current_forecast_type}_{timestamp_script}.json"
            cmd_args.extend(['--output_json', results_json_path])
            
            cmd = build_python_command(script_path, cmd_args)
            logger.info(f"执行内部预测命令 for {current_forecast_type}: {' '.join(cmd)}")
            
            return_code, stdout, stderr = run_process_with_timeout(cmd, timeout=600)
            
            logger.info(f"{current_forecast_type} 预测脚本完成，返回码: {return_code}")
            if stdout: logger.info(f"{current_forecast_type} STDOUT:\n{stdout}")
            if stderr: logger.warning(f"{current_forecast_type} STDERR:\n{stderr}")

            file_created_script = os.path.exists(results_json_path)
            if return_code != 0 or not file_created_script:
                error_msg_script = f"{current_forecast_type.upper()}预测脚本执行失败 (代码: {return_code})."
                if not file_created_script and return_code == 0: error_msg_script = f"{current_forecast_type.upper()}预测脚本执行成功但未创建结果文件."
                # Attempt to get more details from stderr or stdout
                details_script = stderr or stdout or "无详细错误输出。"
                logger.error(f"{error_msg_script} Details: {details_script}")
                raise RuntimeError(f"{error_msg_script} Details: {details_script}")
            
            return results_json_path

        # --- Main Gross Load Forecast ---
        try:
            main_execute_forecast_type = 'load' if calculate_net_load else base_forecast_type
            results_json_load = execute_single_forecast({'forecastType': 'load'})
            latest_prediction_result = results_json_load # Update global var for gross load
        except Exception as e:
            logger.error(f"总负荷预测执行失败: {e}")
            cache.delete_memoized(run_prediction) # Ensure cache is cleared on error
            return jsonify({"success": False, "error": f"总负荷预测失败: {str(e)}"}), 500

        # --- Read Gross Load Results ---
        try:
            with open(results_json_load, 'r', encoding='utf-8') as f:
                main_result_data = json.load(f)
            if main_result_data.get('status') == 'error':
                cache.delete_memoized(run_prediction)
                return jsonify({"success": False, "error": main_result_data.get('error', '总负荷预测返回错误状态')}), 500
        except Exception as e:
            logger.error(f"读取或解析总负荷预测JSON失败: {e}")
            cache.delete_memoized(run_prediction)
            return jsonify({"success": False, "error": f"读取总负荷结果失败: {results_json_load}", "details": str(e)}), 500

        # --- Net Load Calculation (if requested) ---
        if calculate_net_load: # calculate_net_load 为 True 意味着 base_forecast_type 必须是 'load'
            logger.info("开始计算净负荷...")
            df_load = pd.DataFrame(main_result_data.get('predictions', []))
            if df_load.empty or 'datetime' not in df_load.columns or 'predicted' not in df_load.columns:
                logger.error("总负荷预测数据格式不正确或为空，无法计算净负荷。")
            else:
                df_load['datetime'] = pd.to_datetime(df_load['datetime'])
                df_load.set_index('datetime', inplace=True)
                df_load['predicted_gross'] = df_load['predicted'].copy()

                df_pv = pd.DataFrame()
                try:
                    # 强制非概率，非PID，非实时校正的简单PV预测
                    results_json_pv = execute_single_forecast({
                        'forecastType': 'pv', 
                        'probabilistic': False, 
                        'quantiles': None,
                        'realTimeAdjustment': False, 
                        'enablePidCorrection': False 
                    })
                    with open(results_json_pv, 'r', encoding='utf-8') as f:
                        pv_data = json.load(f)
                    if pv_data.get('status') == 'success' and pv_data.get('predictions'):
                        df_pv = pd.DataFrame(pv_data['predictions'])
                        df_pv['datetime'] = pd.to_datetime(df_pv['datetime'])
                        df_pv.set_index('datetime', inplace=True)
                        logger.info("光伏预测成功获取并加载。")
                    else:
                        logger.warning(f"光伏预测未成功或无数据: {pv_data.get('error', '未知错误')}")
                except Exception as e:
                    logger.warning(f"光伏预测执行或读取失败: {e}. 光伏出力将视为0。")
                
                df_wind = pd.DataFrame()
                try:
                     # 强制非概率，非PID，非实时校正的简单Wind预测
                    results_json_wind = execute_single_forecast({
                        'forecastType': 'wind', 
                        'probabilistic': False, 
                        'quantiles': None,
                        'realTimeAdjustment': False,
                        'enablePidCorrection': False
                    })
                    with open(results_json_wind, 'r', encoding='utf-8') as f:
                        wind_data = json.load(f)
                    if wind_data.get('status') == 'success' and wind_data.get('predictions'):
                        df_wind = pd.DataFrame(wind_data['predictions'])
                        df_wind['datetime'] = pd.to_datetime(df_wind['datetime'])
                        df_wind.set_index('datetime', inplace=True)
                        logger.info("风电预测成功获取并加载。")
                    else:
                        logger.warning(f"风电预测未成功或无数据: {wind_data.get('error', '未知错误')}")
                except Exception as e:
                    logger.warning(f"风电预测执行或读取失败: {e}. 风电出力将视为0。")

                if not df_pv.empty and 'predicted' in df_pv.columns:
                    df_load = df_load.merge(df_pv[['predicted']].rename(columns={'predicted':'pv_predicted'}), 
                                            left_index=True, right_index=True, how='left')
                else:
                    df_load['pv_predicted'] = 0
                df_load['pv_predicted'].fillna(0, inplace=True)

                if not df_wind.empty and 'predicted' in df_wind.columns:
                    df_load = df_load.merge(df_wind[['predicted']].rename(columns={'predicted':'wind_predicted'}), 
                                            left_index=True, right_index=True, how='left')
                else:
                    df_load['wind_predicted'] = 0
                df_load['wind_predicted'].fillna(0, inplace=True)
                
                df_load['predicted_net'] = df_load['predicted_gross'] - df_load['pv_predicted'] - df_load['wind_predicted']
                df_load['predicted_net'] = df_load['predicted_net'].clip(lower=0)
                df_load['predicted'] = df_load['predicted_net']
                
                try:
                    df_pv_actual_full = pd.read_csv(f"data/timeseries_pv_{province}.csv", index_col=0, parse_dates=True)
                    df_pv_actual_full.index.name = 'datetime'
                    df_pv_actual = df_pv_actual_full[['pv']].reindex(df_load.index).rename(columns={'pv': 'pv_actual'})
                    df_load = df_load.merge(df_pv_actual, on='datetime', how='left')
                    logger.info(f"成功加载并合并 {df_pv_actual['pv_actual'].notna().sum()} 条实际光伏数据。")
                except FileNotFoundError:
                    logger.warning(f"未找到实际光伏数据文件: data/timeseries_pv_{province}.csv.")
                    df_load['pv_actual'] = np.nan
                except Exception as e:
                    logger.error(f"加载实际光伏数据时出错: {e}")
                    df_load['pv_actual'] = np.nan

                try:
                    df_wind_actual_full = pd.read_csv(f"data/timeseries_wind_{province}.csv", index_col=0, parse_dates=True)
                    df_wind_actual_full.index.name = 'datetime'
                    df_wind_actual = df_wind_actual_full[['wind']].reindex(df_load.index).rename(columns={'wind': 'wind_actual'})
                    df_load = df_load.merge(df_wind_actual, on='datetime', how='left')
                    logger.info(f"成功加载并合并 {df_wind_actual['wind_actual'].notna().sum()} 条实际风电数据。")
                except FileNotFoundError:
                    logger.warning(f"未找到实际风电数据文件: data/timeseries_wind_{province}.csv.")
                    df_load['wind_actual'] = np.nan
                except Exception as e:
                    logger.error(f"加载实际风电数据时出错: {e}")
                    df_load['wind_actual'] = np.nan
                
                df_load.reset_index(inplace=True)
                df_load['datetime'] = df_load['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S')
                
                output_predictions = []
                for _, row in df_load.iterrows():
                    pred_point = {
                        'datetime': row['datetime'],
                        'predicted': row.get('predicted'),
                        'actual': row.get('actual'),
                        'is_peak': row.get('is_peak')
                    }
                    if 'predicted_gross' in row and pd.notna(row['predicted_gross']): 
                        pred_point['predicted_gross'] = row['predicted_gross']
                    if 'pv_predicted' in row and pd.notna(row['pv_predicted']): 
                        pred_point['pv_predicted'] = row['pv_predicted']
                    if 'wind_predicted' in row and pd.notna(row['wind_predicted']): 
                        pred_point['wind_predicted'] = row['wind_predicted']
                    if 'pv_actual' in row and pd.notna(row['pv_actual']): 
                        pred_point['pv_actual'] = row['pv_actual']
                    if 'wind_actual' in row and pd.notna(row['wind_actual']): 
                        pred_point['wind_actual'] = row['wind_actual']
                    output_predictions.append(pred_point)

                main_result_data['predictions'] = output_predictions
                main_result_data['is_net_load'] = True
                main_result_data['forecast_type'] = 'net_load'
                logger.info("净负荷计算完成")

        # --- Final Response ---
        is_cached_flag_name = f"cached_for_{main_execute_forecast_type}" # Dynamic flag name
        is_cached = getattr(run_prediction, is_cached_flag_name, False)
        cache_info = None
        if is_cached:
            increment_cache_hit('predict')
            cache_info = {"hit_count": cache_hits['predict'], "timestamp": datetime.now().isoformat()}
            logger.info(f"缓存命中 (基于 {main_execute_forecast_type}): 预测结果, 总命中数: {cache_hits['predict']}")
        else:
            setattr(run_prediction, is_cached_flag_name, True) # Set the dynamic flag
        
        response_data = {
            "success": True,
            "data": main_result_data,
            "cached": is_cached,
            "cache_info": cache_info
        }
        return jsonify(response_data)

    except ValueError as ve:
        logger.error(f"参数验证错误: {ve}")
        cache.delete_memoized(run_prediction)
        return jsonify({"success": False, "error": str(ve)}), 400
    except RuntimeError as rterr:
        logger.error(f"预测脚本执行时发生运行时错误: {rterr}")
        cache.delete_memoized(run_prediction)
        return jsonify({"success": False, "error": str(rterr)}), 500
    except Exception as e:
        logger.error(f"预测API端点发生未捕获的错误: {str(e)}")
        import traceback
        traceback.print_exc()
        cache.delete_memoized(run_prediction)
        return jsonify({"success": False, "error": "服务器内部错误", "details": str(e)}), 500


@app.route('/api/provinces', methods=['GET'])
@cache.cached(timeout=CACHE_TIMES['provinces'])  # 使用预定义的缓存时间
def get_provinces():
    """获取可用的省份列表 (现在检查所有类型的数据)"""
    
    # 检查请求是否来自缓存
    is_cached = getattr(get_provinces, 'cached', False)
    if is_cached:
        increment_cache_hit('provinces')
        print(f"缓存命中: 省份列表, 总命中数: {cache_hits['provinces']}")
    else:
        # 首次调用，标记此函数
        get_provinces.cached = True
    
    provinces = set() # Use a set to avoid duplicates
    try:
        data_dir = "data"
        if not os.path.isdir(data_dir):
             return jsonify({"success": True, "data": [], "provinces": []}) # Return empty if data dir doesnt exist

        for f in os.listdir(data_dir):
            if f.endswith('.csv'):
                parts = f.replace('.csv', '').split('_')
                # Expecting format like timeseries_TYPE_PROVINCE
                if len(parts) == 3 and parts[0] == 'timeseries' and parts[1] in ['load', 'pv', 'wind']:
                    provinces.add(parts[2]) # Add province name

        # 如果没有找到任何省份，提供默认列表
        province_list = sorted(list(provinces))
        if not province_list:
            province_list = ['上海', '福建', '江苏', '浙江', '安徽']
            print("未从数据目录找到省份，使用默认省份列表")

        return jsonify({
            "success": True,
            "data": province_list, # 保留原先的 data 字段
            "provinces": province_list, # 添加 provinces 字段以匹配前端期望
            "cached": is_cached,  # 标记是否是缓存结果
            "cache_info": {
                "hit_count": cache_hits['provinces'],
                "timestamp": datetime.now().isoformat()
            } if is_cached else None
        })
    except Exception as e:
        print(f"获取省份列表时出错: {e}")
        # 不要缓存错误响应
        cache.delete_memoized(get_provinces)
        return jsonify({
            "success": False,
            "error": "获取省份列表失败",
            "details": str(e)
        }), 500


@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """清除缓存内容 (不再强制要求JSON body)"""
    try:
        # 不再需要解析JSON，直接执行清除操作
        # params = request.get_json() or {}
        # cache_type = params.get('cache_type')

        # 目前只支持清除所有缓存
        cache.clear()
        # 重置所有缓存命中计数
        for key in cache_hits:
            cache_hits[key] = 0
        message = "已清除所有缓存"
            
        return jsonify({
            "success": True,
            "message": message
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "清除缓存失败",
            "details": str(e)
        }), 500

@app.route('/api/cache-stats', methods=['GET'])
def get_cache_statistics():
    """获取缓存使用统计信息"""
    try:
        stats = get_cache_stats()
        return jsonify({
            "success": True,
            "data": {
                "hits": stats['hits'],
                "total_hits": stats['total'],
                "configuration": stats['config'],
                "timestamp": datetime.now().isoformat()
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "获取缓存统计信息失败",
            "details": str(e)
        }), 500

@app.route('/api/train', methods=['POST', 'OPTIONS'])
def train_model():
    # Handle CORS preflight request for POST
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()

    print("接收到训练请求")
    try:
        # Extract parameters
        params = request.get_json()
        print(f"训练参数: {params}")
        
        # Parse and validate
        forecast_type = params.get('forecast_type', 'load')  # Default to load if not specified
        province = params.get('province')
        if not province:
            # 如果省份为空，默认使用上海而不是返回错误
            province = "上海"
            print(f"省份参数为空，自动设置为默认值: {province}")
            params['province'] = province
        
        # Create a unique task ID
        task_id = f"train_{forecast_type}_{province}_{int(time.time())}"
        
        # Add task to our tracking dict
        training_tasks[task_id] = {
            'status': 'running',
            'progress': 0,
            'currentEpoch': 0,
            'totalEpochs': params.get('epochs', 100),
            'trainLoss': [],
            'valLoss': [],
            'logs': [],
            'error': None,
            'eta': '计算中...'
        }
        
        # Define the training task function
        def run_training_task(task_id, params):
            task_info = training_tasks[task_id]
            try:
                # --- Build command with all parameters using helper ---
                script_path = 'scripts/scene_forecasting.py'
                cmd_args = ['--mode', 'train']
                
                # Extract parameters
                forecast_type = params.get('forecast_type', 'load')
                province = params.get('province')
                train_start = params.get('train_start')
                train_end = params.get('train_end')
                epochs = params.get('epochs', 100)
                batch_size = params.get('batch_size', 32)
                learning_rate = params.get('learning_rate', 0.0001)
                retrain = params.get('retrain', False)
                peak_aware = params.get('peak_aware', False)
                # --- 新增：获取训练目标类型 --- 
                train_prediction_type = params.get('train_prediction_type', 'deterministic')
                # --- 新增：获取天气感知训练参数 ---
                weather_aware = params.get('weather_aware', False)
                weather_features = params.get('weather_features')
                weather_data_path = params.get('weather_data_path')
                weather_model_dir = params.get('weather_model_dir')
                # ------------------------------
                
                # Add basic parameters
                cmd_args.extend(['--forecast_type', forecast_type])
                cmd_args.extend(['--province', province])
                # --- 新增：传递训练目标类型参数 ---
                cmd_args.extend(['--train_prediction_type', train_prediction_type])
                # --------------------------------
                
                # --- 添加天气感知训练参数 ---
                if weather_aware and forecast_type in ['load', 'pv', 'wind']:
                    cmd_args.extend(['--weather_aware'])
                    
                    # 处理天气特征参数
                    if weather_features:
                        if isinstance(weather_features, list):
                            weather_features_str = ','.join(weather_features)
                        else:
                            weather_features_str = str(weather_features)
                        cmd_args.extend(['--weather_features', weather_features_str])
                    
                    # 处理天气数据路径
                    if weather_data_path:
                        cmd_args.extend(['--weather_data_path', weather_data_path])
                    
                    # 处理天气模型目录路径
                    if weather_model_dir:
                        cmd_args.extend(['--weather_model_dir', weather_model_dir])
                
                # Add optional parameters
                if train_start:
                    cmd_args.extend(['--train_start', train_start])
                if train_end:
                    cmd_args.extend(['--train_end', train_end])
                if epochs:
                    cmd_args.extend(['--epochs', str(epochs)])
                if batch_size:
                    cmd_args.extend(['--batch_size', str(batch_size)])
                if learning_rate:
                    cmd_args.extend(['--lr', str(learning_rate)])
                if retrain:
                    cmd_args.extend(['--retrain'])
                
                # --- 根据训练类型添加特定参数 ---
                if train_prediction_type == 'deterministic':
                    if peak_aware and forecast_type == 'load': # 峰谷感知仅用于负荷确定性预测
                        cmd_args.extend(['--peak_aware'])
                        if 'peak_start' in params: cmd_args.extend(['--peak_start', str(params['peak_start'])])
                        if 'peak_end' in params: cmd_args.extend(['--peak_end', str(params['peak_end'])])
                        if 'valley_start' in params: cmd_args.extend(['--valley_start', str(params['valley_start'])])
                        if 'valley_end' in params: cmd_args.extend(['--valley_end', str(params['valley_end'])])
                        if 'peak_weight' in params: cmd_args.extend(['--peak_weight', str(params['peak_weight'])])
                        if 'valley_weight' in params: cmd_args.extend(['--valley_weight', str(params['valley_weight'])])
                elif train_prediction_type == 'probabilistic':
                    # 添加概率预测所需的分位数参数
                    if params.get('quantiles'):
                         quantiles_str = ",".join(map(str, params['quantiles'])) if isinstance(params['quantiles'], list) else str(params['quantiles'])
                         cmd_args.extend(['--quantiles', quantiles_str])
                    # 概率预测通常也使用 peak_aware 特征，但不一定使用 peak loss
                    if peak_aware and forecast_type == 'load': 
                         cmd_args.extend(['--peak_aware']) # 传递峰谷感知标志，让训练脚本决定是否用于损失
                elif train_prediction_type == 'interval':
                    # 区间预测通常基于峰谷感知模型
                    if peak_aware and forecast_type == 'load':
                         cmd_args.extend(['--peak_aware'])
                    # 区间预测可能需要置信水平等参数，但通常在预测时指定
                    # 训练时可能需要特定的参数，如果 train_interval_model 需要的话在这里添加
                    pass # 暂无区间训练特有的参数需要从API传递
                # -------------------------------------
                
                # Add historical_days parameter if provided
                if params.get('historical_days') is not None:
                    cmd_args.extend(['--historical_days', str(params['historical_days'])])

                cmd = build_python_command(script_path, cmd_args)
                # --- End building command ---

                print(f"执行训练命令: {' '.join(cmd)}")
                
                # Start training process
                # 使用 Popen 并实时读取输出
                process = subprocess.Popen(cmd, 
                                         stdout=subprocess.PIPE, 
                                         stderr=subprocess.STDOUT, # 合并标准输出和错误
                                         text=True, 
                                         encoding='utf-8', 
                                         errors='replace',
                                         bufsize=1,  # 行缓冲
                                         universal_newlines=True)
                
                # Read the process output line by line in real-time
                for log_line in process.stdout:
                    log_line_stripped = log_line.strip()
                    if log_line_stripped:  # 避免打印空行
                        print(f"训练日志: {log_line_stripped}")  # 实时打印到后端控制台
                        task_info['logs'].append(log_line_stripped)  # 同时保存到任务信息中
                    
                    # 改进Epoch进度解析
                    epoch_match = re.search(r"Epoch (\d+)/(\d+)\s*\|\s*Train Loss:\s*([\d\.e\-]+)\s*\|\s*Val Loss:\s*([\d\.e\-]+)", log_line_stripped)
                    if epoch_match:
                        current_epoch = int(epoch_match.group(1))
                        total_epochs = int(epoch_match.group(2))
                        train_loss = float(epoch_match.group(3))
                        val_loss = float(epoch_match.group(4))
                        
                        task_info['currentEpoch'] = current_epoch
                        task_info['totalEpochs'] = total_epochs
                        task_info['trainLoss'].append(train_loss)
                        task_info['valLoss'].append(val_loss)
                        task_info['progress'] = int((current_epoch / total_epochs) * 100)
                        
                        # 计算预计完成时间
                        elapsed_time = time.time() - task_info.get('start_time', time.time())
                        if current_epoch > 0:
                            time_per_epoch = elapsed_time / current_epoch
                            remaining_epochs = total_epochs - current_epoch
                            eta_seconds = time_per_epoch * remaining_epochs
                            eta = f"{int(eta_seconds // 60)}分{int(eta_seconds % 60)}秒"
                            task_info['eta'] = eta
                    
                    # Check for loss values
                    loss_match = re.search(r"Train Loss: (\d+\.\d+)\s*\|\s*Val Loss: (\d+\.\d+)", log_line_stripped)
                    if loss_match:
                        task_info['trainLoss'].append(float(loss_match.group(1)))
                        task_info['valLoss'].append(float(loss_match.group(2)))
                
                process.stdout.close()
                return_code = process.wait()
                
                if return_code == 0:
                    task_info['status'] = 'completed'
                    task_info['progress'] = 100
                    print(f"训练任务 {task_id} 完成")
                else:
                    task_info['status'] = 'failed'
                    task_info['error'] = f"训练进程异常退出，返回码: {return_code}"
                    print(f"训练任务 {task_id} 失败: {task_info['error']}")
                
            except Exception as e:
                task_info['status'] = 'failed'
                task_info['error'] = str(e)
                print(f"训练任务 {task_id} 异常: {e}")
                import traceback
                task_info['logs'].append(traceback.format_exc())
        
        # Start training in a background thread
        import threading
        task_info = training_tasks[task_id]
        task_info['start_time'] = time.time()
        thread = threading.Thread(target=run_training_task, args=(task_id, params))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "success": True,
            "message": f"训练任务已启动: {forecast_type} 模型 (省份: {province})",
            "task_id": task_id
        })
    except Exception as e:
        print(f"启动训练任务错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "服务器内部错误",
            "details": str(e)
        }), 500

@app.route('/api/training-status/<task_id>', methods=['GET'])
def get_training_status(task_id):
    """获取训练任务的状态"""
    try:
        if task_id in training_tasks:
            return jsonify({
                "success": True,
                "data": training_tasks[task_id]
            })
        else:
            return jsonify({
                "success": False,
                "error": "找不到指定的训练任务"
            }), 404
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "获取训练状态失败",
            "details": str(e)
        }), 500
    
@app.route('/api/integrated-forecast', methods=['POST'])
@cache.cached(timeout=CACHE_TIMES['integrated'], key_prefix=make_cache_key)  # 使用预定义的缓存时间
def integrated_forecast():
    """执行区域多源预测（负荷、光伏、风电）并返回统一结果"""
    
    # 检查请求是否来自缓存
    is_cached = getattr(integrated_forecast, 'cached', False)
    if is_cached:
        increment_cache_hit('integrated')
        print(f"缓存命中: 集成预测, 总命中数: {cache_hits['integrated']}")
    else:
        # 首次调用，标记此函数
        integrated_forecast.cached = True
    
    try:
        params = request.get_json()
        province = params.get('province')
        date = params.get('date')
        
        if not province or not date:
            # 不缓存错误响应
            cache.delete_memoized(integrated_forecast)
            return jsonify({"success": False, "error": "缺少省份或日期参数"}), 400
            
        # 检查是否有缓存文件可用
        results_dir = f"results/integrated/{province}"
        integrated_json = f"{results_dir}/integrated_{date}.json"
        
        # 检查是否需要强制刷新
        force_refresh = params.get('force_refresh', False)
        if force_refresh:
            print(f"强制刷新请求，绕过缓存: 集成预测-{province}-{date}")
        
        # 如果文件已存在且不是强制刷新请求，直接返回文件内容
        if os.path.exists(integrated_json) and not force_refresh:
            try:
                with open(integrated_json, 'r') as f:
                    integrated_result = json.load(f)
                
                # 填充缓存信息
                cache_info = None
                if is_cached:
                    cache_info = {
                        "hit_count": cache_hits['integrated'],
                        "timestamp": datetime.now().isoformat()
                    }
                    
                return jsonify({
                    "success": True, 
                    "data": integrated_result,
                    "cached": is_cached,  # 标记是否为缓存结果
                    "cache_info": cache_info
                })
            except Exception as read_err:
                print(f"读取集成预测缓存文件失败: {read_err}")
                # 如果读取缓存失败，继续执行预测
        
        # 创建结果目录
        os.makedirs(results_dir, exist_ok=True)
        
        # 并行执行三种预测
        forecasts = {}
        forecast_types = ['load', 'pv', 'wind']
        
        for forecast_type in forecast_types:
            # 构建命令
            cmd = ['python', 'scripts/scene_forecasting.py', '--mode', 'forecast',
                   '--forecast_type', forecast_type, '--province', province,
                   '--day_ahead', '--forecast_date', date, 
                   '--output_json', f"{results_dir}/{forecast_type}_{date}.json"]
            
            # 执行预测
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                      text=True, encoding='utf-8')
            process.communicate()
            
            # 读取结果
            try:
                with open(f"{results_dir}/{forecast_type}_{date}.json", 'r') as f:
                    forecasts[forecast_type] = json.load(f)
            except Exception as e:
                forecasts[forecast_type] = {"status": "failed", "error": str(e)}
        
        # 整合结果
        integrated_result = {
            "status": "success",
            "province": province,
            "date": date,
            "forecasts": forecasts,
            "timestamp": datetime.now().isoformat()
        }
        
        # 保存结果
        with open(integrated_json, 'w') as f:
            json.dump(integrated_result, f, indent=4)
            
        return jsonify({
            "success": True, 
            "data": integrated_result,
            "cached": False  # 标记这是一个新生成的结果
        })
        
    except Exception as e:
        # 不缓存错误响应
        cache.delete_memoized(integrated_forecast)
        return jsonify({"success": False, "error": str(e)}), 500
    
@app.route('/api/recognize-scenarios', methods=['POST'])
@cache.cached(timeout=CACHE_TIMES['scenarios'], key_prefix=make_cache_key)  # 使用预定义的缓存时间
def recognize_scenarios():
    """基于多源预测结果识别运行场景"""
    
    # 检查请求是否来自缓存
    is_cached = getattr(recognize_scenarios, 'cached', False)
    if is_cached:
        increment_cache_hit('scenarios')
        print(f"缓存命中: 场景识别, 总命中数: {cache_hits['scenarios']}")
    else:
        # 首次调用，标记此函数
        recognize_scenarios.cached = True
    
    try:
        params = request.get_json()
        province = params.get('province')
        date = params.get('date')
        
        if not province or not date:
            # 不缓存错误响应
            cache.delete_memoized(recognize_scenarios)
            return jsonify({"success": False, "error": "缺少省份或日期参数"}), 400
            
        # 检查是否有缓存文件可用
        results_dir = f"results/integrated/{province}"
        scenario_json = f"{results_dir}/scenarios_{date}.json"
        
        # 检查是否需要强制刷新
        force_refresh = params.get('force_refresh', False)
        if force_refresh:
            print(f"强制刷新请求，绕过缓存: 场景识别-{province}-{date}")
        
        # 如果文件已存在且不是强制刷新请求，直接返回文件内容
        if os.path.exists(scenario_json) and not force_refresh:
            try:
                with open(scenario_json, 'r') as f:
                    scenario_results = json.load(f)
                    
                # 填充缓存信息
                cache_info = None
                if is_cached:
                    cache_info = {
                        "hit_count": cache_hits['scenarios'],
                        "timestamp": datetime.now().isoformat()
                    }
                    
                return jsonify({
                    "success": True, 
                    "data": scenario_results,
                    "cached": is_cached,  # 标记是否为缓存结果
                    "cache_info": cache_info
                })
            except Exception as read_err:
                print(f"读取场景识别缓存文件失败: {read_err}")
                # 如果读取缓存失败，继续执行识别
        
        # 获取整合预测结果
        integrated_json = f"{results_dir}/integrated_{date}.json"
        
        if not os.path.exists(integrated_json):
            # 如果没有集成预测结果，先执行集成预测逻辑（不调用API函数）
            # 创建结果目录
            os.makedirs(results_dir, exist_ok=True)
            
            # 并行执行三种预测
            forecasts = {}
            forecast_types = ['load', 'pv', 'wind']
            
            for forecast_type in forecast_types:
                # 构建命令
                cmd = [sys.executable, '-u', 'scripts/scene_forecasting.py', '--mode', 'forecast',
                      '--forecast_type', forecast_type, '--province', province,
                      '--day_ahead', '--forecast_date', date, 
                      '--output_json', f"{results_dir}/{forecast_type}_{date}.json"]
                
                # 执行预测
                return_code, stdout, stderr = run_process_with_timeout(cmd, timeout=300)
                
                if return_code != 0:
                    print(f"{forecast_type}预测失败: {stderr}")
                    forecasts[forecast_type] = {"status": "failed", "error": stderr}
                else:
                    # 读取结果
                    try:
                        with open(f"{results_dir}/{forecast_type}_{date}.json", 'r', encoding='utf-8') as f:
                            forecasts[forecast_type] = json.load(f)
                    except Exception as e:
                        forecasts[forecast_type] = {"status": "failed", "error": str(e)}
            
            # 整合结果
            integrated_result = {
                "status": "success",
                "province": province,
                "date": date,
                "forecasts": forecasts,
                "timestamp": datetime.now().isoformat()
            }
            
            # 保存结果
            with open(integrated_json, 'w', encoding='utf-8') as f:
                json.dump(integrated_result, f, indent=4)
            
            print(f"集成预测结果已生成并保存至 {integrated_json}")
                
        # 加载预测结果
        try:
            with open(integrated_json, 'r', encoding='utf-8') as f:
                integrated_data = json.load(f)
        except Exception as e:
            # 不缓存错误响应
            cache.delete_memoized(recognize_scenarios)
            return jsonify({"success": False, "error": f"读取集成预测结果失败: {str(e)}"}), 500
            
        # 准备数据
        try:
            load_df = pd.DataFrame(integrated_data['forecasts']['load']['predictions'])
            pv_df = pd.DataFrame(integrated_data['forecasts']['pv']['predictions'])
            wind_df = pd.DataFrame(integrated_data['forecasts']['wind']['predictions'])
        except Exception as e:
            # 不缓存错误响应
            cache.delete_memoized(recognize_scenarios)
            return jsonify({"success": False, "error": f"处理预测数据失败: {str(e)}"}), 500
        
        # 场景识别
        try:
            from utils.scenario_recognizer import ScenarioRecognizer
            recognizer = ScenarioRecognizer()
            scenario_df = recognizer.identify_scenarios(load_df, pv_df, wind_df)
            
            # 生成调节目标
            regulation_df = recognizer.get_regulation_targets(scenario_df, load_df, pv_df, wind_df)
            
            # 确保时间戳正确转换为字符串
            if 'datetime' in scenario_df.columns and pd.api.types.is_datetime64_any_dtype(scenario_df['datetime']):
                scenario_df['datetime'] = scenario_df['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S')
            
            if 'datetime' in regulation_df.columns and pd.api.types.is_datetime64_any_dtype(regulation_df['datetime']):
                regulation_df['datetime'] = regulation_df['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S')
            
            # 清理DataFrame中的NaN/Timestamp值，确保JSON序列化不会出错
            def clean_dataframe_for_json(df):
                df_clean = df.copy()
                # 处理所有列
                for col in df_clean.columns:
                    # 转换日期时间类型
                    if pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                        df_clean[col] = df_clean[col].dt.strftime('%Y-%m-%dT%H:%M:%S')
                    # 替换NaN/None值
                    if df_clean[col].dtype.kind in 'fc':  # 浮点或复数类型
                        df_clean[col] = df_clean[col].replace({np.nan: None, np.inf: None, -np.inf: None})
                return df_clean
            
            # 应用清理函数
            scenario_df_clean = clean_dataframe_for_json(scenario_df)
            regulation_df_clean = clean_dataframe_for_json(regulation_df)
            
            # 转换为可JSON序列化的字典
            scenario_dict = scenario_df_clean.to_dict(orient='records')
            regulation_dict = regulation_df_clean.to_dict(orient='records')
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            # 不缓存错误响应
            cache.delete_memoized(recognize_scenarios)
            return jsonify({"success": False, "error": f"场景识别或生成调节目标失败: {str(e)}"}), 500
        
        # 整合结果
        scenario_results = {
            "status": "success",
            "province": province,
            "date": date,
            "scenarios": scenario_dict,
            "regulation_targets": regulation_dict,
            "timestamp": datetime.now().isoformat()
        }
        
        # 保存结果
        with open(scenario_json, 'w', encoding='utf-8') as f:
            json.dump(scenario_results, f, ensure_ascii=False, indent=4)
            
        return jsonify({
            "success": True, 
            "data": scenario_results,
            "cached": False  # 标记这是一个新生成的结果
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        # 不缓存错误响应
        cache.delete_memoized(recognize_scenarios)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/interval-forecast', methods=['POST'])
@cache.cached(timeout=CACHE_TIMES['integrated'], make_cache_key=make_cache_key)
def api_interval_forecast():
    """
    区间预测API - 通过调用scene_forecasting.py脚本实现
    支持标准预测类型(load, pv, wind)以及净负荷预测(net_load)
    """
    try:
        # 获取请求参数
        params = request.json
        logger.info(f"区间预测API收到参数: {params}")
        
        # 强制刷新缓存检查
        force_refresh = params.get('force_refresh', False)
        if force_refresh:
            logger.info("强制刷新缓存请求")
            cache.delete_memoized(api_interval_forecast)
        
        # 参数验证和提取
        province = params.get('province', '上海')
        forecast_type = params.get('forecastType', params.get('forecast_type', 'load'))
        confidence_level = float(params.get('confidenceLevel', params.get('confidence_level', 0.9)))
        historical_days = int(params.get('historicalDays', params.get('historical_days', 14)))
        interval_minutes = int(params.get('interval', params.get('interval_minutes', 15)))
        
        # 支持两种日期参数命名规范
        forecast_start_date_str = params.get('forecastDate', params.get('forecast_date', 
                                         params.get('start_date', 
                                                (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'))))
        forecast_end_date_str = params.get('forecastEndDate', params.get('forecast_end_date', 
                                       params.get('end_date', forecast_start_date_str)))

        # 日期验证
        try:
            forecast_start_date_dt = pd.to_datetime(forecast_start_date_str).normalize()
            forecast_end_date_dt = pd.to_datetime(forecast_end_date_str).normalize()
            if forecast_end_date_dt < forecast_start_date_dt:
                logger.error(f"结束日期 {forecast_end_date_dt} 不能早于开始日期 {forecast_start_date_dt}")
                return jsonify({'success': False, 'error': '结束日期不能早于开始日期'}), 400
        except ValueError as e:
            logger.error(f"无效的日期格式: {e}")
            return jsonify({'success': False, 'error': f'无效的日期格式: {e}'}), 400

        # 置信水平验证
        if confidence_level <= 0.5 or confidence_level >= 1.0:
            logger.error(f"无效的置信水平: {confidence_level}")
            return jsonify({
                'success': False,
                'error': f'置信水平必须在0.5到1.0之间，当前值: {confidence_level}'
            }), 400

        logger.info(f"将预测以下日期范围: {forecast_start_date_str} 至 {forecast_end_date_str}")

        # 提取天气感知参数
        weather_aware = params.get('weather_aware', False)
        logger.info(f"天气感知参数 (区间预测): {weather_aware}")

        # 提取可再生能源增强预测参数
        renewable_enhanced = params.get('renewable_enhanced', False)
        enable_renewable_prediction = params.get('enable_renewable_prediction', True)
        logger.info(f"可再生能源增强预测参数: {renewable_enhanced}, 启用新能源预测: {enable_renewable_prediction}")

        # 处理净负荷预测特殊情况
        is_net_load = forecast_type == 'net_load'
        actual_forecast_type = 'load' if is_net_load else forecast_type

        # 创建输出目录和文件路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_base_dir = f"results/interval_forecast/{forecast_type}"
        results_province_dir = f"{results_base_dir}/{province}"
        os.makedirs(results_province_dir, exist_ok=True)
        
        # 为多日预测创建唯一的输出文件名
        if forecast_start_date_str == forecast_end_date_str:
            output_filename = f"interval_{forecast_type}_{province}_{forecast_start_date_str}_{timestamp}.json"
        else:
            output_filename = f"interval_{forecast_type}_{province}_{forecast_start_date_str}_to_{forecast_end_date_str}_{timestamp}.json"
        
        results_json_path = os.path.join(results_province_dir, output_filename)
        
        # 构建调用scene_forecasting.py的命令
        script_path = 'scripts/scene_forecasting.py'
        cmd_args = [
            '--mode', 'forecast',
            '--train_prediction_type', 'interval',
            '--day_ahead',
            '--forecast_type', str(actual_forecast_type),
            '--province', str(province),
            '--forecast_date', str(forecast_start_date_str),
            '--forecast_end_date', str(forecast_end_date_str),
            '--confidence_level', str(confidence_level),
            '--historical_days', str(historical_days),
            '--interval', str(interval_minutes),
            '--output_json', str(results_json_path)
        ]
        
        # 添加天气感知参数
        if weather_aware:
            if actual_forecast_type in ['load', 'pv', 'wind']:
                cmd_args.append('--weather_aware')
                logger.info(f"已为{actual_forecast_type}区间预测添加 --weather_aware 参数")
            else:
                logger.warning(f"天气感知功能支持负荷、光伏和风电预测，但收到了{actual_forecast_type}类型请求，已忽略。")
        
        # 添加可再生能源增强预测参数
        if renewable_enhanced:
            if actual_forecast_type == 'load':
                cmd_args.append('--renewable_enhanced')
                if enable_renewable_prediction:
                    # 默认行为是启用新能源预测，只有在明确启用时才添加参数
                    cmd_args.append('--enable_renewable_prediction')
                logger.info(f"已为{actual_forecast_type}区间预测添加可再生能源增强参数 (新能源预测: {enable_renewable_prediction})")
            else:
                logger.warning(f"可再生能源增强功能仅支持负荷预测，但收到了{actual_forecast_type}类型请求，已忽略。")
        
        # 如果强制刷新，添加相应参数（虽然scene_forecasting.py可能不直接支持，但可以通过清理缓存实现）
        if force_refresh:
            # 可以在这里添加清理相关缓存的逻辑
            pass
        
        cmd = build_python_command(script_path, cmd_args)
        logger.info(f"执行区间预测命令: {' '.join(cmd)}")
        
        # 执行脚本
        return_code, stdout, stderr = run_process_with_timeout(cmd, timeout=600)  # 增加超时时间到10分钟
        
        logger.info(f"区间预测脚本完成，返回码: {return_code}")
        if stdout: 
            logger.info(f"STDOUT:\n{stdout}")
        if stderr: 
            logger.warning(f"STDERR:\n{stderr}")

        # 检查执行结果
        file_created = os.path.exists(results_json_path)
        if return_code != 0 or not file_created:
            error_msg = f"区间预测脚本执行失败 (代码: {return_code})."
            if not file_created and return_code == 0: 
                error_msg = "区间预测脚本执行成功但未创建结果文件."
            
            # 尝试从stderr或stdout获取更多详细信息
            details = stderr or stdout or "无详细错误输出。"
            logger.error(f"{error_msg} Details: {details}")
            
            # 根据错误类型返回不同的HTTP状态码
            if "模型" in details or "找不到" in details:
                status_code = 404  # 模型或文件不存在
            elif "内存" in details or "超时" in details:
                status_code = 503  # 服务暂时不可用
            else:
                status_code = 500  # 通用服务器错误
                
            return jsonify({
                "success": False, 
                "error": error_msg, 
                "details": details
            }), status_code

        # 读取并返回结果
        try:
            with open(results_json_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            if result_data.get('status') == 'error':
                cache.delete_memoized(api_interval_forecast)
                return jsonify({
                    "success": False, 
                    "error": result_data.get('error', '区间预测返回错误状态')
                }), 500
            
            # 检查是否是区间预测结果
            if not result_data.get('is_interval_forecast', False):
                logger.warning("返回的结果不是区间预测格式")
            
            # 处理缓存信息
            is_cached = getattr(api_interval_forecast, 'cached', False)
            cache_info = None
            if is_cached:
                increment_cache_hit('integrated')
                cache_info = {
                    "hit_count": cache_hits['integrated'], 
                    "timestamp": datetime.now().isoformat()
                }
                logger.info(f"缓存命中: 区间预测结果, 总命中数: {cache_hits['integrated']}")
            else:
                api_interval_forecast.cached = True
            
            # 如果是净负荷预测，需要计算负荷减去光伏和风电的值
            if is_net_load:
                logger.info("处理净负荷预测计算...")
                
                # 获取同一时间段的光伏和风电预测
                pv_params = params.copy()
                pv_params['forecast_type'] = 'pv'
                wind_params = params.copy()
                wind_params['forecast_type'] = 'wind'
                
                # 执行光伏预测
                pv_script_args = cmd_args.copy()
                pv_script_args_idx_ftype = pv_script_args.index('--forecast_type') + 1 if '--forecast_type' in pv_script_args else -1
                pv_script_args_idx_out = pv_script_args.index('--output_json') + 1 if '--output_json' in pv_script_args else -1

                if pv_script_args_idx_ftype != -1: pv_script_args[pv_script_args_idx_ftype] = 'pv'
                pv_output_path = os.path.join(results_province_dir, f"interval_pv_{province}_{forecast_start_date_str}_to_{forecast_end_date_str}_{timestamp}.json")
                if pv_script_args_idx_out != -1: pv_script_args[pv_script_args_idx_out] = pv_output_path
                
                pv_cmd = build_python_command(script_path, pv_script_args)
                logger.info(f"执行光伏区间预测命令: {' '.join(pv_cmd)}")
                pv_return_code, pv_stdout, pv_stderr = run_process_with_timeout(pv_cmd, timeout=600)
                
                # 执行风电预测
                wind_script_args = cmd_args.copy()
                wind_script_args_idx_ftype = wind_script_args.index('--forecast_type') + 1 if '--forecast_type' in wind_script_args else -1
                wind_script_args_idx_out = wind_script_args.index('--output_json') + 1 if '--output_json' in wind_script_args else -1

                if wind_script_args_idx_ftype != -1: wind_script_args[wind_script_args_idx_ftype] = 'wind'
                wind_output_path = os.path.join(results_province_dir, f"interval_wind_{province}_{forecast_start_date_str}_to_{forecast_end_date_str}_{timestamp}.json")
                if wind_script_args_idx_out != -1: wind_script_args[wind_script_args_idx_out] = wind_output_path

                wind_cmd = build_python_command(script_path, wind_script_args)
                logger.info(f"执行风电区间预测命令: {' '.join(wind_cmd)}")
                wind_return_code, wind_stdout, wind_stderr = run_process_with_timeout(wind_cmd, timeout=600)
                
                # 检查光伏和风电预测是否成功
                pv_data_content = None
                wind_data_content = None
                
                if os.path.exists(pv_output_path):
                    with open(pv_output_path, 'r', encoding='utf-8') as f:
                        pv_data_content = json.load(f)
                else:
                    logger.warning("光伏预测数据不可用，将使用零值")
                
                if os.path.exists(wind_output_path):
                    with open(wind_output_path, 'r', encoding='utf-8') as f:
                        wind_data_content = json.load(f)
                else:
                    logger.warning("风电预测数据不可用，将使用零值")
                
                # 计算净负荷 = 负荷 - 光伏 - 风电
                current_predictions = result_data.get('predictions', [])
                
                # 创建时间戳到预测值的映射
                pv_map = {}
                if pv_data_content and 'predictions' in pv_data_content:
                    for p_pv in pv_data_content['predictions']:
                        pv_map[p_pv['datetime']] = p_pv
                
                wind_map = {}
                if wind_data_content and 'predictions' in wind_data_content:
                    for p_wind in wind_data_content['predictions']:
                        wind_map[p_wind['datetime']] = p_wind
                
                # 计算净负荷值并更新 predicted 字段
                for p_item in current_predictions:
                    dt = p_item['datetime']
                    load_value = p_item.get('predicted', 0)  # 总负荷的点预测
                    
                    pv_value = pv_map.get(dt, {}).get('predicted', 0) or 0
                    wind_value = wind_map.get(dt, {}).get('predicted', 0) or 0
                    
                    # 保存总负荷信息
                    p_item['predicted_gross'] = load_value
                    p_item['pv_predicted'] = pv_value  
                    p_item['wind_predicted'] = wind_value
                    
                    # 计算净负荷点预测并更新 predicted 字段
                    net_load_value = load_value - pv_value - wind_value
                    p_item['predicted'] = max(0, net_load_value)  # 确保不为负
                    
                    # 处理区间预测的上下界
                    load_lower = p_item.get('lower_bound', load_value * 0.9)
                    load_upper = p_item.get('upper_bound', load_value * 1.1)
                    
                    pv_lower = pv_map.get(dt, {}).get('lower_bound', pv_value * 0.9) or 0
                    pv_upper = pv_map.get(dt, {}).get('upper_bound', pv_value * 1.1) or 0
                    
                    wind_lower = wind_map.get(dt, {}).get('lower_bound', wind_value * 0.9) or 0
                    wind_upper = wind_map.get(dt, {}).get('upper_bound', wind_value * 1.1) or 0
                    
                    # 净负荷的区间计算：负荷区间 - 新能源区间（交叉计算）
                    p_item['lower_bound'] = max(0, load_lower - pv_upper - wind_upper)
                    p_item['upper_bound'] = max(0, load_upper - pv_lower - wind_lower)
                
                result_data['predictions'] = current_predictions
                result_data['forecast_type'] = 'net_load'
                result_data['description'] = '净负荷区间预测 (负荷 - 光伏 - 风电)'
                
                with open(results_json_path, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"净负荷计算完成，包含 {len(current_predictions)} 个预测点")

            # 标准化预测项：确保 predicted 字段包含正确的点预测值  
            # 修复：不需要复杂的字段转换，直接使用现有的 predicted 字段
            standardized_predictions = []
            for p_dict_item in result_data.get('predictions', []):
                new_p_item_dict = p_dict_item.copy()
                
                # 确保 predicted 字段存在且不为 None
                if 'predicted' not in new_p_item_dict or new_p_item_dict['predicted'] is None:
                    # 如果 predicted 字段缺失或为 None，尝试从其他字段获取
                    if 'point_forecast' in new_p_item_dict:
                        new_p_item_dict['predicted'] = new_p_item_dict['point_forecast']
                    elif 'net_load' in new_p_item_dict:
                        new_p_item_dict['predicted'] = new_p_item_dict['net_load']
                    elif 'predicted_gross' in new_p_item_dict:
                        new_p_item_dict['predicted'] = new_p_item_dict['predicted_gross']
                    else:
                        logger.warning(f"无法确定预测值，时间点: {new_p_item_dict.get('datetime', 'unknown')}")
                        new_p_item_dict['predicted'] = 0  # 设置默认值
                
                # 清理不需要的字段（可选）
                fields_to_remove = ['point_forecast', 'net_load']
                for field in fields_to_remove:
                    if field in new_p_item_dict:
                        del new_p_item_dict[field]
                
                standardized_predictions.append(new_p_item_dict)

            result_data['predictions'] = standardized_predictions

            # 使用标准化后的预测数据构建响应
            predictions_for_response_chart = standardized_predictions
            
            # 提取时间序列数据用于前端图表
            times = [p.get('datetime') for p in predictions_for_response_chart]
            point_forecasts_chart = [p.get('predicted') for p in predictions_for_response_chart] # Use 'predicted'
            lower_bounds_chart = [p.get('lower_bound') for p in predictions_for_response_chart] # Renamed
            upper_bounds_chart = [p.get('upper_bound') for p in predictions_for_response_chart] # Renamed
            actual_values_chart = [p.get('actual') for p in predictions_for_response_chart if 'actual' in p] # Renamed
            
            hit_rate = None
            if actual_values_chart and len(actual_values_chart) > 0: # Use renamed var
                hits = 0
                total_valid_actuals = 0 # Renamed for clarity
                for p_item_chart in predictions_for_response_chart: # Renamed loop var
                    if 'actual' in p_item_chart and p_item_chart['actual'] is not None:
                        actual_val = p_item_chart['actual'] # Renamed
                        lower_b = p_item_chart.get('lower_bound', float('-inf')) # Renamed
                        upper_b = p_item_chart.get('upper_bound', float('inf')) # Renamed
                        if lower_b <= actual_val <= upper_b:
                            hits += 1
                        total_valid_actuals += 1
                
                if total_valid_actuals > 0:
                    hit_rate = (hits / total_valid_actuals) * 100
                    logger.info(f"区间预测命中率: {hit_rate:.2f}%")
            
            response_data = {
                'success': True,
                'data': result_data, # result_data['predictions'] is now standardized
                'cached': is_cached,
                'cache_info': cache_info,
                'times': times,
                'prediction': point_forecasts_chart,  # Sourced from 'predicted'
                'lower_bound': lower_bounds_chart,
                'upper_bound': upper_bounds_chart,
                'confidence_level': confidence_level,
                'forecast_type': params.get('forecastType', params.get('forecast_type', 'load')), # Use originally requested type
                'province': province
            }
            
            if 'metrics' not in response_data['data'] or not isinstance(response_data['data']['metrics'], dict):
                response_data['data']['metrics'] = {}
            
            script_metrics = result_data.get('metrics', {})
            response_data['data']['metrics'].update(script_metrics)

            if hit_rate is not None:
                response_data['data']['metrics']['hit_rate'] = float(hit_rate)
            
            response_data['data']['interval_statistics'] = result_data.get('interval_statistics', {})

            if actual_values_chart: # Use renamed var
                # Ensure the actual values are consistently sourced for the top-level 'actual' key
                response_data['actual'] = [p.get('actual') for p in predictions_for_response_chart if p.get('actual') is not None]
                logger.info(f"已添加实际值到响应中，包含 {len(response_data['actual'])} 个有效点")
            
            logger.info(f"区间预测API响应成功，包含 {len(predictions_for_response_chart)} 个预测点")
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"读取或解析区间预测JSON失败: {e}")
            cache.delete_memoized(api_interval_forecast)
            return jsonify({
                "success": False, 
            "error": f"读取预测结果失败: {results_json_path}", -
                "details": str(e)
            }), 500
    
    except ValueError as ve:
        try:
            logger.error(f"参数验证错误: {ve}")
            cache.delete_memoized(api_interval_forecast)
            return jsonify({"success": False, "error": str(ve)}), 400
        except Exception as e:
            logger.error(f"区间预测API端点发生未捕获的错误: {str(e)}")
            import traceback
            traceback.print_exc()
            cache.delete_memoized(api_interval_forecast)
            return jsonify({
                "success": False, 
                "error": "服务器内部错误", 
                "details": str(e)
            }), 500

@app.route('/api/historical-results', methods=['GET'])
def get_historical_results():
    """获取历史预测结果列表 - 优化版本，主要处理CSV文件"""
    forecast_type_req = request.args.get('forecastType', 'load')
    province_req = request.args.get('province', '上海')
    start_date_req = request.args.get('startDate', '')
    end_date_req = request.args.get('endDate', '')
    prediction_type_req = request.args.get('predictionType', 'all')
    model_type_req = request.args.get('modelType', 'all')
    
    logger.info(f"历史结果查询参数: forecastType={forecast_type_req}, province={province_req}, predictionType={prediction_type_req}, startDate={start_date_req}, endDate={end_date_req}, modelType={model_type_req}")

    try:
        # 解析日期范围
        start_date_obj = None
        end_date_obj = None
        if start_date_req:
            try:
                start_date_obj = datetime.strptime(start_date_req, '%Y-%m-%d')
            except ValueError:
                return jsonify({
                    'status': 'error',
                    'message': f'开始日期格式错误: {start_date_req}'
                }), 400
        
        if end_date_req:
            try:
                end_date_obj = datetime.strptime(end_date_req, '%Y-%m-%d')
            except ValueError:
                return jsonify({
                    'status': 'error',
                    'message': f'结束日期格式错误: {end_date_req}'
                }), 400
        
        if start_date_obj and end_date_obj and start_date_obj > end_date_obj:
            return jsonify({
                'status': 'error',
                'message': '开始日期不能晚于结束日期'
            }), 400

        results_files = []
        base_results_dir = "results"
        
        # 扫描yearly_forecasts目录下的CSV文件
        yearly_csv_dir = os.path.join(base_results_dir, "yearly_forecasts")
        
        if not os.path.exists(yearly_csv_dir):
            logger.warning(f"年度预测目录不存在: {yearly_csv_dir}")
            return jsonify({
                'status': 'success',
                'results': [],
                'message': '未找到年度预测数据目录'
            })

        # 遍历模型类型目录
        for model_type_folder in os.listdir(yearly_csv_dir):
            model_type_path = os.path.join(yearly_csv_dir, model_type_folder)
            if not os.path.isdir(model_type_path):
                continue
                
            # 如果指定了模型类型且不匹配，跳过
            if model_type_req != 'all' and model_type_folder != model_type_req:
                continue
                
            logger.info(f"扫描模型类型目录: {model_type_path}")
            
            # 遍历预测类型目录
            for forecast_type_folder in os.listdir(model_type_path):
                if forecast_type_folder != forecast_type_req:
                    continue
                    
                forecast_type_path = os.path.join(model_type_path, forecast_type_folder)
                if not os.path.isdir(forecast_type_path):
                    continue
                
                logger.info(f"扫描预测类型目录: {forecast_type_path}")
                
                # 遍历省份目录
                for province_folder in os.listdir(forecast_type_path):
                    if province_folder != province_req:
                        continue
                        
                    province_path = os.path.join(forecast_type_path, province_folder)
                    if not os.path.isdir(province_path):
                        continue
                    
                    logger.info(f"扫描省份目录: {province_path}")
                    
                    # 遍历CSV文件
                    for file in os.listdir(province_path):
                        if not file.endswith('.csv'):
                            continue
                            
                        file_path = os.path.join(province_path, file)
                        logger.debug(f"处理文件: {file_path}")
                        
                        # 解析文件名获取信息
                        file_info = parse_csv_file_info(file, model_type_folder)
                        if not file_info:
                            logger.debug(f"无法解析文件信息: {file}")
                            continue
                        
                        # 应用预测类型过滤
                        if prediction_type_req != 'all' and file_info['prediction_type'] != prediction_type_req:
                            continue
                        
                        # 检查CSV文件内容是否在日期范围内
                        if start_date_obj or end_date_obj:
                            if not check_csv_date_range(file_path, start_date_obj, end_date_obj):
                                continue
                        
                        # 获取文件的实际日期范围
                        actual_date_range = get_csv_date_range(file_path)
                        
                        results_files.append({
                            'path': file_path,
                            'filename': file,
                            'forecast_type': forecast_type_req,
                            'province': province_req,
                            'prediction_type': file_info['prediction_type'],
                            'model_type': model_type_folder,
                            'year': file_info['year'],
                            'actual_start_date': actual_date_range['start_date'] if actual_date_range else None,
                            'actual_end_date': actual_date_range['end_date'] if actual_date_range else None,
                            'size': os.path.getsize(file_path) / 1024,  # KB
                            'last_modified': datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
                        })
        
        # 按模型类型和预测类型分组
        grouped_results = group_results_by_model_and_prediction(results_files)
        
        logger.info(f"历史结果查询完成，找到 {len(results_files)} 个匹配文件")
        
        return jsonify({
            'status': 'success',
            'results': results_files,
            'grouped_results': grouped_results,
            'total_count': len(results_files)
        })
        
    except Exception as e:
        logger.error(f"获取历史结果时出错: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f"获取历史结果失败: {str(e)}"
        }), 500


def parse_csv_file_info(filename, model_type):
    """解析CSV文件名获取预测信息"""
    try:
        # 预期格式: {forecast_type}_{province}_{year}_{prediction_type}.csv
        # 例如: load_上海_2024_interval.csv
        name_without_ext = filename.replace('.csv', '')
        parts = name_without_ext.split('_')
        
        if len(parts) != 4:
            logger.debug(f"文件名格式不符合预期: {filename}")
            return None
        
        forecast_type, province, year, prediction_type = parts
        
        # 验证年份格式
        if not re.match(r'^\d{4}$', year):
            logger.debug(f"年份格式错误: {year}")
            return None
        
        # 标准化预测类型名称
        prediction_type_mapping = {
            'interval': 'interval',
            'rolling': 'rolling',
            'dayahead': 'day-ahead',
            'day-ahead': 'day-ahead',
            'probabilistic': 'probabilistic'
        }
        
        normalized_prediction_type = prediction_type_mapping.get(prediction_type.lower(), prediction_type)
        
        return {
            'forecast_type': forecast_type,
            'province': province,
            'year': year,
            'prediction_type': normalized_prediction_type,
            'model_type': model_type
        }
        
    except Exception as e:
        logger.error(f"解析文件名时出错: {filename}, {e}")
        return None


def check_csv_date_range(file_path, start_date_obj, end_date_obj):
    """检查CSV文件是否包含指定日期范围内的数据"""
    try:
        # 读取CSV文件的前几行和后几行来确定日期范围
        df_head = pd.read_csv(file_path, nrows=5, encoding='utf-8')
        
        if 'datetime' not in df_head.columns:
            return False
            
        # 获取文件的开始日期
        first_date = pd.to_datetime(df_head['datetime'].iloc[0]).date()
        
        # 读取最后几行获取结束日期
        df_tail = pd.read_csv(file_path, encoding='utf-8').tail(5)
        last_date = pd.to_datetime(df_tail['datetime'].iloc[-1]).date()
        
        # 检查日期范围是否有重叠
        file_start = first_date
        file_end = last_date
        
        query_start = start_date_obj.date() if start_date_obj else file_start
        query_end = end_date_obj.date() if end_date_obj else file_end
        
        # 检查是否有重叠
        return not (file_end < query_start or file_start > query_end)
        
    except Exception as e:
        logger.debug(f"检查CSV日期范围时出错: {file_path}, {e}")
        return False


def get_csv_date_range(file_path):
    """获取CSV文件的实际日期范围"""
    try:
        df = pd.read_csv(file_path, usecols=['datetime'], encoding='utf-8')
        if len(df) == 0:
            return None
            
        df['datetime'] = pd.to_datetime(df['datetime'])
        return {
            'start_date': df['datetime'].min().strftime('%Y-%m-%d'),
            'end_date': df['datetime'].max().strftime('%Y-%m-%d')
        }
    except Exception as e:
        logger.debug(f"获取CSV日期范围时出错: {file_path}, {e}")
        return None


def group_results_by_model_and_prediction(results_files):
    """按模型类型和预测类型对结果进行分组"""
    grouped = {}
    
    for result in results_files:
        model_type = result['model_type']
        prediction_type = result['prediction_type']
        
        if model_type not in grouped:
            grouped[model_type] = {}
        
        if prediction_type not in grouped[model_type]:
            grouped[model_type][prediction_type] = []
        
        grouped[model_type][prediction_type].append(result)
    
    return grouped


@app.route('/api/historical-results/view', methods=['GET'])
def view_historical_result():
    """查看特定历史预测结果文件内容 - 优化版本"""
    try:
        file_path = request.args.get('path')
        start_date = request.args.get('startDate', '')
        end_date = request.args.get('endDate', '')
        
        if not file_path:
            return jsonify({'status': 'error', 'message': '未提供文件路径'}), 400
        
        # 安全检查：确保路径在results目录下
        abs_path = os.path.abspath(file_path)
        if not abs_path.startswith(os.path.abspath('results')):
            return jsonify({'status': 'error', 'message': '无效的文件路径'}), 403
        
        if not file_path.endswith('.csv'):
            return jsonify({'status': 'error', 'message': '当前版本仅支持CSV文件'}), 400
        
            # 读取CSV文件
        logger.info(f"正在读取CSV文件: {file_path}")
        
        try:
            # 读取完整的CSV文件
            df = pd.read_csv(file_path, encoding='utf-8')
            logger.info(f"成功读取CSV，行数: {len(df)}, 列: {df.columns.tolist()}")
            
            # 检查必要的列
            if 'datetime' not in df.columns:
                logger.error(f"CSV文件缺少datetime列: {df.columns.tolist()}")
                return jsonify({
                    'status': 'error', 
                    'message': 'CSV文件格式错误：缺少datetime列'
                }), 400
            
            # 转换datetime列
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # 应用日期过滤
            if start_date or end_date:
                original_length = len(df)
                
                if start_date:
                    start_date_obj = pd.to_datetime(start_date)
                    df = df[df['datetime'] >= start_date_obj]
                
                if end_date:
                    end_date_obj = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                    df = df[df['datetime'] <= end_date_obj]
                
                filtered_length = len(df)
                logger.info(f"日期过滤: {original_length} -> {filtered_length} 条记录")
                
                if filtered_length == 0:
                    return jsonify({
                        'status': 'error',
                                    'message': f'在指定日期范围内没有找到数据'
                                }), 404
            
            # 将datetime转换为字符串格式
            df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S')
            
            # 处理NaN值
            df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
            
            # 提取文件信息
            filename = os.path.basename(file_path)
            file_info = parse_csv_file_info(filename, 'convtrans_peak')  # 假设是峰值感知模型
            
            # 构建返回的数据结构
            result = {
                'status': 'success',
                'forecast_type': file_info.get('forecast_type', 'load') if file_info else 'load',
                'province': file_info.get('province', '未知') if file_info else '未知',
                'start_time': df['datetime'].iloc[0] if len(df) > 0 else None,
                'end_time': df['datetime'].iloc[-1] if len(df) > 0 else None,
                'interval_minutes': 15,  # 默认15分钟
                'is_interval_forecast': file_info.get('prediction_type') == 'interval' if file_info else False,
                'is_probabilistic': file_info.get('prediction_type') == 'probabilistic' if file_info else False,
                'predictions': df.to_dict(orient='records'),
                'metrics': {}
            }
            
            # 添加元数据
            result['metadata'] = {
                'file_path': file_path,
                'forecast_type': result['forecast_type'],
                'province': result['province'],
                'prediction_type': file_info.get('prediction_type', '未知') if file_info else '未知',
                'model_type': file_info.get('model_type', 'convtrans_peak') if file_info else 'convtrans_peak',
                'year': file_info.get('year', '未知') if file_info else '未知',
                'total_rows': len(df),
                'filtered_rows': len(df) if start_date or end_date else None
            }
            
            # 根据预测类型添加特定统计信息
            prediction_type = file_info.get('prediction_type') if file_info else 'unknown'
            
            if prediction_type == 'interval' and 'lower_bound' in df.columns and 'upper_bound' in df.columns:
                # 区间预测统计
                interval_widths = pd.to_numeric(df['upper_bound'], errors='coerce') - pd.to_numeric(df['lower_bound'], errors='coerce')
                avg_width = interval_widths.mean()
                
                # 计算命中率
                hit_rate = None
                if 'actual' in df.columns:
                    df_with_actual = df.dropna(subset=['actual'])
                    if len(df_with_actual) > 0:
                        actual_vals = pd.to_numeric(df_with_actual['actual'], errors='coerce')
                        lower_vals = pd.to_numeric(df_with_actual['lower_bound'], errors='coerce')
                        upper_vals = pd.to_numeric(df_with_actual['upper_bound'], errors='coerce')
                        
                        hits = ((actual_vals >= lower_vals) & (actual_vals <= upper_vals)).sum()
                        hit_rate = (hits / len(df_with_actual)) * 100
                
                result['interval_statistics'] = {
                    'average_interval_width': float(avg_width) if not pd.isna(avg_width) else None,
                    'confidence_level': 0.9,  # 默认置信水平
                    'total_predictions': len(df),
                    'hit_rate': float(hit_rate) if hit_rate is not None else None
                }
            
            # 计算基本预测指标
            if 'predicted' in df.columns and 'actual' in df.columns:
                df_valid = df.dropna(subset=['predicted', 'actual'])
                if len(df_valid) > 0:
                    predicted_vals = pd.to_numeric(df_valid['predicted'], errors='coerce')
                    actual_vals = pd.to_numeric(df_valid['actual'], errors='coerce')
                    
                    # 删除无效值
                    valid_mask = ~(predicted_vals.isna() | actual_vals.isna())
                    predicted_vals = predicted_vals[valid_mask]
                    actual_vals = actual_vals[valid_mask]
                    
                    if len(predicted_vals) > 0:
                        mae = np.mean(np.abs(predicted_vals - actual_vals))
                        mape = np.mean(np.abs((predicted_vals - actual_vals) / actual_vals)) * 100
                        rmse = np.sqrt(np.mean((predicted_vals - actual_vals) ** 2))
                        
                        result['metrics'] = {
                            'mae': float(mae),
                            'mape': float(mape),
                            'rmse': float(rmse),
                            'valid_points': int(len(predicted_vals))
                        }
            
            logger.info(f"成功处理CSV文件，返回{len(df)}条记录")
            return jsonify(result)
            
        except pd.errors.EmptyDataError:
            logger.error(f"CSV文件为空: {file_path}")
            return jsonify({
                'status': 'error', 
                'message': 'CSV文件为空'
            }), 400
        except Exception as e:
            logger.error(f"读取CSV文件时出错: {str(e)}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': f'读取CSV文件失败: {str(e)}'
            }), 500
    
    except Exception as e:
        logger.error(f"查看历史结果文件时出错: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f"查看历史结果失败: {str(e)}"
        }), 500

@app.route('/api/weather-forecast', methods=['POST'])
@cache.cached(timeout=CACHE_TIMES['predict'], make_cache_key=make_cache_key)
def weather_aware_forecast():
    """执行天气感知负荷预测并返回结果"""
    global latest_prediction_result

    try:
        # 获取前端传过来的参数
        params = request.get_json()
        logger.info(f"收到天气感知预测请求，参数: {params}")
        
        # === 调试代码：记录天气感知参数 ===
        weather_aware_param = params.get('weatherAware', 'NOT_SET')
        forecast_type_param = params.get('forecastType', 'NOT_SET')
        prediction_type_param = params.get('predictionType', 'NOT_SET')
        logger.warning(f"[DEBUG] 天气感知API接收参数 - weatherAware: {weather_aware_param}, forecastType: {forecast_type_param}, predictionType: {prediction_type_param}")
        
        # 写入调试文件
        debug_entry = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "api": "天气感知预测API (/api/weather-forecast)",
            "weatherAware": weather_aware_param,
            "forecastType": forecast_type_param,
            "predictionType": prediction_type_param,
            "all_params": params
        }
        try:
            with open('debug_api_params.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps(debug_entry, ensure_ascii=False, indent=2) + '\n\n')
        except Exception as debug_err:
            logger.warning(f"写入调试日志失败: {debug_err}")
        # === 调试代码结束 ===

        # --- 参数提取和验证 ---
        prediction_type = params.get('predictionType', 'day-ahead')  # day-ahead or rolling
        forecast_type = params.get('forecastType', 'load')  # 天气感知预测目前只支持负荷预测
        province = params.get('province')
        
        if not province:
            cache.delete_memoized(weather_aware_forecast)
            return jsonify({"success": False, "error": "缺少省份参数(province)"}), 400
        
        if forecast_type not in ['load', 'pv', 'wind']:
            cache.delete_memoized(weather_aware_forecast)
            return jsonify({"success": False, "error": "天气感知预测支持负荷(load)、光伏(pv)和风电(wind)预测"}), 400

        # 验证历史数据天数
        historical_days_str = params.get('historicalDays')
        if historical_days_str is not None:
            try:
                historical_days_val = int(historical_days_str)
                if not (1 <= historical_days_val <= 30):
                    raise ValueError("历史数据天数必须在1-30之间")
            except (ValueError, TypeError):
                cache.delete_memoized(weather_aware_forecast)
                return jsonify({"success": False, "error": "历史数据天数必须是有效整数"}), 400

        # --- 构造命令行参数 ---
        script_path = 'scripts/scene_forecasting.py'
        cmd_args = ['--mode', 'forecast']
        cmd_args.extend(['--forecast_type', forecast_type])
        cmd_args.extend(['--province', province])
        cmd_args.extend(['--weather_aware'])  # 启用天气感知

        # 处理天气特征参数
        weather_features = params.get('weatherFeatures', 'temperature,humidity,pressure,wind_speed,wind_direction,precipitation,solar_radiation')
        if isinstance(weather_features, list):
            weather_features_str = ','.join(weather_features)
        else:
            weather_features_str = str(weather_features)
        cmd_args.extend(['--weather_features', weather_features_str])

        # 处理天气数据路径（可选，通常由脚本自动构建）
        weather_data_path = params.get('weatherDataPath')
        if weather_data_path:
            cmd_args.extend(['--weather_data_path', weather_data_path])

        # 处理天气模型目录路径（可选，通常由脚本自动构建）
        weather_model_dir = params.get('weatherModelDir')
        if weather_model_dir:
            cmd_args.extend(['--weather_model_dir', weather_model_dir])

        # 根据预测类型添加相应参数
        if prediction_type == 'day-ahead':
            cmd_args.extend(['--day_ahead'])
            if params.get('forecastDate'):
                cmd_args.extend(['--forecast_date', params['forecastDate']])
            if params.get('forecastEndDate'):
                cmd_args.extend(['--forecast_end_date', params['forecastEndDate']])
            if params.get('enhancedSmoothing'):
                cmd_args.extend(['--enhanced_smoothing'])
                if params.get('maxDiffPct') is not None:
                    cmd_args.extend(['--max_diff_pct', str(params['maxDiffPct'])])
                if params.get('smoothingWindow') is not None:
                    cmd_args.extend(['--smoothing_window', str(params['smoothingWindow'])])
        elif prediction_type == 'rolling':
            if params.get('startDate'):
                cmd_args.extend(['--forecast_start', params['startDate']])
            if params.get('endDate'):
                cmd_args.extend(['--forecast_end', params['endDate']])
            if params.get('interval'):
                cmd_args.extend(['--interval', str(params['interval'])])
        else:
            cache.delete_memoized(weather_aware_forecast)
            return jsonify({"success": False, "error": f"不支持的预测类型: {prediction_type}"}), 400

        # 添加历史数据天数参数
        if params.get('historicalDays') is not None:
            cmd_args.extend(['--historical_days', str(params['historicalDays'])])

        # 设置输出文件路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S%f')
        results_base_dir = f"results/weather_aware_{prediction_type.replace('-', '_')}/{forecast_type}"
        results_province_dir = f"{results_base_dir}/{province}"
        os.makedirs(results_province_dir, exist_ok=True)
        results_json_path = f"{results_province_dir}/weather_forecast_{timestamp}.json"
        cmd_args.extend(['--output_json', results_json_path])

        # 执行预测命令
        cmd = build_python_command(script_path, cmd_args)
        logger.info(f"执行天气感知预测命令: {' '.join(cmd)}")

        return_code, stdout, stderr = run_process_with_timeout(cmd, timeout=600)

        logger.info(f"天气感知预测脚本完成，返回码: {return_code}")
        if stdout:
            logger.info(f"天气感知预测 STDOUT:\n{stdout}")
        if stderr:
            logger.warning(f"天气感知预测 STDERR:\n{stderr}")

        # 检查结果文件是否创建成功
        file_created = os.path.exists(results_json_path)
        if return_code != 0 or not file_created:
            # 检查是否在测试目录中创建了CSV文件（临时解决方案）
            test_csv_pattern = f"results/weather_aware_day_ahead_test/weather_day_ahead_results_*.csv"
            import glob
            test_files = glob.glob(test_csv_pattern)
            
            if test_files and return_code == 0:
                # 找到了CSV文件，转换为JSON
                latest_csv = max(test_files, key=os.path.getctime)
                logger.info(f"发现天气感知预测CSV文件: {latest_csv}，正在转换为JSON格式")
                
                try:
                    # 读取CSV并转换为JSON格式
                    import pandas as pd
                    df = pd.read_csv(latest_csv)
                    
                    # 确保datetime列格式正确
                    if 'timestamp' in df.columns:
                        df['datetime'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%dT%H:%M:%S')
                        df = df.drop(columns=['timestamp'])
                    elif 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%dT%H:%M:%S')
                    
                    # 重命名列以匹配标准格式
                    if 'predicted' not in df.columns and 'prediction' in df.columns:
                        df = df.rename(columns={'prediction': 'predicted'})
                    
                    # 添加高峰时段标记
                    if 'is_peak' not in df.columns and 'datetime' in df.columns:
                        df['is_peak'] = pd.to_datetime(df['datetime']).dt.hour.apply(
                            lambda x: 1 if 8 <= x <= 20 else 0
                        )
                    
                    # 添加预测日期标识
                    if 'prediction_date' not in df.columns:
                        df['prediction_date'] = params.get('forecastDate', '')
                    
                    # 计算误差百分比（如果有实际值）
                    if 'actual' in df.columns and 'predicted' in df.columns:
                        df['error_pct'] = ((df['predicted'] - df['actual']) / df['actual'] * 100).round(2)
                        df['error_pct'] = df['error_pct'].replace([np.inf, -np.inf], None)
                    
                    # 转换为API期望的JSON格式
                    result_data = {
                        "status": "success",
                        "forecast_type": forecast_type,
                        "province": province,
                        "start_time": df['datetime'].iloc[0] if 'datetime' in df.columns and not df.empty else None,
                        "end_time": df['datetime'].iloc[-1] if 'datetime' in df.columns and not df.empty else None,
                        "interval_minutes": params.get('interval', 15),
                        "historical_days": params.get('historicalDays', 15),
                        "weather_aware": True,
                        "weather_features": weather_features_str.split(',') if weather_features_str else [],
                        "predictions": df.to_dict('records'),
                        "metrics": {
                            "note": "从CSV转换而来，指标可能不完整"
                        }
                    }
                    
                    # 保存为JSON
                    os.makedirs(os.path.dirname(results_json_path), exist_ok=True)
                    with open(results_json_path, 'w', encoding='utf-8') as f:
                        json.dump(result_data, f, ensure_ascii=False, indent=2)
                        
                    logger.info(f"已将CSV结果转换为JSON: {results_json_path}")
                    file_created = True
                    
                except Exception as convert_err:
                    logger.error(f"CSV转换JSON失败: {convert_err}")
                    error_msg = f"天气感知预测脚本执行成功但未创建结果文件，CSV转换也失败."
                    details = stderr or stdout or "无详细错误输出。"
                    logger.error(f"{error_msg} Details: {details}")
                    cache.delete_memoized(weather_aware_forecast)
                    return jsonify({"success": False, "error": f"{error_msg} Details: {details}"}), 500
            else:
                error_msg = f"天气感知预测脚本执行失败 (代码: {return_code})."
                if not file_created and return_code == 0:
                    error_msg = "天气感知预测脚本执行成功但未创建结果文件."
                details = stderr or stdout or "无详细错误输出。"
                logger.error(f"{error_msg} Details: {details}")
                cache.delete_memoized(weather_aware_forecast)
                return jsonify({"success": False, "error": f"{error_msg} Details: {details}"}), 500

        # 读取并返回结果
        try:
            with open(results_json_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            if result_data.get('status') == 'error':
                cache.delete_memoized(weather_aware_forecast)
                return jsonify({"success": False, "error": result_data.get('error', '天气感知预测返回错误状态')}), 500
            
            # 更新全局变量
            latest_prediction_result = results_json_path
            
            # 添加天气感知标识
            result_data['weather_aware'] = True
            result_data['weather_features'] = weather_features_str.split(',')
            
            return jsonify({
                "success": True,
                "data": result_data,
                "message": f"天气感知{forecast_type}预测完成",
                "result_file": results_json_path
            })
            
        except Exception as e:
            logger.error(f"读取或解析天气感知预测JSON失败: {e}")
            cache.delete_memoized(weather_aware_forecast)
            return jsonify({"success": False, "error": f"读取预测结果失败: {results_json_path}", "details": str(e)}), 500

    except Exception as e:
        logger.error(f"天气感知预测请求处理失败: {e}")
        cache.delete_memoized(weather_aware_forecast)
        return jsonify({"success": False, "error": f"天气感知预测失败: {str(e)}"}), 500

@app.route('/api/scenario-aware-uncertainty-forecast', methods=['POST'])
@cache.cached(timeout=CACHE_TIMES['scenarios'], make_cache_key=make_cache_key)
def scenario_aware_uncertainty_forecast_api():
    """执行场景感知的不确定性预测"""
    try:
        # 获取前端传过来的参数
        params = request.get_json()
        logger.info(f"收到场景感知不确定性预测请求，参数: {params}")
        
        # 参数提取和验证
        province = params.get('province')
        forecast_type = params.get('forecastType', 'load')
        start_date = params.get('forecastDate') or params.get('startDate')
        end_date = params.get('forecastEndDate') or params.get('endDate')
        confidence_level = float(params.get('confidenceLevel', 0.9))
        historical_days = int(params.get('historicalDays', 14))
        include_explanations = params.get('includeExplanations', True)
        
        if not province:
            cache.delete_memoized(scenario_aware_uncertainty_forecast_api)
            return jsonify({"success": False, "error": "缺少省份参数(province)"}), 400
            
        if not start_date:
            cache.delete_memoized(scenario_aware_uncertainty_forecast_api)
            return jsonify({"success": False, "error": "缺少预测开始日期"}), 400
        
        # 导入场景感知预测函数
        from scripts.forecast.scenario_aware_uncertainty_forecast import perform_scenario_aware_uncertainty_forecast
        
        # 执行场景感知不确定性预测
        result = perform_scenario_aware_uncertainty_forecast(
            province=province,
            forecast_type=forecast_type,
            start_date=start_date,
            end_date=end_date,
            confidence_level=confidence_level,
            historical_days=historical_days,
            include_explanations=include_explanations
        )
        
        # 检查是否是缓存结果
        is_cached = getattr(scenario_aware_uncertainty_forecast_api, 'cached', False)
        cache_info = None
        if is_cached:
            increment_cache_hit('scenarios')
            cache_info = {"hit_count": cache_hits['scenarios'], "timestamp": datetime.now().isoformat()}
            logger.info(f"缓存命中: 场景感知不确定性预测, 总命中数: {cache_hits['scenarios']}")
        else:
            scenario_aware_uncertainty_forecast_api.cached = True

        # 构建响应
        response_data = {
            "success": True,
            "data": result,
            "cached": is_cached,
            "cache_info": cache_info,
            "message": f"场景感知{forecast_type}不确定性预测完成"
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"场景感知不确定性预测请求处理失败: {e}")
        import traceback
        traceback.print_exc()
        cache.delete_memoized(scenario_aware_uncertainty_forecast_api)
        return jsonify({"success": False, "error": f"场景感知不确定性预测失败: {str(e)}"}), 500

if __name__ == '__main__':
    # 确保结果目录存在
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 启动Flask服务器
    app.run(debug=True, port=5001, threaded=True) 