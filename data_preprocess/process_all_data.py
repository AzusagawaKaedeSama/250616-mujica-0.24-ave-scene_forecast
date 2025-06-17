import os
import sys
import subprocess
import argparse
from datetime import datetime
import glob

def run_command(cmd, description):
    """运行命令并打印输出"""
    print(f"\n{'='*80}")
    print(f"执行 {description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # 实时打印输出
    for line in process.stdout:
        print(line, end='')
    
    # 等待进程完成
    exit_code = process.wait()
    
    if exit_code != 0:
        print(f"\n[错误] {description} 失败，退出代码: {exit_code}")
        return False
    else:
        print(f"\n[成功] {description} 完成")
        return True

def main():
    parser = argparse.ArgumentParser(description='Process all data types including load, renewables, and weather.')
    parser.add_argument('--weather_dir', type=str, default='atmosphere',
                        help='Directory containing NetCDF weather data files (default: atmosphere)')
    parser.add_argument('--weather_file', type=str, default=None,
                        help='Specific NetCDF weather data file to process (if omitted, process all .nc files in weather_dir)')
    parser.add_argument('--skip_excel', action='store_true',
                        help='Skip processing Excel files (load, PV, wind)')
    parser.add_argument('--skip_weather', action='store_true',
                        help='Skip processing weather data')
    parser.add_argument('--variables', type=str, nargs='+', 
                        default=['temperature', 'wind_speed', 'humidity', 'precipitation'],
                        help='Weather variables to extract (default: temperature wind_speed humidity precipitation)')
    parser.add_argument('--provinces', type=str, nargs='*', default=None,
                        help='Specific provinces to process (e.g., 上海 浙江). If omitted, process all provinces.')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Directory to save the output CSV files (default: data)')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    print(f"开始数据处理任务 ({start_time})")
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 确保output目录存在
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    success_results = {}
    
    # 处理Excel数据文件
    if not args.skip_excel:
        # 要处理的数据文件
        wind_file = os.path.join(script_dir, "2024年四省一市风电预测数据.xlsx")
        pv_file = os.path.join(script_dir, "2024年四省一市光伏预测数据.xlsx")
        load_file = os.path.join(script_dir, "2024年四省一市负荷预测数据.xlsx")
        
        # 验证文件存在
        missing_files = []
        for file_path in [wind_file, pv_file, load_file]:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print("错误: 以下文件不存在:")
            for file in missing_files:
                print(f"  - {file}")
            print("\n请确保以下文件存在于 data_preprocess 目录中:")
            print("  - 2024年四省一市风电预测数据.xlsx")
            print("  - 2024年四省一市光伏预测数据.xlsx")
            print("  - 2024年四省一市负荷预测数据.xlsx")
            if args.skip_weather or (args.weather_file is None and args.weather_dir is None):
                sys.exit(1)
            else:
                print("将跳过Excel文件处理，仅处理天气数据。")
        else:
            # 处理风电数据
            success_wind = run_command([
                sys.executable,
                os.path.join(script_dir, "process_renewable_data.py"),
                "--data_type", "wind",
                "--input_file", wind_file,
                "--output_dir", output_dir
            ] + (["--provinces"] + args.provinces if args.provinces else []), 
            "风电数据处理")
            success_results['wind'] = success_wind
            
            # 处理光伏数据
            success_pv = run_command([
                sys.executable,
                os.path.join(script_dir, "process_renewable_data.py"),
                "--data_type", "pv",
                "--input_file", pv_file,
                "--output_dir", output_dir
            ] + (["--provinces"] + args.provinces if args.provinces else []), 
            "光伏数据处理")
            success_results['pv'] = success_pv
            
            # 处理负荷数据
            success_load = run_command([
                sys.executable,
                os.path.join(script_dir, "process_load_data.py"),
                "--input_file", load_file,
                "--output_dir", output_dir
            ] + (["--provinces"] + args.provinces if args.provinces else []), 
            "负荷数据处理")
            success_results['load'] = success_load
    
    # 处理天气数据
    if not args.skip_weather:
        # 确定要处理的天气文件
        weather_files = []
        
        if args.weather_file is not None:
            # 处理单个指定的天气文件
            weather_file = args.weather_file
            if not os.path.isabs(weather_file):
                # 检查在data_preprocess目录下
                potential_path = os.path.join(script_dir, weather_file)
                if os.path.exists(potential_path):
                    weather_files.append(potential_path)
                else:
                    # 检查项目根目录
                    potential_path = os.path.join(project_root, weather_file)
                    if os.path.exists(potential_path):
                        weather_files.append(potential_path)
                    else:
                        print(f"错误: 天气数据文件不存在: {weather_file}")
        else:
            # 处理指定目录下的所有nc文件
            weather_dir = args.weather_dir
            if not os.path.isabs(weather_dir):
                weather_dir = os.path.join(script_dir, weather_dir)
            
            if os.path.exists(weather_dir):
                # 查找目录下所有.nc文件
                weather_files = glob.glob(os.path.join(weather_dir, "*.nc"))
                if not weather_files:
                    print(f"警告: 在 {weather_dir} 目录下未找到任何.nc文件")
            else:
                print(f"错误: 天气数据目录不存在: {weather_dir}")
        
        # 处理每个天气文件
        success_weather_all = True
        for i, weather_file in enumerate(weather_files):
            # 文件描述（用于日志）
            file_desc = os.path.basename(weather_file)
            
            # 构建命令
            cmd = [
                sys.executable,
                os.path.join(script_dir, "process_weather_data.py"),
                "--input_file", weather_file,
                "--output_dir", output_dir
            ]
            
            # 添加变量参数
            if args.variables:
                cmd.extend(["--variables"] + args.variables)
                
            # 添加区域参数
            if args.provinces:
                cmd.extend(["--regions"] + args.provinces)
                
            # 运行命令
            success_weather = run_command(cmd, f"天气数据处理 ({i+1}/{len(weather_files)}: {file_desc})")
            
            # 记录结果
            success_key = f'weather_{i+1}'
            success_results[success_key] = success_weather
            
            # 如果任一文件处理失败，标记整体处理结果为失败
            if not success_weather:
                success_weather_all = False
        
        # 记录总体天气处理结果
        success_results['weather_all'] = success_weather_all
    
    # 汇总结果
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print(f"数据处理任务总结 ({end_time})")
    print(f"总耗时: {duration}")
    print(f"{'='*80}\n")
    
    print("处理结果:")
    if 'wind' in success_results:
        print(f"  - 风电数据: {'成功' if success_results['wind'] else '失败'}")
    if 'pv' in success_results:
        print(f"  - 光伏数据: {'成功' if success_results['pv'] else '失败'}")
    if 'load' in success_results:
        print(f"  - 负荷数据: {'成功' if success_results['load'] else '失败'}")
    
    # 显示天气数据处理结果
    weather_files = [key for key in success_results.keys() if key.startswith('weather_') and key != 'weather_all']
    if weather_files:
        print(f"  - 天气数据总体: {'成功' if success_results.get('weather_all', False) else '失败'}")
        for key in weather_files:
            file_num = key.split('_')[1]
            print(f"    - 文件 {file_num}: {'成功' if success_results[key] else '失败'}")
    
    if all(success_results.values()):
        print("\n所有数据处理成功完成!")
        print(f"输出文件保存在: {output_dir}")
        return 0
    else:
        print("\n部分数据处理失败，请检查上方日志查看具体错误。")
        return 1

if __name__ == "__main__":
    sys.exit(main())