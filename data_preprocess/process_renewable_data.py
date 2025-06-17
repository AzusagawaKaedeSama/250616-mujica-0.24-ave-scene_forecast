import argparse
import os
import sys
from datetime import datetime

# Add project root to path to import DataLoader
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from data.data_loader import DataLoader
except ImportError as e:
    print(f"Error: Could not import DataLoader. Make sure it's in the 'data' directory or adjust sys.path.")
    print(f"Import Error: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Process PV/Wind data into time-series CSV files.')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Path to the Excel data file')
    parser.add_argument('--data_type', type=str, choices=['pv', 'wind'], default='wind',
                        help='Type of data to process: pv (photovoltaic) or wind.')
    parser.add_argument('--provinces', type=str, nargs='*', default=None,
                        help='Specific provinces to process (e.g., 上海 浙江). If omitted, process all found in the file.')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Directory to save the output CSV files (default: data)')

    args = parser.parse_args()

    # 如果未指定输入文件，根据数据类型设置默认值
    if args.input_file is None:
        if args.data_type == 'pv':
            args.input_file = '2024年四省一市光伏预测数据.xlsx'
        else:  # 'wind'
            args.input_file = '2024年四省一市风电预测数据.xlsx'
    
    print(f"--- Starting {args.data_type.upper()} Data Processing ({datetime.now()}) ---")
    print(f"Input File: {args.input_file}")
    print(f"Data Type: {args.data_type}")
    print(f"Target Provinces: {'All' if not args.provinces else ', '.join(args.provinces)}")
    print(f"Output Directory: {args.output_dir}")

    # 处理输入文件路径
    input_file = args.input_file
    if not os.path.isabs(input_file):
        # 检查在data_preprocess目录下
        preprocess_dir_path = os.path.dirname(os.path.abspath(__file__))
        potential_path = os.path.join(preprocess_dir_path, input_file)
        if os.path.exists(potential_path):
            input_file = potential_path
        else:
            # 检查项目根目录
            potential_path = os.path.join(project_root, input_file)
            if os.path.exists(potential_path):
                input_file = potential_path

    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    # 处理输出目录路径
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    try:
        print(f"使用文件: {input_file}")
        print(f"正在使用 DataLoader 处理{args.data_type}数据")
        
        # 初始化DataLoader
        loader = DataLoader(
            data_files=[input_file]
        )

        print("从Excel文件加载数据...")
        loader.load_all_data()

        print("转换数据为时间序列格式...")
        loader.transform_to_timeseries(
            provinces=args.provinces,
            output_dir=output_dir,
            data_type=args.data_type
        )

        print(f"--- {args.data_type.upper()}数据处理成功完成 ({datetime.now()}) ---")
        print(f"输出文件保存在: {output_dir}")

    except FileNotFoundError as e:
        print(f"处理文件时出错: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"数据处理错误: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"错误: Excel文件中缺少预期的表格或列: {e}")
        print("请检查 --input_file 和 DataLoader 预期的表格/列名。")
        sys.exit(1)
    except Exception as e:
        print(f"发生意外错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 