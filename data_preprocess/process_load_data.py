import argparse
import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path to import DataLoader
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from data.data_loader import DataLoader
except ImportError as e:
    print(f"Error: Could not import DataLoader. Make sure it's in the 'data' directory or adjust sys.path.")
    print(f"Import Error: {e}")
    sys.exit(1)

class LoadDataProcessor:
    """负荷数据处理器，用于处理特殊格式的负荷数据"""
    
    def __init__(self, input_file, output_dir, provinces=None):
        self.input_file = input_file
        self.output_dir = output_dir
        self.provinces = provinces
        self.data = {}  # 存储处理后的数据 {province: DataFrame}
        
    def process(self):
        """处理负荷数据"""
        print(f"开始处理负荷数据文件: {self.input_file}")
        
        if not os.path.exists(self.input_file):
            print(f"错误: 文件不存在: {self.input_file}")
            return False
        
        # 读取Excel文件所有sheet
        try:
            xls = pd.ExcelFile(self.input_file)
            sheet_names = xls.sheet_names
            print(f"发现工作表: {sheet_names}")
            
            # 确定要处理的省份
            if self.provinces is None:
                provinces_to_process = sheet_names
            else:
                provinces_to_process = [p for p in self.provinces if p in sheet_names]
                missing = [p for p in self.provinces if p not in sheet_names]
                if missing:
                    print(f"警告: 未找到以下省份的工作表: {missing}")
            
            if not provinces_to_process:
                print("错误: 没有找到有效的省份工作表")
                return False
                
            print(f"将处理以下省份: {provinces_to_process}")
            
            # 处理每个省份的数据
            for province in provinces_to_process:
                print(f"  处理省份: {province}...")
                success = self._process_province(xls, province)
                if not success:
                    print(f"  处理省份 {province} 失败，跳过。")
                    
            # 保存处理后的数据
            return self._save_results()
                
        except Exception as e:
            print(f"处理文件时出错: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _process_province(self, xls, province):
        """处理单个省份的数据"""
        try:
            # 读取工作表
            df = pd.read_excel(xls, sheet_name=province)
            print(f"    读取完成，数据形状: {df.shape}")
            
            # 显示前几列的名称，用于调试
            first_cols = df.columns[:10].tolist()
            print(f"    前10列名称: {first_cols}")
            
            # 检查是否有日期列
            date_cols = [col for col in df.columns if '日期' in str(col) or 'DATE' in str(col).upper()]
            if date_cols:
                date_col = date_cols[0]
                print(f"    找到可能的日期列: {date_col}")
            else:
                # 假设第一列是日期列
                date_col = df.columns[0]
                print(f"    未找到明确的日期列，使用第一列: {date_col}")
            
            # 尝试将日期列转换为datetime格式
            try:
                df['CL_DATE'] = pd.to_datetime(df[date_col])
                print(f"    已将列 '{date_col}' 转换为datetime并重命名为'CL_DATE'")
            except:
                # 如果无法直接转换，尝试使用预定格式
                try:
                    df['CL_DATE'] = pd.to_datetime(df[date_col], format='%Y%m%d')
                    print(f"    已使用格式 '%Y%m%d' 将列 '{date_col}' 转换为datetime")
                except:
                    try:
                        df['CL_DATE'] = pd.to_datetime(df[date_col], format='%Y-%m-%d')
                        print(f"    已使用格式 '%Y-%m-%d' 将列 '{date_col}' 转换为datetime")
                    except Exception as e:
                        print(f"    无法转换列 '{date_col}' 为日期格式: {e}")
                        print(f"    尝试创建日期序列...")
                        # 创建日期序列
                        start_date = datetime(2024, 1, 1)  # 假设从2024年1月1日开始
                        df['CL_DATE'] = [start_date + timedelta(days=i) for i in range(len(df))]
                        print(f"    已创建从 {start_date} 开始的日期序列")
            
            # 检查数值列
            value_cols = [col for col in df.columns if any(f'P{i}' == str(col) for i in range(1, 97))]
            if value_cols:
                print(f"    找到标准格式的数值列: {len(value_cols)} 列")
            else:
                # 尝试查找其他可能的数值列模式
                possible_patterns = ['P', 'F', 'V', '负荷', 'LOAD']
                for pattern in possible_patterns:
                    value_cols = [col for col in df.columns if pattern in str(col)]
                    if len(value_cols) > 90:  # 假设至少有90个点
                        print(f"    使用模式 '{pattern}' 找到 {len(value_cols)} 个可能的数值列")
                        # 重命名列以符合标准格式 (P1-P96)
                        rename_map = {}
                        for i, col in enumerate(sorted(value_cols)[:96], 1):
                            rename_map[col] = f'P{i}'
                        df.rename(columns=rename_map, inplace=True)
                        print(f"    已将 {len(rename_map)} 列重命名为标准格式 (P1-P96)")
                        break
                else:
                    print(f"    错误: 无法找到足够的数值列。")
                    return False
            
            # 添加PARTY_ID列
            if 'PARTY_ID' not in df.columns:
                df['PARTY_ID'] = province
                print(f"    已添加PARTY_ID列，值为: {province}")
            
            # 存储处理后的数据
            self.data[province] = df
            print(f"    省份 {province} 数据处理完成。")
            return True
            
        except Exception as e:
            print(f"    处理省份 {province} 时出错: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_results(self):
        """保存处理结果为时间序列CSV文件"""
        if not self.data:
            print("错误: 没有可保存的数据")
            return False
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        success_count = 0
        for province, df in self.data.items():
            try:
                # 准备数据透视
                point_columns = [f'P{i}' for i in range(1, 97)]
                available_points = [col for col in point_columns if col in df.columns]
                
                if len(available_points) < 90:
                    print(f"  警告: 省份 {province} 只有 {len(available_points)} 个点数据，可能不完整")
                
                # 数据透视
                id_vars = ['PARTY_ID', 'CL_DATE']
                df_long = pd.melt(df,
                                  id_vars=id_vars,
                                  value_vars=available_points,
                                  var_name='IntervalID',
                                  value_name='load')
                print(f"  数据透视完成，生成 {len(df_long)} 行。")
                
                # 提取点号并生成时间戳
                df_long['PointNumber'] = df_long['IntervalID'].str[1:].astype(int)
                df_long['TimeDelta'] = pd.to_timedelta((df_long['PointNumber'] - 1) * 15, unit='m')
                df_long['datetime'] = df_long['CL_DATE'] + df_long['TimeDelta']
                
                # 清理和准备最终数据
                df_final = df_long[['datetime', 'PARTY_ID', 'load']].copy()
                df_final.sort_values(by=['PARTY_ID', 'datetime'], inplace=True)
                df_final.drop_duplicates(subset=['PARTY_ID', 'datetime'], keep='first', inplace=True)
                
                # 保存CSV
                output_filename = f"timeseries_load_{province}.csv"
                output_path = os.path.join(self.output_dir, output_filename)
                print(f"  保存到: {output_path}...")
                df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"  保存成功，共 {len(df_final)} 条记录。")
                success_count += 1
                
            except Exception as e:
                print(f"  保存省份 {province} 数据时出错: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"共成功处理并保存 {success_count}/{len(self.data)} 个省份的数据。")
        return success_count > 0

def main():
    parser = argparse.ArgumentParser(description='Process Load data into time-series CSV files.')
    parser.add_argument('--input_file', type=str, default='2024年四省一市负荷预测数据.xlsx',
                        help='Path to the Excel data file')
    parser.add_argument('--provinces', type=str, nargs='*', default=None,
                        help='Specific provinces to process (e.g., 上海 浙江). If omitted, process all found in the file.')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Directory to save the output CSV files (default: data)')

    args = parser.parse_args()
    
    print(f"--- Starting LOAD Data Processing ({datetime.now()}) ---")
    print(f"Input File: {args.input_file}")
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
        # 使用专门的负荷数据处理器
        processor = LoadDataProcessor(
            input_file=input_file,
            output_dir=output_dir,
            provinces=args.provinces
        )
        
        success = processor.process()
        
        if success:
            print(f"--- LOAD数据处理成功完成 ({datetime.now()}) ---")
        else:
            print(f"--- LOAD数据处理失败 ({datetime.now()}) ---")
            sys.exit(1)
        
        print(f"输出文件保存在: {output_dir}")

    except Exception as e:
        print(f"处理过程中发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 