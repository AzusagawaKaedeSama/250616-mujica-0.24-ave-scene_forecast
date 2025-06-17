"""
负荷数据加载与处理模块
适用于新格式的省市级加密负荷数据文件
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
sys.stdout.reconfigure(encoding='utf-8')

class DataLoader:
    """
    数据加载和转换器，支持从宽格式 Excel (P1-P96) 转换为长格式时间序列 CSV。
    """
    def __init__(self, data_files=None, data_dir='./data', 
                 date_column='CL_DATE', # 源数据中的日期列名
                 party_id_column='PARTY_ID', # 源数据中的地区ID列名
                 value_column_prefix='P'): # 源数据中值列的前缀
        """
        初始化数据加载器
        
        参数:
        data_files (list): 数据文件路径列表，如果为None则扫描data_dir
        data_dir (str): 数据文件所在目录
        date_column (str): 源文件中表示日期的列名.
        party_id_column (str): 源文件中表示地区/参与者ID的列名.
        value_column_prefix (str): 源文件中表示数值的列名的前缀 (P1-P96).
        """
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 如果data_dir是相对路径，将其转换为绝对路径
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(project_root, data_dir.lstrip('./'))
        
        self.data_dir = data_dir
        self.date_column = date_column
        self.party_id_column = party_id_column
        self.value_column_prefix = value_column_prefix
        # 生成 P1 到 P96 的列名列表
        self.point_columns = [f"{self.value_column_prefix}{i}" for i in range(1, 97)]
        print(f"DataLoader 初始化 - 日期列:'{self.date_column}', ID列:'{self.party_id_column}', 值前缀:'{self.value_column_prefix}'")
        print(f"数据目录: {self.data_dir}")

        # 如果未提供具体文件列表，则扫描目录查找所有Excel文件
        if data_files is None:
            self.data_files = self._scan_data_files()
            print(f"扫描目录 '{self.data_dir}' 找到文件: {self.data_files}")
        else:
            # 如果提供的文件路径是相对路径，转换为绝对路径
            self.data_files = [
                os.path.join(project_root, f.lstrip('./')) if not os.path.isabs(f) else f
                for f in data_files
            ]
            
        self.raw_data = {} # 存储各省份的原始宽格式数据 {province: DataFrame}
        
        # 创建数据目录（如果不存在）
        os.makedirs(data_dir, exist_ok=True)
    
    def _scan_data_files(self):
        """扫描数据目录，查找所有Excel文件"""
        data_path = Path(self.data_dir)
        # 同时查找 .xlsx 和 .xls 文件
        excel_files = list(data_path.glob("*.xlsx")) + list(data_path.glob("*.xls"))
        return [str(file) for file in excel_files]
    
    def load_all_data(self):
        """
        加载所有指定文件中的所有省份数据到 self.raw_data.
        假设 Excel 文件中的 Sheet 名称即为省份名称。
        """
        print("\n开始加载原始数据...")
        if not self.data_files:
            print("警告: 未在初始化时提供或扫描到任何数据文件。")
            return self.raw_data

        loaded_provinces = set() # 跟踪已加载的省份，避免重复

        for file_path in self.data_files:
            if not os.path.exists(file_path):
                print(f"警告: 文件不存在 {file_path}, 跳过.")
                continue

            try:
                # 使用 ExcelFile 来获取所有 sheet 名称，避免多次读取文件
                xls = pd.ExcelFile(file_path)
                sheet_names = xls.sheet_names
                print(f"处理文件 '{os.path.basename(file_path)}', 包含 Sheets: {sheet_names}")

                for sheet_name in sheet_names:
                    # 假设 sheet 名称就是省份名称
                    province = sheet_name
                    
                    # 如果该省份数据已从其他文件加载，则跳过
                    if province in loaded_provinces:
                        print(f"  省份 '{province}' 数据已加载，跳过 Sheet '{sheet_name}'。")
                        continue
                        
                    print(f"  正在读取 Sheet: '{sheet_name}' (省份: {province})...")
                    
                    # 尝试读取并解析日期列
                    try:
                         df = pd.read_excel(xls, sheet_name=sheet_name, parse_dates=[self.date_column])
                         # 尝试去除时区信息（如果存在）
                         if pd.api.types.is_datetime64_any_dtype(df[self.date_column]) and df[self.date_column].dt.tz is not None:
                             print(f"    检测到日期列 '{self.date_column}' 包含时区，将移除。")
                             df[self.date_column] = df[self.date_column].dt.tz_localize(None)
                    except ValueError:
                         print(f"    警告: 尝试将 '{self.date_column}' 作为日期解析失败，将作为普通列读取。")
                         df = pd.read_excel(xls, sheet_name=sheet_name)
                         # 后面会再次尝试转换
                    except KeyError:
                         print(f"    警告: 在 Sheet '{sheet_name}' 中未找到指定的日期列 '{self.date_column}'，跳过此 Sheet。")
                         continue # 跳过这个 sheet

                    print(f"    读取完成，数据形状: {df.shape}")

                    # 检查必要的列是否存在 (使用初始化时定义的列名)
                    required_cols = [self.party_id_column, self.date_column] + self.point_columns
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                         print(f"    错误: Sheet '{sheet_name}' 缺失必需列: {missing_cols}. 跳过此Sheet。")
                         continue # 跳过这个 sheet
                        
                    self.raw_data[province] = df
                    loaded_provinces.add(province) # 记录已加载的省份
                    print(f"    Sheet '{sheet_name}' (省份: {province}) 数据已成功加载.")

            except Exception as e:
                print(f"加载文件 '{file_path}' 时发生严重错误: {e}")
                import traceback
                traceback.print_exc()
                
        print(f"原始数据加载完成. 共加载 {len(self.raw_data)} 个省份的数据: {list(self.raw_data.keys())}")
        return self.raw_data

    def transform_to_timeseries(self, provinces=None, output_dir='data', data_type='load', interval_minutes=15):
        """
        修改后：将指定省份的宽格式数据 (P1-P96) 转换为长格式时间序列并分别保存为 CSV。
        
        参数:
        provinces (list): 要转换的省份列表，如果为None则转换所有已加载的省份。
        output_dir (str): 输出CSV文件的目录。
        data_type (str): 数据类型 ('load', 'pv', 'wind'), 用于命名输出文件和值列。
        interval_minutes (int): 每个点代表的时间间隔（分钟）。默认为15。
        """
        print(f"\\n开始转换数据为时间序列格式 (类型: {data_type}, 时间间隔: {interval_minutes} 分钟)...")
        
        if not self.raw_data:
            print("错误: 尚未加载任何原始数据。请先调用 load_all_data() 方法。")
            return
        
        # 确定要转换的省份
        if provinces is None:
            provinces_to_process = list(self.raw_data.keys())
        else:
            provinces_to_process = [p for p in provinces if p in self.raw_data]
            missing_provinces = [p for p in provinces if p not in self.raw_data]
            if missing_provinces:
                print(f"警告: 未加载以下指定省份的数据: {missing_provinces}")
        
        if not provinces_to_process:
            print("错误: 没有有效的省份数据可供转换。")
            return
            
        print(f"将处理以下省份: {provinces_to_process}")
        os.makedirs(output_dir, exist_ok=True)
        
        processed_count = 0
        error_count = 0
        for province in provinces_to_process:
            print(f"  处理省份: {province}...")
            if province not in self.raw_data: # Double check just in case
                print(f"    错误: 在 raw_data 中找不到省份 {province} 的数据。")
                error_count += 1
                continue
                
            df_wide = self.raw_data[province].copy()

            # 再次检查必需列
            required_cols = [self.party_id_column, self.date_column] + self.point_columns
            missing_cols = [col for col in required_cols if col not in df_wide.columns]
            if missing_cols:
                print(f"    错误: 省份 '{province}' 的数据缺失必需列: {missing_cols}. 跳过。")
                error_count += 1
                continue

            # 确保日期列是 datetime 类型 (再次尝试转换)
            try:
                 df_wide[self.date_column] = pd.to_datetime(df_wide[self.date_column])
                 # 去除时区信息
                 if pd.api.types.is_datetime64_any_dtype(df_wide[self.date_column]) and df_wide[self.date_column].dt.tz is not None:
                     df_wide[self.date_column] = df_wide[self.date_column].dt.tz_localize(None)
            except Exception as date_conv_err:
                 print(f"    错误: 无法将列 '{self.date_column}' 转换为日期时间类型: {date_conv_err}. 跳过省份 '{province}'.")
                 error_count += 1
                 continue

            try:
                # 1. 数据透视 (Melt)
                id_vars = [self.party_id_column, self.date_column]
                df_long = pd.melt(df_wide,
                                  id_vars=id_vars,
                                  value_vars=self.point_columns, # 使用 P1 到 P96
                                  var_name='IntervalID',
                                  value_name='value') # 临时值列名
                print(f"    数据透视 (melt) 完成，生成 {len(df_long)} 行。")

                # 2. 清理和转换值
                initial_rows = len(df_long)
                df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce') # 尝试将 value 转为数字
                df_long.dropna(subset=['value'], inplace=True) # 移除转换失败或原始为 NaN 的行
                rows_after_cleaning = len(df_long)
                if initial_rows > rows_after_cleaning:
                    print(f"    清理无效/非数值数据，移除了 {initial_rows - rows_after_cleaning} 行。")
                
                if df_long.empty:
                    print(f"    警告: 清理后省份 '{province}' 没有有效数据，跳过保存。")
                    error_count += 1
                    continue

                # 3. 生成时间戳
                # 从 IntervalID (如 'P1', 'P96') 中提取序号 (1 到 96)
                try:
                    # 假定前缀固定，如果前缀变化则需要更复杂的逻辑
                    prefix_len = len(self.value_column_prefix)
                    df_long['PointNumber'] = df_long['IntervalID'].str[prefix_len:].astype(int)
                except Exception as point_err:
                    print(f"    错误: 无法从 'IntervalID' 列提取点序号 (预期格式: {self.value_column_prefix}数字): {point_err}")
                    print(f"      IntervalID 示例: {df_long['IntervalID'].unique()[:5]}") 
                    error_count += 1
                    continue 

                # 计算时间偏移 (点1 -> 0分钟, 点2 -> interval_minutes, ...)
                df_long['TimeDelta'] = pd.to_timedelta((df_long['PointNumber'] - 1) * interval_minutes, unit='m')

                # 计算最终的 datetime
                df_long['datetime'] = df_long[self.date_column] + df_long['TimeDelta']
                print(f"    时间戳生成完成。")

                # 4. 列重命名与选择
                target_value_column = data_type # 'load', 'pv', or 'wind'
                # 使用初始化时定义的 party_id_column
                df_final = df_long[['datetime', self.party_id_column, 'value']].copy()
                df_final.rename(columns={'value': target_value_column, 
                                        self.party_id_column: 'PARTY_ID'}, # 统一输出的ID列名为PARTY_ID
                                inplace=True)

                # 按时间和ID排序 (可选，但推荐)
                df_final.sort_values(by=['PARTY_ID', 'datetime'], inplace=True)
                # 移除潜在的重复时间戳（如果源数据有重叠日期）
                df_final.drop_duplicates(subset=['PARTY_ID', 'datetime'], keep='first', inplace=True)

                # 5. 保存CSV
                output_filename = f"timeseries_{data_type}_{province}.csv"
                output_path = os.path.join(output_dir, output_filename)
                print(f"    正在保存到: {output_path}...")
                # 使用 utf-8-sig 编码以确保 Excel 能正确识别中文并处理BOM头
                df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"      保存成功，共 {len(df_final)} 条记录。")
                processed_count += 1

            except Exception as e:
                print(f"    处理省份 '{province}' 时发生严重错误: {e}")
                import traceback
                traceback.print_exc()
                error_count += 1

        print(f"\\n时间序列转换处理完成. 成功处理 {processed_count} 个省份，{error_count} 个省份处理失败。")
        print(f"输出文件位于: {output_dir}")

    def get_timeseries_data(self, province, data_type='load', start_date=None, end_date=None):
        """
        (示例) 获取指定省份和类型的已处理时间序列数据。
        需要确保文件已通过 transform_to_timeseries 生成。
        """
        filename = f"timeseries_{data_type}_{province}.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"错误: 时间序列文件不存在: {filepath}。请先运行 transform_to_timeseries。")
            return None
            
        try:
            df = pd.read_csv(filepath, parse_dates=['datetime'])
            
            # 根据日期过滤 (如果需要)
            if start_date:
                df = df[df['datetime'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['datetime'] <= pd.to_datetime(end_date)]
                
            print(f"成功加载时间序列数据: {filepath}, 形状: {df.shape}")
            return df
        except Exception as e:
            print(f"读取时间序列文件 {filepath} 时出错: {e}")
            return None