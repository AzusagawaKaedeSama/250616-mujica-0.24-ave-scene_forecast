      
# torch_models.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import time
from torch.nn import Transformer, TransformerEncoderLayer, TransformerEncoder
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.parameter import Parameter
from torch.distributions import Normal
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler
import math
from models.abstract_model import AbstractTimeSeriesModel
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import traceback

# 改进的学习率调度器
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    """
    余弦退火学习率调度器，带预热阶段
    """
    def lr_lambda(current_step):
        # 预热阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # 退火阶段
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        # 确保学习率不低于最小值
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class TemporalDataset(Dataset):
    """统一时序数据集类"""
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        # 确保targets的形状正确
        if len(targets.shape) == 1:
            # 如果targets是1D，将其重塑为[num_samples, 1]
            targets = targets.reshape(-1, 1)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        
    def forward(self, pred, true):
        # 调整维度以确保兼容
        if pred.shape != true.shape:
            # 情况1: pred是[batch, pred_len]，true是[batch]
            if len(pred.shape) == 2 and len(true.shape) == 1:
                # 只使用第一个预测值进行比较，或者将true扩展为[batch, pred_len]
                # 方案A: 只用第一个预测值
                # pred = pred[:, 0]
                
                # 方案B: 将true扩展为[batch, pred_len]
                true = true.unsqueeze(1).expand(-1, pred.shape[1])
            
            # 情况2: pred是[batch]，true是[batch, pred_len]
            elif len(pred.shape) == 1 and len(true.shape) == 2:
                # 将pred扩展为[batch, pred_len]
                pred = pred.unsqueeze(1).expand(-1, true.shape[1])
        
        # 计算MSE损失
        mse_loss = self.mse(pred, true)
        
        # 计算MAPE损失，避免除以0
        epsilon = 1e-8
        mape_loss = torch.mean(torch.abs((true - pred) / (torch.abs(true) + epsilon)))
        
        return self.alpha * mse_loss + (1 - self.alpha) * mape_loss

class TemporalConvLSTM(nn.Module):
    """时空融合预测模型"""
    def __init__(self, input_size, seq_len=96, pred_len=4):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 空间特征提取
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),  # 添加BatchNorm
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2)
        )
        
        # 时间特征提取
        self.temporal_lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True # 双向LSTM
        )
        
        # 预测头
        lstm_output_dim = 128 * 2 # 双向 LSTM 输出维度是 hidden_size * 2
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim * (seq_len//2), 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, pred_len)
        )
        
    def forward(self, x):
        # 输入形状: (batch_size, seq_len, num_features)
        batch_size = x.size(0)
        
        # 空间特征提取
        x = x.permute(0, 2, 1)  # (batch, features, seq)
        x = self.spatial_conv(x)  # (batch, 64, seq//2)
        
        # 时间特征提取
        x = x.permute(0, 2, 1)  # (batch, seq//2, 64)
        lstm_out, _ = self.temporal_lstm(x)  # (batch, seq//2, 128)
        
        # 特征融合
        x = lstm_out.contiguous().view(batch_size, -1)  # (batch, seq//2 * 128)
        
        # print(f"中间特征形状: {x.shape}")
        
        # 预测输出 - 注意我们不使用额外的激活函数，直接输出原始值
        return self.fc(x)

class TorchForecaster:
    """PyTorch预测器封装类"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PyTorch 使用设备: {self.device}")
        print(f"CUDA 是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
            print(f"GPU 内存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
            print(f"当前内存使用量: {torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024:.2f} GB")
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def optimized_train(self, data_dict):
        """优化的训练流程"""
        # 初始化模型
        input_size = data_dict['train'][0].shape[-1]
        self.model = TemporalConvLSTM(
            input_size=input_size,
            seq_len=self.config['seq_length'],
            pred_len=self.config['pred_length']
        ).to(self.device)
        
        # 自动优化批量大小和学习率
        self.find_optimal_batch_size(data_dict)
        
        # 设置优化器
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config['lr'], weight_decay=1e-4)
        
        # 余弦退火学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config['epochs'], eta_min=self.config['lr'] / 10
        )
        
        # 使用组合损失函数
        criterion = CombinedLoss(alpha=0.7)
        
        # 创建数据加载器
        train_dataset = TemporalDataset(*data_dict['train'])
        val_dataset = TemporalDataset(*data_dict['val'])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size']*2,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        # 初始化混合精度训练
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()
        
        # 训练监控变量
        best_val_loss = float('inf')
        early_stop_counter = 0
        train_losses = []
        val_losses = []
        
        # 训练循环
        from time import time
        total_start = time()
        
        for epoch in range(self.config['epochs']):
            epoch_start = time()
            
            # 训练阶段
            self.forecaster.model.train()
            train_loss = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                # 处理维度不匹配
                if batch_y.dim() == 1 and self.config['pred_length'] > 1:
                    batch_y = batch_y.unsqueeze(1).expand(-1, self.config['pred_length'])
                
                self.optimizer.zero_grad(set_to_none=True)
                
                # 使用混合精度
                with autocast():
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                
                # 梯度缩放
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(self.optimizer)
                scaler.update()
                
                train_loss += loss.item() * batch_x.size(0)
            
            # 计算平均训练损失
            train_loss /= len(train_loader.dataset)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    
                    # 处理维度不匹配
                    if batch_y.dim() == 1 and self.config['pred_length'] > 1:
                        batch_y = batch_y.unsqueeze(1).expand(-1, self.config['pred_length'])
                    
                    # 对验证也使用混合精度
                    with autocast():
                        outputs = self.model(batch_x)
                        loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item() * batch_x.size(0)
            
            # 计算平均验证损失
            val_loss /= len(val_loader.dataset)
            
            # 记录损失
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # 更新学习率
            self.scheduler.step()
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                early_stop_counter = 0
                print(f"[+] 模型改进，已保存检查点")
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.config['patience']:
                    print(f"⚠ {self.config['patience']}个epoch没有改进，提前停止训练")
                    break
            
            epoch_time = time() - epoch_start
            print(f"Epoch {epoch+1}/{self.config['epochs']} | "
                f"训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f} | "
                f"学习率: {self.optimizer.param_groups[0]['lr']:.2e} | "
                f"耗时: {epoch_time:.2f}秒")
            
            # 打印GPU内存使用情况
            if torch.cuda.is_available():
                used_mem = torch.cuda.memory_allocated() / 1024**3
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"GPU内存使用: {used_mem:.2f}GB / {total_mem:.2f}GB")
        
        total_time = time() - total_start
        print(f"训练完成，总耗时: {total_time:.2f}秒，最佳验证损失: {best_val_loss:.4f}")
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_model.pth'))
        return self.model

    def find_optimal_batch_size(self, data_dict, initial_batch_size=32, max_batch_size=512):
        """查找最佳批量大小"""
        # 使用配置中的批量大小作为初始值
        if initial_batch_size is None:
            initial_batch_size = self.config['batch_size']
        
        batch_size = initial_batch_size
        original_lr = self.config['lr']
        
        # 获取输入形状
        if not hasattr(self, 'model') or self.model is None:
            print("警告：模型尚未初始化，无法进行批量大小优化")
            return batch_size
        
        print(f"开始寻找最佳批量大小（初始：{batch_size}）...")
        
        while batch_size < max_batch_size:
            try:
                # 尝试更大的批量大小
                next_batch_size = batch_size * 2
                
                # 创建一个测试批次
                dummy_input = torch.zeros((next_batch_size, self.config['seq_length'], 
                                        data_dict['train'][0].shape[-1])).to(self.device)
                
                # 测试前向传播
                self.model(dummy_input)
                
                # 如果成功，更新批量大小
                batch_size = next_batch_size
                torch.cuda.empty_cache()  # 清空缓存
                
                print(f"  测试批量大小 {batch_size} 成功")
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    print(f"  批量大小 {batch_size*2} 内存不足")
                    break
                else:
                    print(f"  出现非内存错误: {e}")
                    break
        
        # 根据批量大小调整学习率
        self.config['batch_size'] = batch_size
        self.config['lr'] = original_lr * (batch_size / initial_batch_size)  # 按比例调整学习率
        
        print(f"已优化批量大小为: {batch_size}, 学习率调整为: {self.config['lr']:.2e}")
        return batch_size

    def train(self, data_dict):
        """训练入口方法"""
        return self.optimized_train(data_dict)


    def train_backup(self, data_dict):
        """完整训练流程"""
        # 初始化模型
        input_size = data_dict['train'][0].shape[-1]
        self.model = TemporalConvLSTM(
            input_size=input_size,
            seq_len=self.config['seq_length'],
            pred_len=self.config['pred_length']
        ).to(self.device)
        
        print(f"模型已移至 {self.device} 设备")
        if torch.cuda.is_available():
            print(f"训练开始时 GPU 内存使用量: {torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024:.2f} GB")
        
        # 设置优化器
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config['lr'], weight_decay=1e-4)
        
        # 学习率调度 - 使用余弦退火
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config['epochs'],
            eta_min=self.config['lr'] / 10
        )
        
        # 使用组合损失函数
        criterion = CombinedLoss(alpha=0.7)
        
        # 创建数据加载器
        train_dataset = TemporalDataset(*data_dict['train'])
        val_dataset = TemporalDataset(*data_dict['val'])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size']*2,
            num_workers=2,
            pin_memory=True
        )
        
        best_val_loss = float('inf')
        early_stop_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config['epochs']):
            # 训练阶段
            self.forecaster.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(batch_x)
                
                # 确保形状匹配
                if outputs.shape != batch_y.shape:
                    print(f"警告: 输出形状 {outputs.shape} 与目标形状 {batch_y.shape} 不匹配")
                    # 可能需要调整形状
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度裁剪以避免梯度爆炸
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    
                    outputs = self.model(batch_x)
                    val_loss += criterion(outputs, batch_y).item() * batch_x.size(0)
            
            # 计算平均损失
            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录损失
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # 早停机制
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.config['patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # 打印训练信息
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{self.config['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.2e}")
            # 在每个epoch结束后打印GPU内存使用情况
            if torch.cuda.is_available():
                print(f"Epoch {epoch+1} GPU 内存使用量: {torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024:.2f} GB")
        
        # 训练结束后绘制损失曲线
        os.makedirs('results', exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/training_loss.png')
        plt.close()
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_model.pth'))

    def improved_train(self, data_dict):
        """改进的训练流程"""
        # 初始化模型 - 使用增强版模型
        input_size = data_dict['train'][0].shape[-1]
        self.model = EnhancedTemporalModel(
            input_size=input_size,
            seq_len=self.config['seq_length'],
            pred_len=self.config['pred_length']
        ).to(self.device)
        
        print(f"模型已移至 {self.device} 设备")
        
        # 设置优化器 - 使用RAdam优化器
        try:
            self.optimizer = optim.RAdam(
                self.model.parameters(), 
                lr=self.config['lr'], 
                weight_decay=1e-4
            )
        except AttributeError:
            # 如果RAdam不可用，回退到AdamW
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.config['lr'], 
                weight_decay=1e-4
            )
        
        # 使用自动权重损失函数
        criterion = AutoWeightedLoss().to(self.device)
        
        # 创建数据加载器
        train_dataset = TemporalDataset(*data_dict['train'])
        val_dataset = TemporalDataset(*data_dict['val'])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size']*2,
            num_workers=2,
            pin_memory=True
        )
        
        # 使用改进的学习率调度器
        total_steps = self.config['epochs'] * len(train_loader)
        warmup_steps = total_steps // 10  # 10%的步骤用于预热
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 训练监控
        best_val_loss = float('inf')
        early_stop_counter = 0
        train_losses = []
        val_losses = []
        
        # 创建结果目录
        os.makedirs('results', exist_ok=True)
        
        for epoch in range(self.config['epochs']):
            # 训练阶段
            self.forecaster.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(batch_x)
                
                # 确保形状匹配 - 处理维度问题
                if outputs.shape != batch_y.shape:
                    if len(outputs.shape) == 2 and len(batch_y.shape) == 2:
                        # 只有最后一个维度可能不匹配
                        min_dim = min(outputs.shape[1], batch_y.shape[1])
                        outputs = outputs[:, :min_dim]
                        batch_y = batch_y[:, :min_dim]
                    else:
                        print(f"警告: 输出形状 {outputs.shape} 与目标形状 {batch_y.shape} 不匹配")
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度裁剪以避免梯度爆炸
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.scheduler.step()  # 每批次更新学习率
                
                train_loss += loss.item() * batch_x.size(0)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            val_preds = []
            val_trues = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    
                    outputs = self.model(batch_x)
                    
                    # 确保形状匹配
                    if outputs.shape != batch_y.shape:
                        if len(outputs.shape) == 2 and len(batch_y.shape) == 2:
                            min_dim = min(outputs.shape[1], batch_y.shape[1])
                            outputs = outputs[:, :min_dim]
                            batch_y = batch_y[:, :min_dim]
                    
                    # 收集预测和真实值用于计算指标
                    val_preds.append(outputs.cpu().numpy())
                    val_trues.append(batch_y.cpu().numpy())
                    
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
            
            # 计算平均损失
            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)
            
            # 记录损失
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # 计算性能指标
            val_preds = np.concatenate(val_preds, axis=0)
            val_trues = np.concatenate(val_trues, axis=0)
            
            val_mae = np.mean(np.abs(val_preds - val_trues))
            val_rmse = np.sqrt(np.mean((val_preds - val_trues) ** 2))
            
            # 打印训练信息
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{self.config['epochs']} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f} | LR: {lr:.2e}")
            
            # 早停机制
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                early_stop_counter = 0
                
                # 保存当前最佳模型的预测结果可视化
                self._plot_validation_results(val_trues, val_preds, epoch+1)
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.config['patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # 训练结束后绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/training_loss.png')
        plt.close()

        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        return self.model

    # 添加到TorchForecaster类的辅助方法
    def _plot_validation_results(self, y_true, y_pred, epoch):
        """绘制验证集预测结果"""
        # 扁平化数组便于绘图
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # 确保长度一致
        min_len = min(len(y_true_flat), len(y_pred_flat))
        y_true_flat = y_true_flat[:min_len]
        y_pred_flat = y_pred_flat[:min_len]
        
        # 计算指标
        mae = np.mean(np.abs(y_true_flat - y_pred_flat))
        rmse = np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2))
        mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (np.abs(y_true_flat) + 1e-10))) * 100
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 绘制预测对比
        plt.subplot(2, 1, 1)
        plt.plot(y_true_flat[:96], 'b-', label='真实值', linewidth=2)
        plt.plot(y_pred_flat[:96], 'r--', label='预测值', linewidth=2)
        plt.title(f'验证集预测结果 (Epoch {epoch}) - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%')
        plt.xlabel('时间步 (15分钟间隔)')
        plt.ylabel('负荷')
        plt.legend()
        plt.grid(True)
        
        # 绘制误差
        plt.subplot(2, 1, 2)
        error = y_true_flat[:96] - y_pred_flat[:96]
        plt.bar(range(len(error)), error, color='g', alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('预测误差')
        plt.xlabel('时间步 (15分钟间隔)')
        plt.ylabel('误差')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'results/validation_epoch_{epoch}.png')
        plt.close()

    def evaluate(self, data_dict):
        """模型评估"""
        test_dataset = TemporalDataset(*data_dict['test'])
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size']*2,
            num_workers=2,
            pin_memory=True
        )
        
        self.model.eval()
        predictions = []  # 确保这是一个列表，而不是NumPy数组
        actuals = []      # 确保这是一个列表，而不是NumPy数组
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x).detach().cpu().numpy()
                
                # 如果是多维输出，我们可能需要选择特定的维度
                if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                    # 这里我们假设预测的是多个时间步，我们取所有时间步
                    for i in range(outputs.shape[0]):
                        predictions.extend(outputs[i])
                else:
                    # 单一预测值，直接添加
                    predictions.extend(outputs.flatten())
                
                # 同样处理实际值
                if len(batch_y.shape) > 1 and batch_y.shape[1] > 1:
                    for i in range(batch_y.shape[0]):
                        actuals.extend(batch_y[i].cpu().numpy())
                else:
                    actuals.extend(batch_y.cpu().numpy().flatten())
        
        # 转换为numpy数组（只在最后返回前转换）
        predictions_array = np.array(predictions)
        actuals_array = np.array(actuals)
        
        # 计算指标
        mae = np.mean(np.abs(predictions_array - actuals_array))
        rmse = np.sqrt(np.mean((predictions_array - actuals_array)**2))
        mape = np.mean(np.abs((actuals_array - predictions_array) / (np.abs(actuals_array) + 1e-8))) * 100
        
        print(f"\nTest Metrics:")
        print(f"MAE: {mae:.2f} MW")
        print(f"RMSE: {rmse:.2f} MW")
        print(f"MAPE: {mape:.2f}%")
        
        # 可视化结果
        self._plot_results(actuals_array, predictions_array)
        
        return predictions_array, actuals_array
    
    def _plot_results(self, actuals, predictions, num_samples=96):
        """结果可视化"""
        plt.figure(figsize=(15, 6))
        
        # 选择最近24小时数据
        plot_len = min(len(actuals), num_samples)
        plot_actual = actuals[:plot_len]
        plot_pred = predictions[:plot_len]
        
        plt.plot(plot_actual, label='实际负荷', alpha=0.8)
        plt.plot(plot_pred, '--', label='预测负荷')
        
        plt.title('负荷预测结果 (24小时)')
        plt.xlabel('时间步 (15分钟间隔)')
        plt.ylabel('电力负荷 (MW)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # 保存图像
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/forecast_results.png')
        plt.close()

    def enhanced_evaluate(self, data_dict):
        """增强版评估方法"""
        test_dataset = TemporalDataset(*data_dict['test'])
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size']*2,
            num_workers=2,
            pin_memory=True
        )
        
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x).detach().cpu().numpy()
                
                # 保存结果
                predictions.append(outputs)
                actuals.append(batch_y.numpy())
        
        # 合并结果
        predictions = np.vstack(predictions)
        actuals = np.vstack(actuals)
        
        # 确保形状一致性
        if predictions.shape != actuals.shape:
            min_shape = [min(p, a) for p, a in zip(predictions.shape, actuals.shape)]
            predictions = predictions[:min_shape[0], :min_shape[1]] if len(predictions.shape) > 1 else predictions[:min_shape[0]]
            actuals = actuals[:min_shape[0], :min_shape[1]] if len(actuals.shape) > 1 else actuals[:min_shape[0]]
        
        # 计算指标
        metrics = self._calculate_metrics(actuals, predictions)
        
        # 打印指标
        print("\n==== 测试集评估结果 ====")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        # 可视化结果
        self._enhanced_plot_results(actuals, predictions)
        
        # 返回结果，便于进一步分析
        return predictions, actuals, metrics

    def _calculate_metrics(self, y_true, y_pred):
        """计算多种评估指标"""
        # 确保是一维数组
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # 截取至相同长度
        min_len = min(len(y_true_flat), len(y_pred_flat))
        y_true_flat = y_true_flat[:min_len]
        y_pred_flat = y_pred_flat[:min_len]
        
        # 均方误差（MSE）
        mse = np.mean((y_true_flat - y_pred_flat) ** 2)
        
        # 均方根误差（RMSE）
        rmse = np.sqrt(mse)
        
        # 平均绝对误差（MAE）
        mae = np.mean(np.abs(y_true_flat - y_pred_flat))
        
        # 平均绝对百分比误差（MAPE）
        # 避免除以零
        epsilon = 1e-8
        mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (np.abs(y_true_flat) + epsilon))) * 100
        
        # 对称平均绝对百分比误差（SMAPE）
        # 更稳健的指标
        smape = 200 * np.mean(np.abs(y_pred_flat - y_true_flat) / (np.abs(y_pred_flat) + np.abs(y_true_flat) + epsilon))
        
        # R^2 (确定系数)
        if np.var(y_true_flat) == 0:
            r2 = 0  # 避免除以零
        else:
            r2 = 1 - (np.sum((y_true_flat - y_pred_flat) ** 2) / (np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)))
        
        # 返回所有指标
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'SMAPE': smape,
            'R^2': r2
        }

    def _enhanced_plot_results(self, actuals, predictions, num_samples=192):
        """增强版可视化结果，带有多图表布局和更详细的指标"""
        # 创建子图布局
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
        
        # 扁平化数组以便于绘图
        y_true_flat = actuals.flatten()
        y_pred_flat = predictions.flatten()
        
        # 确保长度一致
        min_len = min(len(y_true_flat), len(y_pred_flat))
        y_true_flat = y_true_flat[:min_len]
        y_pred_flat = y_pred_flat[:min_len]
        
        # 计算误差
        errors = y_true_flat - y_pred_flat
        
        # 选择要显示的数据点数量
        plot_len = min(len(y_true_flat), num_samples)
        
        # 1. 整体对比图
        ax1 = plt.subplot(gs[0, :])
        ax1.plot(range(plot_len), y_true_flat[:plot_len], 'b-', label='真实值', linewidth=2)
        ax1.plot(range(plot_len), y_pred_flat[:plot_len], 'r--', label='预测值', linewidth=2)
        
        # 添加误差区域
        ax1.fill_between(range(plot_len), 
                        y_true_flat[:plot_len], 
                        y_pred_flat[:plot_len], 
                        color='gray', alpha=0.3, label='误差')
        
        # 添加图例和标签
        ax1.set_title('负荷预测结果对比（实际值 vs 预测值）', fontsize=14)
        ax1.set_xlabel('时间步（15分钟间隔）', fontsize=12)
        ax1.set_ylabel('负荷（标准化）', fontsize=12)
        ax1.legend(loc='best', fontsize=12)
        ax1.grid(True)
        
        # 添加垂直线标记完整天数
        for day in range(1, plot_len // 96 + 1):
            ax1.axvline(x=day*96, color='gray', linestyle='--', alpha=0.5)
            
        # 2. 误差分布图
        ax2 = plt.subplot(gs[1, 0])
        ax2.hist(errors, bins=50, alpha=0.75, color='steelblue')
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_title('预测误差分布', fontsize=14)
        ax2.set_xlabel('误差值', fontsize=12)
        ax2.set_ylabel('频率', fontsize=12)
        ax2.grid(True)
        
        # 3. 误差时间序列图
        ax3 = plt.subplot(gs[1, 1])
        ax3.plot(range(plot_len), errors[:plot_len], 'g-', linewidth=1)
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_title('预测误差随时间变化', fontsize=14)
        ax3.set_xlabel('时间步（15分钟间隔）', fontsize=12)
        ax3.set_ylabel('误差值', fontsize=12)
        ax3.grid(True)
        
        # 4. 相关性散点图
        ax4 = plt.subplot(gs[2, 0])
        ax4.scatter(y_true_flat, y_pred_flat, alpha=0.5, color='darkblue')
        
        # 添加理想的对角线
        min_val = min(np.min(y_true_flat), np.min(y_pred_flat))
        max_val = max(np.max(y_true_flat), np.max(y_pred_flat))
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax4.set_title('预测值 vs 实际值相关性', fontsize=14)
        ax4.set_xlabel('实际值', fontsize=12)
        ax4.set_ylabel('预测值', fontsize=12)
        ax4.grid(True)
        
        # 5. 性能指标显示
        ax5 = plt.subplot(gs[2, 1])
        ax5.axis('off')  # 不显示坐标轴
        
        # 计算指标
        metrics = self._calculate_metrics(y_true_flat, y_pred_flat)
        
        # 创建指标文本
        metrics_text = "\n".join([
            f"MSE: {metrics['MSE']:.4f}",
            f"RMSE: {metrics['RMSE']:.4f}",
            f"MAE: {metrics['MAE']:.4f}",
            f"MAPE: {metrics['MAPE']:.2f}%",
            f"SMAPE: {metrics['SMAPE']:.2f}%",
            f"R²: {metrics['R^2']:.4f}"
        ])
        
        # 在图中添加指标文本
        ax5.text(0.5, 0.5, metrics_text, 
                horizontalalignment='center',
                verticalalignment='center', 
                transform=ax5.transAxes,
                fontsize=14,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.5))
        
        # 调整布局并保存
        plt.tight_layout()
        plt.savefig('results/enhanced_forecast_results.png', dpi=300, bbox_inches='tight')
        
        # 为了进一步分析，再创建每日负荷对比图
        if plot_len >= 96:  # 至少有一天的数据
            plt.figure(figsize=(15, 8))
            
            for day in range(min(plot_len // 96, 3)):  # 最多显示3天
                start_idx = day * 96
                end_idx = start_idx + 96
                
                plt.subplot(3, 1, day+1)
                plt.plot(range(96), y_true_flat[start_idx:end_idx], 'b-', label='真实值', linewidth=2)
                plt.plot(range(96), y_pred_flat[start_idx:end_idx], 'r--', label='预测值', linewidth=2)
                
                # 添加时间标记（每4小时）
                hour_ticks = list(range(0, 96, 16))  # 每16个点为4小时
                hour_labels = [f'{h//4}:00' for h in range(0, 24, 4)]
                plt.xticks(hour_ticks, hour_labels)
                
                plt.title(f'第{day+1}天负荷曲线对比', fontsize=14)
                plt.xlabel('时间', fontsize=12)
                plt.ylabel('负荷', fontsize=12)
                plt.legend(loc='best')
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('results/daily_forecast_comparison.png', dpi=300, bbox_inches='tight')
        
        plt.close('all')


class TorchConvTransformer(AbstractTimeSeriesModel):
    """包装TorchForecaster，使其符合AbstractTimeSeriesModel接口"""
    model_type = 'convtrans'
    
    def __init__(self, input_shape=None, **kwargs):
        super().__init__()
        # 配置默认值
        self.config = {
            'seq_length': 96,
            'pred_length': 1,
            'batch_size': 32,
            'lr': 5e-4,
            'epochs': 20,
            'patience': 10
        }
        
        # 更新配置
        if kwargs:
            self.config.update(kwargs)
            
        # 保存输入形状
        self._input_shape = input_shape
        
        # 如果提供了input_shape，设置seq_length
        if input_shape is not None:
            self.config['seq_length'] = input_shape[0]
            
        # 创建实际的模型
        self.forecaster = TorchForecaster(self.config)
        
    @property
    def input_shape(self):
        """实现抽象方法：返回模型输入形状"""
        return self._input_shape
    
    @property
    def output_shape(self):
        """实现抽象方法：返回模型输出形状"""
        return (self.config['pred_length'],)
        
    def train(self, X_train, y_train, X_val, y_val, epochs=None, batch_size=None, save_dir=None, callbacks=None):
        """训练模型"""
        if epochs:
            self.config['epochs'] = epochs
        if batch_size:
            self.config['batch_size'] = batch_size
            
        # 准备数据字典
        data_dict = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_val, y_val)  # 暂时用验证集代替测试集
        }
        
        # 训练模型
        self.forecaster.train(data_dict)
        
        if save_dir:
            self.save(save_dir)
            
        return self.forecaster.model
    
    def predict(self, X):
        """进行预测"""
        # 检查模型是否已训练
        if self.forecaster.model is None:
            raise ValueError("模型尚未训练")
            
        # 准备数据
        if isinstance(X, np.ndarray):
            # 转换为数据集
            dataset = TemporalDataset(X, np.zeros((len(X), self.config['pred_length'])))
            dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False)
            
            # 预测
            self.forecaster.model.eval()
            predictions = []
            with torch.no_grad():
                for batch_x, _ in dataloader:
                    batch_x = batch_x.to(self.forecaster.device)
                    outputs = self.forecaster.model(batch_x).detach().cpu().numpy()
                    predictions.append(outputs)
                    
            return np.vstack(predictions)
        else:
            raise ValueError("输入必须是numpy数组")
    
    def save(self, save_dir='models/convtrans'):
        """保存模型"""
        import os
        import json
        
        # 创建目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型参数
        model_path = os.path.join(save_dir, f'{self.model_type}_model.pth')
        torch.save(self.forecaster.model.state_dict(), model_path)
        
        # 保存配置
        config_path = os.path.join(save_dir, f'{self.model_type}_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f)
            
        # 保存输入形状
        input_shape_path = os.path.join(save_dir, 'input_shape.json')
        with open(input_shape_path, 'w') as f:
            json.dump(list(self._input_shape) if self._input_shape else None, f)
            
        print(f"模型已保存到目录: {save_dir}")
    
    @classmethod
    def load(cls, save_dir='models/convtrans'):
        """加载模型"""

        # 检查文件是否存在
        model_path = os.path.join(save_dir, f'{cls.model_type}_model.pth')
        config_path = os.path.join(save_dir, f'{cls.model_type}_config.json')
        input_shape_path = os.path.join(save_dir, 'input_shape.json')
        
        if not all(os.path.exists(p) for p in [model_path, config_path, input_shape_path]):
            raise FileNotFoundError(f"模型文件缺失，无法加载模型")
            
        # 加载配置
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # 加载输入形状
        with open(input_shape_path, 'r') as f:
            input_shape = json.load(f)
            if input_shape:
                input_shape = tuple(input_shape)
        
        # 创建模型实例
        model_instance = cls(input_shape, **config)
        
        # 初始化模型架构
        input_size = input_shape[1] if input_shape else 1
        model_instance.forecaster.model = TemporalConvLSTM(
            input_size=input_size,
            seq_len=config['seq_length'],
            pred_len=config['pred_length']
        ).to(model_instance.forecaster.device)
        
        # 加载模型参数
        model_instance.forecaster.model.load_state_dict(
            torch.load(model_path, map_location=model_instance.forecaster.device, weights_only=False)
        )
        
        return model_instance

class EnhancedTemporalModel(nn.Module):
    """增强版时空融合预测模型，添加注意力机制和残差连接"""
    def __init__(self, input_size, seq_len=96, pred_len=4, 
                 batch_norm_momentum=0.1, batch_norm_eps=1e-5):
        """
        初始化增强的时序模型
        
        Args:
            input_size: 输入特征数
            seq_len: 输入序列长度
            pred_len: 预测时间步数
            batch_norm_momentum: BatchNorm层的动量参数
            batch_norm_eps: BatchNorm层的epsilon参数
        """
        super(EnhancedTemporalModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 添加BatchNorm提升训练稳定性
        self.input_norm = nn.BatchNorm1d(input_size, momentum=batch_norm_momentum, eps=batch_norm_eps)
        
        # 一维卷积层，负责捕获局部时间模式
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32, momentum=batch_norm_momentum, eps=batch_norm_eps),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64, momentum=batch_norm_momentum, eps=batch_norm_eps),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32, momentum=batch_norm_momentum, eps=batch_norm_eps),
            nn.ReLU()
        )
        
        # 增加一个Dropout层，防止过拟合
        self.dropout1 = nn.Dropout(0.2)
        
        # LSTM层捕获长期依赖关系，使用双向以捕获未来和过去的上下文
        self.lstm = nn.LSTM(32, 64, batch_first=True, bidirectional=True)
        
        # 增加一个Dropout层
        self.dropout2 = nn.Dropout(0.2)
        
        # 全连接层，将序列映射到预测目标
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * seq_len, 128),
            nn.LayerNorm(128, eps=batch_norm_eps),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, momentum=batch_norm_momentum, eps=batch_norm_eps),
            nn.ReLU(),
            nn.Linear(64, pred_len)
        )
        
    def forward(self, x):
        """
        前向传播过程
        
        Args:
            x: 输入特征 (batch_size, seq_len, features)
            
        Returns:
            torch.Tensor: 预测结果 (batch_size, pred_len)
        """
        # 检查输入格式是否正确
        if x.dim() != 3:
            raise ValueError(f"输入维度必须为3 (batch_size, seq_len, features)，实际为: {x.shape}")
            
        batch_size, seq_len, n_features = x.shape
        
        # 检查NaN/Inf输入以便调试
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("警告: 输入数据包含NaN或Inf值")
            x = torch.clamp(x.clone(), -1e9, 1e9)  # 限制数值范围
            x[torch.isnan(x)] = 0.0  # 将NaN替换为0
            
        try:
            # 首先，对输入的每个特征进行归一化
            input_reshaped = x.reshape(-1, n_features).t()  # 转置以使特征成为第一维度
            input_normed = self.input_norm(input_reshaped)
            input_normed = input_normed.t().reshape(batch_size, seq_len, n_features)  # 恢复原始形状
            
            # 重新排列输入，进行卷积操作
            input_conv = input_normed.permute(0, 2, 1)  # 变为(batch_size, features, seq_len)
            
            # 应用卷积层
            conv_out = self.conv_block(input_conv)  # (batch_size, 32, seq_len)
            
            # 重新排列以应用LSTM
            lstm_in = conv_out.permute(0, 2, 1)  # 变为(batch_size, seq_len, 32)
            lstm_in = self.dropout1(lstm_in)
            
            # 应用LSTM
            lstm_out, _ = self.lstm(lstm_in)  # (batch_size, seq_len, 128)
            lstm_out = self.dropout2(lstm_out)
            
            # 展平输出
            fc_in = lstm_out.reshape(batch_size, -1)  # (batch_size, seq_len*128)
            
            # 应用全连接层得到预测
            output = self.fc_layers(fc_in)  # (batch_size, pred_len)
            
            # 检查输出是否有NaN或Inf
            if torch.isnan(output).any() or torch.isinf(output).any():
                print("警告: 模型输出包含NaN或Inf值")
                # 尝试修复：替换为有界值
                output = torch.nan_to_num(output, nan=0.0, posinf=1e3, neginf=-1e3)
                
            return output
            
        except Exception as e:
            print(f"前向传播出错: {e}")
            traceback.print_exc()
            # 在出错的情况下，返回全零张量作为备用
            return torch.zeros(batch_size, self.pred_len, device=x.device)

class AutoWeightedLoss(nn.Module):
    """自动学习权重的多任务损失函数"""
    def __init__(self):
        super(AutoWeightedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
        # 可学习的权重参数
        self.mse_weight = nn.Parameter(torch.ones(1, requires_grad=True))
        self.mae_weight = nn.Parameter(torch.ones(1, requires_grad=True))
        self.mape_weight = nn.Parameter(torch.ones(1, requires_grad=True))
        
    def forward(self, pred, true):
        # 计算各损失
        mse = self.mse_loss(pred, true)
        mae = self.mae_loss(pred, true)
        if pred.shape != true.shape:
            if len(pred.shape) == 2 and len(true.shape) == 1:
                # 方案1：取预测的第一个时间步
                pred = pred[:, 0]
                
        # MAPE损失
        epsilon = 1e-8
        mape = torch.mean(torch.abs((true - pred) / (torch.abs(true) + epsilon)))
        
        # 动态调整权重
        # 权重的平方确保非负性，并除以和以确保归一化
        weights_sum = self.mse_weight**2 + self.mae_weight**2 + self.mape_weight**2 + 1e-8
        
        mse_weight = (self.mse_weight**2) / weights_sum
        mae_weight = (self.mae_weight**2) / weights_sum
        mape_weight = (self.mape_weight**2) / weights_sum
        
        # 组合损失
        loss = mse_weight * mse + mae_weight * mae + mape_weight * mape
        return loss
    
class PeakAwareLoss(nn.Module):
    """高峰感知损失函数，对高峰和低谷时段的误差赋予不同权重"""
    
    def __init__(self, base_criterion=None, peak_weight=5.0, valley_weight=1.5,
                 peak_hours=(8, 20), valley_hours = (0,7), is_workday_fn=None):
        """
        初始化高峰感知损失函数
        
        Args:
            base_criterion (nn.Module): 基础损失函数，如果为None则使用MSELoss
            peak_weight (float): 高峰时段的权重倍数
            valley_weight (float): 低谷时段的权重倍数
            peak_hours (tuple): 高峰时段的开始和结束小时
            valley_hours (tuple): 低谷时段的开始和结束小时
            is_workday_fn (callable): 判断是否为工作日的函数
        """
        super(PeakAwareLoss, self).__init__()
        self.base_criterion = base_criterion if base_criterion is not None else nn.MSELoss(reduction='none')
        self.peak_weight = peak_weight
        self.valley_weight = valley_weight
        self.peak_start, self.peak_end = peak_hours
        self.valley_start, self.valley_end = valley_hours
        self.is_workday_fn = is_workday_fn
    
    def forward(self, pred, true, timestamps=None, is_peak=None, is_valley=None):
        """
        计算加权损失
        
        Args:
            pred (torch.Tensor): 预测值
            true (torch.Tensor): 真实值
            timestamps (torch.Tensor): 时间戳信息
            is_peak (torch.Tensor): 是否为高峰时段的标记
            is_valley (torch.Tensor): 是否为低谷时段的标记
        
        Returns:
            torch.Tensor: 加权损失
        """
        # 确保张量维度匹配
        if pred.dim() > true.dim():
            pred = pred.squeeze(-1)  # 从[batch_size, 1]变为[batch_size]
        elif true.dim() > pred.dim():
            true = true.squeeze(-1)  # 从[batch_size, 1]变为[batch_size]
        
        # 检查并处理NaN/Inf
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            # 用真实值替换NaN/Inf
            pred = torch.where(torch.isnan(pred) | torch.isinf(pred), true.detach(), pred)
            print("警告: 预测值中发现NaN/Inf，已替换为真实值")

        # 计算基础损失
        base_loss = self.base_criterion(pred, true)
        
        # 如果没有提供高峰/低谷标记，使用平均损失
        if is_peak is None and is_valley is None and timestamps is None:
            return torch.mean(base_loss)
            
        # 使用提供的高峰标记
        if is_peak is not None:
            weights = torch.ones_like(base_loss)
            weights = torch.where(is_peak == 1, self.peak_weight * weights, weights)
            
            if is_valley is not None:
                weights = torch.where(is_valley == 1, self.valley_weight * weights, weights)
                
            return torch.mean(weights * base_loss)
        
        # 使用时间戳推断高峰/低谷
        if timestamps is not None:
            weights = torch.ones_like(base_loss)
            
            # 提取小时信息
            if isinstance(timestamps, torch.Tensor) and timestamps.dim() > 0:
                # 假设timestamps是小时值
                hours = timestamps
                
                # 识别高峰时段
                peak_mask = (hours >= self.peak_start) & (hours <= self.peak_end)
                weights[peak_mask] = self.peak_weight
                
                # 识别低谷时段
                valley_mask = (hours >= self.valley_start) & (hours <= self.valley_end)
                weights[valley_mask] = self.valley_weight
            
            return torch.mean(weights * base_loss)
        
        # 默认情况
        return torch.mean(base_loss)

# 工作日判断辅助函数
def is_workday(timestamp):
    """判断是否为工作日（周一至周五）"""
    if isinstance(timestamp, (str, np.datetime64)):
        timestamp = pd.Timestamp(timestamp)
    return timestamp.dayofweek < 5


# class PeakAwareConvTransformer(TorchConvTransformer):
#     """具有高峰感知能力的卷积-Transformer模型"""
#     model_type = 'convtrans_peak'
    
#     def __init__(self, input_shape=None, **kwargs):
#         super().__init__(input_shape, **kwargs)
        
#         # 高峰感知相关配置
#         self.use_peak_loss = kwargs.get('use_peak_loss', True)
#         self.peak_weight = kwargs.get('peak_weight', 5.0)
#         self.valley_weight = kwargs.get('valley_weight', 1.5)
#         self.peak_hours = kwargs.get('peak_hours', (8, 20))
#         self.valley_hours = kwargs.get('valley_hours', (0, 7))
    
#     def train_with_peak_awareness_v0(self, X_train, y_train, X_val, y_val, 
#                                  train_is_peak=None, val_is_peak=None,
#                                  epochs=None, batch_size=None, save_dir=None):
#         """使用高峰感知损失函数进行训练"""
#         if epochs:
#             self.config['epochs'] = epochs
#         if batch_size:
#             self.config['batch_size'] = batch_size
        
#         # 创建数据集和加载器
#         train_dataset = PeakAwareDataset(X_train, y_train, train_is_peak)
#         val_dataset = PeakAwareDataset(X_val, y_val, val_is_peak)
        
#         train_loader = torch.utils.data.DataLoader(
#             train_dataset, 
#             batch_size=self.config['batch_size'],
#             shuffle=True,
#             num_workers=2,
#             pin_memory=True,
#             drop_last=True  # 丢弃最后一个不完整批次，避免BatchNorm错误
#         )
        
#         val_loader = torch.utils.data.DataLoader(
#             val_dataset,
#             batch_size=self.config['batch_size']*2,
#             shuffle=False,
#             num_workers=2,
#             pin_memory=True,
#             drop_last=True  # 丢弃最后一个不完整批次，避免BatchNorm错误
#         )
        
#         # 检查数据集大小是否足够
#         min_batch_size = 2  # 批标准化要求至少2个样本
#         if len(train_dataset) < min_batch_size or len(val_dataset) < min_batch_size:
#             raise ValueError(f"数据集太小，无法使用批标准化。训练集大小: {len(train_dataset)}，验证集大小: {len(val_dataset)}，最小要求: {min_batch_size}")
        
#         # 确保批大小合适
#         if self.config['batch_size'] < min_batch_size:
#             print(f"警告: 批大小 {self.config['batch_size']} 太小，已调整为 {min_batch_size}")
#             self.config['batch_size'] = min_batch_size
        
#         # 初始化模型（如果尚未初始化）
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         if self.forecaster.model is None:
#             self.forecaster.model = TemporalConvLSTM(
#                 input_size=X_train.shape[2],
#                 seq_len=self.config['seq_length'],
#                 pred_len=self.config['pred_length']
#             ).to(device)
        
#         # 设置优化器
#         optimizer = torch.optim.AdamW(
#             self.forecaster.model.parameters(), 
#             lr=self.config['lr'],
#             weight_decay=1e-4
#         )
        
#         # 设置学习率调度器
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, 
#             T_max=self.config['epochs'],
#             eta_min=self.config['lr'] / 10
#         )
        
#         # 设置高峰感知损失函数
#         criterion = PeakAwareLoss(
#             base_criterion=nn.MSELoss(reduction='none'),
#             peak_weight=self.peak_weight,
#             valley_weight=self.peak_weight,
#             peak_hours=self.peak_hours,
#             valley_hours=self.valley_hours,
#             is_workday_fn=is_workday
#         )
        
#         # 训练监控变量
#         best_val_loss = float('inf')
#         early_stop_counter = 0
#         train_losses = []
#         val_losses = []
        
#         # 使用新的API初始化GradScaler
#         try:
#             scaler = torch.amp.GradScaler('cuda')
#         except:
#             # 兼容旧版PyTorch
#             from torch.cuda.amp import GradScaler
#             scaler = GradScaler()
        
#         # 训练循环
#         for epoch in range(self.config['epochs']):
#             # 训练阶段
#             self.forecaster.model.train()
#             train_loss = 0
#             num_batches = 0
            
#             for batch_x, batch_y, batch_is_peak, batch_is_valley in train_loader:
#                 batch_x = batch_x.to(device)
#                 batch_y = batch_y.to(device)
#                 batch_is_peak = batch_is_peak.to(device)
#                 batch_is_valley = batch_is_valley.to(device)
                
#                 optimizer.zero_grad()
#                 outputs = self.forecaster.model(batch_x)
                
#                 # 使用高峰感知损失函数
#                 loss = criterion(outputs, batch_y, is_peak=batch_is_peak)
                
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(self.forecaster.model.parameters(), 1.0)
#                 optimizer.step()
                
#                 train_loss += loss.item()
#                 num_batches += 1
            
#             # 计算平均训练损失
#             train_loss /= num_batches
            
#             # 验证阶段
#             self.forecaster.model.eval()
#             val_loss = 0
#             num_val_batches = 0
            
#             with torch.no_grad():
#                 for batch_x, batch_y, batch_is_peak, batch_is_valley in val_loader:
#                     batch_x = batch_x.to(device)
#                     batch_y = batch_y.to(device)
#                     batch_is_peak = batch_is_peak.to(device)
#                     batch_is_valley = batch_is_valley.to(device)
                    
#                     outputs = self.forecaster.model(batch_x)
#                     loss = criterion(outputs, batch_y, is_peak=batch_is_peak)
                    
#                     val_loss += loss.item()
#                     num_val_batches += 1
            
#             # 计算平均验证损失
#             if num_val_batches > 0:
#                 val_loss /= num_val_batches
#             else:
#                 print("警告: 验证过程中没有批次数据，跳过计算验证损失")
#                 val_loss = float('inf')  # 设为无穷大，确保不会被错误地识别为最佳模型
            
#             # 更新学习率
#             scheduler.step()
            
#             # 记录损失
#             train_losses.append(train_loss)
#             val_losses.append(val_loss)
            
#             # 打印进度
#             lr = optimizer.param_groups[0]['lr']
#             print(f"Epoch {epoch+1}/{self.config['epochs']} | "
#                   f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.2e}")
                            
#             if val_loss < 0.005:      # 当验证损失优于阈值时提前停止
#                 print("验证损失已低于阈值，提前停止训练。")
#                 break

#             # 早停检查
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 if save_dir:
#                     os.makedirs(save_dir, exist_ok=True)
#                     torch.save(self.forecaster.model.state_dict(), f"{save_dir}/best_model.pth")
#                 early_stop_counter = 0
#                 print(f"[+] 模型改进，已保存检查点")
#             else:
#                 early_stop_counter += 1
#                 if early_stop_counter >= self.config['patience']:
#                     print(f"⚠ {self.config['patience']}个epoch没有改进，提前停止训练")
#                     break

        
#         # 训练结束后绘制损失曲线
#         if save_dir:
#             plt.figure(figsize=(10, 6))
#             plt.plot(train_losses, label='Train Loss')
#             plt.plot(val_losses, label='Validation Loss')
#             plt.title('Training and Validation Loss (with Peak Awareness)')
#             plt.xlabel('Epoch')
#             plt.ylabel('Loss')
#             plt.legend()
#             plt.grid(True)
#             plt.savefig(f"{save_dir}/training_loss.png")
#             plt.close()
            
#             # 加载最佳模型
#             best_model_path = f"{save_dir}/best_model.pth"
#             if os.path.exists(best_model_path):
#                 self.forecaster.model.load_state_dict(torch.load(best_model_path))
        
#         # 返回训练历史字典
#         return {
#             'train_loss': train_losses,
#             'val_loss': val_losses,
#             'epochs': list(range(1, len(train_losses) + 1))
#         }

#     def train_with_peak_awareness(self, X_train, y_train, X_val, y_val, 
#                                  train_is_peak=None, val_is_peak=None,
#                                  epochs=None, batch_size=None, save_dir=None):
#         """使用高峰感知损失函数进行训练"""
#         if epochs:
#             self.config['epochs'] = epochs
#         if batch_size:
#             self.config['batch_size'] = batch_size
        
#         # 创建数据集和加载器
#         train_dataset = PeakAwareDataset(X_train, y_train, train_is_peak)
#         val_dataset = PeakAwareDataset(X_val, y_val, val_is_peak)
        
#         train_loader = torch.utils.data.DataLoader(
#             train_dataset, 
#             batch_size=self.config['batch_size'],
#             shuffle=True,
#             num_workers=2,
#             pin_memory=True,
#             drop_last=True  # 丢弃最后一个不完整批次，避免BatchNorm错误
#         )
        
#         val_loader = torch.utils.data.DataLoader(
#             val_dataset,
#             batch_size=min(self.config['batch_size']*2, max(min_batch_size, len(val_dataset) // 2)),  # 确保验证批次大小适合数据集大小
#             shuffle=False,
#             num_workers=2,
#             pin_memory=True,
#             drop_last=False  # 不丢弃最后一个批次，确保所有验证样本都被使用
#         )
        
#         # 打印验证加载器信息
#         print(f"验证数据加载器批次大小: {val_loader.batch_size}, 批次数量: {len(val_loader)}")
        
#         # 检查数据集大小是否足够
#         min_batch_size = 2  # 批标准化要求至少2个样本
#         if len(train_dataset) < min_batch_size:
#             raise ValueError(f"训练集太小，无法使用批标准化。训练集大小: {len(train_dataset)}，最小要求: {min_batch_size}")
        
#         # 确保批大小合适
#         if self.config['batch_size'] < min_batch_size:
#             print(f"警告: 批大小 {self.config['batch_size']} 太小，已调整为 {min_batch_size}")
#             self.config['batch_size'] = min_batch_size
        
#         # 初始化模型（如果尚未初始化）
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         if self.forecaster.model is None:
#             self.forecaster.model = TemporalConvLSTM(
#                 input_size=X_train.shape[2],
#                 seq_len=self.config['seq_length'],
#                 pred_len=self.config['pred_length']
#             ).to(device)
        
#         # 设置优化器
#         optimizer = torch.optim.AdamW(
#             self.forecaster.model.parameters(), 
#             lr=self.config['lr'],
#             weight_decay=1e-4
#         )
        
#         # 设置学习率调度器
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, 
#             T_max=self.config['epochs'],
#             eta_min=self.config['lr'] / 10
#         )
        
#         # 设置高峰感知损失函数
#         criterion = PeakAwareLoss(
#             base_criterion=nn.MSELoss(reduction='none'),
#             peak_weight=self.peak_weight,
#             valley_weight=self.peak_weight,
#             peak_hours=self.peak_hours,
#             valley_hours=self.valley_hours,
#             is_workday_fn=is_workday
#         )
        
#         # 训练监控变量
#         best_val_loss = float('inf')
#         early_stop_counter = 0
#         train_losses = []
#         val_losses = []
#         scaler = GradScaler()
        
#         # 训练循环
#         for epoch in range(self.config['epochs']):
#             # 训练阶段
#             self.forecaster.model.train()
#             train_loss = 0
#             num_batches = 0
            
#             for batch_x, batch_y, batch_is_peak, batch_is_valley in train_loader:
#                 batch_x = batch_x.to(device)
#                 batch_y = batch_y.to(device)
#                 batch_is_peak = batch_is_peak.to(device)
#                 batch_is_valley = batch_is_valley.to(device)
                
#                 optimizer.zero_grad()
#                 outputs = self.forecaster.model(batch_x)
                
#                 # 使用高峰感知损失函数
#                 loss = criterion(outputs, batch_y, is_peak=batch_is_peak)

#                 scaler.scale(loss).backward()
#                 scaler.step(optimizer)
#                 scaler.update()
#                 # loss.backward()
#                 # torch.nn.utils.clip_grad_norm_(self.forecaster.model.parameters(), 1.0)
#                 # optimizer.step()
                
#                 train_loss += loss.item()
#                 num_batches += 1
            
#             # 计算平均训练损失
#             if num_batches > 0:
#                 train_loss /= num_batches
            
#             # 验证阶段
#             self.forecaster.model.eval()
#             val_loss = 0
#             num_val_batches = 0
            
#             with torch.no_grad():
#                 for batch_x, batch_y, batch_is_peak, batch_is_valley in val_loader:
#                     batch_x = batch_x.to(device)
#                     batch_y = batch_y.to(device)
#                     batch_is_peak = batch_is_peak.to(device)
#                     batch_is_valley = batch_is_valley.to(device)
                    
#                     outputs = self.forecaster.model(batch_x)
#                     loss = criterion(outputs, batch_y, is_peak=batch_is_peak)
                    
#                     val_loss += loss.item()
#                     num_val_batches += 1
            
#             # 计算平均验证损失
#             if num_val_batches > 0:
#                 val_loss /= num_val_batches
#             else:
#                 print("警告: 验证过程中没有批次数据，跳过计算验证损失")
#                 val_loss = float('inf')  # 设为无穷大，确保不会被错误地识别为最佳模型
            
#             # 更新学习率
#             scheduler.step()
            
#             # 记录损失
#             train_losses.append(train_loss)
#             val_losses.append(val_loss)
            
#             # 打印进度
#             lr = optimizer.param_groups[0]['lr']
#             print(f"Epoch {epoch+1}/{self.config['epochs']} | "
#                   f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.2e}")
                            
#             if val_loss < 0.005:      # 当验证损失优于阈值时提前停止
#                 print("验证损失已低于阈值，提前停止训练。")
#                 break
            
#             # 早停检查
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 if save_dir:
#                     os.makedirs(save_dir, exist_ok=True)
#                     torch.save(self.forecaster.model.state_dict(), f"{save_dir}/best_model.pth")
#                 early_stop_counter = 0
#                 print(f"[+] 模型改进，已保存检查点")
#             else:
#                 early_stop_counter += 1
#                 if early_stop_counter >= self.config['patience']:
#                     print(f"⚠ {self.config['patience']}个epoch没有改进，提前停止训练")
#                     break

        
#         # 训练结束后绘制损失曲线
#         if save_dir:
#             plt.figure(figsize=(10, 6))
#             plt.plot(train_losses, label='Train Loss')
#             plt.plot(val_losses, label='Validation Loss')
#             plt.title('Training and Validation Loss (with Peak Awareness)')
#             plt.xlabel('Epoch')
#             plt.ylabel('Loss')
#             plt.legend()
#             plt.grid(True)
#             plt.savefig(f"{save_dir}/training_loss.png")
#             plt.close()
            
#             # 加载最佳模型
#             best_model_path = f"{save_dir}/best_model.pth"
#             if os.path.exists(best_model_path):
#                 self.forecaster.model.load_state_dict(torch.load(best_model_path))
        
#         # 返回训练历史字典
#         return {
#             'train_loss': train_losses,
#             'val_loss': val_losses,
#             'epochs': list(range(1, len(train_losses) + 1))
#         }
    

#     def predict(self, X):
#         """进行预测，与父类相同"""
#         return super().predict(X)
class PeakAwareConvTransformer(TorchConvTransformer):
    """具有高峰感知能力的卷积-Transformer模型"""
    model_type = 'convtrans_peak'
    
    def __init__(self, input_shape=None, **kwargs):
        super().__init__(input_shape, **kwargs)
        
        # 高峰感知相关配置
        self.use_peak_loss = kwargs.get('use_peak_loss', True)
        self.peak_weight = kwargs.get('peak_weight', 5.0)
        self.valley_weight = kwargs.get('valley_weight', 1.5)
        self.peak_hours = kwargs.get('peak_hours', (8, 20))
        self.valley_hours = kwargs.get('valley_hours', (0, 7))
    
    def train_with_peak_awareness_v0(self, X_train, y_train, X_val, y_val, 
                                 train_is_peak=None, val_is_peak=None,
                                 epochs=None, batch_size=None, save_dir=None):
        """使用高峰感知损失函数进行训练"""
        if epochs:
            self.config['epochs'] = epochs
        if batch_size:
            self.config['batch_size'] = batch_size
        
        # 创建数据集和加载器
        train_dataset = PeakAwareDataset(X_train, y_train, train_is_peak)
        val_dataset = PeakAwareDataset(X_val, y_val, val_is_peak)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config['batch_size']*2,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # 初始化模型（如果尚未初始化）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.forecaster.model is None:
            self.forecaster.model = TemporalConvLSTM(
                input_size=X_train.shape[2],
                seq_len=self.config['seq_length'],
                pred_len=self.config['pred_length']
            ).to(device)
        
        # 设置优化器
        optimizer = torch.optim.AdamW(
            self.forecaster.model.parameters(), 
            lr=self.config['lr'],
            weight_decay=1e-4
        )
        
        # 设置学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.config['epochs'],
            eta_min=self.config['lr'] / 10
        )
        
        # 设置高峰感知损失函数
        criterion = PeakAwareLoss(
            base_criterion=nn.MSELoss(reduction='none'),
            peak_weight=self.peak_weight,
            valley_weight=self.peak_weight,
            peak_hours=self.peak_hours,
            valley_hours=self.valley_hours,
            is_workday_fn=is_workday
        )
        
        # 训练监控变量
        best_val_loss = float('inf')
        early_stop_counter = 0
        train_losses = []
        val_losses = []
        scaler = GradScaler()
        
        # 训练循环
        for epoch in range(self.config['epochs']):
            # 训练阶段
            self.forecaster.model.train()
            train_loss = 0
            num_batches = 0
            
            for batch_x, batch_y, batch_is_peak, batch_is_valley in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_is_peak = batch_is_peak.to(device)
                batch_is_valley = batch_is_valley.to(device)
                
                optimizer.zero_grad()
                outputs = self.forecaster.model(batch_x)
                
                # 使用高峰感知损失函数
                loss = criterion(outputs, batch_y, is_peak=batch_is_peak)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.forecaster.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            # 计算平均训练损失
            train_loss /= num_batches
            
            # 验证阶段
            self.forecaster.model.eval()
            val_loss = 0
            num_val_batches = 0
            
            with torch.no_grad():
                for batch_x, batch_y, batch_is_peak, batch_is_valley in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    batch_is_peak = batch_is_peak.to(device)
                    batch_is_valley = batch_is_valley.to(device)
                    
                    outputs = self.forecaster.model(batch_x)
                    loss = criterion(outputs, batch_y, is_peak=batch_is_peak)
                    
                    val_loss += loss.item()
                    num_val_batches += 1
            
            # 计算平均验证损失
            val_loss /= num_val_batches
            
            # 更新学习率
            scheduler.step()
            
            # 记录损失
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # 打印进度
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{self.config['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.2e}")
                            
            if val_loss < 0.005:      # 当验证损失优于阈值时提前停止
                print("验证损失已低于阈值，提前停止训练。")
                break

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(self.forecaster.model.state_dict(), f"{save_dir}/best_model.pth")
                early_stop_counter = 0
                print(f"[+] 模型改进，已保存检查点")
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.config['patience']:
                    print(f"⚠ {self.config['patience']}个epoch没有改进，提前停止训练")
                    break

        
        # 训练结束后绘制损失曲线
        if save_dir:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.title('Training and Validation Loss (with Peak Awareness)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{save_dir}/training_loss.png")
            plt.close()
            
            # 加载最佳模型
            best_model_path = f"{save_dir}/best_model.pth"
            if os.path.exists(best_model_path):
                self.forecaster.model.load_state_dict(torch.load(best_model_path))
        
        # 返回训练历史字典
        return {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'epochs': list(range(1, len(train_losses) + 1))
        }

    def train_with_peak_awareness(self, X_train, y_train, X_val, y_val, 
                                 train_is_peak=None, val_is_peak=None,
                                 epochs=None, batch_size=None, save_dir=None):
        """使用高峰感知损失函数进行训练"""
        if epochs:
            self.config['epochs'] = epochs
        if batch_size:
            self.config['batch_size'] = batch_size
        
        # 创建数据集和加载器
        train_dataset = PeakAwareDataset(X_train, y_train, train_is_peak)
        val_dataset = PeakAwareDataset(X_val, y_val, val_is_peak)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config['batch_size']*2,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # 初始化模型（如果尚未初始化）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.forecaster.model is None:
            self.forecaster.model = TemporalConvLSTM(
                input_size=X_train.shape[2],
                seq_len=self.config['seq_length'],
                pred_len=self.config['pred_length']
            ).to(device)
        
        # 设置优化器
        optimizer = torch.optim.AdamW(
            self.forecaster.model.parameters(), 
            lr=self.config['lr'],
            weight_decay=1e-4
        )
        
        # 设置学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.config['epochs'],
            eta_min=self.config['lr'] / 10
        )
        
        # 设置高峰感知损失函数
        criterion = PeakAwareLoss(
            base_criterion=nn.MSELoss(reduction='none'),
            peak_weight=self.peak_weight,
            valley_weight=self.peak_weight,
            peak_hours=self.peak_hours,
            valley_hours=self.valley_hours,
            is_workday_fn=is_workday
        )
        
        # 训练监控变量
        best_val_loss = float('inf')
        early_stop_counter = 0
        train_losses = []
        val_losses = []
        scaler = GradScaler()
        
        # 训练循环
        for epoch in range(self.config['epochs']):
            # 训练阶段
            self.forecaster.model.train()
            train_loss = 0
            num_batches = 0
            
            for batch_x, batch_y, batch_is_peak, batch_is_valley in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_is_peak = batch_is_peak.to(device)
                batch_is_valley = batch_is_valley.to(device)
                
                optimizer.zero_grad()
                outputs = self.forecaster.model(batch_x)
                
                # 使用高峰感知损失函数
                loss = criterion(outputs, batch_y, is_peak=batch_is_peak)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.forecaster.model.parameters(), 1.0)
                # optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            # 计算平均训练损失
            if num_batches > 0:
                train_loss /= num_batches
            
            # 验证阶段
            self.forecaster.model.eval()
            val_loss = 0
            num_val_batches = 0
            
            with torch.no_grad():
                for batch_x, batch_y, batch_is_peak, batch_is_valley in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    batch_is_peak = batch_is_peak.to(device)
                    batch_is_valley = batch_is_valley.to(device)
                    
                    outputs = self.forecaster.model(batch_x)
                    loss = criterion(outputs, batch_y, is_peak=batch_is_peak)
                    
                    val_loss += loss.item()
                    num_val_batches += 1
            
            # 计算平均验证损失
            val_loss /= num_val_batches
            
            # 更新学习率
            scheduler.step()
            
            # 记录损失
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # 打印进度
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{self.config['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.2e}")
                            
            if val_loss < 0.005:      # 当验证损失优于阈值时提前停止
                print("验证损失已低于阈值，提前停止训练。")
                break
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(self.forecaster.model.state_dict(), f"{save_dir}/best_model.pth")
                early_stop_counter = 0
                print(f"[+] 模型改进，已保存检查点")
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.config['patience']:
                    print(f"⚠ {self.config['patience']}个epoch没有改进，提前停止训练")
                    break

        
        # 训练结束后绘制损失曲线
        if save_dir:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.title('Training and Validation Loss (with Peak Awareness)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{save_dir}/training_loss.png")
            plt.close()
            
            # 加载最佳模型
            best_model_path = f"{save_dir}/best_model.pth"
            if os.path.exists(best_model_path):
                self.forecaster.model.load_state_dict(torch.load(best_model_path))
        
        # 返回训练历史字典
        return {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'epochs': list(range(1, len(train_losses) + 1))
        }
    

    def predict(self, X):
        """进行预测，与父类相同"""
        return super().predict(X)
    
    

# --- Probabilistic Forecasting Components ---

def quantile_loss(preds, target, quantiles):
    """
    分位数回归损失函数
    
    Args:
        preds: 预测值，形状 (batch_size, num_quantiles)
        target: 真实值，形状 (batch_size, 1)
        quantiles: 分位数列表，例如 [0.1, 0.5, 0.9]
        
    Returns:
        损失值
    """
    assert preds.shape[1] == len(quantiles), "预测维度与分位数数量不匹配"
    
    # 将target扩展以匹配preds的形状
    target = target.expand(-1, len(quantiles))
    
    errors = target - preds
    loss = torch.max((torch.tensor(quantiles, device=preds.device) - 1) * errors, 
                     torch.tensor(quantiles, device=preds.device) * errors)
    return torch.mean(loss)


class ProbabilisticConvTransformer(PeakAwareConvTransformer):
    """支持概率预测的卷积-Transformer模型"""
    model_type = 'prob_convtrans'
    
    def __init__(self, input_shape=None, quantiles=[0.5], **kwargs):
        super().__init__(input_shape, **kwargs)
        
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        # ---- 重要: 在模型初始化后修改最后一层 ----
        # 需要确保 forecaster 和 model 已经由父类初始化
        if hasattr(self.forecaster, 'model') and self.forecaster.model is not None:
            try:
                # 直接检查并修改TemporalConvLSTM的fc层
                if hasattr(self.forecaster.model, 'fc'):
                    # 获取输入特征数
                    if isinstance(self.forecaster.model.fc, nn.Sequential):
                        # 如果fc是Sequential，找到最后一个线性层
                        last_layer = None
                        for layer in self.forecaster.model.fc:
                            if isinstance(layer, nn.Linear):
                                last_layer = layer
                        
                        if last_layer:
                            in_features = last_layer.in_features
                            # 创建新的最后一层
                            new_layer = nn.Linear(in_features, self.num_quantiles)
                            # 替换最后一个线性层
                            layers = list(self.forecaster.model.fc)
                            for i, layer in enumerate(layers):
                                if isinstance(layer, nn.Linear):
                                    layers[i] = new_layer
                            self.forecaster.model.fc = nn.Sequential(*layers)
                            print(f"模型fc Sequential中的线性层已修改为输出 {self.num_quantiles} 个分位数")
                    else:
                        # 如果fc是直接的线性层
                        in_features = self.forecaster.model.fc.in_features
                        self.forecaster.model.fc = nn.Linear(in_features, self.num_quantiles)
                        print(f"模型fc层已修改为输出 {self.num_quantiles} 个分位数")
                else:
                    # 尝试寻找其他可能的输出层
                    print("警告: 模型没有fc属性，尝试寻找其他输出层")
                    
                    # 尝试使用TemporalConvLSTM的已知结构
                    # 查看模型中最后一层的名称
                    model_attrs = dir(self.forecaster.model)
                    output_attrs = [attr for attr in model_attrs if any(x in attr.lower() for x in ['output', 'fc', 'final', 'head', 'pred'])]
                    
                    if output_attrs:
                        print(f"发现可能的输出层属性: {output_attrs}")
                        for attr in output_attrs:
                            try:
                                layer = getattr(self.forecaster.model, attr)
                                if isinstance(layer, nn.Linear) or isinstance(layer, nn.Sequential):
                                    # 尝试修改该层
                                    print(f"尝试修改 {attr} 层...")
                                    if isinstance(layer, nn.Linear):
                                        in_features = layer.in_features
                                        setattr(self.forecaster.model, attr, nn.Linear(in_features, self.num_quantiles))
                                        print(f"已成功修改 {attr} 层为输出 {self.num_quantiles} 个分位数")
                                        break
                            except Exception as e:
                                print(f"修改 {attr} 层时出错: {e}")
                    else:
                        print("未找到可能的输出层属性")
                        
            except Exception as e:
                print(f"错误: 修改模型输出层失败: {e}")
        else:
            print("警告: 概率模型初始化时父模型尚未完全初始化，最后一层可能未修改")

        # 更新配置
        self.config['quantiles'] = self.quantiles
        self.config['pred_length'] = self.num_quantiles # Output length is now number of quantiles
    
    # --- 覆盖训练方法以使用分位数损失 ---
    def train_probabilistic(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, save_dir=None):
        # 训练代码，专门针对概率预测
        print(f"开始概率模型训练 (Quantiles: {self.quantiles})...")
        
        # 确认是否使用GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"PyTorch 使用设备: {device}")
        
        if device.type == 'cuda':
            print(f"CUDA 是否可用: {torch.cuda.is_available()}")
            print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
            print(f"GPU 内存总量: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"当前内存使用量: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.FloatTensor(y_val).to(device)
        
        # 数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 确保模型在正确的设备上
        self.forecaster.model = self.forecaster.model.to(device)
        
        # 优化器
        optimizer = optim.Adam(self.forecaster.model.parameters(), lr=self.config.get('lr', 0.001))
        
        # 使用混合精度训练加速
        scaler = GradScaler()
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config.get('patience', 10)
        
        # 初始化训练损失记录
        train_losses = []
        
        for epoch in range(epochs):
            # 训练模式
            self.forecaster.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                # 使用混合精度训练
                with autocast(device_type=device.type):
                    outputs = self.forecaster.model(batch_X)
                    loss = quantile_loss(outputs, batch_y, self.quantiles)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}")
            
            if save_dir:
                self.save(save_dir)
                early_stop_counter = 0
                print(f"[+] 概率模型改进，已保存检查点")
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"⚠ {patience}个epoch没有改进，提前停止训练")
                    break

        if save_dir:
            # Plot losses
            plt.figure(figsize=(10, 6))
            plt.plot(list(range(1, len(train_losses) + 1)), train_losses, label='Train Loss')
            plt.title('Probabilistic Model Training Loss (Quantile Loss)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{save_dir}/prob_training_loss.png")
            plt.close()
            
            # Load best model
            best_model_path = os.path.join(save_dir, f'{self.model_type}_model.pth')
            if os.path.exists(best_model_path):
                 self.load(save_dir) # Reload the best state

        return {'train_loss': train_losses, 'val_loss': best_val_loss, 'epochs': list(range(1, len(train_losses) + 1))}

    def predict_probabilistic(self, X):
        """进行概率预测"""
        if self.forecaster.model is None:
            raise ValueError("模型尚未训练")

        self.forecaster.model.eval()
        predictions = {}
        device = self.forecaster.device
        
        dataset = TemporalDataset(X, np.zeros((len(X), 1))) # Dummy target
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'] * 2, shuffle=False)
        
        all_outputs = []
        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(device)
                with autocast(device_type=device.type):
                     outputs = self.forecaster.model(batch_x).detach().cpu().numpy() # Shape: [batch, num_quantiles]
                all_outputs.append(outputs)
        
        all_outputs = np.vstack(all_outputs)
        
        # 组织结果
        results = {}
        for i, q in enumerate(self.quantiles):
             q_label = f'p{int(q * 100)}'
             results[q_label] = all_outputs[:, i]
             
        return results

    def _ensure_model_initialized(self, input_size):
         """Helper to initialize the model if it's None"""
         if self.forecaster.model is None:
             print("内部模型未初始化，正在尝试初始化...")
             self.forecaster.model = TemporalConvLSTM( # Or the base model you intend to use
                 input_size=input_size,
                 seq_len=self.config['seq_length'],
                 pred_len=self.num_quantiles # Crucial: pred_len is num_quantiles now
             )
             # Immediately modify the last layer after base initialization
             self._modify_final_layer_for_quantiles(input_size)


    def _modify_final_layer_for_quantiles(self, input_size):
         """Helper to modify the final layer after initialization"""
         if hasattr(self.forecaster, 'model') and self.forecaster.model is not None:
             try:
                 # 直接检查并修改TemporalConvLSTM的fc层
                 if hasattr(self.forecaster.model, 'fc'):
                     # 获取输入特征数
                     if isinstance(self.forecaster.model.fc, nn.Sequential):
                         # 如果fc是Sequential，找到最后一个线性层
                         last_layer = None
                         for layer in self.forecaster.model.fc:
                             if isinstance(layer, nn.Linear):
                                 last_layer = layer
                         
                         if last_layer:
                             in_features = last_layer.in_features
                             # 创建新的最后一层
                             new_layer = nn.Linear(in_features, self.num_quantiles)
                             # 替换最后一个线性层
                             layers = list(self.forecaster.model.fc)
                             for i, layer in enumerate(layers):
                                 if isinstance(layer, nn.Linear):
                                     layers[i] = new_layer
                             self.forecaster.model.fc = nn.Sequential(*layers)
                             print(f"(Helper)模型fc Sequential中的线性层已修改为输出 {self.num_quantiles} 个分位数")
                     else:
                         # 如果fc是直接的线性层
                         in_features = self.forecaster.model.fc.in_features
                         self.forecaster.model.fc = nn.Linear(in_features, self.num_quantiles)
                         print(f"(Helper)模型fc层已修改为输出 {self.num_quantiles} 个分位数")
                 else:
                     # 尝试寻找其他可能的输出层
                     print("警告(Helper): 模型没有fc属性，尝试使用提供的input_size")
                     
                     # 尝试直接修改fc属性
                     try:
                         # 使用提供的input_size创建一个新的线性层
                         print(f"(Helper)尝试使用input_size={input_size}创建新的线性层")
                         self.forecaster.model.fc = nn.Linear(128, self.num_quantiles)  # 使用128作为默认输入特征维度
                         print(f"(Helper)成功创建新的fc层，输出 {self.num_quantiles} 个分位数")
                     except Exception as layer_err:
                         print(f"(Helper)创建新fc层失败: {layer_err}")
                         
                         # 尝试获取所有可能的输出层属性
                         model_attrs = dir(self.forecaster.model)
                         output_attrs = [attr for attr in model_attrs if any(x in attr.lower() for x in ['output', 'fc', 'final', 'head', 'pred'])]
                         
                         if output_attrs:
                             print(f"(Helper)发现可能的输出层属性: {output_attrs}")
                             for attr in output_attrs:
                                 try:
                                     layer = getattr(self.forecaster.model, attr)
                                     if isinstance(layer, nn.Linear) or isinstance(layer, nn.Sequential):
                                         # 尝试修改该层
                                         print(f"(Helper)尝试修改 {attr} 层...")
                                         if isinstance(layer, nn.Linear):
                                             in_features = layer.in_features
                                             setattr(self.forecaster.model, attr, nn.Linear(in_features, self.num_quantiles))
                                             print(f"(Helper)已成功修改 {attr} 层为输出 {self.num_quantiles} 个分位数")
                                             break
                                 except Exception as e:
                                     print(f"(Helper)修改 {attr} 层时出错: {e}")
                         else:
                             print("(Helper)未找到可能的输出层属性")
             except Exception as e:
                 print(f"错误(Helper): 自动修改模型最后一层失败: {e}")
         else:
             print("错误(Helper): 无法修改最后一层，模型未初始化")


    # Override save and load to handle the probabilistic model type
    def save(self, save_dir='models/prob_convtrans'):
        super().save(save_dir) # Use parent save but default to new directory

    @classmethod
    def load(cls, save_dir='models/prob_convtrans'):
        """加载概率模型"""
        # 检查文件是否存在
        model_path = os.path.join(save_dir, f'{cls.model_type}_model.pth')
        config_path = os.path.join(save_dir, f'{cls.model_type}_config.json')
        input_shape_path = os.path.join(save_dir, 'input_shape.json')
        
        if not all(os.path.exists(p) for p in [model_path, config_path, input_shape_path]):
            raise FileNotFoundError(f"概率模型文件缺失: {save_dir}")
            
        with open(config_path, 'r') as f: config = json.load(f)
        with open(input_shape_path, 'r') as f: input_shape = tuple(json.load(f)) if json.load(f) else None
        
        # --- 重要: 使用加载的 quantiles 初始化 ---
        quantiles = config.get('quantiles', [0.1, 0.5, 0.9]) # Get quantiles from config
        
        # --- 创建 ProbabilisticConvTransformer 实例 ---
        model_instance = cls(input_shape, quantiles=quantiles, **config)
        
        # --- 确保模型内部结构已根据输入和分位数初始化 ---
        if input_shape:
            model_instance._ensure_model_initialized(input_shape[1])
        else:
            # Handle case where input_shape might be missing (less ideal)
             print("警告: 加载模型时缺少输入形状信息，可能导致错误。")
             # Attempt initialization with a default feature size (e.g., 1)
             # This is risky and might need adjustment
             model_instance._ensure_model_initialized(1) 

        # --- 加载状态字典 ---
        model_instance.forecaster.model.load_state_dict(
            torch.load(model_path, map_location=model_instance.forecaster.device)
        )
        model_instance.forecaster.model.to(model_instance.forecaster.device) # Ensure model is on correct device
        
        print(f"概率模型 {cls.model_type} 已从 {save_dir} 加载")
        return model_instance


# --- END Probabilistic Forecasting Components ---

# 高峰感知数据集
class PeakAwareDataset(torch.utils.data.Dataset):
    """包含高峰和低谷标记的时序数据集"""
    def __init__(self, sequences, targets, is_peak=None, is_valley=None):
        self.sequences = torch.FloatTensor(sequences)
        if len(targets.shape) == 1:
            targets = targets.reshape(-1, 1)
        self.targets = torch.FloatTensor(targets)
        
        # 处理高峰标记
        if is_peak is None:
            self.is_peak = torch.zeros(len(sequences), dtype=torch.bool)
        else:
            self.is_peak = torch.BoolTensor(is_peak)
            
        # 处理低谷标记
        if is_valley is None:
            self.is_valley = torch.zeros(len(sequences), dtype=torch.bool)
        else:
            self.is_valley = torch.BoolTensor(is_valley)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (self.sequences[idx], self.targets[idx], 
                self.is_peak[idx], self.is_valley[idx])


class IntervalPeakAwareConvTransformer(PeakAwareConvTransformer):
    """基于峰谷感知的区间预测模型，通过误差分布创建预测区间"""
    model_type = 'convtrans_peak'
    
    def __init__(self, input_shape=None, quantiles=[0.025, 0.05, 0.1, 0.5, 0.9, 0.95, 0.975], **kwargs):
        super().__init__(input_shape, **kwargs)
        
        self.quantiles = quantiles  # 用于区间预测的分位数
        self.error_stats = {}  # 存储误差统计信息
        self.config['quantiles'] = self.quantiles
        
        # 指定不同时段的误差组
        self.period_types = ['all', 'peak', 'valley', 'normal']
        
        # 初始化各时段的误差统计数据结构
        for period in self.period_types:
            self.error_stats[period] = {
                'errors': [],  # 存储误差值
                'mae': None,  # 平均绝对误差
                'std': None,  # 误差标准差
                'quantiles': {}  # 各分位数对应的误差值
            }
    
    def train_with_error_capturing(self, X_train, y_train, X_val, y_val, 
                                  train_is_peak=None, val_is_peak=None,
                                  train_is_valley=None, val_is_valley=None,
                                  epochs=None, batch_size=None, save_dir=None):
        """训练模型，同时捕获预测误差用于后续区间预测"""
        # 使用父类的峰谷感知训练方法
        results = self.train_with_peak_awareness(
            X_train, y_train, X_val, y_val, 
            train_is_peak, val_is_peak,
            epochs, batch_size, save_dir
        )
        
        # 训练后，使用验证集获取误差分布
        print("开始捕获预测误差分布...")
        
        # 确保is_valley数据可用
        if train_is_valley is None and train_is_peak is not None:
            train_is_valley = ~train_is_peak
        if val_is_valley is None and val_is_peak is not None:
            val_is_valley = ~val_is_peak
            
        # 获取验证集上的预测结果
        val_pred = self.predict(X_val)
        
        # 计算误差
        errors = y_val - val_pred
        
        # 保存全部数据的误差统计
        self._calculate_error_stats(errors, 'all')
        
        # 如果有峰谷标记，分别计算各时段的误差统计
        if val_is_peak is not None:
            peak_errors = errors[val_is_peak]
            self._calculate_error_stats(peak_errors, 'peak')
            
            if val_is_valley is not None:
                valley_errors = errors[val_is_valley]
                self._calculate_error_stats(valley_errors, 'valley')
                
                # 计算普通时段的误差（既不是峰也不是谷）
                normal_mask = ~(val_is_peak | val_is_valley)
                if np.any(normal_mask):
                    normal_errors = errors[normal_mask]
                    self._calculate_error_stats(normal_errors, 'normal')
        
        print("误差分布捕获完成")
        
        # 保存误差统计信息
        if save_dir:
            error_stats_path = os.path.join(save_dir, f'{self.model_type}_error_stats.json')
            try:
                # 将numpy数组转换为Python列表以便保存为JSON
                serializable_stats = {}
                for period, stats in self.error_stats.items():
                    serializable_stats[period] = {
                        'mae': float(stats['mae']) if stats['mae'] is not None else None,
                        'std': float(stats['std']) if stats['std'] is not None else None,
                        'quantiles': {str(q): float(val) for q, val in stats['quantiles'].items()}
                    }
                
                with open(error_stats_path, 'w') as f:
                    json.dump(serializable_stats, f, indent=2)
                print(f"误差统计信息已保存到 {error_stats_path}")
            except Exception as e:
                print(f"保存误差统计信息时出错: {e}")
        
        return results
    
    def _calculate_error_stats(self, errors, period_type):
        """计算误差的统计信息"""
        if len(errors) == 0:
            print(f"警告: {period_type}时段没有可用的误差数据")
            return
        
        # 存储原始误差
        self.error_stats[period_type]['errors'] = errors
        
        # 计算平均绝对误差和标准差
        self.error_stats[period_type]['mae'] = np.mean(np.abs(errors))
        self.error_stats[period_type]['std'] = np.std(errors)
        
        # 计算各分位数对应的误差值
        for q in self.quantiles:
            self.error_stats[period_type]['quantiles'][q] = np.quantile(errors, q)
            
        print(f"{period_type}时段误差统计: MAE={self.error_stats[period_type]['mae']:.2f}, STD={self.error_stats[period_type]['std']:.2f}")
        for q in self.quantiles:
            print(f"  {int(q*100)}%分位数误差: {self.error_stats[period_type]['quantiles'][q]:.2f}")
    
    def predict_interval(self, X, is_peak=None, is_valley=None):
        """使用误差分布进行区间预测
        
        Args:
            X: 输入数据
            is_peak: 各样本是否为峰时段，布尔数组
            is_valley: 各样本是否为谷时段，布尔数组
            
        Returns:
            dict: 包含点预测和各分位数区间预测的字典
        """
        # 首先获取点预测
        point_pred = self.predict(X)
        
        # 检查是否已有误差统计
        if not self.error_stats or 'all' not in self.error_stats or self.error_stats['all']['mae'] is None:
            print("警告: 模型尚未通过train_with_error_capturing方法训练，无法使用误差分布进行区间预测")
            print("仅返回点预测结果")
            return {"p50": point_pred}
        
        # 初始化结果
        intervals = {"p50": point_pred}  # 中位数就是点预测
        
        # 初始化样本的时段类型
        if is_peak is None and is_valley is None:
            # 如果没有提供峰谷信息，则全部使用'all'类型的误差统计
            period_masks = {
                'all': np.ones(len(point_pred), dtype=bool)
            }
        else:
            # 根据峰谷标记划分样本
            if is_peak is None:
                is_peak = np.zeros(len(point_pred), dtype=bool)
            if is_valley is None:
                is_valley = np.zeros(len(point_pred), dtype=bool)
                
            period_masks = {
                'peak': is_peak,
                'valley': is_valley,
                'normal': ~(is_peak | is_valley)
            }
            
            # 如果某类型没有样本，则从period_masks中移除
            period_masks = {k: v for k, v in period_masks.items() if np.any(v)}
        
        # 为每个分位数计算预测区间
        for q in self.quantiles:
            if q == 0.5:  # 中位数已经有了，跳过
                continue
                
            # 初始化该分位数的预测结果数组
            q_pred = np.zeros_like(point_pred)
            
            # 根据不同时段应用相应的误差分布
            for period, mask in period_masks.items():
                if not np.any(mask):
                    continue
                
                # 如果特定时段的统计信息不存在，则使用全部数据的统计
                if period not in self.error_stats or self.error_stats[period]['mae'] is None:
                    period = 'all'
                
                # 获取该分位数的误差值
                if q in self.error_stats[period]['quantiles']:
                    error_q = self.error_stats[period]['quantiles'][q]
                else:
                    print(f"警告: {period}时段缺少{q}分位数统计，使用all时段代替")
                    error_q = self.error_stats['all']['quantiles'].get(q, 0)
                
                # 为该时段的样本计算预测值
                q_pred[mask] = point_pred[mask] + error_q
            
            # 保存结果
            intervals[f"p{int(q*100)}"] = q_pred
        
        return intervals
    
    def save(self, save_dir=None):
        """保存模型和误差统计信息"""
        if save_dir is None:
            save_dir = f'models/{self.model_type}'
            
        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 先使用父类方法保存模型
        super().save(save_dir)
        
        # 保存误差统计信息
        if self.error_stats:
            error_stats_path = os.path.join(save_dir, f'{self.model_type}_error_stats.json')
            try:
                # 将numpy数组转换为Python列表以便保存为JSON
                serializable_stats = {}
                for period, stats in self.error_stats.items():
                    serializable_stats[period] = {
                        'mae': float(stats['mae']) if stats['mae'] is not None else None,
                        'std': float(stats['std']) if stats['std'] is not None else None,
                        'quantiles': {str(q): float(val) for q, val in stats['quantiles'].items() if isinstance(val, (int, float, np.number))}
                    }
                
                with open(error_stats_path, 'w') as f:
                    json.dump(serializable_stats, f, indent=2)
                print(f"误差统计信息已保存到 {error_stats_path}")
            except Exception as e:
                print(f"保存误差统计信息时出错: {e}")
    
    @classmethod
    def load(cls, save_dir=None):
        """加载模型和误差统计信息
        
        覆盖父类方法，确保加载的是 convtrans_peak 模型文件，
        因为区间预测模型与峰谷感知确定性模型共享相同的权重。
        """
        if save_dir is None:
            # --- 修改：显式指向 convtrans_peak 目录 --- 
            # save_dir = f'models/{cls.model_type}'
            # 假设 forecast_type 和 dataset_id 可以从 save_dir 推断或需要传递
            # 为了简单起见，我们假设 save_dir 已经是完整的路径，例如 'models/convtrans_peak/load/上海'
            # 如果不是，这里的逻辑需要更复杂
            model_base_type_to_load = 'convtrans_peak' # 显式指定要加载的基础模型类型
            print(f"IntervalPeakAwareConvTransformer.load: 显式加载 '{model_base_type_to_load}' 类型的文件")
        else:
             # 如果提供了 save_dir，也强制使用 convtrans_peak 类型查找文件
             model_base_type_to_load = 'convtrans_peak'
             print(f"IntervalPeakAwareConvTransformer.load: 显式加载 '{model_base_type_to_load}' 类型的文件，从指定目录 {save_dir}")

        # --- 加载模型配置和形状 --- 
        config_path = os.path.join(save_dir, f'{model_base_type_to_load}_config.json')
        input_shape_path = os.path.join(save_dir, 'input_shape.json')
        model_path = os.path.join(save_dir, f'{model_base_type_to_load}_model.pth')

        if not all(os.path.exists(p) for p in [model_path, config_path, input_shape_path]):
            # 打印具体哪个文件缺失
            missing = [p for p in [model_path, config_path, input_shape_path] if not os.path.exists(p)]
            raise FileNotFoundError(f"模型文件缺失于 {save_dir}，无法加载模型。缺失文件: {missing}")
            
        with open(config_path, 'r') as f: config = json.load(f)
        with open(input_shape_path, 'r') as f: 
            input_shape_data = json.load(f)
            input_shape = tuple(input_shape_data) if input_shape_data else None
        
        # --- 创建当前类的实例，但使用加载的配置 --- 
        # 注意：这里的 quantiles 参数可能需要从 config 中获取或使用类默认值
        # --- 修改：使用显式默认列表作为备选项 --- 
        default_quantiles = [0.025, 0.05, 0.1, 0.5, 0.9, 0.95, 0.975] # 与 __init__ 保持一致
        quantiles = config.get('quantiles', default_quantiles) # 使用 config 或显式默认值
        # ---------------------------------------
        model_instance = cls(input_shape=input_shape, quantiles=quantiles, **config)
        
        # --- 显式创建基础 PyTorch 模型架构 --- 
        try:
            if input_shape:
                input_size = input_shape[1]
            else:
                # Fallback if input_shape is missing (less ideal)
                input_size = config.get('input_size', 1) # 假设 config 可能包含此信息，或使用默认值
                print(f"警告: 使用推断或默认 input_size={input_size} 初始化模型")

            seq_len = config.get('seq_length', 96)
            # 区间预测模型基于点预测结构，其输出维度为1
            pred_len = 1 

            # --- 修改: 创建兼容的TemporalConvLSTM实例 --- 
            # 创建自定义版本的TemporalConvLSTM，使用单向LSTM来匹配保存的权重
            class CompatibleTemporalConvLSTM(nn.Module):
                """兼容版时空融合预测模型，使用单向LSTM"""
                def __init__(self, input_size, seq_len=96, pred_len=1):
                    super().__init__()
                    self.seq_len = seq_len
                    self.pred_len = pred_len
                    
                    # 空间特征提取
                    self.spatial_conv = nn.Sequential(
                        nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=2),
                        nn.Dropout(0.2)
                    )
                    
                    # 时间特征提取 - 使用单向LSTM而非双向
                    self.temporal_lstm = nn.LSTM(
                        input_size=64,
                        hidden_size=128,
                        num_layers=2,
                        batch_first=True,
                        dropout=0.3,
                        bidirectional=False  # 使用单向LSTM
                    )
                    
                    # 预测头
                    lstm_output_dim = 128  # 单向LSTM输出维度是hidden_size
                    self.fc = nn.Sequential(
                        nn.Linear(lstm_output_dim * (seq_len//2), 256),
                        nn.BatchNorm1d(256),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(256, pred_len)
                    )
                    
                def forward(self, x):
                    # 输入形状: (batch_size, seq_len, num_features)
                    batch_size = x.size(0)
                    
                    # 空间特征提取
                    x = x.permute(0, 2, 1)  # (batch, features, seq)
                    x = self.spatial_conv(x)  # (batch, 64, seq//2)
                    
                    # 时间特征提取
                    x = x.permute(0, 2, 1)  # (batch, seq//2, 64)
                    lstm_out, _ = self.temporal_lstm(x)  # (batch, seq//2, 128)
                    
                    # 特征融合
                    x = lstm_out.contiguous().view(batch_size, -1)  # (batch, seq//2 * 128)
                    
                    # 预测输出
                    return self.fc(x)

            # 使用兼容版本的模型
            model_instance.forecaster.model = CompatibleTemporalConvLSTM(
                 input_size=input_size,
                 seq_len=seq_len,
                 pred_len=pred_len
            ).to(model_instance.forecaster.device)
            
            print("已创建兼容版TemporalConvLSTM模型架构 (使用单向LSTM匹配保存的权重)")
            # --- 修改结束 ---

        except Exception as init_err:
            raise RuntimeError(f"加载模型时无法创建内部模型架构: {init_err}")
        # --- 模型架构创建结束 ---

        # --- 加载状态字典 --- 
        try:
            model_instance.forecaster.model.load_state_dict(
                 torch.load(model_path, map_location=model_instance.forecaster.device)
            )
            model_instance.forecaster.model.to(model_instance.forecaster.device)
            print(f"模型权重已从 {model_path} 加载")
        except Exception as load_err:
            raise RuntimeError(f"加载模型权重时出错: {load_err}")
        # ---------------------

        # --- 加载误差统计信息 --- 
        error_stats_path = os.path.join(save_dir, f'{model_base_type_to_load}_error_stats.json') # 使用基础模型类型查找误差文件
        if os.path.exists(error_stats_path):
            try:
                with open(error_stats_path, 'r') as f:
                    error_stats = json.load(f)
                    model_instance.error_stats = {period: {k: v for k, v in stats.items()} for period, stats in error_stats.items()}
                    print(f"已加载误差统计信息从 {error_stats_path}")
            except Exception as e:
                print(f"加载误差统计信息时出错: {e}")
        else:
            print(f"警告: 未找到误差统计信息文件 {error_stats_path}")
        # ----------------------
        
        return model_instance

# # 添加一个新的天气增强模型类
# class WeatherEnhancedConvTransformer(nn.Module):
#     """结合卷积和Transformer的天气增强模型"""
    
#     def __init__(self, 
#                  input_dim=4,           # 基础输入维度（不含天气特征）
#                  seq_len=96,            # 输入序列长度
#                  pred_len=4,            # 预测长度
#                  weather_dim=0,         # 天气特征维度
#                  d_model=64,            # 模型内部维度
#                  n_heads=4,             # 注意力头数
#                  dropout=0.1,           # Dropout率
#                  activation='gelu',     # 激活函数类型
#                  use_bn=True,           # 是否使用批归一化
#                  batch_norm_momentum=0.1, # BatchNorm动量参数
#                  batch_norm_eps=1e-5,     # BatchNorm的epsilon参数
#                  use_skip=True,         # 是否使用跳跃连接
#                  include_peaks=False,   # 是否包含峰值信息
#                  features_fusion_method='concat'  # 特征融合方法
#                 ):
#         """
#         初始化天气增强的卷积Transformer模型
        
#         Args:
#             input_dim: 基础输入特征维度（不含天气特征）
#             seq_len: 输入序列长度
#             pred_len: 预测时间步
#             weather_dim: 天气特征维度
#             d_model: 模型内部维度
#             n_heads: 注意力头数
#             dropout: Dropout比率
#             activation: 激活函数类型
#             use_bn: 是否使用BatchNorm
#             batch_norm_momentum: BatchNorm动量参数 
#             batch_norm_eps: BatchNorm的epsilon参数
#             use_skip: 是否使用跳跃连接
#             include_peaks: 是否包含峰值感知特征
#             features_fusion_method: 特征融合方法（concat或attention）
#         """
#         super(WeatherEnhancedConvTransformer, self).__init__()
        
#         # 保存配置
#         self.input_dim = input_dim
#         self.seq_len = seq_len
#         self.pred_len = pred_len
#         self.weather_dim = weather_dim
#         self.d_model = d_model
#         self.use_bn = use_bn
#         self.use_skip = use_skip
#         self.include_peaks = include_peaks
#         self.features_fusion_method = features_fusion_method
#         self.batch_norm_momentum = batch_norm_momentum
#         self.batch_norm_eps = batch_norm_eps
        
#         # 计算总输入维度
#         self.total_input_dim = input_dim + weather_dim
#         if include_peaks:
#             # 峰值特征: 是否为峰值时段(0/1)、到最近峰值的距离、峰值幅度
#             self.total_input_dim += 3
            
#         # 数据归一化层
#         self.input_norm = nn.BatchNorm1d(self.total_input_dim, 
#                                          momentum=batch_norm_momentum, 
#                                          eps=batch_norm_eps)
        
#         # 对总输入进行编码
#         self.feature_encoder = nn.Linear(self.total_input_dim, d_model)
        
#         # 卷积编码器 - 捕获局部时间模式
#         self.conv_encoder = nn.Sequential(
#             nn.Conv1d(d_model, d_model*2, kernel_size=3, padding=1),
#             nn.BatchNorm1d(d_model*2, momentum=batch_norm_momentum, eps=batch_norm_eps) if use_bn else nn.Identity(),
#             getattr(nn, activation)(),
#             nn.Dropout(dropout),
            
#             nn.Conv1d(d_model*2, d_model, kernel_size=3, padding=1),
#             nn.BatchNorm1d(d_model, momentum=batch_norm_momentum, eps=batch_norm_eps) if use_bn else nn.Identity(),
#             getattr(nn, activation)()
#         )
        
#         # 位置编码
#         self.pos_encoder = PositionalEncoding(d_model, dropout)
        
#         # Transformer编码器层
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=n_heads,
#             dim_feedforward=d_model*4,
#             dropout=dropout,
#             activation=activation,
#             batch_first=True,
#             norm_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
#         # 输出层 - 预测未来时间步
#         self.prediction_head = nn.Sequential(
#             nn.Linear(d_model * seq_len, d_model * 4),
#             nn.LayerNorm(d_model * 4, eps=batch_norm_eps),
#             getattr(nn, activation)(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model * 4, d_model * 2),
#             nn.LayerNorm(d_model * 2, eps=batch_norm_eps),
#             getattr(nn, activation)(),
#             nn.Linear(d_model * 2, pred_len)
#         )
        
#         # 设置初始学习率和优化器
#         self.learning_rate = 1e-4
#         self.optimizer = None
#         self.scheduler = None
#         self.train_losses = []
#         self.val_losses = []
#         self.best_val_loss = float('inf')
#         self.loss_function = nn.MSELoss()
#         self.device = None
#         self.model_initialized = False
        
#         # 初始化权重
#         self._initialize_weights()

#     def _initialize_weights(self):
#         """初始化模型权重，使用Xavier和Kaiming初始化提高训练稳定性"""
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 # Kaiming初始化卷积层
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 # BatchNorm层标准初始化
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 # Xavier初始化线性层
#                 nn.init.xavier_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
    
#     def _ensure_model_initialized(self, x):
#         """确保模型在使用前正确初始化，检查设备并打印调试信息"""
#         if not self.model_initialized:
#             try:
#                 # 检测可用设备并将模型移至合适设备
#                 if torch.cuda.is_available():
#                     self.device = torch.device('cuda')
#                     # 获取GPU信息
#                     gpu_count = torch.cuda.device_count()
#                     gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
#                     gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3) if gpu_count > 0 else 0
#                     print(f"使用GPU: {gpu_name}, 显存: {gpu_mem:.2f}GB")
#                 else:
#                     self.device = torch.device('cpu')
#                     print("使用CPU进行训练")
                
#                 self.to(self.device)
                
#                 # 打印模型结构和参数数量
#                 total_params = sum(p.numel() for p in self.parameters())
#                 print(f"模型总参数量: {total_params}")
                
#                 # 打印输入形状信息
#                 batch_size = x.shape[0]
#                 print(f"批量大小: {batch_size}, 序列长度: {self.seq_len}, 总输入维度: {self.total_input_dim}")
                
#                 # 设置BatchNorm层参数
#                 for m in self.modules():
#                     if isinstance(m, nn.BatchNorm1d):
#                         # 如果批量大小为1，需要启用track_running_stats=False
#                         if batch_size == 1:
#                             m.track_running_stats = False
#                             print("检测到BatchSize=1，已禁用BatchNorm统计跟踪")
                
#                 # 梯度检查钩子 - 在调试时用于检测NaN/Inf
#                 if os.environ.get('MODEL_DEBUG_MODE') == '1':
#                     for name, param in self.named_parameters():
#                         def hook(grad, name=name):
#                             if torch.isnan(grad).any() or torch.isinf(grad).any():
#                                 print(f"检测到梯度问题 - {name}: NaN={torch.isnan(grad).any()}, Inf={torch.isinf(grad).any()}")
#                             return grad
#                         param.register_hook(hook)
                
#                 self.model_initialized = True
#             except Exception as e:
#                 print(f"初始化模型时出错: {e}")
#                 traceback.print_exc()
#                 # 尝试回退到CPU训练
#                 print("尝试回退到CPU训练...")
#                 self.device = torch.device('cpu')
#                 self.to(self.device)
#                 self.model_initialized = True

#     def forward(self, x):
#         """
#         前向传播过程
        
#         Args:
#             x: 输入张量 (batch_size, seq_len, features)
            
#         Returns:
#             torch.Tensor: 预测结果 (batch_size, pred_len)
#         """
#         # 确保模型已初始化
#         self._ensure_model_initialized(x)
        
#         # 获取批量大小和检查输入格式
#         batch_size, seq_len, n_features = x.shape
        
#         if seq_len != self.seq_len:
#             print(f"警告: 输入序列长度 {seq_len} 与模型配置的序列长度 {self.seq_len} 不匹配")
        
#         if n_features != self.total_input_dim:
#             print(f"警告: 输入特征数 {n_features} 与模型配置的特征数 {self.total_input_dim} 不匹配")
        
#         # 检查NaN/Inf输入
#         if torch.isnan(x).any() or torch.isinf(x).any():
#             print("警告: 输入数据包含NaN或Inf值，将被替换")
#             x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
#         try:
#             # 应用特征归一化
#             x_reshaped = x.reshape(-1, n_features).t()  # [features, batch_size*seq_len]
#             x_normed = self.input_norm(x_reshaped)
#             x = x_normed.t().reshape(batch_size, seq_len, n_features)  # [batch_size, seq_len, features]
            
#             # 特征编码
#             x = self.feature_encoder(x)  # [batch_size, seq_len, d_model]
            
#             # 应用卷积编码器 - 需要改变维度顺序
#             x_conv = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
#             x_conv_out = self.conv_encoder(x_conv)  # [batch_size, d_model, seq_len]
            
#             # 跳跃连接
#             if self.use_skip:
#                 x_conv_out = x_conv_out + x_conv  # 残差连接
                
#             # 变回序列格式
#             x = x_conv_out.transpose(1, 2)  # [batch_size, seq_len, d_model]
            
#             # 位置编码
#             x = self.pos_encoder(x)  # [batch_size, seq_len, d_model]
            
#             # Transformer编码器
#             x = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]
            
#             # 展平并预测
#             x_flat = x.reshape(batch_size, -1)  # [batch_size, seq_len*d_model]
#             output = self.prediction_head(x_flat)  # [batch_size, pred_len]
            
#             # 检查输出的NaN/Inf
#             if torch.isnan(output).any() or torch.isinf(output).any():
#                 print("警告: 模型输出包含NaN或Inf值")
#                 output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
            
#             return output
            
#         except Exception as e:
#             print(f"前向传播出错: {e}")
#             traceback.print_exc()
#             # 出错时返回零张量
#             return torch.zeros(batch_size, self.pred_len, device=self.device)

#     def to(self, device):
#         """
#         将模型移动到指定设备
        
#         参数:
#         device: PyTorch设备对象或字符串
        
#         返回:
#         self: 返回自身以支持链式调用
#         """
#         if self.forecaster.model is not None:
#             self.forecaster.model = self.forecaster.model.to(device)
#             print(f"[WeatherEnhancedConvTransformer] 模型已移动到设备: {device}")
#         self.forecaster.device = device
#         return self
    
#     def train_epoch(self, train_loader, criterion, optimizer, epoch, device, gradient_clip_value=1.0, save_dir=None):
#         """训练一个epoch"""
#         self.to(device)
#         self.forecaster.model.train()
        
#         total_loss = 0.0
#         total_samples = 0
#         model_path = os.path.join(save_dir, "model.pth")
#         torch.save(self.forecaster.model.state_dict(), model_path)
#         print(f"[WeatherEnhancedConvTransformer] 模型已保存到: {model_path}")
        
#         # 保存配置
#         config_path = os.path.join(save_dir, "config.json")
#         with open(config_path, 'w') as f:
#             json.dump({
#                 **self.config,
#                 'model_type': self.model_type,
#                 'weather_feature_count': self.weather_feature_count,
#                 'use_weather_features': self.use_weather_features,
#                 'input_shape': self.input_shape if hasattr(self, 'input_shape') else None
#             }, f, indent=4)
        
#         print(f"[WeatherEnhancedConvTransformer] 配置已保存到: {config_path}")
    
#     @classmethod
#     def load(cls, save_dir=None):
#         """加载模型"""
#         if save_dir is None:
#             save_dir = f"models/{cls.model_type}"
        
#         # 加载配置
#         config_path = os.path.join(save_dir, "config.json")
#         if not os.path.exists(config_path):
#             raise FileNotFoundError(f"找不到配置文件: {config_path}")
        
#         with open(config_path, 'r') as f:
#             config = json.load(f)
        
#         # 创建模型实例
#         input_shape = config.get('input_shape')
#         instance = cls(input_shape=input_shape, **config)
        
#         # 加载模型参数
#         model_path = os.path.join(save_dir, "model.pth")
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         # 初始化模型结构
#         if input_shape is not None and len(input_shape) > 1:
#             instance._ensure_model_initialized(input_shape[1])
        
#         # 加载模型权重
#         if os.path.exists(model_path) and instance.forecaster.model is not None:
#             instance.forecaster.model.load_state_dict(torch.load(model_path, map_location=device))
#             instance.forecaster.model.to(device)
#             print(f"[WeatherEnhancedConvTransformer] 模型已从 {model_path} 加载并移动到设备: {device}")
        
#         return instance

class WeatherAwareConvTransformer(PeakAwareConvTransformer):
    """基于PeakAwareConvTransformer的天气增强模型
    
    保持原有的高精度架构，增加天气特征处理能力
    """
    model_type = 'convtrans_weather'
    
    def __init__(self, input_shape=None, weather_features=None, **kwargs):
        """
        初始化天气感知模型
        
        参数:
        input_shape: 输入形状 (seq_length, total_features)
        weather_features: 天气特征列表，如['temperature', 'humidity', 'wind_speed']
        """
        self.weather_features = weather_features or []
        self.weather_dim = len(self.weather_features)
        
        # 调用父类初始化
        super().__init__(input_shape=input_shape, **kwargs)
        
        print(f"初始化天气感知模型，天气特征数: {self.weather_dim}")
        if self.weather_features:
            print(f"天气特征: {self.weather_features}")
    
    def _create_model(self, input_size):
        """创建包含天气特征处理的模型"""
        # 使用父类的模型创建逻辑，但增加天气特征处理
        model = super()._create_model(input_size)
        
        # 如果有天气特征，可以在这里添加特殊的天气特征处理层
        # 目前保持简单，直接使用原有架构处理所有特征
        
        return model
    
    def train_with_weather_awareness(self, X_train, y_train, X_val, y_val,
                                   train_is_peak=None, val_is_peak=None,
                                   epochs=None, batch_size=None, save_dir=None):
        """
        使用天气感知训练
        
        实际上调用父类的峰值感知训练，因为天气特征已经包含在输入中
        """
        print("开始天气感知训练...")
        
        # 检查输入数据中是否包含天气特征
        if X_train.shape[2] > 3:  # 基础特征通常是3个（时间特征）
            print(f"检测到 {X_train.shape[2] - 3} 个额外特征（可能包含天气特征）")
        
        # 调用父类的峰值感知训练方法
        return self.train_with_peak_awareness(
            X_train, y_train, X_val, y_val,
            train_is_peak=train_is_peak, val_is_peak=val_is_peak,
            epochs=epochs, batch_size=batch_size, save_dir=save_dir
        )
    
    def save(self, save_dir='models/convtrans_weather'):
        """保存天气感知模型"""
        # 调用父类保存方法
        super().save(save_dir=save_dir)
        
        # 额外保存天气特征信息
        import json
        weather_config = {
            'weather_features': self.weather_features,
            'weather_dim': self.weather_dim,
            'model_type': self.model_type
        }
        
        config_path = os.path.join(save_dir, 'weather_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(weather_config, f, indent=2, ensure_ascii=False)
        
        print(f"天气配置已保存到: {config_path}")
    
    @classmethod
    def load(cls, save_dir='models/convtrans_weather'):
        """加载天气感知模型"""
        import json
        
        # 先尝试加载天气配置
        weather_config_path = os.path.join(save_dir, 'weather_config.json')
        weather_features = []
        
        if os.path.exists(weather_config_path):
            try:
                with open(weather_config_path, 'r', encoding='utf-8') as f:
                    weather_config = json.load(f)
                    weather_features = weather_config.get('weather_features', [])
                print(f"加载天气配置: {weather_features}")
            except Exception as e:
                print(f"加载天气配置失败: {e}")
        
        # 加载基础模型
        instance = super().load(save_dir=save_dir)
        
        # 设置天气特征信息
        if hasattr(instance, '__class__'):
            instance.__class__ = cls
        instance.weather_features = weather_features
        instance.weather_dim = len(weather_features)
        instance.model_type = 'convtrans_weather'
        
        return instance

    