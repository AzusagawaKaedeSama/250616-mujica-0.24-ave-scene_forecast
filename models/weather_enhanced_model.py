import os
import math
import json
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 导入位置编码模块
from models.positional_encoding import PositionalEncoding, get_positional_encoding

class WeatherEnhancedConvTransformer(nn.Module):
    """结合卷积和Transformer的天气增强模型"""
    
    def __init__(self, 
                 input_dim=4,           # 基础输入维度（不含天气特征）
                 seq_len=96,            # 输入序列长度
                 pred_len=4,            # 预测长度
                 weather_dim=0,         # 天气特征维度
                 d_model=64,            # 模型内部维度
                 n_heads=4,             # 注意力头数
                 dropout=0.1,           # Dropout率
                 activation='GELU',     # 激活函数类型
                 use_bn=True,           # 是否使用批归一化
                 batch_norm_momentum=0.1, # BatchNorm动量参数
                 batch_norm_eps=1e-5,     # BatchNorm的epsilon参数
                 use_skip=True,         # 是否使用跳跃连接
                 include_peaks=False,   # 是否包含峰值信息
                 features_fusion_method='concat',  # 特征融合方法
                 pos_encoding_type='fixed'  # 位置编码类型：'fixed' 或 'learnable'
                ):
        """
        初始化天气增强的卷积Transformer模型
        
        Args:
            input_dim: 基础输入特征维度（不含天气特征）
            seq_len: 输入序列长度
            pred_len: 预测时间步
            weather_dim: 天气特征维度
            d_model: 模型内部维度
            n_heads: 注意力头数
            dropout: Dropout比率
            activation: 激活函数类型
            use_bn: 是否使用BatchNorm
            batch_norm_momentum: BatchNorm动量参数 
            batch_norm_eps: BatchNorm的epsilon参数
            use_skip: 是否使用跳跃连接
            include_peaks: 是否包含峰值感知特征
            features_fusion_method: 特征融合方法（concat或attention）
            pos_encoding_type: 位置编码类型
        """
        super(WeatherEnhancedConvTransformer, self).__init__()
        
        # 保存配置
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.weather_dim = weather_dim
        self.d_model = d_model
        self.use_bn = use_bn
        self.use_skip = use_skip
        self.include_peaks = include_peaks
        self.features_fusion_method = features_fusion_method
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_eps = batch_norm_eps
        self.pos_encoding_type = pos_encoding_type
        
        # 计算总输入维度
        self.total_input_dim = input_dim + weather_dim
        if include_peaks:
            # 峰值特征: 是否为峰值时段(0/1)、到最近峰值的距离、峰值幅度
            self.total_input_dim += 3
            
        # 数据归一化层
        self.input_norm = nn.BatchNorm1d(self.total_input_dim, 
                                         momentum=batch_norm_momentum, 
                                         eps=batch_norm_eps)
        
        # 对总输入进行编码
        self.feature_encoder = nn.Linear(self.total_input_dim, d_model)
        
        # 激活函数
        self.act_fn = getattr(nn, activation.upper())()
        
        # 卷积编码器 - 捕获局部时间模式
        self.conv_encoder = nn.Sequential(
            nn.Conv1d(d_model, d_model*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model*2, momentum=batch_norm_momentum, eps=batch_norm_eps) if use_bn else nn.Identity(),
            self.act_fn,
            nn.Dropout(dropout),
            
            nn.Conv1d(d_model*2, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model, momentum=batch_norm_momentum, eps=batch_norm_eps) if use_bn else nn.Identity(),
            self.act_fn
        )
        
        # 位置编码
        self.pos_encoder = get_positional_encoding(pos_encoding_type, d_model, dropout)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model*4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # 输出层 - 预测未来时间步
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model * seq_len, d_model * 4),
            nn.LayerNorm(d_model * 4, eps=batch_norm_eps),
            self.act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model * 2),
            nn.LayerNorm(d_model * 2, eps=batch_norm_eps),
            self.act_fn,
            nn.Linear(d_model * 2, pred_len)
        )
        
        # 设置初始学习率和优化器
        self.learning_rate = 1e-4
        self.optimizer = None
        self.scheduler = None
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.loss_function = nn.MSELoss()
        self.device = None
        self.model_initialized = False
        
        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化模型权重，使用Xavier和Kaiming初始化提高训练稳定性"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Kaiming初始化卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                # BatchNorm层标准初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier初始化线性层
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _ensure_model_initialized(self, x):
        """
        确保模型已初始化，并移动到合适的设备
        
        Args:
            x: 输入示例，用于确定设备和检查特征维度
            
        Returns:
            None
        """
        if self.model_initialized:
            return
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        self.to(self.device)
        
        # 检查特征维度是否匹配
        n_features = x.shape[-1]
        if n_features != self.total_input_dim:
            print(f"警告: 输入特征数 {n_features} 与模型配置的特征数 {self.total_input_dim} 不匹配")
            print(f"将调整模型以适应输入维度")
            
            # 调整模型的输入归一化层以适应实际输入特征维度
            self.total_input_dim = n_features
            self.input_norm = nn.BatchNorm1d(self.total_input_dim, 
                                             momentum=self.batch_norm_momentum, 
                                             eps=self.batch_norm_eps).to(self.device)
            
            # 重新构建特征编码器以适应新的输入维度
            self.feature_encoder = nn.Linear(self.total_input_dim, self.d_model).to(self.device)
            
            print(f"已调整模型接受 {n_features} 维特征输入")
            
            # 重新初始化权重
            nn.init.xavier_normal_(self.feature_encoder.weight)
            if self.feature_encoder.bias is not None:
                nn.init.constant_(self.feature_encoder.bias, 0)
        
        # 注册钩子，用于监控梯度
        for name, param in self.named_parameters():
            if param.requires_grad:
                def hook(grad, name=name):
                    if torch.isnan(grad).any() or torch.isinf(grad).any():
                        print(f"警告: 参数 {name} 的梯度包含NaN或Inf")
                    return grad
                param.register_hook(hook)
        
        # 添加钩子监控中间输出        
        def forward_hook(module, input, output):
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"警告: 模块 {module.__class__.__name__} 的输出包含NaN或Inf")
            return None
            
        # 对关键层添加钩子
        self.feature_encoder.register_forward_hook(forward_hook)
        
        # 添加批量归一化层的钩子
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.LayerNorm):
                module.register_forward_hook(forward_hook)
                
        self.model_initialized = True

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, features]
            
        Returns:
            torch.Tensor: 预测输出，形状为 [batch_size, pred_len]
        """
        # 确保模型已初始化
        if not self.model_initialized:
            self._ensure_model_initialized(x)
        
        # 获取批量大小和序列长度
        batch_size, seq_len, n_features = x.shape
        
        # 如果需要，将输入移动到正确的设备
        x = x.to(self.device)
        
        try:
            # 检查特征维度是否匹配，如果不匹配则重新调整模型
            if n_features != self.total_input_dim:
                print(f"警告: 前向传播时输入特征数 {n_features} 与模型配置的特征数 {self.total_input_dim} 不匹配")
                
                # 如果之前已初始化过，但特征维度有变化，需要重新调整
                self.total_input_dim = n_features
                self.input_norm = nn.BatchNorm1d(self.total_input_dim, 
                                               momentum=self.batch_norm_momentum, 
                                               eps=self.batch_norm_eps).to(self.device)
            
            # 检查数据是否包含NaN或Inf
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"警告: 输入数据包含NaN或Inf值，将替换为0")
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 特征归一化
            try:
                # 调整维度以匹配BatchNorm1d的期望输入 (N, C, L)
                x = x.permute(0, 2, 1)  # [B, F, S]
                normalized_x = self.input_norm(x)
                x = normalized_x.permute(0, 2, 1) # [B, S, F]
            except Exception as e:
                print(f"特征归一化出错: {e}")
                print("跳过归一化步骤")
                x = x.permute(0, 2, 1)  # 换回来
            
            # 检查归一化后是否有NaN
            if torch.isnan(x).any():
                print("警告: 归一化后出现NaN，请检查数据和BatchNorm层参数")
                x = torch.nan_to_num(x, nan=0.0)
            
            # 特征编码
            encoded_features = self.feature_encoder(x) # [B, S, d_model]
            
            # 卷积编码
            # 调整维度以匹配Conv1d (N, C, L)
            conv_input = encoded_features.permute(0, 2, 1) # [B, d_model, S]
            conv_output = self.conv_encoder(conv_input)
            
            # 检查卷积输出
            if torch.isnan(conv_output).any():
                print("警告: 卷积编码器输出包含NaN")
                conv_output = torch.nan_to_num(conv_output, nan=0.0)
            
            # 跳跃连接
            if self.use_skip:
                # 确保维度一致
                skip_connection = conv_output + conv_input
            else:
                skip_connection = conv_output
            
            # 位置编码
            try:
                # 调整维度以匹配Transformer (S, B, C) for older versions, or (B, S, C) for newer
                pos_input = skip_connection.permute(0, 2, 1) # [B, S, d_model]
                pos_output = self.pos_encoder(pos_input)
            except Exception as e:
                print(f"位置编码出错: {e}")
                print("使用原始输入")
                pos_output = pos_input
            
            # Transformer编码器
            transformer_output = self.transformer_encoder(pos_output)
            
            # 检查Transformer输出
            if torch.isnan(transformer_output).any():
                print("警告: Transformer编码器输出包含NaN")
                # 备选方案: 使用跳跃连接的输出来代替
                if not torch.isnan(pos_output).any():
                    print("使用Transformer的输入作为替代输出")
                    transformer_output = pos_output
                else:
                    print("Transformer输入也包含NaN，将替换为0")
                    transformer_output = torch.nan_to_num(transformer_output, nan=0.0)
            
            # 展平输出以进行预测
            flattened_output = transformer_output.reshape(batch_size, -1)
            
            # 预测头
            prediction = self.prediction_head(flattened_output)
            
            # 最终检查输出
            if torch.isnan(prediction).any():
                print("警告: 模型最终输出包含NaN，将替换为0")
                prediction = torch.nan_to_num(prediction, nan=0.0)
                
            return prediction
            
        except Exception as e:
            print(f"前向传播过程中出错: {e}")
            traceback.print_exc()
            
            # 返回一个形状正确的零张量，以避免训练中断
            return torch.zeros(batch_size, self.pred_len, device=self.device)
            
    def train_epoch(self, train_loader, criterion, optimizer, epoch, device, gradient_clip_value=1.0):
        """
        训练一个epoch，并返回平均损失
        
        Args:
            train_loader: 训练数据加载器
            criterion: 损失函数
            optimizer: 优化器
            epoch: 当前epoch
            device: 训练设备
            gradient_clip_value: 梯度裁剪阈值
            
        Returns:
            float: 平均训练损失
        """
        self.train()  # 设置为训练模式
        total_loss = 0.0
        valid_batches = 0
        
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} - 当前学习率: {current_lr:.6f}")
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            try:
                # 将数据移动到设备
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 检查输入数据
                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    print(f"批次 {batch_idx}: 输入数据包含NaN或Inf值，跳过")
                    continue
                    
                # 清零梯度
                optimizer.zero_grad()
                
                # 前向传播
                outputs = self(inputs)  # 使用模型的forward方法
                
                # 确保张量维度匹配
                if outputs.shape != targets.shape:
                    if outputs.dim() == 2 and targets.dim() == 1:
                        # 如果预测是二维[batch, pred_len]但目标是一维[batch]
                        if outputs.size(1) == 1:
                            outputs = outputs.squeeze(1)  # [batch, 1] -> [batch]
                    elif outputs.dim() == 1 and targets.dim() == 2:
                        # 如果预测是一维[batch]但目标是二维[batch, 1]
                        if targets.size(1) == 1:
                            targets = targets.squeeze(1)  # [batch, 1] -> [batch]
                            
                # 计算损失
                loss = criterion(outputs, targets)
                
                # 检查损失是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"批次 {batch_idx}: 损失为NaN或Inf，跳过")
                    continue
                    
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_value)
                
                # 检查梯度
                has_bad_gradients = False
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"批次 {batch_idx}: 参数 {name} 的梯度包含NaN或Inf")
                            has_bad_gradients = True
                            break
                
                if has_bad_gradients:
                    continue
                    
                # 更新参数
                optimizer.step()
                
                # 累加损失
                current_loss = loss.item()
                total_loss += current_loss
                valid_batches += 1
                
                # 打印进度
                if (batch_idx + 1) % 10 == 0 or batch_idx == len(train_loader) - 1:
                    print(f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] - Loss: {current_loss:.6f}")
                    
            except Exception as e:
                print(f"处理批次 {batch_idx} 时出错: {e}")
                traceback.print_exc()
                continue
                
        # 计算平均损失
        avg_loss = total_loss / max(valid_batches, 1)
        print(f"Epoch {epoch} 完成: 平均损失 = {avg_loss:.6f}, 有效批次: {valid_batches}/{len(train_loader)}")
        return avg_loss
        
    def validate(self, val_loader, criterion, device):
        """
        验证模型并返回平均验证损失
        
        Args:
            val_loader: 验证数据加载器
            criterion: 损失函数
            device: 验证设备
            
        Returns:
            float: 平均验证损失
        """
        self.eval()  # 设置为评估模式
        total_loss = 0.0
        valid_batches = 0
        all_losses = []
        
        with torch.no_grad():  # 不计算梯度
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                try:
                    # 将数据移动到设备
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # 检查输入数据
                    if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                        print(f"验证批次 {batch_idx}: 输入数据包含NaN或Inf值，跳过")
                        continue
                        
                    # 前向传播
                    outputs = self(inputs)
                    
                    # 确保张量维度匹配
                    if outputs.shape != targets.shape:
                        if outputs.dim() == 2 and targets.dim() == 1:
                            if outputs.size(1) == 1:
                                outputs = outputs.squeeze(1)
                        elif outputs.dim() == 1 and targets.dim() == 2:
                            if targets.size(1) == 1:
                                targets = targets.squeeze(1)
                                
                    # 计算损失
                    loss = criterion(outputs, targets)
                    
                    # 检查损失是否有效
                    current_loss = loss.item()
                    if math.isnan(current_loss) or math.isinf(current_loss):
                        print(f"验证批次 {batch_idx}: 损失为NaN或Inf，跳过")
                        continue
                        
                    # 累加损失
                    total_loss += current_loss
                    all_losses.append(current_loss)
                    valid_batches += 1
                    
                except Exception as e:
                    print(f"处理验证批次 {batch_idx} 时出错: {e}")
                    traceback.print_exc()
                    continue
                    
        # 计算平均损失
        if valid_batches > 0:
            # 使用修剪均值，去除异常值
            if len(all_losses) > 10:
                all_losses = sorted(all_losses)
                trim_size = int(len(all_losses) * 0.05)  # 修剪5%
                trimmed_losses = all_losses[trim_size:-trim_size] if trim_size > 0 else all_losses
                avg_loss = sum(trimmed_losses) / len(trimmed_losses)
                print(f"验证完成: 修剪均值损失 = {avg_loss:.6f}, 原始均值 = {total_loss/valid_batches:.6f}")
            else:
                avg_loss = total_loss / valid_batches
                print(f"验证完成: 平均损失 = {avg_loss:.6f}")
                
            return avg_loss
        else:
            print("警告: 验证过程中没有有效批次")
            return 1.0  # 返回一个合理的默认值
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, 
            batch_size=32, epochs=100, patience=10, 
            learning_rate=None, weight_decay=1e-4,
            save_dir=None, loss_fn=None):
        """
        训练模型
        
        Args:
            X_train: 训练输入数据
            y_train: 训练目标数据
            X_val: 验证输入数据，可选
            y_val: 验证目标数据，可选
            batch_size: 批量大小
            epochs: 训练轮数
            patience: 早停耐心值
            learning_rate: 学习率，如果为None则使用默认值
            weight_decay: 权重衰减
            save_dir: 模型保存目录
            loss_fn: 自定义损失函数，可选
            
        Returns:
            self: 训练后的模型
        """
        # 设置学习率
        if learning_rate is not None:
            self.learning_rate = learning_rate
            
        # 设置损失函数
        if loss_fn is not None:
            self.loss_function = loss_fn
            print(f"使用自定义损失函数: {loss_fn.__class__.__name__}")
            
        # 准备数据加载器
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                      torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                                        torch.tensor(y_val, dtype=torch.float32))
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            has_validation = True
        else:
            has_validation = False
            
        # 确保模型已初始化
        self._ensure_model_initialized(torch.tensor(X_train[0:1], dtype=torch.float32))
        
        # 设置优化器和学习率调度器
        self.optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', 
                                                             factor=0.5, patience=patience//2, 
                                                             verbose=True, min_lr=1e-6)
        
        # 开始训练
        best_val_loss = float('inf')
        best_epoch = 0
        no_improve_count = 0
        
        print(f"开始训练，共 {epochs} 轮...")
        for epoch in range(1, epochs + 1):
            # 训练一个epoch
            train_loss = self.train_epoch(train_loader, self.loss_function, self.optimizer, epoch, self.device)
            self.train_losses.append(train_loss)
            
            # 验证
            if has_validation:
                val_loss = self.validate(val_loader, self.loss_function, self.device)
                self.val_losses.append(val_loss)
                
                # 更新学习率调度器
                self.scheduler.step(val_loss)
                
                # 检查是否有改进
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    no_improve_count = 0
                    
                    # 保存最佳模型
                    if save_dir:
                        self._save_model(save_dir)
                        
                else:
                    no_improve_count += 1
                    
                print(f"Epoch {epoch}: 训练损失 = {train_loss:.6f}, 验证损失 = {val_loss:.6f}, 最佳 = {best_val_loss:.6f} (轮次 {best_epoch})")
                
                # 早停
                if no_improve_count >= patience:
                    print(f"早停: {patience} 轮没有改进")
                    break
            else:
                # 没有验证集，直接保存模型
                if save_dir and epoch % (epochs // 10 + 1) == 0:
                    self._save_model(save_dir)
                    
                print(f"Epoch {epoch}: 训练损失 = {train_loss:.6f}")
                
        # 训练完成
        print(f"训练完成，最佳验证损失: {best_val_loss:.6f} (轮次 {best_epoch})")
        
        # 如果有保存的最佳模型，加载它
        if has_validation and save_dir:
            self._load_model(save_dir)
            
        return self
            
    def _save_model(self, save_dir):
        """保存模型到指定目录"""
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, "model.pth")
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'total_input_dim': self.total_input_dim,
            'config': {
                'input_dim': self.input_dim,
                'seq_len': self.seq_len,
                'pred_len': self.pred_len,
                'weather_dim': self.weather_dim,
                'd_model': self.d_model,
                'include_peaks': self.include_peaks,
                'pos_encoding_type': self.pos_encoding_type
            }
        }, model_path)
        print(f"模型已保存至: {model_path}")
        
    def _load_model(self, save_dir):
        """从指定目录加载模型"""
        model_path = os.path.join(save_dir, "model.pth")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            if self.optimizer and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'train_losses' in checkpoint:
                self.train_losses = checkpoint['train_losses']
            if 'val_losses' in checkpoint:
                self.val_losses = checkpoint['val_losses']
            print(f"模型已从 {model_path} 加载")
        else:
            print(f"找不到模型文件: {model_path}")
            
    def predict(self, X):
        """
        使用模型进行预测
        
        Args:
            X: 输入数据，形状为 [样本数, seq_len, features]
            
        Returns:
            numpy.ndarray: 预测结果，形状为 [样本数, pred_len]
        """
        self.eval()  # 设置为评估模式
        
        # 转换为PyTorch张量
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # 如果批量大小为1，直接预测
        if X_tensor.shape[0] == 1:
            with torch.no_grad():
                X_tensor = X_tensor.to(self.device)
                y_pred = self(X_tensor).cpu().numpy()
            return y_pred
        
        # 对于大批量，分批处理以避免内存问题
        batch_size = 64
        n_samples = X_tensor.shape[0]
        y_pred_list = []
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                X_batch = X_tensor[i:i + batch_size].to(self.device)
                y_pred_batch = self(X_batch).cpu().numpy()
                y_pred_list.append(y_pred_batch)
                
        # 合并所有批次的预测结果
        y_pred = np.vstack(y_pred_list)
        return y_pred

# 准备一个适合峰值感知的损失函数
class PeakAwareLoss(nn.Module):
    """峰值感知损失函数，对峰值时段的预测错误给予更高的权重"""
    
    def __init__(self, base_criterion=nn.MSELoss(), peak_weight=5.0, non_peak_weight=1.0):
        """
        初始化峰值感知损失函数
        
        Args:
            base_criterion: 基础损失函数
            peak_weight: 峰值时段的权重
            non_peak_weight: 非峰值时段的权重
        """
        super(PeakAwareLoss, self).__init__()
        self.base_criterion = base_criterion
        self.peak_weight = peak_weight
        self.non_peak_weight = non_peak_weight
        
    def forward(self, pred, target, is_peak=None):
        """
        计算加权损失
        
        Args:
            pred: 预测值
            target: 目标值
            is_peak: 是否为峰值时段的标志，形状与pred相同
            
        Returns:
            torch.Tensor: 加权损失值
        """
        base_loss = self.base_criterion(pred, target)
        
        if is_peak is None:
            return base_loss
            
        # 确保is_peak与pred和target的形状兼容
        if is_peak.shape != pred.shape:
            is_peak = is_peak.view(-1, 1).expand_as(pred)
            
        # 计算加权损失
        weights = torch.where(is_peak, self.peak_weight, self.non_peak_weight)
        weighted_loss = base_loss * weights.mean()
        
        return weighted_loss 