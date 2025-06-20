#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch GPU和CUDA加速测试脚本
简单测试GPU可用性、张量操作和模型训练
"""

import torch
import torch.nn as nn
import time
import numpy as np

def test_gpu_availability():
    """测试GPU可用性"""
    print("=" * 50)
    print("GPU和CUDA可用性测试")
    print("=" * 50)
    
    # 检查CUDA是否可用
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        # 获取GPU数量
        gpu_count = torch.cuda.device_count()
        print(f"GPU数量: {gpu_count}")
        
        # 获取当前GPU
        current_device = torch.cuda.current_device()
        print(f"当前GPU设备: {current_device}")
        
        # 获取GPU名称
        gpu_name = torch.cuda.get_device_name(current_device)
        print(f"GPU名称: {gpu_name}")
        
        # 获取GPU内存信息
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
        memory_cached = torch.cuda.memory_reserved(current_device) / 1024**3
        print(f"已分配内存: {memory_allocated:.2f} GB")
        print(f"缓存内存: {memory_cached:.2f} GB")
        
        return True
    else:
        print("CUDA不可用，将使用CPU")
        return False

def test_tensor_operations(gpu_available):
    """测试张量操作性能"""
    print("\n" + "=" * 50)
    print("张量操作性能测试")
    print("=" * 50)
    
    # 创建更大的测试数据以获得更明显的性能差异
    size = 2000
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    
    if gpu_available:
        # 测试CPU操作
        start_time = time.time()
        c_cpu = torch.mm(a, b)
        cpu_time = time.time() - start_time
        print(f"CPU矩阵乘法耗时: {cpu_time:.4f} 秒")
        
        # 测试GPU操作
        a_gpu = a.cuda()
        b_gpu = b.cuda()
        
        # 预热GPU
        torch.mm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        
        # 多次运行以获得更准确的时间
        num_runs = 5
        gpu_times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            c_gpu = torch.mm(a_gpu, b_gpu)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            gpu_times.append(gpu_time)
        
        avg_gpu_time = sum(gpu_times) / len(gpu_times)
        print(f"GPU矩阵乘法平均耗时: {avg_gpu_time:.4f} 秒")
        
        # 计算加速比
        if avg_gpu_time > 0:
            speedup = cpu_time / avg_gpu_time
            print(f"GPU加速比: {speedup:.2f}x")
        else:
            print("GPU时间过短，无法计算准确加速比")
        
        # 验证结果一致性
        c_cpu_from_gpu = c_gpu.cpu()
        is_close = torch.allclose(c_cpu, c_cpu_from_gpu, atol=1e-5)
        print(f"结果一致性检查: {'通过' if is_close else '失败'}")
        
    else:
        # 仅CPU测试
        start_time = time.time()
        c = torch.mm(a, b)
        cpu_time = time.time() - start_time
        print(f"CPU矩阵乘法耗时: {cpu_time:.4f} 秒")

def test_model_training(gpu_available):
    """测试模型训练性能"""
    print("\n" + "=" * 50)
    print("模型训练性能测试")
    print("=" * 50)
    
    # 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(100, 200)
            self.fc2 = nn.Linear(200, 100)
            self.fc3 = nn.Linear(100, 1)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # 创建模型和数据
    model = SimpleModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 生成训练数据
    batch_size = 1000
    num_epochs = 10
    X = torch.randn(batch_size, 100)
    y = torch.randn(batch_size, 1)
    
    if gpu_available:
        # 移动模型和数据到GPU
        model = model.cuda()
        X = X.cuda()
        y = y.cuda()
        print("使用GPU训练")
    else:
        print("使用CPU训练")
    
    # 训练模型
    start_time = time.time()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    if gpu_available:
        torch.cuda.synchronize()
    
    training_time = time.time() - start_time
    print(f"训练总耗时: {training_time:.4f} 秒")
    print(f"平均每轮耗时: {training_time/num_epochs:.4f} 秒")

def test_memory_usage(gpu_available):
    """测试内存使用情况"""
    print("\n" + "=" * 50)
    print("内存使用测试")
    print("=" * 50)
    
    if gpu_available:
        # 清空GPU缓存
        torch.cuda.empty_cache()
        
        # 创建大张量测试内存
        try:
            # 尝试分配1GB内存
            large_tensor = torch.randn(1024, 1024, 256).cuda()  # 约1GB
            print("成功分配1GB GPU内存")
            
            # 获取内存使用情况
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"分配后GPU内存: {allocated:.2f} GB")
            print(f"保留GPU内存: {reserved:.2f} GB")
            
            # 释放内存
            del large_tensor
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            print(f"GPU内存不足: {e}")
    
    # 测试CPU内存
    try:
        large_tensor_cpu = torch.randn(1024, 1024, 256)  # 约1GB
        print("成功分配1GB CPU内存")
        del large_tensor_cpu
    except RuntimeError as e:
        print(f"CPU内存不足: {e}")

def main():
    """主函数"""
    print("PyTorch GPU和CUDA加速测试")
    print(f"PyTorch版本: {torch.__version__}")
    
    # 测试GPU可用性
    gpu_available = test_gpu_availability()
    
    # 测试张量操作
    test_tensor_operations(gpu_available)
    
    # 测试模型训练
    test_model_training(gpu_available)
    
    # 测试内存使用
    test_memory_usage(gpu_available)
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)

if __name__ == "__main__":
    main() 