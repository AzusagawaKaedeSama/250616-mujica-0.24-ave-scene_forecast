import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    位置编码器，为序列提供位置信息
    
    在Transformer模型中，由于自注意力机制本身不包含位置信息，
    需要额外添加位置编码，使模型能够识别序列中元素的相对或绝对位置。
    """
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        初始化位置编码器
        
        Args:
            d_model: 模型的隐藏维度
            dropout: Dropout率
            max_len: 支持的最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 计算正弦和余弦位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用正弦
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用余弦
        
        # 添加批次维度并转置 [max_len, 1, d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # 注册为非参数缓冲区（不会被优化器更新）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播：将位置编码添加到输入张量
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            带位置信息的张量 [batch_size, seq_len, d_model]
        """
        # 添加位置编码（只使用对应序列长度的部分）
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    可学习的位置编码器，与固定的正弦位置编码不同，这个编码可以通过训练进行优化
    """
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        初始化可学习的位置编码器
        
        Args:
            d_model: 模型的隐藏维度
            dropout: Dropout率
            max_len: 支持的最大序列长度
        """
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建可学习的位置编码嵌入
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_len, d_model))
        
        # 初始化位置嵌入
        self._init_weights()
    
    def _init_weights(self):
        """初始化位置嵌入权重"""
        nn.init.xavier_normal_(self.position_embeddings)
    
    def forward(self, x):
        """
        前向传播：将可学习的位置编码添加到输入张量
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            带位置信息的张量 [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        # 添加位置编码（只使用对应序列长度的部分）
        x = x + self.position_embeddings[:, :seq_len, :]
        return self.dropout(x)


def get_positional_encoding(encoding_type, d_model, dropout=0.1, max_len=5000):
    """
    获取指定类型的位置编码器
    
    Args:
        encoding_type: 位置编码类型，'fixed'或'learnable'
        d_model: 模型的隐藏维度
        dropout: Dropout率
        max_len: 支持的最大序列长度
        
    Returns:
        位置编码器模块
    """
    if encoding_type.lower() == 'learnable':
        return LearnablePositionalEncoding(d_model, dropout, max_len)
    else:  # 默认使用固定的正弦位置编码
        return PositionalEncoding(d_model, dropout, max_len) 