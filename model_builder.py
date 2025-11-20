
# model_builder.py

import torch
import torch.nn as nn
from config import LSTM_UNITS, DENSE_UNITS

# 尝试导入 LimiXPredictor，如果失败则说明 LimiX 未安装或环境问题
try:
    from limix_ldm.predict import LimiXPredictor
    import numpy as np
    print("LimiXPredictor 导入成功。")
except ImportError:
    print("LimiXPredictor 导入失败。请确保已安装 LimiX 库及其依赖。")
    LimiXPredictor = None
    np = None

# --- PyTorch LSTM Model --- #
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, DENSE_UNITS)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(DENSE_UNITS, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # out shape: (batch_size, seq_len, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output from the last time step
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# --- PyTorch Attention-LSTM Model --- #
class AttentionLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(AttentionLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        
        # Attention Mechanism
        # For self-attention, query, key, value are all from the LSTM output
        self.attention_linear = nn.Linear(hidden_size, hidden_size)
        self.context_linear = nn.Linear(hidden_size * 2, hidden_size) # For combining LSTM output and context
        
        # Decoder (can be another LSTM or directly Dense layers)
        self.fc1 = nn.Linear(hidden_size, DENSE_UNITS)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(DENSE_UNITS, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Self-Attention
        # query, key, value from lstm_out
        query = lstm_out
        key = lstm_out
        value = lstm_out
        
        # Calculate attention scores
        # scores shape: (batch_size, seq_len, seq_len)
        scores = torch.bmm(query, key.transpose(1, 2))
        attention_weights = torch.softmax(scores, dim=-1)
        
        # context_vector shape: (batch_size, seq_len, hidden_size)
        context_vector = torch.bmm(attention_weights, value)
        
        # Combine LSTM output with context vector (e.g., concatenate and pass through linear)
        # For simplicity, let's take the last step's context vector and combine with last LSTM output
        # Or, we can average context_vector across seq_len
        # Here, we'll use the context vector of the last time step
        attended_output = context_vector[:, -1, :]
        
        out = self.fc1(attended_output)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def build_lstm_model(input_size, output_size):
    """
    构建 PyTorch LSTM 模型。
    :param input_size: 输入特征的维度。
    :param output_size: 目标变量的维度。
    :return: PyTorch LSTM 模型实例。
    """
    model = LSTMModel(input_size, LSTM_UNITS, output_size)
    return model

def build_attention_lstm_model(input_size, output_size):
    """
    构建 PyTorch Attention-LSTM 模型。
    :param input_size: 输入特征的维度。
    :param output_size: 目标变量的维度。
    :return: PyTorch Attention-LSTM 模型实例。
    """
    model = AttentionLSTMModel(input_size, LSTM_UNITS, output_size)
    return model

def build_limix_model(input_dim, output_dim):
    """
    构建 LimiX 模型。由于 LimiX 是一个预训练模型，这里主要是初始化其预测器。
    需要下载 LimiX 模型文件。
    :param input_dim: 输入特征的维度。
    :param output_dim: 目标变量的维度。
    :return: LimiXPredictor 实例。
    """ 
    if LimiXPredictor is None:
        raise ImportError("LimiXPredictor 未成功导入，无法构建 LimiX 模型。")

    # LimiX 模型需要预先下载。这里假设模型文件路径为 \'limix_model/LimiX-16M.ckpt\'
    # 您需要根据 LimiX 的官方文档下载相应的模型文件并放置到正确路径。
    # 例如：wget https://www.limix.ai/models/LimiX-16M.ckpt -P limix_model/
    model_path = 'limix_model/LimiX-16M.ckpt'
    
    # 检查模型文件是否存在
    import os
    if not os.path.exists(model_path):
        print(f"警告: LimiX 模型文件 {model_path} 不存在。请从 LimiX 官方网站下载并放置到正确路径。")
        print("尝试下载 LimiX-16M.ckpt...")
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            import requests
            url = "https://www.limix.ai/models/LimiX-16M.ckpt"
            response = requests.get(url, stream=True)
            response.raise_for_status() # 检查请求是否成功
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"LimiX 模型文件已下载到 {model_path}")
        except Exception as e:
            print(f"下载 LimiX 模型文件失败: {e}")
            raise FileNotFoundError(f"LimiX 模型文件 {model_path} 不存在且下载失败，无法构建 LimiX 模型。")

    # LimiXPredictor 的初始化需要 device 参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # categorical_features_indices 需要根据您的数据实际情况提供
    # 假设所有输入特征都是数值型，如果没有分类特征，可以设置为 None 或空列表
    categorical_features_indices = [] # 示例：假设没有分类特征

    # LimiXPredictor 的 inference_config 参数，通常是一个列表或字符串，指向配置文件
    # 暂时使用默认配置，实际应用中可能需要根据 LimiX 文档进行配置
    inference_config = "default"

    limix_predictor = LimiXPredictor(
        device=device,
        model_path=model_path,
        task_type='Regression', # 您的任务是回归
        categorical_features_indices=categorical_features_indices,
        inference_config=inference_config
    )
    print("LimiXPredictor 初始化成功。")
    return limix_predictor

