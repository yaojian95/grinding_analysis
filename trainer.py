
# trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from config import EPOCHS, BATCH_SIZE

def train_pytorch_model(model, X_train, y_train, X_test, y_test):
    """
    训练 PyTorch 深度学习模型。
    :param model: PyTorch 模型实例。
    :param X_train: 训练集特征 (numpy array)。
    :param y_train: 训练集标签 (numpy array)。
    :param X_test: 测试集特征 (numpy array)。
    :param y_test: 测试集标签 (numpy array)。
    :return: 训练好的模型。
    """
    print("\n开始训练 PyTorch 模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 转换为 PyTorch Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    # 创建 DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(EPOCHS):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # 在测试集上评估
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

    print("PyTorch 模型训练完成。")
    return model

def evaluate_pytorch_model(model, X_test, y_test, scaler, target_indices, total_cols_for_scaling):
    """
    评估 PyTorch 模型并进行预测，反归一化结果。
    :param model: 训练好的 PyTorch 模型实例。
    :param X_test: 测试集特征 (numpy array)。
    :param y_test: 测试集标签 (numpy array)。
    :param scaler: 用于反归一化的MinMaxScaler对象。
    :param target_indices: 目标变量在原始数据中的索引。
    :param total_cols_for_scaling: 参与归一化的所有列的总数，用于反归一化。
    :return: 反归一化后的实际值和预测值。
    """
    print("\n开始进行 PyTorch 模型预测...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        predictions_tensor = model(X_test_tensor)
        predictions = predictions_tensor.cpu().numpy()
    print("PyTorch 模型预测完成。")

    # 反归一化预测结果
    dummy_predictions = np.zeros((len(predictions), total_cols_for_scaling))
    dummy_predictions[:, target_indices] = predictions
    inversed_predictions = scaler.inverse_transform(dummy_predictions)[:, target_indices]

    # 同样处理y_test
    dummy_y_test = np.zeros((len(y_test), total_cols_for_scaling))
    dummy_y_test[:, target_indices] = y_test
    inversed_y_test = scaler.inverse_transform(dummy_y_test)[:, target_indices]

    return inversed_y_test, inversed_predictions

def evaluate_limix_model(limix_predictor, X_context, y_context, X_test_limix, y_test_limix, scaler, target_indices, total_cols_for_scaling):
    """
    评估 LimiX 模型并进行预测，反归一化结果。
    :param limix_predictor: LimiXPredictor 实例。
    :param X_context: 用于 LimiX few-shot 预测的上下文特征。
    :param y_context: 用于 LimiX few-shot 预测的上下文标签。
    :param X_test_limix: 待预测的测试集特征。
    :param y_test_limix: 待预测的测试集标签（用于对比）。
    :param scaler: 用于反归一化的MinMaxScaler对象。
    :param target_indices: 目标变量在原始数据中的索引。
    :param total_cols_for_scaling: 参与归一化的所有列的总数，用于反归一化。
    :return: 反归一化后的实际值和预测值。
    """
    print("\n开始进行LimiX模型预测...")
    # LimiX 的 predict 方法需要 x_train, y_train, x_test
    # 这里的 x_train, y_train 对应于我们的 X_context, y_context
    # x_test 对应于我们的 X_test_limix
    predictions = limix_predictor.predict(x_train=X_context, y_train=y_context, x_test=X_test_limix)
    print("LimiX模型预测完成。")

    # 反归一化预测结果
    dummy_predictions = np.zeros((len(predictions), total_cols_for_scaling))
    dummy_predictions[:, target_indices] = predictions
    inversed_predictions = scaler.inverse_transform(dummy_predictions)[:, target_indices]

    # 同样处理y_test_limix
    dummy_y_test = np.zeros((len(y_test_limix), total_cols_for_scaling))
    dummy_y_test[:, target_indices] = y_test_limix
    inversed_y_test = scaler.inverse_transform(dummy_y_test)[:, target_indices]

    return inversed_y_test, inversed_predictions

