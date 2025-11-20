
# main.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import os

from data_loader import load_and_preprocess_data, create_dataset_for_lstm, create_dataset_for_limix
from model_builder import build_lstm_model, build_attention_lstm_model, build_limix_model # 导入 LimiX 模型构建函数
from trainer import train_pytorch_model, evaluate_pytorch_model, evaluate_limix_model
from visualizer import plot_predictions
from config import (
    TIME_STEP, TEST_SPLIT, MODEL_PATH, DATA_FILE,
)

def main(model_choice = 'lstm'):
    print("\n--- 工业生产时序数据多输出预测项目 ---")

    # 0. 确保结果目录存在
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("limix_model", exist_ok=True)

    # 1. 数据加载和预处理
    print("\n阶段1: 数据加载和预处理...")
    scaled_data, scaler, input_feature_indices, target_indices, existing_features, existing_targets = load_and_preprocess_data()
    print(f"用于模型训练的数据形状: {scaled_data.shape}")

    # 2. 模型选择与预测
    print("\n阶段2: 模型选择与预测...")
    # model_choice = "limix" # 默认使用 LimiX 模型，用户可以在实际运行时修改

    if model_choice == "lstm":
        # 创建数据集 (LSTM)
        X, y = create_dataset_for_lstm(scaled_data, input_feature_indices, target_indices)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=42, shuffle=False)
        
        # 构建模型
        print("构建 PyTorch LSTM 模型。")
        input_size = X_train.shape[2] # (batch_size, seq_len, input_size)
        output_size = y_train.shape[1]
        model = build_lstm_model(input_size, output_size)
        print(model)

        # 训练和评估
        print("\n阶段2.1: 训练 PyTorch 模型...")
        model = train_pytorch_model(model, X_train, y_train, X_test, y_test)
        torch.save(model.state_dict(), MODEL_PATH) # 保存 PyTorch 模型状态字典
        print(f"模型已保存至 {MODEL_PATH}")
        print("\n阶段2.2: 评估 PyTorch 模型并进行预测...")
        inversed_y_test, inversed_predictions = evaluate_pytorch_model(model, X_test, y_test, scaler, target_indices, scaled_data.shape[1])

    elif model_choice == "attention_lstm":
        # 创建数据集 (Attention LSTM)
        X, y = create_dataset_for_lstm(scaled_data, input_feature_indices, target_indices)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, random_state=42, shuffle=False)

        # 构建模型
        print("构建 PyTorch 注意力机制LSTM模型。")
        input_size = X_train.shape[2]
        output_size = y_train.shape[1]
        model = build_attention_lstm_model(input_size, output_size)
        print(model)

        # 训练和评估
        print("\n阶段2.1: 训练 PyTorch 模型...")
        model = train_pytorch_model(model, X_train, y_train, X_test, y_test)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"模型已保存至 {MODEL_PATH}")
        print("\n阶段2.2: 评估 PyTorch 模型并进行预测...")
        inversed_y_test, inversed_predictions = evaluate_pytorch_model(model, X_test, y_test, scaler, target_indices, scaled_data.shape[1])

    elif model_choice == "limix":
        # 创建数据集 (LimiX)
        X_all, y_all = create_dataset_for_limix(scaled_data, input_feature_indices, target_indices)
        
        # 划分训练集和测试集
        train_size = int(len(X_all) * (1 - TEST_SPLIT))
        X_context, X_test_limix = X_all[:train_size], X_all[train_size:]
        y_context, y_test_limix = y_all[:train_size], y_all[train_size:]

        print(f"LimiX 上下文数据X形状: {X_context.shape}, y形状: {y_context.shape}")
        print(f"LimiX 测试数据X形状: {X_test_limix.shape}, y形状: {y_test_limix.shape}")

        # 构建 LimiX 模型
        print("构建 LimiX 模型。")
        input_dim = X_context.shape[1]
        output_dim = y_context.shape[1]
        limix_predictor = build_limix_model(input_dim, output_dim)

        # LimiX 是 few-shot 预测，不需要显式的 train_model 步骤
        print("\n阶段2.1: LimiX 模型进行预测...")
        inversed_y_test, inversed_predictions = evaluate_limix_model(limix_predictor, X_context, y_context, X_test_limix, y_test_limix, scaler, target_indices, scaled_data.shape[1])

    else:
        print("无效的模型选择，程序退出。")
        return

    # 3. 结果可视化
    print("\n阶段3: 结果可视化...")
    plot_predictions(inversed_y_test, inversed_predictions, existing_targets)

    print("\n--- 项目运行完成 --- ")

if __name__ == "__main__":
    main()

