
# run_eda.py

import pandas as pd
import numpy as np
import os

from data_loader import generate_dummy_data
from exploratory_data_analysis import perform_descriptive_analysis, perform_correlation_analysis, perform_anomaly_detection
from config import (
    DATA_FILE, ALL_INPUT_FEATURES, TARGET_COLUMNS, DESCRIPTIVE_STATS_PATH,
    CORRELATION_MATRIX_PATH, CORRELATION_HEATMAP_PATH, ANOMALY_DETECTION_PLOT_PATH,
    ANOMALY_CONTAMINATION
)

def run_exploratory_data_analysis():
    print("\n--- 工业生产时序数据探索性分析 ---")

    # 0. 确保结果目录存在
    os.makedirs("results", exist_ok=True)

    # 1. 数据加载
    print("\n阶段1: 数据加载...")
    try:
        raw_data = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"示例数据文件 \'{DATA_FILE}\' 未找到。将生成模拟数据用于演示。")
        raw_data = generate_dummy_data()
        raw_data.to_csv(DATA_FILE) # 保存生成的模拟数据
        print(f"模拟数据已保存到 {DATA_FILE}")

    # 2. 执行数据分析
    print("\n阶段2: 执行数据分析功能...")
    
    # 描述性统计分析
    perform_descriptive_analysis(raw_data, output_path=DESCRIPTIVE_STATS_PATH)

    # 相关性分析
    perform_correlation_analysis(raw_data, ALL_INPUT_FEATURES, TARGET_COLUMNS, 
                                 output_path=CORRELATION_MATRIX_PATH, plot_path=CORRELATION_HEATMAP_PATH)

    # 异常检测
    numeric_cols = raw_data.select_dtypes(include=np.number).columns.tolist()
    features_for_anomaly = [col for col in numeric_cols if col not in TARGET_COLUMNS]
    data_with_anomalies = perform_anomaly_detection(raw_data.copy(), features_for_anomaly, 
                                                    contamination=ANOMALY_CONTAMINATION, 
                                                    plot_path=ANOMALY_DETECTION_PLOT_PATH)
    
    print("\n--- 探索性数据分析完成 --- ")

if __name__ == "__main__":
    run_exploratory_data_analysis()

