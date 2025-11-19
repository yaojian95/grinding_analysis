# exploratory_data_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import ccf
from config import (
    ALL_INPUT_FEATURES, TARGET_COLUMNS, DESCRIPTIVE_STATS_PATH, 
    CORRELATION_MATRIX_PATH, CORRELATION_HEATMAP_PATH, 
    ANOMALY_DETECTION_PLOT_PATH, ANOMALY_CONTAMINATION
)
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def perform_descriptive_analysis(data_df, output_path=None):
    """
    对DataFrame进行描述性统计分析，并打印结果。可选地保存到文件。
    :param data_df: 包含所有原始数据的DataFrame。
    :param output_path: 描述性统计结果的保存路径（CSV格式）。如果为None，则只打印。
    """
    print("\n--- 描述性统计分析 ---")
    description = data_df.describe(include='all')
    print(description)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        description.to_csv(output_path)
        print(f"描述性统计结果已保存到: {output_path}")
    return description

def perform_correlation_analysis(data_df, input_features, target_columns, threshold=0.5, method='all'):
    """
    对输入特征和目标变量进行相关性分析，并可视化相关性矩阵。
    :param data_df: 包含所有原始数据的DataFrame。
    :param input_features: 输入特征的列名列表。
    :param target_columns: 目标变量的列名列表。
    :param threshold: float, 强相关判定阈值
    :param method: str, 相关性计算方法：'pearson' | 'spearman' | 'kendall' | 'all'
                  - pearson: 皮尔逊相关系数（线性相关）
                  - spearman: 斯皮尔曼等级相关（可检测非线性单调关系）
                  - kendall: 肯德尔 Tau 相关系数（非参数，对异常值更稳健）
                  - all: 同时计算以上所有方法
    """
    print("\n=== 相关性分析 ===")
    
    # 筛选出实际存在于数据中的特征和目标
    existing_input_features = [f for f in input_features if f in data_df.columns]
    existing_target_columns = [t for t in target_columns if t in data_df.columns]

    if not existing_input_features or not existing_target_columns:
        print("警告: 无法进行相关性分析，因为数据中缺少输入特征或目标变量。")
        return None
    
    # 设置要计算的相关系数方法
    methods = ['pearson', 'spearman', 'kendall'] if method == 'all' else [method.lower()]
    if not all(m in ['pearson', 'spearman', 'kendall'] for m in methods):
        print(f"警告: 未知的相关系数计算方法 {method}，将使用 pearson")
        methods = ['pearson']
    
    # 合并要分析的列
    cols_for_correlation = existing_input_features + existing_target_columns
    
    results = {}
    for corr_method in methods:
        print(f"\n--- {corr_method.capitalize()} 相关性分析 ---")
        correlation_matrix = data_df[cols_for_correlation].corr(method=corr_method)
        results[corr_method] = correlation_matrix
        
        # 输出绝对相关系数超过阈值的变量对（上三角，不含对角）
        try:
            th = float(threshold)
        except Exception:
            th = 0.5

        strong_pairs = []
        cols = correlation_matrix.columns.tolist()
        n = len(cols)
        for i in range(n):
            for j in range(i + 1, n):
                val = correlation_matrix.iloc[i, j]
                if pd.notnull(val) and abs(val) >= th:
                    strong_pairs.append((cols[i], cols[j], val))

        if strong_pairs:
            print(f"相关对 (|r| >= {th}): 共 {len(strong_pairs)} 对")
            for a, b, v in strong_pairs:
                print(f"{a} <-> {b}: {corr_method[0].upper()} = {v:.3f}")
        else:
            print(f"未发现 |r| >= {th} 的强相关对")

        # 绘制热力图
        plt.figure(figsize=(16, 12))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(f'{corr_method.capitalize()} 相关性矩阵')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        plt.close()

    # 如果只计算了一种方法，直接返回矩阵；否则返回字典
    return results[methods[0]] if len(methods) == 1 else results

def perform_anomaly_detection(data_df, features_for_anomaly, contamination=0.01, plot_path=None):
    """
    使用 Isolation Forest 进行异常检测，并可视化异常点。
    :param data_df: 包含所有原始数据的DataFrame。
    :param features_for_anomaly: 用于异常检测的特征列名列表。
    :param contamination: 数据集中异常值的比例（0到0.5之间）。
    :param plot_path: 异常检测结果图的保存路径（PNG格式）。
    :return: 带有 'anomaly' 列的DataFrame，其中 -1 表示异常，1 表示正常。
    """
    print("\n--- 异常检测 (Isolation Forest) ---")
    
    # 筛选出实际存在于数据中的特征
    existing_features = [f for f in features_for_anomaly if f in data_df.columns]

    if not existing_features:
        print("警告: 无法进行异常检测，因为数据中缺少用于异常检测的特征。")
        return data_df

    X = data_df[existing_features].copy()
    
    # 填充缺失值，Isolation Forest 不支持 NaN
    X.fillna(X.mean(), inplace=True)

    model = IsolationForest(contamination=contamination, random_state=42)
    data_df['anomaly'] = model.fit_predict(X)

    n_anomalies = data_df[data_df['anomaly'] == -1].shape[0]
    print(f"检测到 {n_anomalies} 个异常点 (占总数据的 {n_anomalies / len(data_df) * 100:.2f}%)...")

    if plot_path:
        plt.figure(figsize=(18, 6))
        plt.plot(data_df.index, data_df[existing_features[0]], label='主要特征', alpha=0.7)
        anomalies = data_df[data_df['anomaly'] == -1]
        plt.scatter(anomalies.index, anomalies[existing_features[0]], color='red', label='异常点', s=50, marker='o')
        plt.title(f'{existing_features[0]} 异常检测结果')
        plt.xlabel('时间')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        print(f"异常检测结果图已保存到: {plot_path}")
        plt.close()

    return data_df

def perform_mutual_information_analysis(data_df, input_features, target_columns, output_path=None):
    """
    计算输入特征与目标变量之间的互信息。
    :param data_df: 包含所有原始数据的DataFrame。
    :param input_features: 输入特征的列名列表。
    :param target_columns: 目标变量的列名列表。
    :param output_path: 互信息结果的保存路径（CSV格式）。如果为None，则只打印。
    """
    print("\n--- 互信息分析 ---")

    existing_input_features = [f for f in input_features if f in data_df.columns]
    existing_target_columns = [t for t in target_columns if t in data_df.columns]

    if not existing_input_features or not existing_target_columns:
        print("警告: 无法进行互信息分析，因为数据中缺少输入特征或目标变量。")
        return None

    # 互信息需要数值型数据，且不能有NaN
    df_mi = data_df[existing_input_features + existing_target_columns].dropna()
    if df_mi.empty:
        print("警告: 用于互信息分析的数据在去除NaN后为空。")
        return None

    mi_results = pd.DataFrame(columns=existing_target_columns, index=existing_input_features)

    for target in existing_target_columns:
        mi_scores = mutual_info_regression(df_mi[existing_input_features], df_mi[target])
        mi_results[target] = mi_scores
    
    print("互信息分数 (输入特征 vs 目标变量):\n")
    print(mi_results.sort_values(by=existing_target_columns[0], ascending=False))

    # if output_path:
    #     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #     mi_results.to_csv(output_path)
    #     print(f"互信息结果已保存到: {output_path}")

    return mi_results

def perform_cross_correlation_analysis(data_df, input_features, target_columns, max_lag=20, plot_path=None):
    """
    计算输入特征与目标变量之间的时滞互相关，并可视化。
    :param data_df: 包含所有原始数据的DataFrame。
    :param input_features: 输入特征的列名列表。
    :param target_columns: 目标变量的列名列表。
    :param max_lag: 最大滞后步数。
    :param plot_path: 时滞互相关图的保存路径（PNG格式）。
    """
    print("\n--- 时滞互相关分析 ---")

    existing_input_features = [f for f in input_features if f in data_df.columns]
    existing_target_columns = [t for t in target_columns if t in data_df.columns]

    if not existing_input_features or not existing_target_columns:
        print("警告: 无法进行时滞互相关分析，因为数据中缺少输入特征或目标变量。")
        return None

    # 时滞互相关需要数值型数据，且不能有NaN
    df_ccf = data_df[existing_input_features + existing_target_columns].dropna()
    if df_ccf.empty:
        print("警告: 用于时滞互相关分析的数据在去除NaN后为空。")
        return None
    if plot_path:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    for target in existing_target_columns:
        plt.figure(figsize=(15, 8))
        for feature in existing_input_features:
            # 确保序列是平稳的，这里简单差分处理，实际可能需要更复杂的平稳性检验和处理
            # 或者直接使用原始数据，但解释性会受影响
            series1 = df_ccf[feature].diff().dropna()
            series2 = df_ccf[target].diff().dropna()
            
            if len(series1) > max_lag and len(series2) > max_lag:
                ccf_values = ccf(series1, series2, adjusted=False, nlags=max_lag)
                # 调整 x 轴范围以匹配 ccf_values 的长度
                lags = np.arange(-max_lag, max_lag + 1)
                # ccf 函数返回的是从 lag 0 到 max_lag 的值，我们需要手动构建对称的滞后轴
                # statsmodels.tsa.stattools.ccf 返回的是 (max_lag + 1) 个值，从 lag 0 到 max_lag
                # 为了绘制对称图，我们通常需要计算负滞后
                # 这里简化处理，假设 ccf_values 已经包含了正负滞后信息或者我们只关注正滞后
                # 如果需要完整的双向滞后，需要更复杂的计算或使用其他库
                # 暂时只绘制正滞后，或者根据 ccf_values 的实际含义调整
                # 修正：ccf 返回的是从0到nlags的互相关系数，所以需要手动构建对称的滞后轴
                # 这里使用 np.concatenate((ccf_values[::-1], ccf_values[1:])) 是一个常见的绘制方法，但需要注意 ccf_values 的实际含义
                # 为了避免混淆，我们只绘制正滞后，或者明确说明
                # 考虑到用户可能希望看到双向滞后，这里尝试一个更通用的绘制方法
                # 假设 ccf_values 已经包含了从0到max_lag的互相关系数
                # 我们可以手动构建负滞后部分，或者使用 statsmodels.graphics.tsaplots.plot_acf/plot_pacf 的变体
                # 为了简单起见，我们只绘制 ccf_values (正滞后) 并标记
                plt.plot(np.arange(len(ccf_values)), ccf_values, label=feature)
            else:
                print(f"警告: {feature} 或 {target} 序列太短，无法计算时滞互相关。")

        plt.title(f'{target} 与输入特征的时滞互相关 (差分后)')
        plt.xlabel('滞后 (Lag)')
        plt.ylabel('互相关系数')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.savefig(plot_path.replace(".png", f"__{target}.png"))
        # plt.close()
        # print(f"时滞互相关图已保存到: {plot_path.replace(".png", f"__{target}.png")}")

