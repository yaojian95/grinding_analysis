
# data_loader.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from config import (
    DATA_FILE, TIME_STEP,
    LAG_FEATURES, LAG_STEPS, ROLLING_WINDOW_FEATURES, ROLLING_WINDOW_SIZE, ROLLING_WINDOW_STATS,
    USE_PCA, PCA_N_COMPONENTS, VARIANCE_THRESHOLD, BALL_MILL
)
from scipy.signal import savgol_filter

BALL_MILL_SET = [1, 2, 3]
BALL_MILL_SET.remove(BALL_MILL)

def load_data():
    """
    加载原始数据，并处理CSV文件的特殊格式要求。
    返回原始数据DataFrame。
    """
    try:
        # 首先读取第三行作为列名（注意pandas的索引从0开始）
        # 然后从第五行开始读取数据
        # 设置header=2表示使用第三行（索引为2）作为列名
        # 设置skiprows=[3]表示跳过第四行（索引为3）
        # 设置index_col=0表示使用第一列作为索引
        # 设置parse_dates=True表示解析索引为日期时间类型

        data = pd.read_csv(DATA_FILE, 
                           header=2,  # 使用第三行（索引为2）作为列名
                           skiprows=[3],  # 跳过第四行（索引为3）
                           index_col=0, 
                           parse_dates=True)
        # 重命名索引列，确保它被正确识别为时间戳
        data.index.name = 'timestamp'

        input_features = [col for col in data.columns 
                if '%s号' % BALL_MILL_SET[0] not in col and '%s号' % BALL_MILL_SET[1] not in col
                and '+80' not in col and '-200' not in col]
        
        target_features = [col for col in data.columns if '%s号' % BALL_MILL in col and ('-200目' in col or '+80目' in col)]

        print("使用球磨机%d的输入变量" % BALL_MILL)
        print("对应的目标变量:")
        print(target_features)
        return data[input_features+target_features], input_features, target_features
    except FileNotFoundError:
        print(f"示例数据文件 '{DATA_FILE}' 未找到。请确保文件路径正确。")

def load_and_preprocess_data():
    """
    加载、预处理数据，并准备用于模型训练。
    包括缺失值处理、特征工程、PCA降维和归一化。
    :return: 归一化后的数据、scaler对象、输入特征索引、目标索引、存在的特征和目标列表。
    """

    raw_data, input_features, target_features = load_data()
    #填充缺失值
    raw_data.ffill(inplace=True)
    raw_data.bfill(inplace=True)

    # # 特征工程, 暂未加入
    # data_with_features = create_lag_features(raw_data, LAG_FEATURES, LAG_STEPS)
    # data_with_features = create_rolling_window_features(data_with_features, ROLLING_WINDOW_FEATURES, ROLLING_WINDOW_SIZE, ROLLING_WINDOW_STATS)

    #数据清洗：去除异常值；去除常量列；去除噪声; see grind_data_process.ipynb
    df_rolling_combined, masks = filter_outliers(raw_data, method='rolling_iqr', combine_with_global=True, rolling_window=240, iqr_multiplier=1.5, impute=True)
    data_removed = remove_constant_columns(df_rolling_combined)
    data_with_features = denoise_dataframe(data_removed, method='rolling_median', window=240)

    # 更新输入特征列表以包含新创建的特征
    current_features = [col for col in data_with_features.columns if col not in target_features]

    # PCA降维 (可选)
    pca_model = None
    if USE_PCA:
        data_processed, pca_model = apply_pca(data_with_features, current_features, n_components=PCA_N_COMPONENTS)
        # PCA之后，特征列表变为PC_1, PC_2, ...
        current_features = [col for col in data_processed.columns if col.startswith('PC_')]
    else:
        data_processed = data_with_features

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_processed)
    scaled_df = pd.DataFrame(scaled_data, columns=data_processed.columns, index=data_processed.index)

    # 获取输入特征和目标的索引
    existing_features = [f for f in current_features if f in scaled_df.columns]
    existing_targets = [t for t in target_features if t in scaled_df.columns]
    input_feature_indices = [scaled_df.columns.get_loc(col) for col in existing_features]
    target_indices = [scaled_df.columns.get_loc(col) for col in existing_targets]

    return scaled_data, scaler, input_feature_indices, target_indices, existing_features, existing_targets

def create_dataset_for_lstm(scaled_data, input_feature_indices, target_indices):
    """
    根据归一化数据和时间步长创建X（特征）和y（标签）数据集，适用于LSTM模型。
    """
    X, y = [], []
    for i in range(len(scaled_data) - TIME_STEP):
        X.append(scaled_data[i:(i + TIME_STEP), input_feature_indices])
        y.append(scaled_data[i + TIME_STEP, target_indices])

    return np.array(X), np.array(y)

def create_dataset_for_limix(scaled_data, input_feature_indices, target_indices):
    """
    根据归一化数据创建X（特征）和y（标签）数据集，适用于LimiX模型。
    LimiX的预测模式是 few-shot，即在预测 x_test 时，需要提供 x_train 和 y_train 作为上下文。
    这里我们准备数据，使得 X 是当前时间步的特征，y 是下一个时间步的目标。
    """
    X_features, y_targets = [], []
    for i in range(len(scaled_data) - 1):
        X_features.append(scaled_data[i, input_feature_indices])
        y_targets.append(scaled_data[i + 1, target_indices])
    
    return np.array(X_features), np.array(y_targets)

def create_lag_features(data_df, features, lags):
    """
    为指定的特征创建滞后特征。
    :param data_df: 原始DataFrame。
    :param features: 需要创建滞后特征的列名列表。
    :param lags: 滞后步长列表，例如 [1, 2, 3] 表示创建滞后1、2、3步的特征。
    :return: 包含滞后特征的DataFrame。
    """
    print("\n--- 特征工程: 创建滞后特征 ---")
    df_lagged = data_df.copy()
    for feature in features:
        for lag in lags:
            df_lagged[f"{feature}_lag_{lag}"] = df_lagged[feature].shift(lag)
    return df_lagged

def create_rolling_window_features(data_df, features, window_size, stats):
    """
    为指定的特征创建滚动窗口统计特征。
    :param data_df: 原始DataFrame。
    :param features: 需要创建滚动窗口特征的列名列表。
    :param window_size: 滚动窗口的大小。
    :param stats: 需要计算的统计量列表，例如 ['mean', 'std', 'min', 'max']。
    :return: 包含滚动窗口特征的DataFrame。
    """
    print("\n--- 特征工程: 创建滚动窗口特征 ---")
    df_rolled = data_df.copy()
    for feature in features:
        for stat in stats:
            if stat == 'mean':
                df_rolled[f"{feature}_rolling_mean_{window_size}"] = df_rolled[feature].rolling(window=window_size).mean()
            elif stat == 'std':
                df_rolled[f"{feature}_rolling_std_{window_size}"] = df_rolled[feature].rolling(window=window_size).std()
            elif stat == 'min':
                df_rolled[f"{feature}_rolling_min_{window_size}"] = df_rolled[feature].rolling(window=window_size).min()
            elif stat == 'max':
                df_rolled[f"{feature}_rolling_max_{window_size}"] = df_rolled[feature].rolling(window=window_size).max()
            # 可以添加更多统计量
    return df_rolled


def remove_constant_columns(df, variance_threshold=VARIANCE_THRESHOLD):
    """
    删除在整个时间序列中不发生变化的列（常量列）。
    :param df: pandas DataFrame
    :return: 剔除常量列后的 DataFrame
    """
    # 数值列：使用方差阈值判断
    num_vars = df.select_dtypes(include=[np.number])
    low_var_cols = []
    if not num_vars.empty:
        try:
            variances = num_vars.var(axis=0, skipna=True)
            low_var_cols = variances[variances <= variance_threshold].index.tolist()
        except Exception:
            low_var_cols = []

    # 非数值列：基于唯一值判断（去除NaN后唯一值<=1视为常量）
    nonnum = df.select_dtypes(exclude=[np.number])
    const_nonnum = [col for col in nonnum.columns if nonnum[col].dropna().nunique() <= 1]

    # 合并候选列并排除目标列，避免误删
    cols_to_drop = [c for c in set(low_var_cols + const_nonnum)]
    if cols_to_drop:
        # 细化打印信息，分别列出因方差阈值和因唯一值被判定的列
        dropped_from_variance = [c for c in cols_to_drop if c in low_var_cols]
        dropped_from_const = [c for c in cols_to_drop if c in const_nonnum]
        if dropped_from_variance:
            print(f"--- 因低方差被移除的列 ({len(dropped_from_variance)})---")
            for i, col in enumerate(dropped_from_variance, 1):
                print(f"  {i}. {col}")
            print("---" + "-" * 20)
        if dropped_from_const:
            print(f"--- 因常量/唯一值被移除的列 ({len(dropped_from_const)}): {dropped_from_const} ---")
        # print(f"--- 共移除列 ({len(cols_to_drop)}): {cols_to_drop} ---")
        return df.drop(columns=cols_to_drop)
    else:
        print("--- 未发现需移除的低方差/常量列 ---")
    return df


def compute_global_iqr(series_num, iqr_multiplier=1.5):
    """
    计算序列的全局 IQR 边界并返回 (g_lower, g_upper, q1, q3, iqr)，
    若计算失败则返回 (None, None, None, None, None)。
    """
    try:
        q1 = series_num.quantile(0.25)
        q3 = series_num.quantile(0.75)
        iqr = q3 - q1
        g_lower = q1 - iqr_multiplier * iqr
        g_upper = q3 + iqr_multiplier * iqr
        return g_lower, g_upper, q1, q3, iqr
    except Exception:
        return None, None, None, None, None


def filter_outliers(df, method='rolling_iqr', rolling_window=25, iqr_multiplier=1.5, impute=True, combine_with_global=False):
    """
    对 DataFrame 的每一列进行异常值检测与过滤。

    参数:
      - df: pandas DataFrame
      - method: 'iqr' | 'zscore' | 'rolling_iqr'
      - rolling_window: 用于 rolling_iqr 的窗口大小
      - iqr_multiplier: IQR 方法的乘数 k
      - impute: 若为 True，则用时间插值或前后填充修复被过滤的点

    返回:
      - filtered_df: 过滤并可选插值后的 DataFrame（数值列为 float）
      - outlier_masks: dict, 每列对应一个布尔 Series，True 表示被识别为异常点
    """
    filtered = {}
    masks = {}
    for col in df.columns:
        series_orig = df[col]
        series_num = pd.to_numeric(series_orig, errors='coerce')

        outlier_mask = pd.Series(False, index=series_num.index)
        if method == 'iqr' and not series_num.dropna().empty:
            # 使用统一的全局 IQR helper，便于复用与测试
            g_lower, g_upper, q1, q3, iqr = compute_global_iqr(series_num, iqr_multiplier=iqr_multiplier)
            if g_lower is not None and g_upper is not None:
                outlier_mask = (series_num < g_lower) | (series_num > g_upper)
            else:
                # 若无法计算全局边界，则退化为无异常（更安全的默认行为）
                outlier_mask = pd.Series(False, index=series_num.index)
        elif method == 'zscore' and not series_num.dropna().empty:
            mean = series_num.mean()
            std = series_num.std()
            if std != 0 and not np.isnan(std):
                z = (series_num - mean) / std
                outlier_mask = z.abs() > 3.0
        elif method == 'rolling_iqr' and not series_num.dropna().empty:
            try:
                # 计算基于滚动窗口的上下界
                q1 = series_num.rolling(window=rolling_window, min_periods=1, center=True).quantile(0.25)
                q3 = series_num.rolling(window=rolling_window, min_periods=1, center=True).quantile(0.75)
                iqr = q3 - q1
                lower = q1 - iqr_multiplier * iqr
                upper = q3 + iqr_multiplier * iqr
                outlier_mask = (series_num < lower) | (series_num > upper)

                # 决定是否需要预计算全局边界：用于填充 NaN 或合并全局检测
                need_global_bounds = outlier_mask.isnull().any() or combine_with_global
                global_bounds_available = False
                g_lower = g_upper = None
                if need_global_bounds:
                    g_lower, g_upper, _, _, _ = compute_global_iqr(series_num, iqr_multiplier=iqr_multiplier)
                    global_bounds_available = (g_lower is not None and g_upper is not None)

                # 若滚动边界在某些位置为 NaN，则用全局边界来填充判断
                if outlier_mask.isnull().any() and global_bounds_available:
                    outlier_mask = outlier_mask.fillna((series_num < g_lower) | (series_num > g_upper))

                # 若要求与全局 IQR 合并，则在全局边界可用的情况下合并（并集）
                if combine_with_global and global_bounds_available:
                    try:
                        global_mask = (series_num < g_lower) | (series_num > g_upper)
                        outlier_mask = outlier_mask | global_mask
                    except Exception:
                        pass
            except Exception:
                # 若滚动计算失败，回退为全局 IQR（使用辅助函数）
                g_lower, g_upper, _, _, _ = compute_global_iqr(series_num, iqr_multiplier=iqr_multiplier)
                if g_lower is not None and g_upper is not None:
                    outlier_mask = (series_num < g_lower) | (series_num > g_upper)
                else:
                    # 极端情况下仍然回退为简单阈值判断（空掩码）
                    outlier_mask = pd.Series(False, index=series_num.index)

        masks[col] = outlier_mask.fillna(False)

        series_filtered = series_num.copy()
        n_outliers = int(masks[col].sum())
        if n_outliers > 0:
            series_filtered.loc[masks[col]] = np.nan
            if impute:
                try:
                    series_filtered = series_filtered.interpolate(method='time', limit_direction='both')
                except Exception:
                    series_filtered = series_filtered.ffill().bfill()

        filtered[col] = series_filtered

    filtered_df = pd.DataFrame(filtered, index=df.index)
    # 对于无法转换为数值的列（全为非数值），filtered_df 会有 NaN，保留原列
    for col in df.columns:
        if filtered_df[col].isna().all():
            filtered_df[col] = df[col]

    return filtered_df, masks


def denoise_dataframe(df, method='rolling_median', window=5, polyorder=2, ema_alpha=None, min_periods=1):
    """
    对 DataFrame 中的数值列进行去噪平滑。

    支持的方法:
      - 'rolling_median' : 中位数滚动滤波（对脉冲噪声鲁棒）
      - 'rolling_mean'   : 滚动均值（简单低通）
      - 'ema'            : 指数加权移动平均（EWMA），参数 ema_alpha 控制平滑程度
      - 'savgol'         : Savitzky-Golay 滤波（保留波形特征），依赖 scipy.signal.savgol_filter，若不可用则退回到 rolling_mean

    参数:
      - df: pandas DataFrame
      - method: str, 上述方法之一
      - window: 滑动窗口大小（奇数更适合 savgol）
      - polyorder: savgol 的多项式阶数（仅在 method='savgol' 时使用）
      - ema_alpha: EMA 的平滑系数（0<alpha<=1）。若为 None，则按 window 大小自动计算 alpha = 2/(window+1)
      - min_periods: rolling 时允许的最小非空观测数

    返回:
      - denoised_df: 去噪后的 DataFrame（非数值列原样返回）
    """
    denoised = {}

    # 仅处理数值列，其他列原样返回
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    nonnum_cols = [c for c in df.columns if c not in num_cols]

    for col in num_cols:
        s = pd.to_numeric(df[col], errors='coerce')

        if method == 'rolling_median':
            smoothed = s.rolling(window=window, min_periods=min_periods, center=True).median()
        elif method == 'rolling_mean':
            smoothed = s.rolling(window=window, min_periods=min_periods, center=True).mean()
        elif method == 'ema':
            if ema_alpha is None:
                alpha = 2.0 / (window + 1.0)
            else:
                alpha = float(ema_alpha)
            # ewm 本身能处理 NaN，但为了更平滑，先做一次向前/向后填充的插值作为输入
            s_fill = s.interpolate(method='time', limit_direction='both')
            smoothed = s_fill.ewm(alpha=alpha, adjust=False).mean()
        elif method == 'savgol':   
            # Savgol 对 NaN 不友好，先做时间插值
            s_fill = s.interpolate(method='time', limit_direction='both')
            # window 必须为正奇数且 polyorder < window
            w = int(window)
            if w % 2 == 0:
                w += 1
            if w <= polyorder:
                w = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
            try:
                filtered_vals = savgol_filter(s_fill.values, window_length=w, polyorder=polyorder, mode='interp')
                smoothed = pd.Series(filtered_vals, index=s.index)
            except Exception:
                # 回退到 rolling_mean
                smoothed = s_fill.rolling(window=window, min_periods=min_periods, center=True).mean()
        else:
            raise ValueError(f"未知的去噪方法: {method}")

        # 对于输入中仍有 NaN 的位置，保留原始值（避免全部被覆盖）
        # 使用插值后再填充边缘
        smoothed = smoothed.ffill().bfill()

        # 将非数值或完全 NaN 的列保持原样
        if smoothed.isna().all():
            denoised[col] = df[col]
        else:
            denoised[col] = smoothed

    # 合并数值列的去噪结果与非数值列
    denoised_df = pd.DataFrame(denoised, index=df.index)
    for col in nonnum_cols:
        denoised_df[col] = df[col]

    # 保持原来的列顺序
    denoised_df = denoised_df[df.columns]
    return denoised_df

    return data

def apply_pca(df, input_features, n_components=0.95):
    """
    对输入特征应用PCA进行降维。
    :param df: 包含输入特征的DataFrame。
    :param input_features: 输入特征的列名列表。
    :param n_components: PCA保留的主成分数量或方差比例。
    :return: 降维后的DataFrame和PCA模型。
    """
    print(f"应用PCA进行特征降维，目标主成分数/方差: {n_components}")
    pca = PCA(n_components=n_components)
    # 确保只对存在的特征进行PCA
    existing_features = [f for f in input_features if f in df.columns]
    features_pca = pca.fit_transform(df[existing_features])
    print(f"PCA降维后特征数量: {pca.n_components_}")
    
    # 创建新的DataFrame
    pca_df = pd.DataFrame(features_pca, index=df.index, columns=[f'PC_{i+1}' for i in range(pca.n_components_)])
    
    # 将PCA处理后的特征与原始目标变量合并
    df_pca_combined = pd.concat([pca_df, df[[col for col in df.columns if col not in existing_features]]], axis=1)
    
    return df_pca_combined, pca


