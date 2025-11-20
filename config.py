import os 
# config.py

# 数据文件路径
# 使用os.path.join构建路径
DATA_FILE = os.path.join("..", "..", "grinding_data", "Ball Mill Section 1 Tag List_2023Q1.csv")

# 目标变量
TARGET_COLUMNS = ['磨浮工段1号球磨机旋流器溢流+80目', '磨浮工段1号球磨机旋流器溢流-200目']

# 所有可能的输入特征（工况参数）
ALL_INPUT_FEATURES = [
    '磨浮工段1号球磨机砂泵功率', '磨浮工段1号球磨机砂泵转速', '磨浮工段1号球磨机泵池液位', 
    '磨浮工段1号球磨机返砂含水量', '磨浮工段1号球磨机给矿处理量', '磨浮工段1号球磨机给矿含水量', '磨浮工段1号球磨机磨机功率', 
    '磨浮工段1号球磨机磨矿浓度', '磨浮工段1号球磨机前给水', '磨浮工段1号球磨机上浆固体量', '磨浮工段1号球磨机上浆含水量',
    '磨浮工段1号球磨机上浆总流量', '磨浮工段1号球磨机水量（检化）', '磨浮工段1号球磨机旋流器返砂矿量',
    '磨浮工段1号球磨机旋流器返砂浓度', '磨浮工段1号球磨机旋流器工作压力', '磨浮工段1号球磨机旋流器上浆（给矿矿浆）浓度',
    '磨浮工段1号球磨机旋流器上浆流量', '磨浮工段1号球磨机溢流固体量',
]

BALL_MILL = 1 # 球磨机编号


# 模型参数
TIME_STEP = 60
LSTM_UNITS = 100
DENSE_UNITS = 50
EPOCHS = 1000
BATCH_SIZE = 128
TEST_SPLIT = 0.2 # 训练集和测试集的划分比例

# LimiX 模型配置
LIMIX_MODEL_PATH = 'limix_model/LimiX-16M.ckpt'
LIMIX_INFERENCE_CONFIG = 'default' # 或指向一个配置文件路径
LIMIX_OUTLIER_REMOVE_STD = 12
LIMIX_SOFTMAX_TEMPERATURE = 0.9

# 文件路径配置
RESULTS_DIR = "results"
MODELS_DIR = "models"

PLOT_PATH = os.path.join(RESULTS_DIR, "predictions.png")
MODEL_PATH = os.path.join(MODELS_DIR, "pytorch_lstm_model.pth") # PyTorch模型保存路径

# 数据分析结果保存路径
DESCRIPTIVE_STATS_PATH = os.path.join(RESULTS_DIR, "descriptive_statistics.csv")
CORRELATION_MATRIX_PATH = os.path.join(RESULTS_DIR, "pearson_correlation_matrix.csv")
CORRELATION_HEATMAP_PATH = os.path.join(RESULTS_DIR, "pearson_correlation_heatmap.png")
MUTUAL_INFO_PATH = os.path.join(RESULTS_DIR, "mutual_information_scores.csv")
CROSS_CORRELATION_PLOT_PATH = os.path.join(RESULTS_DIR, "cross_correlation_plots.png")
ANOMALY_DETECTION_PLOT_PATH = os.path.join(RESULTS_DIR, "anomaly_detection_plot.png")
ANOMALY_CONTAMINATION = 0.01 # 异常检测中异常值的比例

# 特征工程配置
LAG_FEATURES = [
    "磨浮工段1号球磨机磨机功率",
    "磨浮工段1号球磨机给矿处理量",
    "磨浮工段1号球磨机旋流器工作压力"
] # 需要创建滞后特征的列名列表
LAG_STEPS = [1, 3, 6, 12] # 滞后步长列表 (例如，滞后1小时，3小时，6小时，12小时)

ROLLING_WINDOW_FEATURES = [
    "磨浮工段1号球磨机磨机功率",
    "磨浮工段1号球磨机给矿处理量"
] # 需要创建滚动窗口统计特征的列名列表
ROLLING_WINDOW_SIZE = 5 # 滚动窗口的大小 (例如，过去5个时间步)
ROLLING_WINDOW_STATS = ["mean", "std", "min", "max"] # 滚动窗口统计量列表

# PCA 特征降维配置
USE_PCA = False # 是否启用PCA
PCA_N_COMPONENTS = 0.95 # PCA保留的主成分数量或方差比例 (例如，0.95表示保留95%的方差)

# 常量/低方差列的方差阈值（小于等于该值的数值列会被视为常量并移除）
VARIANCE_THRESHOLD = 1e-8

