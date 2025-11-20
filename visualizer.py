
# visualizer.py

import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import numpy as np
from config import PLOT_PATH, RESULTS_DIR
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']        # 黑体
plt.rcParams['axes.unicode_minus'] = False 

def plot_predictions(inversed_y_test, inversed_predictions, target_features):
    """
    可视化模型的预测结果。
    :param inversed_y_test: 反归一化后的实际值。
    :param inversed_predictions: 反归一化后的预测值。
    """
    plt.figure(figsize=(18, 7))

    # 绘制第一个目标变量的预测图
    plt.subplot(1, 2, 1)
    plt.plot(inversed_y_test[:, 0], color='blue', label=f'Actual {target_features[0]}' if len(target_features) > 0 else 'Actual Target 1')
    plt.plot(inversed_predictions[:, 0], color='red', label=f'Predicted {target_features[0]}' if len(target_features) > 0 else 'Predicted Target 1')
    plt.title(f'{target_features[0]} Prediction' if len(target_features) > 0 else 'Target 1 Prediction')
    plt.xlabel('Time')
    plt.ylabel('Proportion')
    plt.legend()

    # 绘制第二个目标变量的预测图
    plt.subplot(1, 2, 2)
    plt.plot(inversed_y_test[:, 1], color='blue', label=f'Actual {target_features[1]}' if len(target_features) > 1 else 'Actual Target 2')
    plt.plot(inversed_predictions[:, 1], color='red', label=f'Predicted {target_features[1]}' if len(target_features) > 1 else 'Predicted Target 2')
    plt.title(f'{target_features[1]} Prediction' if len(target_features) > 1 else 'Target 2 Prediction')
    plt.xlabel('Time')
    plt.ylabel('Proportion')
    plt.legend()

    plt.tight_layout()

    # 确保保存路径存在
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.savefig(PLOT_PATH)
    plt.show()
    print(f"预测图表已保存至 {PLOT_PATH}")


def plot_dataframe_all_columns(df, filtered_df=None, outlier_masks=None, 
                               output_name = None, out_dir=None, cols_per_fig=10, figsize=(16, 12), dpi=150, plot=True):
    """
    将 DataFrame 的所有列每 N 列绘制到一张大图中，每列一个子图并保存图像。

    如果提供了 `filtered_df` 和可选的 `outlier_masks`，在每个子图中会展示原始（灰）与过滤后（蓝）的对比，并将异常点标为红色。

    返回:
      - 若提供了 `filtered_df`，返回该 DataFrame；否则返回 None。
    """
    if out_dir is None:
        out_dir = os.path.join(RESULTS_DIR, 'all_columns')
    os.makedirs(out_dir, exist_ok=True)

    color_ori = 'lightgray' if filtered_df is not None else 'blue'
    cols = list(df.columns)
    total = len(cols)
    parts = math.ceil(total / cols_per_fig)

    for part in range(parts):
        start = part * cols_per_fig
        end = min(start + cols_per_fig, total)
        subset = cols[start:end]

        n_plots = len(subset)
        ncols = 2
        nrows = math.ceil(n_plots / ncols) if n_plots > 0 else 1

        if plot:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)
            if isinstance(axes, plt.Axes):
                axes = [axes]
            else:
                axes = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]

        for i, col in enumerate(subset):
            series_orig = pd.to_numeric(df[col], errors='coerce')

            if filtered_df is not None and col in filtered_df.columns:
                series_filtered = pd.to_numeric(filtered_df[col], errors='coerce')
                if outlier_masks is not None and col in outlier_masks:
                    mask = outlier_masks[col].reindex(series_orig.index).fillna(False)
                else:
                    try:
                        mask = (~series_orig.isna()) & (~series_filtered.isna()) & (~np.isclose(series_orig.fillna(0), series_filtered.fillna(0)))
                    except Exception:
                        mask = (series_orig != series_filtered)
            else:
                series_filtered = None
                mask = None

            if plot:
                ax = axes[i]
                ax.plot(df.index, series_orig, color=color_ori, linewidth=0.8, label='原始')
                if series_filtered is not None:
                    ax.plot(df.index, series_filtered, color='tab:blue', linewidth=1.0, label='过滤后')
                if mask is not None and mask.any():
                    ax.scatter(df.index[mask], series_orig[mask], color='red', s=10, label=f'异常点 ({int(mask.sum())})')

                ax.set_title(col)
                ax.grid(True, linestyle='--', alpha=0.4)
                try:
                    if pd.api.types.is_datetime64_any_dtype(df.index) or hasattr(df.index, 'tz'):
                        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
                except Exception:
                    pass
                ax.legend(loc='upper right', fontsize='small')

        # 隐藏多余子图
        if plot:
            for j in range(n_plots, len(axes)):
                axes[j].axis('off')

            plt.tight_layout()
            out_path = os.path.join(out_dir, f'{output_name}_part_{part+1}.png') if output_name else os.path.join(out_dir, f'all_columns_part_{part+1}.png')
            fig.savefig(out_path)
            plt.close(fig)
            # print(f"保存: {out_path}")

    # return filtered_df if filtered_df is not None else None