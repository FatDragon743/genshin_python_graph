"""
简要说明：原神抽卡概率与统计工具（精简说明）

主要功能：
- `GenshinWishProbability`：计算单抽概率、累积概率、生成统计并绘图。
- 从 Excel 读取：`find_pulls_from_excel`、`read_full_wish_records_from_excel`（依赖 pandas 和 openpyxl）。
- 分析与绘图：`plot_full_analysis`、`plot_merged_analysis`。
- GUI：`create_gui` 启动 Tkinter 界面，支持读取文件、生成图表并保存。

依赖：numpy、matplotlib、pandas（openpyxl）、tkinter（标准库）。
用法：运行 `python test3.py` 启动 GUI，或在代码中导入类/函数用于无界面分析。
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
import tkinter as tk
from tkinter import ttk, messagebox
import warnings
import os
from datetime import datetime
import glob
import re
import tempfile
try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

# 字体设置，防止中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.monospace'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
warnings.filterwarnings('ignore')


class GenshinWishProbability:
    """原神抽卡概率计算与可视化类"""

    def __init__(self):
        self.BASE_RATE = 0.006
        self.SOFT_PITY_START = 74
        self.HARD_PITY = 90
        self.RATE_INCREASE = 0.06
        self.MAX_PROB = 1.0

    def calculate_probability(self, pulls_since_last):
        probabilities = []
        for i in range(pulls_since_last, self.HARD_PITY + 1):
            if i < self.SOFT_PITY_START:
                prob = self.BASE_RATE
            elif i < self.HARD_PITY - 1:
                prob = self.BASE_RATE + self.RATE_INCREASE * (i - self.SOFT_PITY_START + 1)
                prob = min(prob, self.MAX_PROB)
            else:
                prob = 1.0
            probabilities.append(prob)
        return probabilities

    def calculate_cumulative_prob(self, probabilities):
        cumulative_probs = []
        prob_no_5star = 1.0
        for prob in probabilities:
            prob_no_5star *= (1 - prob)
            cumulative_probs.append(1 - prob_no_5star)
        return cumulative_probs

    def plot_probability_curve(self, pulls_since_last, probabilities, cumulative_probs, stats=None):
        """绘制概率曲线：垂直堆叠，每张子图一行。若提供 `stats`，在底部显示统计文本。"""
        # 垂直布局：每个子图占一行，行数根据是否显示统计信息而定
        nrows = 3 if stats is not None else 2
        # 每行高度增加以生成更长的图像，便于查看
        per_row_height = 6.5
        fig_height = max(10, per_row_height * nrows)
        fig, axes = plt.subplots(nrows, 1, figsize=(12, fig_height), constrained_layout=True)
        if nrows == 3:
            ax1, ax2, ax3 = axes
            ax3.axis('off')
        else:
            ax1, ax2 = axes

        x_values = list(range(pulls_since_last, pulls_since_last + len(probabilities)))

        bars = ax1.bar(x_values, probabilities, color='skyblue', edgecolor='black', linewidth=0.5)
        for i, (x, prob) in enumerate(zip(x_values, probabilities)):
            if x >= self.SOFT_PITY_START and prob > self.BASE_RATE and x < self.HARD_PITY:
                bars[i].set_color('lightcoral')
            elif x == self.HARD_PITY:
                bars[i].set_color('gold')

        ax1.axvline(x=pulls_since_last, color='green', linestyle='--', alpha=0.7, label=f'当前: {pulls_since_last}抽')
        ax1.axvline(x=self.SOFT_PITY_START, color='orange', linestyle='--', alpha=0.7,
                    label=f'软保底开始: {self.SOFT_PITY_START}抽')
        ax1.axvline(x=self.HARD_PITY, color='red', linestyle='--', alpha=0.7, label=f'硬保底: {self.HARD_PITY}抽')

        ax1.set_xlabel('抽数')
        ax1.set_ylabel('单抽出金概率')
        ax1.set_title(f'原神抽卡概率分析 (当前已连续{pulls_since_last}抽未出5星)')
        ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(pulls_since_last - 1, min(self.HARD_PITY + 5, pulls_since_last + len(probabilities)))
        ax1.set_ylim(0, 1.05)

        for i, prob in enumerate(probabilities):
            if prob > 0.1 or i == 0 or i == len(probabilities) - 1 or x_values[i] == self.SOFT_PITY_START:
                ax1.text(x_values[i], prob + 0.01, f'{prob:.1%}', ha='center', va='bottom', fontsize=9, rotation=0)

        ax2.plot(x_values, cumulative_probs, 'o-', linewidth=2, markersize=4, color='darkblue')
        for target_prob in [0.5, 0.75, 0.9, 0.99]:
            for i, cum_prob in enumerate(cumulative_probs):
                if cum_prob >= target_prob:
                    ax2.axhline(y=target_prob, color='gray', linestyle=':', alpha=0.5)
                    ax2.axvline(x=x_values[i], color='gray', linestyle=':', alpha=0.5)
                    ax2.text(x_values[i] + 0.5, target_prob - 0.03, f'{target_prob:.0%} ({x_values[i]}抽)', fontsize=9)
                    break

        ax2.set_xlabel('抽数')
        ax2.set_ylabel('累积出金概率')
        ax2.set_title('累积概率曲线')
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(pulls_since_last - 1, min(self.HARD_PITY + 5, pulls_since_last + len(probabilities)))
        ax2.set_ylim(0, 1.05)

        for i in range(0, len(cumulative_probs), max(1, len(cumulative_probs) // 10)):
            ax2.text(x_values[i], cumulative_probs[i] + 0.02, f'{cumulative_probs[i]:.1%}', ha='center', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        if stats is not None:
            stats_lines = [
                f"原神抽卡概率分析报告",
                "",
                f"当前状态: 已连续 {stats['current_pulls']} 抽未出5星",
                f"当前单抽出金概率: {stats['current_prob']:.2%}",
                f"下一抽出金概率: {stats['next_pull_prob']:.2%}",
                f"距离软保底开始还需: {stats['pulls_to_soft_pity']} 抽",
                f"距离硬保底还需: {stats['pulls_to_hard_pity']} 抽",
                f"数学期望(还需): {stats['expected_pulls']:.1f} 抽",
                "",
                "累积概率分析:"
            ]
            for target in [25, 50, 75, 90, 99]:
                key = f'pulls_to_{target}%'
                if key in stats:
                    value = stats[key]
                    if isinstance(value, str):
                        stats_lines.append(f"{target}%: {value}")
                    else:
                        stats_lines.append(f"{target}%: 还需 {value} 抽")

            stats_text = "\n".join(stats_lines)
            # 在底部独立一行显示统计文本，使整体呈现长页式布局
            ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=11, va='top', family='sans-serif',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.95, pad=0.8))

        plt.tight_layout()
        plt.close(fig)
        return fig

    def generate_statistics(self, pulls_since_last, probabilities, cumulative_probs):
        x_values = list(range(pulls_since_last, pulls_since_last + len(probabilities)))
        stats = {
            'current_pulls': pulls_since_last,
            'pulls_to_soft_pity': max(0, self.SOFT_PITY_START - pulls_since_last),
            'pulls_to_hard_pity': max(0, self.HARD_PITY - pulls_since_last),
            'current_prob': probabilities[0] if probabilities else 0,
            'next_pull_prob': probabilities[1] if len(probabilities) > 1 else 0,
        }
        target_probs = [0.25, 0.5, 0.75, 0.9, 0.99]
        for target in target_probs:
            for i, cum_prob in enumerate(cumulative_probs):
                if cum_prob >= target:
                    stats[f'pulls_to_{int(target*100)}%'] = x_values[i] - pulls_since_last
                    break
            else:
                stats[f'pulls_to_{int(target*100)}%'] = '>保底前剩余抽数'
        expected_value = 0
        for i, prob in enumerate(probabilities):
            prob_of_getting_here = 1.0
            for j in range(i):
                prob_of_getting_here *= (1 - probabilities[j])
            expected_value += (i + 1) * prob_of_getting_here * prob
        stats['expected_pulls'] = expected_value
        return stats

    def print_statistics(self, stats):
        print("=" * 50)
        print("原神抽卡概率分析报告")
        print("=" * 50)
        print(f"当前状态: 已连续 {stats['current_pulls']} 抽未出5星")
        print(f"当前单抽出金概率: {stats['current_prob']:.2%}")
        print(f"下一抽出金概率: {stats['next_pull_prob']:.2%}")
        print(f"距离软保底开始还需: {stats['pulls_to_soft_pity']} 抽")
        print(f"距离硬保底还需: {stats['pulls_to_hard_pity']} 抽")
        print(f"数学期望(还需): {stats['expected_pulls']:.1f} 抽")
        print("-" * 50)
        print("累积概率分析:")
        for target in [25, 50, 75, 90, 99]:
            key = f'pulls_to_{target}%'
            if key in stats:
                value = stats[key]
                if isinstance(value, str):
                    print(f"  {target}%概率出金: {value}")
                else:
                    print(f"  {target}%概率出金: 还需 {value} 抽 (总计 {stats['current_pulls'] + value} 抽)")
        print("=" * 50)


def find_pulls_from_excel(search_dir=None):
    """在当前目录查找原神祈愿记录 Excel 并返回距离上一个5星的抽数。

    返回: (pulls_since_last, info_str) 其中 pulls_since_last 为整数或 None，info_str 为文件名或错误说明
    """
    search_dir = search_dir or os.getcwd()
    pattern = os.path.join(search_dir, "原神祈愿记录*.xlsx")
    files = glob.glob(pattern)
    if not files:
        return None, "未找到匹配的祈愿记录文件"

    latest = max(files, key=os.path.getmtime)
    try:
        import pandas as pd
    except ImportError:
        return None, "缺少 pandas 库，请先安装: pip install pandas openpyxl"

    try:
        df = pd.read_excel(latest, engine='openpyxl')
    except Exception as e:
        return None, f"读取文件失败: {e}"

    n = len(df)
    if n == 0:
        return None, f"文件 {os.path.basename(latest)} 为空"

    cols = list(df.columns)
    pref = [c for c in cols if re.search(r'rank|rar|星|等级|rarity|评级', str(c), re.I)]
    idx_found = None

    def cell_indicates_5(val):
        if pd.isna(val):
            return False
        if isinstance(val, (int, float)):
            try:
                return int(val) == 5
            except Exception:
                return False
        s = str(val)
        if re.search(r'5\s*星|五星|5★|5-star|5星级', s):
            return True
        if re.fullmatch(r"\s*5\s*", s):
            return True
        return False

    if pref:
        for i in range(n - 1, -1, -1):
            for c in pref:
                try:
                    if cell_indicates_5(df.iloc[i][c]):
                        idx_found = i
                        break
                except Exception:
                    continue
            if idx_found is not None:
                break

    if idx_found is None:
        for i in range(n - 1, -1, -1):
            row = df.iloc[i]
            for val in row:
                try:
                    if cell_indicates_5(val):
                        idx_found = i
                        break
                except Exception:
                    continue
            if idx_found is not None:
                break

    if idx_found is None:
        return n, f"未检测到5星记录，使用总抽数 {n} 作为近似（文件: {os.path.basename(latest)})"

    pulls_since_last = (n - 1 - idx_found)
    return pulls_since_last, os.path.basename(latest)


def read_full_wish_records_from_excel(search_dir=None):
    """读取 Excel 中的所有抽卡记录，返回 (wish_counts, characters, info_str) 或 (None, None, error)

    wish_counts: 每次获得5星之间的抽数序列（列表）
    characters: 对应每次5星获得的角色/物品名称列表
    info_str: 源文件名或错误说明
    """
    search_dir = search_dir or os.getcwd()
    pattern = os.path.join(search_dir, "原神祈愿记录*.xlsx")
    files = glob.glob(pattern)
    if not files:
        return None, None, "未找到匹配的祈愿记录文件"

    latest = max(files, key=os.path.getmtime)
    try:
        import pandas as pd
    except ImportError:
        return None, None, "缺少 pandas 库，请先安装: pip install pandas openpyxl"

    try:
        df = pd.read_excel(latest, engine='openpyxl')
    except Exception as e:
        return None, None, f"读取文件失败: {e}"

    if df is None or len(df) == 0:
        return None, None, f"文件 {os.path.basename(latest)} 为空或无数据"

    cols = list(df.columns)
    # 试图识别名称列和星级/稀有度列
    name_col_candidates = [c for c in cols if re.search(r'name|名称|物品|item|角色', str(c), re.I)]
    rare_col_candidates = [c for c in cols if re.search(r'rank|rar|星|等级|rarity|评级', str(c), re.I)]

    name_col = name_col_candidates[0] if name_col_candidates else None
    rare_col = rare_col_candidates[0] if rare_col_candidates else None

    def cell_indicates_5(val):
        if pd.isna(val):
            return False
        if isinstance(val, (int, float)):
            try:
                return int(val) == 5
            except Exception:
                return False
        s = str(val)
        if re.search(r'5\s*星|五星|5★|5-star|5星级', s):
            return True
        if re.fullmatch(r"\s*5\s*", s):
            return True
        # 一些导出可能用文字描述 "5" 或 "五星"
        return False

    wish_counts = []
    characters = []
    count = 0

    # 处理行顺序：假设表格从最早到最近，如果是相反顺序也可工作（结果依赖文件格式）
    for idx, row in df.iterrows():
        count += 1
        # 获取名称
        name = None
        if name_col is not None:
            try:
                name = row[name_col]
            except Exception:
                name = None
        else:
            # 尝试从常见列中获取第一个非空字符串
            for c in cols:
                v = row[c]
                if pd.notna(v) and isinstance(v, str) and len(str(v).strip()) > 0:
                    name = v
                    break

        # 检查是否为5星
        is5 = False
        if rare_col is not None:
            try:
                is5 = cell_indicates_5(row[rare_col])
            except Exception:
                is5 = False
        else:
            # 回退：在该行所有列搜索是否有5星标识
            for c in cols:
                try:
                    if cell_indicates_5(row[c]):
                        is5 = True
                        break
                except Exception:
                    continue

        if is5:
            wish_counts.append(count)
            characters.append(str(name) if name is not None else '未知')
            count = 0

    if len(wish_counts) == 0:
        return None, None, f"未在文件中检测到5星记录（文件: {os.path.basename(latest)})"

    return wish_counts, characters, os.path.basename(latest)


def plot_full_analysis(wish_counts, characters):
    """基于 test.py 的分析绘图（6 子图），返回 matplotlib Figure 对象"""
    import matplotlib.pyplot as plt
    import numpy as np

    # 垂直布局：6 个子图依次向下排列，网页风格
    nrows = 6
    # 使用更高的每行高度生成长图，减小子图压缩
    per_row_height = 6.5
    fig_height = max(24, per_row_height * nrows)
    fig, axes = plt.subplots(nrows, 1, figsize=(12, fig_height), constrained_layout=True)
    fig.suptitle('原神抽卡数据分析（来自文件）', fontsize=16, fontweight='bold')

    # 1. 抽数分布直方图
    ax1 = axes[0]
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    hist_data, bins_out, patches = ax1.hist(wish_counts, bins=bins, edgecolor='black', alpha=0.7, color='skyblue')
    for i, (count, patch) in enumerate(zip(hist_data, patches)):
        ax1.text(patch.get_x() + patch.get_width()/2, count + 0.5, str(int(count)), ha='center', va='bottom')
    ax1.set_xlabel('抽数区间', fontsize=12)
    ax1.set_ylabel('出现次数', fontsize=12)
    ax1.set_title('抽数分布直方图', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([5, 15, 25, 35, 45, 55, 65, 75, 85])

    # 2. 累积概率分布图
    ax2 = axes[1]
    sorted_counts = np.sort(wish_counts)
    cumulative_prob = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    ax2.plot(sorted_counts, cumulative_prob * 100, 'o-', linewidth=2, markersize=4, color='coral')
    key_points = [20, 30, 40, 50, 60, 70, 80]
    for point in key_points:
        idx = np.searchsorted(sorted_counts, point)
        if idx < len(sorted_counts):
            prob = cumulative_prob[idx] * 100
            ax2.plot(point, prob, 'ro', markersize=8)
            ax2.text(point + 2, prob - 5, f'{prob:.1f}%', fontsize=10)
    ax2.set_xlabel('抽数', fontsize=12)
    ax2.set_ylabel('累积概率 (%)', fontsize=12)
    ax2.set_title('累积概率分布', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 90)
    ax2.set_ylim(0, 105)

    # 3. 角色类型分布（简单按是否常驻）
    ax3 = axes[2]
    standard_chars = {"刻晴", "迪卢克", "莫娜", "琴", "七七", "提纳里", "迪希雅"}
    standard_count = sum(1 for c in characters if c in standard_chars)
    limited_count = len(characters) - standard_count
    labels = ['限定角色', '常驻角色']
    sizes = [limited_count, standard_count]
    colors = ['lightgreen', 'lightcoral']
    explode = (0.05, 0)
    ax3.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    ax3.set_title('角色类型分布', fontsize=14)
    ax3.axis('equal')

    # 4. 平均抽数走势（每 batch_size 个 5 星为一批）
    ax4 = axes[3]
    batch_size = 10
    batch_means = []
    batch_indices = []
    for i in range(0, len(wish_counts), batch_size):
        batch = wish_counts[i:i+batch_size]
        if len(batch) > 0:
            batch_means.append(np.mean(batch))
            batch_indices.append(i//batch_size + 1)
    ax4.plot(batch_indices, batch_means, 'o-', linewidth=2, markersize=8, color='purple')
    ax4.axhline(y=np.mean(wish_counts), color='red', linestyle='--', alpha=0.7, label=f'总平均: {np.mean(wish_counts):.1f}')
    ax4.axhline(y=62.5, color='green', linestyle=':', alpha=0.7, label='理论期望: 62.5')
    ax4.set_xlabel('批次 (每10次5星)', fontsize=12)
    ax4.set_ylabel('平均抽数', fontsize=12)
    ax4.set_title('平均抽数走势', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_xticks(batch_indices)

    # 5. 角色出现频次柱状图（前10名）
    ax5 = axes[4]
    char_counts = {}
    for char in characters:
        char_counts[char] = char_counts.get(char, 0) + 1
    top_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    char_names = [char for char, _ in top_chars]
    char_freq = [count for _, count in top_chars]
    colors_bar = ['gold' if char in standard_chars else 'lightblue' for char in char_names]
    bars = ax5.barh(range(len(char_names)), char_freq, color=colors_bar, edgecolor='black')
    ax5.set_yticks(range(len(char_names)))
    ax5.set_yticklabels(char_names, fontsize=10)
    ax5.invert_yaxis()
    ax5.set_xlabel('出现次数', fontsize=12)
    ax5.set_title('角色出现频次 Top 10', fontsize=14)
    for bar, count in zip(bars, char_freq):
        width = bar.get_width()
        ax5.text(width + 0.1, bar.get_y() + bar.get_height()/2, str(count), va='center', fontsize=10)

    # 6. 统计指标表格
    ax6 = axes[5]
    ax6.axis('off')
    mean_count = np.mean(wish_counts)
    median_count = np.median(wish_counts)
    std_count = np.std(wish_counts)
    min_count = min(wish_counts)
    max_count = max(wish_counts)
    soft_guarantee = sum(1 for count in wish_counts if count >= 74) / len(wish_counts) * 100
    early_lucky = sum(1 for count in wish_counts if count <= 20) / len(wish_counts) * 100
    table_data = [
        ["指标", "你的数据", "理论参考"],
        ["平均抽数", f"{mean_count:.1f}", "~62.5"],
        ["中位数抽数", f"{median_count:.1f}", "~75"],
        ["标准差", f"{std_count:.1f}", "~22"],
        ["最早出金", f"{min_count}抽", "1抽"],
        ["最晚出金", f"{max_count}抽", "90抽"],
        ["软保底率(≥74)", f"{soft_guarantee:.1f}%", "~50%"],
        ["20抽内出金", f"{early_lucky:.1f}%", "~15%"],
        ["总抽数", f"{sum(wish_counts)}", "-"],
        ["5星总数", f"{len(wish_counts)}", "-"]
    ]
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center', colWidths=[0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4B8BBE')
        else:
            cell.set_facecolor('#f2f2f2')
        cell.set_edgecolor('gray')
    ax6.set_title('关键统计指标', fontsize=14, y=0.95)

    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_merged_analysis(pulls, probabilities, cumulative_probs, stats, wish_counts=None, characters=None, file_info=None):
    """将概率图与完整分析合并为垂直布局：概率图在上，完整分析逐项向下排列。"""
    # 根据是否包含完整分析，计算行数（2 行概率 + 6 行完整分析 或 1 行注释）
    has_full = (wish_counts is not None and characters is not None)
    nrows = 2 + (6 if has_full else 1)
    per_row_height = 6.5
    fig_height = max(14, per_row_height * nrows)
    fig, axes = plt.subplots(nrows, 1, figsize=(14, fig_height), constrained_layout=True)

    ax_prob_top = axes[0]
    ax_prob_bot = axes[1]

    x_values = list(range(pulls, pulls + len(probabilities)))
    bars = ax_prob_top.bar(x_values, probabilities, color='skyblue', edgecolor='black', linewidth=0.5)
    for i, (x, prob) in enumerate(zip(x_values, probabilities)):
        if x >= 74 and prob > 0.006 and x < 90:
            bars[i].set_color('lightcoral')
        elif x == 90:
            bars[i].set_color('gold')
    ax_prob_top.axvline(x=pulls, color='green', linestyle='--', alpha=0.7, label=f'当前: {pulls}抽')
    ax_prob_top.axvline(x=74, color='orange', linestyle='--', alpha=0.7, label='软保底开始: 74抽')
    ax_prob_top.axvline(x=90, color='red', linestyle='--', alpha=0.7, label='硬保底: 90抽')
    ax_prob_top.set_title(f'单抽概率 (已连续 {pulls} 抽未出5星)')
    ax_prob_top.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_prob_top.grid(True, alpha=0.3)
    ax_prob_top.legend()

    for i, prob in enumerate(probabilities):
        if prob > 0.1 or i == 0 or i == len(probabilities) - 1 or x_values[i] == 74:
            ax_prob_top.text(x_values[i], prob + 0.01, f'{prob:.1%}', ha='center', va='bottom', fontsize=9, rotation=0)

    ax_prob_bot.plot(x_values, cumulative_probs, 'o-', linewidth=2, markersize=4, color='darkblue')
    for target_prob in [0.5, 0.75, 0.9, 0.99]:
        for i, cum_prob in enumerate(cumulative_probs):
            if cum_prob >= target_prob:
                ax_prob_bot.axhline(y=target_prob, color='gray', linestyle=':', alpha=0.5)
                ax_prob_bot.axvline(x=x_values[i], color='gray', linestyle=':', alpha=0.5)
                ax_prob_bot.text(x_values[i] + 0.5, target_prob - 0.03, f'{target_prob:.0%} ({x_values[i]}抽)', fontsize=9)
                break
    ax_prob_bot.set_title('累积出金概率')
    ax_prob_bot.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_prob_bot.grid(True, alpha=0.3)

    # 右侧区域改为向下展开的完整分析
    if has_full:
        ax1 = axes[2]
        bins = [0,10,20,30,40,50,60,70,80,90]
        hist_data, _, patches = ax1.hist(wish_counts, bins=bins, edgecolor='black', alpha=0.7, color='skyblue')
        for count, patch in zip(hist_data, patches):
            ax1.text(patch.get_x() + patch.get_width()/2, count + 0.5, str(int(count)), ha='center', va='bottom')
        ax1.set_title('抽数分布直方图')

        ax2 = axes[3]
        sorted_counts = np.sort(wish_counts)
        cumulative_prob = np.arange(1, len(sorted_counts)+1)/len(sorted_counts)
        ax2.plot(sorted_counts, cumulative_prob*100, 'o-', color='coral')
        ax2.set_title('累积概率分布')

        ax3 = axes[4]
        standard_chars = {"刻晴", "迪卢克", "莫娜", "琴", "七七", "提纳里", "迪希雅"}
        standard_count = sum(1 for c in characters if c in standard_chars)
        limited_count = len(characters) - standard_count
        ax3.pie([limited_count, standard_count], labels=['限定','常驻'], autopct='%1.1f%%')
        ax3.set_title('角色类型分布')

        ax4 = axes[5]
        batch_size = 10
        batch_means = []
        batch_indices = []
        for i in range(0, len(wish_counts), batch_size):
            batch = wish_counts[i:i+batch_size]
            if len(batch) > 0:
                batch_means.append(np.mean(batch))
                batch_indices.append(i//batch_size + 1)
        ax4.plot(batch_indices, batch_means, 'o-', color='purple')
        ax4.set_title('平均抽数走势')

        ax5 = axes[6]
        char_counts = {}
        for ch in characters:
            char_counts[ch] = char_counts.get(ch, 0) + 1
        top = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        names = [t[0] for t in top]
        freqs = [t[1] for t in top]
        ax5.barh(range(len(names)), freqs, color='lightblue')
        ax5.set_yticks(range(len(names)))
        ax5.set_yticklabels(names)
        ax5.invert_yaxis()
        ax5.set_title('角色出现频次 Top10')

        ax6 = axes[7]
        ax6.axis('off')
        mean_count = np.mean(wish_counts)
        median_count = np.median(wish_counts)
        std_count = np.std(wish_counts)
        min_count = min(wish_counts)
        max_count = max(wish_counts)
        table_data = [
            ["指标","你的数据","参考"],
            ["平均抽数", f"{mean_count:.1f}", "~62.5"],
            ["中位数", f"{median_count:.1f}", "~75"],
            ["标准差", f"{std_count:.1f}", "~22"],
            ["最早出金", f"{min_count}抽", "1抽"],
            ["最晚出金", f"{max_count}抽", "90抽"]
        ]
        table = ax6.table(cellText=table_data, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        if file_info:
            fig.text(0.5, 0.01, f'数据来源: {file_info}', fontsize=9, ha='center')

    else:
        ax_note = axes[2]
        ax_note.axis('off')
        note = "未检测到完整抽卡记录文件。\n仅显示概率分析。\n如需完整分析，请将 '原神祈愿记录*.xlsx' 放到当前文件夹。"
        ax_note.text(0.5, 0.5, note, ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.close(fig)
    return fig


def create_gui():
    root = tk.Tk()
    root.title("原神抽卡概率计算器")
    root.geometry("560x380")

    style = ttk.Style()
    style.theme_use('clam')

    fig_container = {'fig': None, 'pulls': None}

    title_label = ttk.Label(root, text="原神抽卡概率分析器", font=("Arial", 16, "bold"))
    title_label.pack(pady=10)

    input_frame = ttk.Frame(root)
    input_frame.pack(pady=10)

    ttk.Label(input_frame, text="距离上一个5星的抽数:", font=("Arial", 11)).pack(side=tk.LEFT, padx=5)
    pulls_var = tk.StringVar(value="0")
    pulls_entry = ttk.Entry(input_frame, textvariable=pulls_var, width=10, font=("Arial", 11))
    pulls_entry.pack(side=tk.LEFT, padx=5)
    ttk.Label(input_frame, text="抽", font=("Arial", 11)).pack(side=tk.LEFT)

    file_info_var = tk.StringVar(value="")
    def read_from_file():
        pulls_from_file, info = find_pulls_from_excel()
        if pulls_from_file is None:
            messagebox.showwarning("读取失败", info)
            file_info_var.set(info)
            return
        # 限制到 0-89（若大于89则提醒并取89）
        shown_pulls = pulls_from_file
        if pulls_from_file >= 90:
            shown_pulls = 89
            messagebox.showinfo("提示", f"检测到距离上次5星 {pulls_from_file} 抽（>=90），为符合显示限制，使用 {shown_pulls} 抽进行分析。")
        pulls_var.set(str(shown_pulls))
        file_info_var.set(f"从文件读取: {info}，原始: {pulls_from_file} 抽")

    file_btn = ttk.Button(input_frame, text="从文件读取抽数", command=read_from_file, width=18)
    file_btn.pack(side=tk.LEFT, padx=8)
    file_label = ttk.Label(root, textvariable=file_info_var, font=("Arial", 9), foreground="gray")
    file_label.pack()
    # 启动时尝试自动读取（若存在文件）
    try:
        read_from_file()
    except Exception:
        pass

    info_label = ttk.Label(root, text="请输入0-89之间的整数", font=("Arial", 10), foreground="gray")
    info_label.pack(pady=5)

    def calculate_probability():
        try:
            pulls = int(pulls_var.get())
            if pulls < 0 or pulls >= 90:
                messagebox.showerror("输入错误", "请输入0-89之间的整数！")
                return

            calculator = GenshinWishProbability()
            probabilities = calculator.calculate_probability(pulls)
            cumulative_probs = calculator.calculate_cumulative_prob(probabilities)
            stats = calculator.generate_statistics(pulls, probabilities, cumulative_probs)
            calculator.print_statistics(stats)
            fig = calculator.plot_probability_curve(pulls, probabilities, cumulative_probs, stats)
            fig_container['fig'] = fig
            fig_container['pulls'] = pulls
            save_btn.config(state=tk.NORMAL)
            # 保存临时图像并尝试在滚动窗口打开（便于查看长图）
            try:
                tmp = tempfile.gettempdir()
                tmp_path = os.path.join(tmp, f"wish_tmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                fig.savefig(tmp_path, dpi=150, bbox_inches='tight')
                try:
                    if Image is None or ImageTk is None:
                        raise ImportError('Pillow 未安装')
                    display_image_in_scroll_window(tmp_path)
                except Exception:
                    pass
            except Exception:
                pass
            messagebox.showinfo("成功", "图表已生成！\n现在可以点击'保存为图片'按钮保存图表。")

        except ValueError:
            messagebox.showerror("输入错误", "请输入有效的整数！")

    def save_figure():
        if fig_container['fig'] is None:
            messagebox.showwarning("警告", "请先生成图表！")
            return
        try:
            save_dir = os.path.expanduser("~/Pictures/GenshinWishAnalysis")
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pulls = fig_container['pulls']
            filename = f"wish_analysis_pulls{pulls}_{timestamp}.png"
            filepath = os.path.join(save_dir, filename)
            # 保存高分辨率长图
            fig_container['fig'].savefig(filepath, dpi=300, bbox_inches='tight')
            messagebox.showinfo("保存成功", f"图片已保存！\n路径: {filepath}")
            # 尝试在带滚动条的新窗口中打开（需要 pillow）
            try:
                if Image is None or ImageTk is None:
                    raise ImportError('Pillow 未安装')
                display_image_in_scroll_window(filepath)
            except Exception:
                # 无法展示时静默返回（用户可在文件管理器中打开）
                pass
        except Exception as e:
            messagebox.showerror("保存失败", f"保存图片时出错: {str(e)}")

    button_frame = ttk.Frame(root)
    button_frame.pack(pady=10)

    def generate_merged():
        try:
            # 尝试读取完整文件记录（可选）
            wish_counts, characters, info = read_full_wish_records_from_excel()
        except Exception:
            wish_counts, characters, info = None, None, None

        try:
            pulls = int(pulls_var.get())
            if pulls < 0 or pulls >= 90:
                messagebox.showerror("输入错误", "请输入0-89之间的整数！")
                return
        except ValueError:
            messagebox.showerror("输入错误", "请输入有效的整数！")
            return

        calculator = GenshinWishProbability()
        probabilities = calculator.calculate_probability(pulls)
        cumulative_probs = calculator.calculate_cumulative_prob(probabilities)
        stats = calculator.generate_statistics(pulls, probabilities, cumulative_probs)
        calculator.print_statistics(stats)

        try:
            fig = plot_merged_analysis(pulls, probabilities, cumulative_probs, stats, wish_counts, characters, info)
            fig_container['fig'] = fig
            fig_container['pulls'] = pulls
            save_btn.config(state=tk.NORMAL)
            # 自动保存到临时文件并在可滚动窗口展示（便于查看长图）
            try:
                tmp = tempfile.gettempdir()
                tmp_path = os.path.join(tmp, f"wish_tmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
                fig.savefig(tmp_path, dpi=150, bbox_inches='tight')
                try:
                    if Image is None or ImageTk is None:
                        raise ImportError('Pillow 未安装')
                    display_image_in_scroll_window(tmp_path)
                except Exception:
                    pass
            except Exception:
                pass
            messagebox.showinfo("成功", "已生成合并分析图表！")
        except Exception as e:
            messagebox.showerror("错误", f"生成合并图表时出错: {e}")

    generate_btn = ttk.Button(button_frame, text="生成合并分析", command=generate_merged, width=24)
    generate_btn.pack(side=tk.LEFT, padx=5)

    save_btn = ttk.Button(button_frame, text="保存为图片", command=save_figure, width=15, state=tk.DISABLED)
    save_btn.pack(side=tk.LEFT, padx=5)

    rule_frame = ttk.LabelFrame(root, text="原神抽卡概率规则", padding=10)
    rule_frame.pack(pady=10, padx=20, fill=tk.X)

    rule_text = """• 基础概率: 0.6% (每抽)
• 软保底: 74抽开始概率提升
• 硬保底: 90抽必出5星
• 概率提升: 74抽后每抽增加约6%概率"""

    rule_label = ttk.Label(rule_frame, text=rule_text, justify=tk.LEFT, font=("Arial", 9))
    rule_label.pack()

    author_label = ttk.Label(root, text="基于原神实际抽卡机制模拟", font=("Arial", 8), foreground="gray")
    author_label.pack(side=tk.BOTTOM, pady=5)

    root.mainloop()


def display_image_in_scroll_window(image_path):
    """在 Tkinter 可滚动窗口中显示大图（需要 Pillow）。"""
    if Image is None or ImageTk is None:
        raise ImportError('Pillow 未安装，请运行: pip install pillow')

    win = tk.Toplevel()
    win.title(os.path.basename(image_path))

    # 加载图片
    img = Image.open(image_path)
    width, height = img.size

    canvas = tk.Canvas(win, width=min(1200, width), height=min(800, height))
    hbar = ttk.Scrollbar(win, orient=tk.HORIZONTAL, command=canvas.xview)
    vbar = ttk.Scrollbar(win, orient=tk.VERTICAL, command=canvas.yview)
    canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

    hbar.pack(side=tk.BOTTOM, fill=tk.X)
    vbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

    # 将图像转换为 PhotoImage（可能需要缩放为适合显示器）
    photo = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor='nw', image=photo)
    canvas.image = photo
    canvas.config(scrollregion=(0, 0, width, height))

    # 允许使用鼠标滚轮滚动
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), 'units')

    canvas.bind_all('<MouseWheel>', _on_mousewheel)
    # 当窗口关闭时释放图像引用
    def on_close():
        try:
            canvas.image = None
        except Exception:
            pass
        win.destroy()

    win.protocol('WM_DELETE_WINDOW', on_close)


def main():
    """主函数：直接启动图形界面"""
    create_gui()


if __name__ == "__main__":
    try:
        import matplotlib
        import tkinter
        main()
    except ImportError as e:
        print(f"错误：缺少必要的库 - {e}")
        print("请使用以下命令安装所需库:")
        print("pip install numpy matplotlib pandas openpyxl")
        print("或者如果你使用Anaconda:")
        print("conda install numpy matplotlib pandas openpyxl")

