"""抽卡概率计算与可视化（自 test3 迁出）。"""

from __future__ import annotations

import os
import tempfile
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

import tkinter as tk
from tkinter import ttk

from .constants import STANDARD_CHARS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.monospace"] = ["Microsoft YaHei", "SimHei", "SimSun"]
warnings.filterwarnings("ignore")


class GenshinWishProbability:
    """原神抽卡概率计算（支持不同卡池规则）。"""

    def __init__(
        self,
        base_rate: float = 0.006,
        soft_pity: int = 74,
        hard_pity: int = 90,
        rate_increase: float = 0.06,
    ):
        self.BASE_RATE = float(base_rate)
        self.SOFT_PITY_START = int(soft_pity)
        self.HARD_PITY = int(hard_pity)
        self.RATE_INCREASE = float(rate_increase)
        self.MAX_PROB = 1.0

    @classmethod
    def for_pool(cls, pool_key: str = "character") -> "GenshinWishProbability":
        from .constants import POOL_PITY_RULES

        rule = POOL_PITY_RULES.get(pool_key) or POOL_PITY_RULES["character"]
        return cls(
            base_rate=rule["base_rate"],
            soft_pity=rule["soft_pity"],
            hard_pity=rule["hard_pity"],
            rate_increase=rule["rate_increase"],
        )

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
            prob_no_5star *= 1 - prob
            cumulative_probs.append(1 - prob_no_5star)
        return cumulative_probs

    def plot_probability_curve(self, pulls_since_last, probabilities, cumulative_probs, stats=None):
        nrows = 3 if stats is not None else 2
        per_row_height = 6.5
        fig_height = max(10, per_row_height * nrows)
        fig, axes = plt.subplots(nrows, 1, figsize=(12, fig_height), constrained_layout=True)
        if nrows == 3:
            ax1, ax2, ax3 = axes
            ax3.axis("off")
        else:
            ax1, ax2 = axes

        x_values = list(range(pulls_since_last, pulls_since_last + len(probabilities)))

        bars = ax1.bar(x_values, probabilities, color="skyblue", edgecolor="black", linewidth=0.5)
        for i, (x, prob) in enumerate(zip(x_values, probabilities)):
            if x >= self.SOFT_PITY_START and prob > self.BASE_RATE and x < self.HARD_PITY:
                bars[i].set_color("lightcoral")
            elif x == self.HARD_PITY:
                bars[i].set_color("gold")

        ax1.axvline(x=pulls_since_last, color="green", linestyle="--", alpha=0.7, label=f"当前: {pulls_since_last}抽")
        ax1.axvline(
            x=self.SOFT_PITY_START,
            color="orange",
            linestyle="--",
            alpha=0.7,
            label=f"软保底开始: {self.SOFT_PITY_START}抽",
        )
        ax1.axvline(x=self.HARD_PITY, color="red", linestyle="--", alpha=0.7, label=f"硬保底: {self.HARD_PITY}抽")

        ax1.set_xlabel("抽数")
        ax1.set_ylabel("单抽出金概率")
        ax1.set_title(f"原神抽卡概率分析 (当前已连续{pulls_since_last}抽未出5星)")
        ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(pulls_since_last - 1, min(self.HARD_PITY + 5, pulls_since_last + len(probabilities)))
        ax1.set_ylim(0, 1.05)

        for i, prob in enumerate(probabilities):
            if prob > 0.1 or i == 0 or i == len(probabilities) - 1 or x_values[i] == self.SOFT_PITY_START:
                ax1.text(x_values[i], prob + 0.01, f"{prob:.1%}", ha="center", va="bottom", fontsize=9)

        ax2.plot(x_values, cumulative_probs, "o-", linewidth=2, markersize=4, color="darkblue")
        ax2.plot([], [], "o", color="darkblue", label="累积概率")
        for target_prob in [0.25, 0.5, 0.75, 0.9, 0.99]:
            for i, cum_prob in enumerate(cumulative_probs):
                if cum_prob >= target_prob:
                    ax2.axhline(y=target_prob, color="gray", linestyle="--", alpha=0.6)
                    ax2.axvline(x=x_values[i], color="gray", linestyle="--", alpha=0.6)
                    ax2.text(
                        x_values[i] + 0.6,
                        target_prob - 0.03,
                        f"{int(target_prob*100)}% ({x_values[i]}抽)",
                        fontsize=9,
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                    )
                    break
        sample_step = max(1, len(cumulative_probs) // 20)
        for i in range(0, len(cumulative_probs), sample_step):
            xp = x_values[i]
            yp = cumulative_probs[i]
            ax2.plot(xp, yp, "o", color="navy")
            ax2.text(xp, yp + 0.03, f"{yp:.1%}", ha="center", fontsize=8)
            ax2.axvline(x=xp, color="lightgray", linestyle=":", linewidth=0.8)
            ax2.axhline(y=yp, color="lightgray", linestyle=":", linewidth=0.8)
        ax2.legend()
        ax2.set_xlabel("抽数")
        ax2.set_ylabel("累积出金概率")
        ax2.set_title("累积概率曲线")
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(pulls_since_last - 1, min(self.HARD_PITY + 5, pulls_since_last + len(probabilities)))
        ax2.set_ylim(0, 1.05)

        if stats is not None:
            stats_lines = [
                "原神抽卡概率分析报告",
                "",
                f"当前状态: 已连续 {stats['current_pulls']} 抽未出5星",
                f"当前单抽出金概率: {stats['current_prob']:.2%}",
                f"下一抽出金概率: {stats['next_pull_prob']:.2%}",
                f"距离软保底开始还需: {stats['pulls_to_soft_pity']} 抽",
                f"距离硬保底还需: {stats['pulls_to_hard_pity']} 抽",
                f"数学期望(还需): {stats['expected_pulls']:.1f} 抽",
                "",
                "累积概率分析:",
            ]
            for target in [25, 50, 75, 90, 99]:
                key = f"pulls_to_{target}%"
                if key in stats:
                    value = stats[key]
                    if isinstance(value, str):
                        stats_lines.append(f"{target}%: {value}")
                    else:
                        stats_lines.append(f"{target}%: 还需 {value} 抽")
            stats_text = "\n".join(stats_lines)
            ax3.text(
                0.02,
                0.98,
                stats_text,
                transform=ax3.transAxes,
                fontsize=11,
                va="top",
                family="sans-serif",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.95, pad=0.8),
            )
            short_text = "\n".join(stats_lines[:8])
            ax1.text(
                0.02,
                0.95,
                short_text,
                transform=ax1.transAxes,
                fontsize=9,
                va="top",
                family="sans-serif",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, pad=0.6),
            )

        plt.tight_layout()
        plt.close(fig)
        return fig

    def generate_statistics(self, pulls_since_last, probabilities, cumulative_probs):
        x_values = list(range(pulls_since_last, pulls_since_last + len(probabilities)))
        stats = {
            "current_pulls": pulls_since_last,
            "pulls_to_soft_pity": max(0, self.SOFT_PITY_START - pulls_since_last),
            "pulls_to_hard_pity": max(0, self.HARD_PITY - pulls_since_last),
            "current_prob": probabilities[0] if probabilities else 0,
            "next_pull_prob": probabilities[1] if len(probabilities) > 1 else 0,
        }
        for target in [0.25, 0.5, 0.75, 0.9, 0.99]:
            for i, cum_prob in enumerate(cumulative_probs):
                if cum_prob >= target:
                    stats[f"pulls_to_{int(target*100)}%"] = x_values[i] - pulls_since_last
                    break
            else:
                stats[f"pulls_to_{int(target*100)}%"] = ">保底前剩余抽数"
        expected_value = 0
        for i, prob in enumerate(probabilities):
            prob_of_getting_here = 1.0
            for j in range(i):
                prob_of_getting_here *= 1 - probabilities[j]
            expected_value += (i + 1) * prob_of_getting_here * prob
        stats["expected_pulls"] = expected_value
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
            key = f"pulls_to_{target}%"
            if key in stats:
                value = stats[key]
                if isinstance(value, str):
                    print(f"  {target}%概率出金: {value}")
                else:
                    print(
                        f"  {target}%概率出金: 还需 {value} 抽 (总计 {stats['current_pulls'] + value} 抽)"
                    )
        print("=" * 50)


def plot_full_analysis(wish_counts, characters):
    nrows = 6
    per_row_height = 6.5
    fig_height = max(24, per_row_height * nrows)
    fig, axes = plt.subplots(nrows, 1, figsize=(12, fig_height), constrained_layout=True)
    fig.suptitle("原神抽卡数据分析（来自文件）", fontsize=16, fontweight="bold")

    ax1 = axes[0]
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    hist_data, _, patches = ax1.hist(wish_counts, bins=bins, edgecolor="black", alpha=0.7, color="skyblue")
    for count, patch in zip(hist_data, patches):
        ax1.text(patch.get_x() + patch.get_width() / 2, count + 0.5, str(int(count)), ha="center", va="bottom")
    ax1.set_xlabel("抽数区间")
    ax1.set_ylabel("出现次数")
    ax1.set_title("抽数分布直方图")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    sorted_counts = np.sort(wish_counts)
    cumulative_prob = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    ax2.plot(sorted_counts, cumulative_prob * 100, "o-", linewidth=2, markersize=5, color="coral", label="经验累积概率")
    for target_prob in [0.25, 0.5, 0.75, 0.9, 0.99]:
        idx = np.searchsorted(cumulative_prob, target_prob)
        if idx < len(sorted_counts):
            xk = sorted_counts[idx]
            yk = cumulative_prob[idx] * 100
            ax2.axhline(y=target_prob * 100, color="gray", linestyle="--", alpha=0.6)
            ax2.axvline(x=xk, color="gray", linestyle="--", alpha=0.6)
            ax2.text(xk + 1, yk - 6, f"{int(target_prob*100)}% ({xk}抽)", fontsize=9, bbox=dict(facecolor="white", alpha=0.8))
    ax2.set_xlabel("抽数")
    ax2.set_ylabel("累积概率 (%)")
    ax2.set_title("累积概率分布")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 90)
    ax2.set_ylim(0, 105)
    ax2.legend()

    ax3 = axes[2]
    standard_count = sum(1 for c in characters if c in STANDARD_CHARS)
    limited_count = len(characters) - standard_count
    ax3.pie(
        [limited_count, standard_count],
        explode=(0.05, 0),
        labels=["限定角色", "常驻角色"],
        colors=["lightgreen", "lightcoral"],
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
    )
    ax3.set_title("角色类型分布")

    ax4 = axes[3]
    counts = np.array(wish_counts)
    n = len(counts)
    x_idx = np.arange(1, n + 1)
    ax4.scatter(x_idx, counts, color="lightgray", s=30, label="每次5星抽数")
    cum_avg = [np.mean(counts[:i]) for i in range(1, n + 1)]
    ax4.plot(x_idx, cum_avg, "-o", color="purple", linewidth=2, markersize=6, label="累计平均")
    window = 10
    if n >= 2:
        mov_avg = [np.mean(counts[max(0, i - window) : i]) for i in range(1, n + 1)]
        ax4.plot(x_idx, mov_avg, "-s", color="tab:blue", linewidth=1.5, markersize=4, alpha=0.9, label=f"{window} 次滑动平均")
    final_cum = cum_avg[-1] if cum_avg else np.mean(counts)
    ax4.axhline(y=final_cum, color="red", linestyle="--", alpha=0.9, label=f"最终累计平均: {final_cum:.1f}")
    ax4.axhline(y=np.mean(counts), color="orange", linestyle="-.", alpha=0.7, label=f"总体平均: {np.mean(counts):.1f}")
    ax4.axhline(y=62.5, color="green", linestyle=":", alpha=0.7, label="理论期望: 62.5")
    ax4.set_xlabel("第 x 个 5星")
    ax4.set_ylabel("抽数")
    ax4.set_title("每个5星抽数与累计平均走势")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    ax5 = axes[4]
    char_counts = {}
    for char in characters:
        char_counts[char] = char_counts.get(char, 0) + 1
    top_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    char_names = [c for c, _ in top_chars]
    char_freq = [c for _, c in top_chars]
    colors_bar = ["gold" if c in STANDARD_CHARS else "lightblue" for c in char_names]
    bars = ax5.barh(range(len(char_names)), char_freq, color=colors_bar, edgecolor="black")
    ax5.set_yticks(range(len(char_names)))
    ax5.set_yticklabels(char_names, fontsize=10)
    ax5.invert_yaxis()
    ax5.set_xlabel("出现次数")
    ax5.set_title("角色出现频次 Top 20")
    for bar, count in zip(bars, char_freq):
        ax5.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, str(count), va="center", fontsize=10)

    ax6 = axes[5]
    ax6.axis("off")
    mean_count = np.mean(wish_counts)
    median_count = np.median(wish_counts)
    std_count = np.std(wish_counts)
    table_data = [
        ["指标", "你的数据", "理论参考"],
        ["平均抽数", f"{mean_count:.1f}", "~62.5"],
        ["中位数抽数", f"{median_count:.1f}", "~75"],
        ["标准差", f"{std_count:.1f}", "~22"],
        ["最早出金", f"{min(wish_counts)}抽", "1抽"],
        ["最晚出金", f"{max(wish_counts)}抽", "90抽"],
        ["软保底率(≥74)", f"{sum(1 for c in wish_counts if c >= 74)/len(wish_counts)*100:.1f}%", "~50%"],
        ["20抽内出金", f"{sum(1 for c in wish_counts if c <= 20)/len(wish_counts)*100:.1f}%", "~15%"],
        ["总抽数", f"{sum(wish_counts)}", "-"],
        ["5星总数", f"{len(wish_counts)}", "-"],
    ]
    table = ax6.table(cellText=table_data, cellLoc="center", loc="center", colWidths=[0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    ax6.set_title("关键统计指标", fontsize=14, y=0.95)

    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_merged_analysis(pulls, probabilities, cumulative_probs, stats, wish_counts=None, characters=None, file_info=None):
    has_full = wish_counts is not None and characters is not None and len(wish_counts) > 0
    nrows = 2 + (6 if has_full else 1)
    per_row_height = 6.5
    fig_height = max(14, per_row_height * nrows)
    fig, axes = plt.subplots(nrows, 1, figsize=(14, fig_height), constrained_layout=True)

    ax_prob_top = axes[0]
    ax_prob_bot = axes[1]
    x_values = list(range(pulls, pulls + len(probabilities)))
    bars = ax_prob_top.bar(x_values, probabilities, color="skyblue", edgecolor="black", linewidth=0.5)
    for i, (x, prob) in enumerate(zip(x_values, probabilities)):
        if x >= 74 and prob > 0.006 and x < 90:
            bars[i].set_color("lightcoral")
        elif x == 90:
            bars[i].set_color("gold")
    ax_prob_top.axvline(x=pulls, color="green", linestyle="--", alpha=0.7, label=f"当前: {pulls}抽")
    ax_prob_top.axvline(x=74, color="orange", linestyle="--", alpha=0.7, label="软保底开始: 74抽")
    ax_prob_top.axvline(x=90, color="red", linestyle="--", alpha=0.7, label="硬保底: 90抽")
    ax_prob_top.set_title(f"单抽概率 (已连续 {pulls} 抽未出5星)")
    ax_prob_top.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_prob_top.grid(True, alpha=0.3)
    ax_prob_top.legend()

    ax_prob_bot.plot(x_values, cumulative_probs, "o-", linewidth=2, markersize=4, color="darkblue", label="累积概率")
    for target_prob in [0.25, 0.5, 0.75, 0.9, 0.99]:
        for i, cum_prob in enumerate(cumulative_probs):
            if cum_prob >= target_prob:
                ax_prob_bot.axhline(y=target_prob, color="gray", linestyle="--", alpha=0.6)
                ax_prob_bot.axvline(x=x_values[i], color="gray", linestyle="--", alpha=0.6)
                ax_prob_bot.text(
                    x_values[i] + 0.6,
                    target_prob - 0.03,
                    f"{int(target_prob*100)}% ({x_values[i]}抽)",
                    fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )
                break
    ax_prob_bot.set_title("累积出金概率")
    ax_prob_bot.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_prob_bot.grid(True, alpha=0.3)
    ax_prob_bot.legend()

    if stats is not None:
        stats_lines = [
            "原神抽卡概率分析报告",
            "",
            f"当前状态: 已连续 {stats['current_pulls']} 抽未出5星",
            f"当前单抽出金概率: {stats['current_prob']:.2%}",
            f"下一抽出金概率: {stats['next_pull_prob']:.2%}",
            f"距离软保底开始还需: {stats['pulls_to_soft_pity']} 抽",
            f"距离硬保底还需: {stats['pulls_to_hard_pity']} 抽",
            f"数学期望(还需): {stats['expected_pulls']:.1f} 抽",
        ]
        ax_prob_top.text(
            0.02,
            0.95,
            "\n".join(stats_lines),
            transform=ax_prob_top.transAxes,
            fontsize=9,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, pad=0.6),
        )

    if has_full:
        ax1 = axes[2]
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        hist_data, _, patches = ax1.hist(wish_counts, bins=bins, edgecolor="black", alpha=0.7, color="skyblue")
        for count, patch in zip(hist_data, patches):
            ax1.text(patch.get_x() + patch.get_width() / 2, count + 0.5, str(int(count)), ha="center", va="bottom")
        ax1.set_title("抽数分布直方图")

        ax2 = axes[3]
        sorted_counts = np.sort(wish_counts)
        cumulative_prob = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
        ax2.plot(sorted_counts, cumulative_prob * 100, "o-", color="coral", label="经验累积概率")
        ax2.set_title("累积概率分布")
        ax2.legend()

        ax3 = axes[4]
        standard_count = sum(1 for c in characters if c in STANDARD_CHARS)
        limited_count = len(characters) - standard_count
        ax3.pie([limited_count, standard_count], labels=["限定", "常驻"], autopct="%1.1f%%")
        ax3.set_title("角色类型分布")

        ax4 = axes[5]
        counts = np.array(wish_counts)
        n = len(counts)
        x_idx = np.arange(1, n + 1)
        ax4.scatter(x_idx, counts, color="lightgray", s=30, label="每次5星抽数")
        cum_avg = [np.mean(counts[:i]) for i in range(1, n + 1)]
        ax4.plot(x_idx, cum_avg, "-o", color="purple", linewidth=2, markersize=6, label="累计平均")
        ax4.axhline(y=62.5, color="green", linestyle=":", alpha=0.7, label="理论期望: 62.5")
        ax4.set_title("每个5星抽数与累计平均走势")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        ax5 = axes[6]
        char_counts = {}
        for ch in characters:
            char_counts[ch] = char_counts.get(ch, 0) + 1
        top = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        names = [t[0] for t in top]
        freqs = [t[1] for t in top]
        bars = ax5.barh(range(len(names)), freqs, color="lightblue", edgecolor="black")
        ax5.set_yticks(range(len(names)))
        ax5.set_yticklabels(names)
        ax5.invert_yaxis()
        ax5.set_title("角色出现频次 Top20")
        for bar, count in zip(bars, freqs):
            ax5.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, str(count), va="center", fontsize=9)

        ax6 = axes[7]
        ax6.axis("off")
        table_data = [
            ["指标", "你的数据", "参考"],
            ["平均抽数", f"{np.mean(wish_counts):.1f}", "~62.5"],
            ["中位数", f"{np.median(wish_counts):.1f}", "~75"],
            ["标准差", f"{np.std(wish_counts):.1f}", "~22"],
            ["最早出金", f"{min(wish_counts)}抽", "1抽"],
            ["最晚出金", f"{max(wish_counts)}抽", "90抽"],
        ]
        table = ax6.table(cellText=table_data, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        if file_info:
            fig.text(0.5, 0.01, f"数据来源: {file_info}", fontsize=9, ha="center")
    else:
        ax_note = axes[2]
        ax_note.axis("off")
        note = "未检测到完整抽卡记录。\n仅显示概率分析。\n请先同步线上记录或放入本地 xlsx。"
        ax_note.text(0.5, 0.5, note, ha="center", va="center", fontsize=12, bbox=dict(facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.close(fig)
    return fig


def save_fig_temp(fig, dpi=150) -> str:
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, f"wish_tmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def display_image_in_scroll_window(image_path):
    if Image is None or ImageTk is None:
        raise ImportError("Pillow 未安装，请运行: pip install pillow")

    win = tk.Toplevel()
    win.title(os.path.basename(image_path))
    img = Image.open(image_path)
    width, height = img.size
    screen_w = win.winfo_screenwidth()
    max_display_w = int(screen_w * 0.92)
    if width > max_display_w:
        ratio = max_display_w / width
        disp_w = max_display_w
        disp_h = int(height * ratio)
        display_img = img.resize((disp_w, disp_h), Image.LANCZOS)
    else:
        display_img = img.copy()
        disp_w, disp_h = display_img.size

    canvas = tk.Canvas(win, width=min(disp_w, max_display_w), height=min(800, disp_h))
    vbar = ttk.Scrollbar(win, orient=tk.VERTICAL, command=canvas.yview)
    canvas.configure(yscrollcommand=vbar.set)
    vbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
    photo = ImageTk.PhotoImage(display_img)
    canvas.create_image(0, 0, anchor="nw", image=photo)
    canvas.image = photo
    canvas.config(scrollregion=(0, 0, disp_w, disp_h))

    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def on_close():
        try:
            canvas.image = None
        except Exception:
            pass
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", on_close)
