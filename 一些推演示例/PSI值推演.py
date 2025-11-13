# -*- coding: utf-8 -*-
"""
PSI 值手撕脚本（带详细推演和图形输出）
作者：面试专用
功能：一步步展示 PSI 指标的构造、计算与解读
"""

import math
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def print_divider(title: str) -> None:
    print("\n" + "=" * 30)
    print(title)
    print("=" * 30)


def make_demo_samples(seed: int = 2025) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    baseline = rng.normal(loc=650, scale=45, size=1200)
    shifted = rng.normal(loc=620, scale=55, size=1200)
    baseline = np.clip(baseline, 450, 900)
    shifted = np.clip(shifted, 450, 900)
    return baseline, shifted


def build_bin_edges(data: np.ndarray, n_bins: int) -> np.ndarray:
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(data, quantiles)
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    return np.unique(edges)


def calculate_bin_distribution(data: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    counts, _ = np.histogram(data, bins=edges)
    proportions = counts / counts.sum()
    return counts, proportions


def calculate_psi_row(expected_pct: float, actual_pct: float, epsilon: float = 1e-6) -> float:
    safe_expected = max(expected_pct, epsilon)
    safe_actual = max(actual_pct, epsilon)
    return (safe_actual - safe_expected) * math.log(safe_actual / safe_expected)


def assemble_psi_table(
    baseline_counts: np.ndarray,
    baseline_pct: np.ndarray,
    shifted_counts: np.ndarray,
    shifted_pct: np.ndarray,
    edges: np.ndarray,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for i in range(len(baseline_counts)):
        psi_value = calculate_psi_row(baseline_pct[i], shifted_pct[i])
        rows.append(
            {
                "分箱序号": i + 1,
                "区间下界": round(edges[i], 2),
                "区间上界": round(edges[i + 1], 2),
                "基线样本数": baseline_counts[i],
                "基线占比": round(baseline_pct[i], 4),
                "对比样本数": shifted_counts[i],
                "对比占比": round(shifted_pct[i], 4),
                "单箱PSI": round(psi_value, 4),
            }
        )
    return pd.DataFrame(rows)


def interpret_psi(total_psi: float) -> str:
    if total_psi < 0.1:
        return "PSI < 0.1：特征分布基本稳定，模型可继续使用。"
    if total_psi < 0.25:
        return "0.1 ≤ PSI < 0.25：分布出现轻微漂移，建议密切关注或局部调参。"
    return "PSI ≥ 0.25：分布明显漂移，应排查数据或考虑重新训练模型。"


if __name__ == "__main__":
    print_divider("Step 1. 准备基线样本与对比样本")
    baseline_sample, shifted_sample = make_demo_samples()
    print(f"基线样本数量: {baseline_sample.size}, 均值: {baseline_sample.mean():.2f}, 标准差: {baseline_sample.std():.2f}")
    print(f"对比样本数量: {shifted_sample.size}, 均值: {shifted_sample.mean():.2f}, 标准差: {shifted_sample.std():.2f}")

    bin_options = [5, 8, 10]
    psi_results: Dict[int, Dict[str, pd.DataFrame]] = {}

    for idx, n_bins in enumerate(bin_options, start=1):
        print_divider(f"Step 2.{idx} 分箱数量 {n_bins}")
        bin_edges = build_bin_edges(baseline_sample, n_bins=n_bins)
        print(f"分箱边界: {np.round(bin_edges, 2)}")
        baseline_counts, baseline_pct = calculate_bin_distribution(baseline_sample, bin_edges)
        shifted_counts, shifted_pct = calculate_bin_distribution(shifted_sample, bin_edges)
        psi_table = assemble_psi_table(baseline_counts, baseline_pct, shifted_counts, shifted_pct, bin_edges)
        print(psi_table)

        total_psi = psi_table["单箱PSI"].sum()
        psi_results[n_bins] = {"table": psi_table, "total": total_psi}
        print(f"总 PSI = {total_psi:.4f}")
        print(interpret_psi(total_psi))

    print_divider("Step 3. 可视化不同分箱 PSI 贡献")
    fig, axes = plt.subplots(1, len(bin_options), figsize=(5 * len(bin_options), 5), sharey=True)
    if len(bin_options) == 1:
        axes = [axes]

    for ax, n_bins in zip(axes, bin_options):
        plot_table = psi_results[n_bins]["table"]
        positions = np.arange(len(plot_table))
        colors = ["#2E86AB" if value <= 0 else "#D35400" for value in plot_table["单箱PSI"]]
        ax.bar(positions, plot_table["单箱PSI"], color=colors)
        ax.axhline(y=0, color="#333333", linestyle="--", linewidth=1)
        ax.set_title(f"{n_bins} 分箱", fontsize=14, fontweight="bold")
        ax.set_xlabel("分箱序号", fontsize=11)
        ax.set_xticks(positions)
        ax.set_xticklabels(plot_table["分箱序号"].astype(str))
        if ax is axes[0]:
            ax.set_ylabel("单箱 PSI 值", fontsize=11)
        for pos, val in zip(positions, plot_table["单箱PSI"]):
            offset = 0.002 * np.sign(val) if val != 0 else 0.002
            ax.text(pos, val + offset, f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=9)

    fig.tight_layout()
    fig.savefig("PSI_可视化.png", dpi=300, bbox_inches="tight")
    print("图表已保存为 'PSI_可视化.png'")
    plt.show()
