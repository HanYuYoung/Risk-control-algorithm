import numpy as np
import matplotlib.pyplot as plt
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==============================
# 1. 数据（10个样本，1个连续特征，二分类）
# ==============================
X = np.array([2.1, 3.5, 1.8, 7.2, 4.9, 2.7, 6.3, 3.1, 8.0, 5.5])
y = np.array([    0,    0,    0,    1,    0,    0,    1,    0,    1,    1])  # 标签

print("原始数据 X:", X)
print("标签 y:    ", y)
print("-" * 50)

# ==============================
# 2. 构建直方图（LightGBM 风格）
# ==============================
num_bins = 4  # 桶数量（实际 LightGBM 自动决定，这里模拟 4 个）

# 使用 np.histogram 自动分箱 + 统计
hist_count, bin_edges = np.histogram(X, bins=num_bins, range=(X.min()-0.1, X.max()+0.1))
print(f"直方图桶边界: {bin_edges}")
print(f"每个桶样本数: {hist_count}")

# 手动统计每个桶里 0 和 1 的数量
bin_labels_0 = []
bin_labels_1 = []
bin_centers = []

for i in range(num_bins):
    left = bin_edges[i]
    right = bin_edges[i+1]
    mask = (X >= left) & (X < right)
    samples_in_bin = X[mask]
    labels_in_bin = y[mask]
    
    count_0 = np.sum(labels_in_bin == 0)
    count_1 = np.sum(labels_in_bin == 1)
    
    bin_labels_0.append(count_0)
    bin_labels_1.append(count_1)
    bin_centers.append((left + right) / 2)
    
    print(f"桶 {i}: [{left:.1f}, {right:.1f}) → {len(samples_in_bin)} 个样本 → 0类: {count_0}, 1类: {count_1}")

print("-" * 50)

# ==============================
# 3. 打印「直方图表」（LightGBM 内部长这样）
# ==============================
print("LightGBM 眼里的「直方图」表格：")
print("桶号 | 范围       | 样本数 | 0类 | 1类")
print("-" * 35)
for i in range(num_bins):
    left = bin_edges[i]
    right = bin_edges[i+1]
    total = bin_labels_0[i] + bin_labels_1[i]
    print(f"{i:>2}   | {left:>4.1f}~{right:<4.1f} | {total:>4}   | {bin_labels_0[i]:>2}  | {bin_labels_1[i]:>2}")

print("-" * 50)

# ==============================
# 4. 用 plt 画出「直方图可视化」
# ==============================
fig, ax = plt.subplots(figsize=(10, 6))

# 底部堆积：先画 0 类（深蓝）
bottom = np.array([0] * num_bins)
bars0 = ax.bar(bin_centers, bin_labels_0, width=(bin_edges[1]-bin_edges[0])*0.8,
               label='Class 0', color='navy', edgecolor='black')

# 再堆积 1 类（橙色）
bars1 = ax.bar(bin_centers, bin_labels_1, bottom=bin_labels_0, width=(bin_edges[1]-bin_edges[0])*0.8,
               label='Class 1', color='orange', edgecolor='black')

# 在柱子上标数字
for i, (b0, b1) in enumerate(zip(bin_labels_0, bin_labels_1)):
    total_height = b0 + b1
    if b0 > 0:
        ax.text(bin_centers[i], b0/2, str(b0), ha='center', va='center', color='white', fontweight='bold')
    if b1 > 0:
        ax.text(bin_centers[i], b0 + b1/2, str(b1), ha='center', va='center', color='white', fontweight='bold')
    ax.text(bin_centers[i], total_height + 0.1, f"Total:{total_height}", ha='center', va='bottom', fontsize=9)

ax.set_xlabel("Feature x1 value")
ax.set_ylabel("Number of samples in bin")
ax.set_title("LightGBM Histogram (Binning + Label Count)", fontsize=14, fontweight='bold')
ax.set_xticks(bin_edges)
ax.set_xticklabels([f"{e:.1f}" for e in bin_edges], rotation=45)
ax.legend()
ax.grid(True, axis='y', alpha=0.3)

# 显示桶边界线
for edge in bin_edges[1:-1]:
    ax.axvline(edge, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# ==============================
# 5. 模拟「只试桶边界分裂」
# ==============================
print("LightGBM 只会尝试这 3 个分裂点（桶边界）：")
for i in range(1, num_bins):
    left_bins = sum(hist_count[:i])
    right_bins = sum(hist_count[i:])
    print(f"  分裂 {i}: 左 = 前 {i} 个桶 ({left_bins} 个样本) | 右 = 后 {num_bins-i} 个桶 ({right_bins} 个样本)")