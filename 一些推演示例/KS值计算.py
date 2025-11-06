# -*- coding: utf-8 -*-
"""
KS 值手撕脚本（带详细步骤输出）
作者：面试专用
功能：一步步打印阈值、累计比例、KS 变化过程
"""
import matplotlib.pyplot as plt
import numpy as np

# ==================== 示例数据 ====================
pos_scores = [0.9, 0.8, 0.7, 0.85]   # 正样本得分（4个）
neg_scores = [0.3, 0.4, 0.2, 0.35]  # 负样本得分（4个）

print("=== 输入数据 ===")
print(f"正样本得分: {pos_scores}")
print(f"负样本得分: {neg_scores}")
print()

# ==================== Step 1: 合并 & 排序去重 ====================
all_scores = pos_scores + neg_scores
thresholds = sorted(set(all_scores))

print("=== Step 1: 合并所有得分并排序去重 ===")
print(f"合并后: {all_scores}")
print(f"去重排序后阈值 candidates: {thresholds}")
print()

# ==================== Step 2: 初始化统计量 ====================
KS_max = 0
n_pos = len(pos_scores)
n_neg = len(neg_scores)

print("=== Step 2: 初始化 ===")
print(f"正样本总数 n_pos = {n_pos}")
print(f"负样本总数 n_neg = {n_neg}")
print(f"初始 KS_max = {KS_max}")
print()

# ==================== Step 3: 遍历每个阈值 t ====================
print("=== Step 3: 遍历每个阈值，计算累计分布和 KS ===")
print("-" * 80)
print(f"{'阈值 t':<8} {'≤t 正样本数':<12} {'F1(t)':<8} {'≤t 负样本数':<12} {'F0(t)':<8} {'|F1-F0|':<8} {'KS_max 更新'}")
print("-" * 80)

# 用于存储可视化数据
plot_data = {
    'thresholds': [],
    'F1': [],
    'F0': [],
    'KS': []
}

for i, t in enumerate(thresholds):
    # 计算正样本中 ≤ t 的数量
    cum_pos_count = sum(1 for s in pos_scores if s <= t)
    cum_pos = cum_pos_count / n_pos
    
    # 计算负样本中 ≤ t 的数量
    cum_neg_count = sum(1 for s in neg_scores if s <= t)
    cum_neg = cum_neg_count / n_neg
    
    # 当前 KS 值
    ks_current = abs(cum_pos - cum_neg)
    
    # 更新最大 KS
    old_ks_max = KS_max
    KS_max = max(KS_max, ks_current)
    
    # 存储数据用于可视化
    plot_data['thresholds'].append(t)
    plot_data['F1'].append(cum_pos)
    plot_data['F0'].append(cum_neg)
    plot_data['KS'].append(ks_current)
    
    # 打印本轮详细计算过程
    updated = "← 更新!" if KS_max > old_ks_max else ""
    print(f"{t:<8.2f} {cum_pos_count:<12} {cum_pos:<8.3f} {cum_neg_count:<12} {cum_neg:<8.3f} "
          f"{ks_current:<8.3f} {KS_max:<.3f} {updated}")

print("-" * 80)

# ==================== Step 4: 输出最终结果 ====================
best_t = None
for t in thresholds:
    cum_pos_count = sum(1 for s in pos_scores if s <= t)
    cum_pos = cum_pos_count / n_pos
    cum_neg_count = sum(1 for s in neg_scores if s <= t)
    cum_neg = cum_neg_count / n_neg
    if abs(cum_pos - cum_neg) == KS_max:
        best_t = t
        break

print()
print("=== 最终结果 ===")
print(f"最大 KS 值: {KS_max:.4f}")
print(f"达到最大 KS 的最佳阈值: {best_t}")
print(f"此时正样本累计比例 F1 = {cum_pos:.3f}")
print(f"此时负样本累计比例 F0 = {cum_neg:.3f}")
print()

# ==================== 可视化：绘制 CDF 曲线 ====================
print("=== 可视化 ===")
print("正在绘制 CDF 曲线...")

plt.figure(figsize=(10, 6))
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 绘制两条 CDF 曲线
plt.plot(plot_data['thresholds'], plot_data['F1'], 'o-', label='正样本累计分布 F1(t)', 
         linewidth=2, markersize=8, color='#2E86AB')
plt.plot(plot_data['thresholds'], plot_data['F0'], 's-', label='负样本累计分布 F0(t)', 
         linewidth=2, markersize=8, color='#A23B72')

# 找到 KS 最大值对应的阈值和累计比例
ks_max_idx = np.argmax(plot_data['KS'])
best_t_plot = plot_data['thresholds'][ks_max_idx]
best_F1 = plot_data['F1'][ks_max_idx]
best_F0 = plot_data['F0'][ks_max_idx]

# 标注 KS 最大值位置
plt.plot([best_t_plot, best_t_plot], [best_F0, best_F1], 
         'r--', linewidth=2, label=f'KS 最大值 = {KS_max:.4f}')
plt.plot(best_t_plot, best_F1, 'ro', markersize=12, markerfacecolor='none', 
         markeredgewidth=2, label=f'最佳阈值 = {best_t_plot:.2f}')
plt.plot(best_t_plot, best_F0, 'ro', markersize=12, markerfacecolor='none', 
         markeredgewidth=2)

# 添加垂直距离标注
mid_y = (best_F1 + best_F0) / 2
plt.annotate(f'KS = {KS_max:.4f}', 
             xy=(best_t_plot, mid_y), 
             xytext=(best_t_plot + 0.1, mid_y),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=12, color='red', weight='bold')

# 设置图表属性
plt.xlabel('得分阈值', fontsize=12)
plt.ylabel('累计比例', fontsize=12)
plt.title('KS 统计量可视化：两条 CDF 曲线', fontsize=14, weight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='best', fontsize=10)
plt.ylim(0, 1.05)

# 添加辅助线
plt.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
plt.axhline(y=1, color='k', linewidth=0.5, alpha=0.3)

plt.tight_layout()
plt.savefig('KS_可视化.png', dpi=300, bbox_inches='tight')
print("图表已保存为 'KS_可视化.png'")
plt.show()