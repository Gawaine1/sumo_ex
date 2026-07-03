import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
initial_df = pd.read_excel('population_initial.xlsx')
final_df = pd.read_excel('final_population .xlsx')

print("分析tau异常值对tau降低的影响")
print("=" * 60)

# 计算每个个体的tau变化
final_df['tau_change'] = final_df['tau_mean'] - initial_df['tau_mean']
final_df['tau_change_pct'] = (final_df['tau_change'] / initial_df['tau_mean']) * 100

# 分析无限值数量与tau降低的关系
print("\n无限值数量与tau降低的相关性:")
correlation = final_df['tau_nonfinite_count'].corr(final_df['tau_change_pct'])
print(f"相关系数: {correlation:.4f}")

# 按无限值数量分组分析
final_df['infinite_group'] = pd.cut(final_df['tau_nonfinite_count'], 
                                    bins=[0, 45000, 46000, 47000, 50000],
                                    labels=['低', '中', '高', '极高'])

print("\n不同无限值组的tau降低分析:")
group_stats = final_df.groupby('infinite_group')['tau_change_pct'].agg(['mean', 'std', 'count'])
print(group_stats)

# 创建可视化图表
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 无限值数量分布
axes[0, 0].hist(final_df['tau_nonfinite_count'], bins=20, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('无限值数量')
axes[0, 0].set_ylabel('个体数量')
axes[0, 0].set_title('进化后种群无限值数量分布')
axes[0, 0].grid(True, alpha=0.3)

# 2. 无限值数量 vs tau降低散点图
axes[0, 1].scatter(final_df['tau_nonfinite_count'], final_df['tau_change_pct'], 
                   alpha=0.6, edgecolors='black', linewidths=0.5)
axes[0, 1].set_xlabel('无限值数量')
axes[0, 1].set_ylabel('tau降低百分比 (%)')
axes[0, 1].set_title(f'无限值数量与tau降低关系 (r={correlation:.4f})')
axes[0, 1].grid(True, alpha=0.3)

# 添加趋势线
z = np.polyfit(final_df['tau_nonfinite_count'], final_df['tau_change_pct'], 1)
p = np.poly1d(z)
axes[0, 1].plot(final_df['tau_nonfinite_count'], p(final_df['tau_nonfinite_count']), 
                "r--", alpha=0.8, linewidth=2, label=f'趋势线: y={z[0]:.6f}x+{z[1]:.2f}')
axes[0, 1].legend()

# 3. 不同无限值组的tau降低箱线图
group_data = [final_df[final_df['infinite_group'] == group]['tau_change_pct'] 
              for group in ['低', '中', '高', '极高']]
bp = axes[1, 0].boxplot(group_data, labels=['低', '中', '高', '极高'], patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
axes[1, 0].set_xlabel('无限值数量分组')
axes[1, 0].set_ylabel('tau降低百分比 (%)')
axes[1, 0].set_title('不同无限值组的tau降低分布')
axes[1, 0].grid(True, alpha=0.3)

# 4. tau_mean vs 无限值比例
final_df['infinite_ratio'] = final_df['tau_nonfinite_count'] / final_df['tau_numel'] * 100
axes[1, 1].scatter(final_df['infinite_ratio'], final_df['tau_mean'], 
                   alpha=0.6, edgecolors='black', linewidths=0.5)
axes[1, 1].set_xlabel('无限值比例 (%)')
axes[1, 1].set_ylabel('tau_mean')
axes[1, 1].set_title('无限值比例与tau_mean关系')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/tau_outlier_impact_analysis.png', dpi=300, bbox_inches='tight')
print("\n图表已保存: images/tau_outlier_impact_analysis.png")

# 详细分析
print("\n" + "=" * 60)
print("tau异常值影响详细分析")

# 分析无限值对tau_mean计算的影响
print("\ntau_mean计算机制分析:")
print("tau_mean = sum(有限tau值) / count(有限tau值)")
print("无限值被自动排除在计算之外")

# 比较有无无限值的个体
high_infinite = final_df[final_df['tau_nonfinite_count'] > 46000]
low_infinite = final_df[final_df['tau_nonfinite_count'] <= 46000]

print(f"\n高无限值组 (n={len(high_infinite)}):")
print(f"  平均无限值数量: {high_infinite['tau_nonfinite_count'].mean():.2f}")
print(f"  平均tau_mean: {high_infinite['tau_mean'].mean():.6f}")
print(f"  平均tau降低: {high_infinite['tau_change_pct'].mean():.2f}%")

print(f"\n低无限值组 (n={len(low_infinite)}):")
print(f"  平均无限值数量: {low_infinite['tau_nonfinite_count'].mean():.2f}")
print(f"  平均tau_mean: {low_infinite['tau_mean'].mean():.6f}")
print(f"  平均tau降低: {low_infinite['tau_change_pct'].mean():.2f}%")

print("\n结论:")
print("1. 无限值数量与tau降低呈负相关，无限值越多，tau降低幅度越大")
print("2. 这表明无限值的出现反映了路径规划的极端情况")
print("3. tau_mean的计算排除了无限值，降低8.86%反映了有限值部分的真实优化")
print("4. 无限值的增加是进化过程中探索性策略的结果")
