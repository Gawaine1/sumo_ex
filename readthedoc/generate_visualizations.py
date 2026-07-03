import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import os

# 创建images目录
os.makedirs('images', exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
initial_df = pd.read_excel('population_initial.xlsx')
final_df = pd.read_excel('final_population .xlsx')

print("生成可视化图表...")

# 1. 相关性热力图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 初始种群相关性
initial_corr = initial_df[['rho', 'tau_mean', 'rewards_mean']].corr()
sns.heatmap(initial_corr, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=axes[0])
axes[0].set_title('初始种群参数相关性', fontsize=14, fontweight='bold')

# 进化后种群相关性
final_corr = final_df[['rho', 'tau_mean', 'rewards_mean']].corr()
sns.heatmap(final_corr, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=axes[1])
axes[1].set_title('进化后种群参数相关性', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('images/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("[OK] 相关性热力图已生成")

# 2. 个体变化热力图
comparison_df = pd.DataFrame({
    'individual_id': range(100),
    'rho_change_pct': ((final_df['rho'].values - initial_df['rho'].values) / initial_df['rho'].values * 100),
    'tau_change_pct': ((final_df['tau_mean'].values - initial_df['tau_mean'].values) / initial_df['tau_mean'].values * 100),
    'rewards_change_pct': ((final_df['rewards_mean'].values - initial_df['rewards_mean'].values) / initial_df['rewards_mean'].values * 100)
})

fig, ax = plt.subplots(figsize=(12, 8))
change_data = comparison_df[['rho_change_pct', 'tau_change_pct', 'rewards_change_pct']].T
sns.heatmap(change_data, cmap='RdYlGn', center=0, 
            cbar_kws={'label': '变化百分比 (%)'}, ax=ax)
ax.set_xlabel('个体ID', fontsize=12)
ax.set_ylabel('参数', fontsize=12)
ax.set_title('个体参数变化热力图', fontsize=14, fontweight='bold')
ax.set_yticklabels(['rho变化%', 'tau变化%', 'rewards变化%'])

plt.tight_layout()
plt.savefig('images/individual_change_heatmap.png', dpi=300, bbox_inches='tight')
print("[OK] 个体变化热力图已生成")

# 3. 聚类可视化
features = final_df[['rho', 'tau_mean', 'rewards_mean']].values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(features_scaled)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
cluster_names = ['高优化组', '中等优化组', '稳定优化组']

for i in range(3):
    mask = clusters == i
    ax.scatter(features_scaled[mask, 0], features_scaled[mask, 1], features_scaled[mask, 2],
               c=colors[i], label=cluster_names[i], s=50, alpha=0.7)

ax.set_xlabel('rho (标准化)', fontsize=12)
ax.set_ylabel('tau_mean (标准化)', fontsize=12)
ax.set_zlabel('rewards_mean (标准化)', fontsize=12)
ax.set_title('个体聚类结果 (3D空间)', fontsize=14, fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig('images/clustering_visualization.png', dpi=300, bbox_inches='tight')
print("[OK] 聚类可视化已生成")

# 4. 异常个体分析
comparison_df['cluster'] = clusters
# 计算z-score
comparison_df['rho_z'] = np.abs(stats.zscore(comparison_df['rho_change_pct']))
comparison_df['tau_z'] = np.abs(stats.zscore(comparison_df['tau_change_pct']))
comparison_df['rewards_z'] = np.abs(stats.zscore(comparison_df['rewards_change_pct']))
comparison_df['is_outlier'] = (comparison_df['rho_z'] > 2) | (comparison_df['tau_z'] > 2) | (comparison_df['rewards_z'] > 2)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# rho变化分布
axes[0, 0].hist(comparison_df['rho_change_pct'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].axvline(comparison_df['rho_change_pct'].mean(), color='red', linestyle='--', linewidth=2, label='均值')
axes[0, 0].axvline(comparison_df['rho_change_pct'].median(), color='green', linestyle='--', linewidth=2, label='中位数')
axes[0, 0].set_xlabel('rho变化百分比 (%)', fontsize=11)
axes[0, 0].set_ylabel('频数', fontsize=11)
axes[0, 0].set_title('rho变化分布', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# tau变化分布
axes[0, 1].hist(comparison_df['tau_change_pct'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
axes[0, 1].axvline(comparison_df['tau_change_pct'].mean(), color='red', linestyle='--', linewidth=2, label='均值')
axes[0, 1].axvline(comparison_df['tau_change_pct'].median(), color='green', linestyle='--', linewidth=2, label='中位数')
axes[0, 1].set_xlabel('tau变化百分比 (%)', fontsize=11)
axes[0, 1].set_ylabel('频数', fontsize=11)
axes[0, 1].set_title('tau变化分布', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# rewards变化分布
axes[1, 0].hist(comparison_df['rewards_change_pct'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
axes[1, 0].axvline(comparison_df['rewards_change_pct'].mean(), color='red', linestyle='--', linewidth=2, label='均值')
axes[1, 0].axvline(comparison_df['rewards_change_pct'].median(), color='green', linestyle='--', linewidth=2, label='中位数')
axes[1, 0].set_xlabel('rewards变化百分比 (%)', fontsize=11)
axes[1, 0].set_ylabel('频数', fontsize=11)
axes[1, 0].set_title('rewards变化分布', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 异常个体散点图
normal_mask = ~comparison_df['is_outlier']
outlier_mask = comparison_df['is_outlier']
axes[1, 1].scatter(comparison_df.loc[normal_mask, 'rho_change_pct'], 
                   comparison_df.loc[normal_mask, 'rewards_change_pct'],
                   c='blue', alpha=0.6, label='正常个体', s=50)
axes[1, 1].scatter(comparison_df.loc[outlier_mask, 'rho_change_pct'],
                   comparison_df.loc[outlier_mask, 'rewards_change_pct'],
                   c='red', alpha=0.8, label='异常个体', s=80, marker='^')
axes[1, 1].set_xlabel('rho变化百分比 (%)', fontsize=11)
axes[1, 1].set_ylabel('rewards变化百分比 (%)', fontsize=11)
axes[1, 1].set_title('异常个体识别 (rho vs rewards)', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)
axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('images/outlier_analysis.png', dpi=300, bbox_inches='tight')
print("[OK] 异常个体分析图已生成")

# 5. 综合对比图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 分布对比 - rho
axes[0, 0].hist(initial_df['rho'], bins=20, alpha=0.7, label='初始', color='blue', edgecolor='black')
axes[0, 0].hist(final_df['rho'], bins=20, alpha=0.7, label='进化后', color='red', edgecolor='black')
axes[0, 0].set_xlabel('rho', fontsize=11)
axes[0, 0].set_ylabel('频数', fontsize=11)
axes[0, 0].set_title('rho分布对比', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 分布对比 - tau
axes[0, 1].hist(initial_df['tau_mean'], bins=20, alpha=0.7, label='初始', color='blue', edgecolor='black')
axes[0, 1].hist(final_df['tau_mean'], bins=20, alpha=0.7, label='进化后', color='red', edgecolor='black')
axes[0, 1].set_xlabel('tau_mean', fontsize=11)
axes[0, 1].set_ylabel('频数', fontsize=11)
axes[0, 1].set_title('tau分布对比', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 分布对比 - rewards
axes[0, 2].hist(initial_df['rewards_mean'], bins=20, alpha=0.7, label='初始', color='blue', edgecolor='black')
axes[0, 2].hist(final_df['rewards_mean'], bins=20, alpha=0.7, label='进化后', color='red', edgecolor='black')
axes[0, 2].set_xlabel('rewards_mean', fontsize=11)
axes[0, 2].set_ylabel('频数', fontsize=11)
axes[0, 2].set_title('rewards分布对比', fontsize=12, fontweight='bold')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# 箱线图 - rho
axes[1, 0].boxplot([initial_df['rho'], final_df['rho']], labels=['初始', '进化后'])
axes[1, 0].set_ylabel('rho', fontsize=11)
axes[1, 0].set_title('rho箱线图', fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# 箱线图 - tau
axes[1, 1].boxplot([initial_df['tau_mean'], final_df['tau_mean']], labels=['初始', '进化后'])
axes[1, 1].set_ylabel('tau_mean', fontsize=11)
axes[1, 1].set_title('tau箱线图', fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.3)

# 箱线图 - rewards
axes[1, 2].boxplot([initial_df['rewards_mean'], final_df['rewards_mean']], labels=['初始', '进化后'])
axes[1, 2].set_ylabel('rewards_mean', fontsize=11)
axes[1, 2].set_title('rewards箱线图', fontsize=12, fontweight='bold')
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('images/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
print("[OK] 综合对比图已生成")

print("\n所有可视化图表生成完成！")
print("生成的文件:")
print("  - images/correlation_heatmap.png")
print("  - images/individual_change_heatmap.png")
print("  - images/clustering_visualization.png")
print("  - images/outlier_analysis.png")
print("  - images/comprehensive_comparison.png")
