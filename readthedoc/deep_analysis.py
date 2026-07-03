import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
initial_df = pd.read_excel('population_initial.xlsx')
final_df = pd.read_excel('final_population .xlsx')

print("=" * 100)
print("深度数据分析报告")
print("=" * 100)

# 1. 个体级别的详细分析
print("\n" + "=" * 100)
print("1. 个体级别的详细变化分析")
print("=" * 100)

# 合并数据进行个体对比
comparison_df = pd.DataFrame({
    'individual_id': range(100),
    'rho_initial': initial_df['rho'].values,
    'rho_final': final_df['rho'].values,
    'rho_change': final_df['rho'].values - initial_df['rho'].values,
    'rho_change_pct': ((final_df['rho'].values - initial_df['rho'].values) / initial_df['rho'].values * 100),
    'tau_mean_initial': initial_df['tau_mean'].values,
    'tau_mean_final': final_df['tau_mean'].values,
    'tau_change': final_df['tau_mean'].values - initial_df['tau_mean'].values,
    'tau_change_pct': ((final_df['tau_mean'].values - initial_df['tau_mean'].values) / initial_df['tau_mean'].values * 100),
    'rewards_mean_initial': initial_df['rewards_mean'].values,
    'rewards_mean_final': final_df['rewards_mean'].values,
    'rewards_change': final_df['rewards_mean'].values - initial_df['rewards_mean'].values,
    'rewards_change_pct': ((final_df['rewards_mean'].values - initial_df['rewards_mean'].values) / initial_df['rewards_mean'].values * 100)
})

print("\n个体变化统计:")
print(f"rho变化 - 最大提升: {comparison_df['rho_change_pct'].max():.2f}%, 最大下降: {comparison_df['rho_change_pct'].min():.2f}%")
print(f"tau变化 - 最大提升: {comparison_df['tau_change_pct'].max():.2f}%, 最大下降: {comparison_df['tau_change_pct'].min():.2f}%")
print(f"rewards变化 - 最大提升: {comparison_df['rewards_change_pct'].max():.2f}%, 最大下降: {comparison_df['rewards_change_pct'].min():.2f}%")

print("\n个体变化分布:")
print(f"rho提升的个体数: {(comparison_df['rho_change'] > 0).sum()}/100")
print(f"tau降低的个体数: {(comparison_df['tau_change'] < 0).sum()}/100")
print(f"rewards提升的个体数: {(comparison_df['rewards_change'] > 0).sum()}/100")

# 找出最优和最差个体
print("\n最优个体分析:")
best_individual = comparison_df.loc[comparison_df['rewards_change'].idxmax()]
print(f"个体ID: {int(best_individual['individual_id'])}")
print(f"  rho: {best_individual['rho_initial']:.6f} -> {best_individual['rho_final']:.6f} ({best_individual['rho_change_pct']:.2f}%)")
print(f"  tau: {best_individual['tau_mean_initial']:.4f} -> {best_individual['tau_mean_final']:.4f} ({best_individual['tau_change_pct']:.2f}%)")
print(f"  rewards: {best_individual['rewards_mean_initial']:.6f} -> {best_individual['rewards_mean_final']:.6f} ({best_individual['rewards_change_pct']:.2f}%)")

print("\n最差个体分析:")
worst_individual = comparison_df.loc[comparison_df['rewards_change'].idxmin()]
print(f"个体ID: {int(worst_individual['individual_id'])}")
print(f"  rho: {worst_individual['rho_initial']:.6f} -> {worst_individual['rho_final']:.6f} ({worst_individual['rho_change_pct']:.2f}%)")
print(f"  tau: {worst_individual['tau_mean_initial']:.4f} -> {worst_individual['tau_mean_final']:.4f} ({worst_individual['tau_change_pct']:.2f}%)")
print(f"  rewards: {worst_individual['rewards_mean_initial']:.6f} -> {worst_individual['rewards_mean_final']:.6f} ({worst_individual['rewards_change_pct']:.2f}%)")

# 2. 相关性分析
print("\n" + "=" * 100)
print("2. 参数相关性分析")
print("=" * 100)

# 初始种群相关性
print("\n初始种群相关性矩阵:")
initial_corr = initial_df[['rho', 'tau_mean', 'rewards_mean']].corr()
print(initial_corr)

print("\n进化后种群相关性矩阵:")
final_corr = final_df[['rho', 'tau_mean', 'rewards_mean']].corr()
print(final_corr)

# 变化量相关性
print("\n变化量相关性矩阵:")
change_corr = comparison_df[['rho_change', 'tau_change', 'rewards_change']].corr()
print(change_corr)

# 3. 统计检验
print("\n" + "=" * 100)
print("3. 统计显著性检验")
print("=" * 100)

# rho的t检验
rho_t_stat, rho_p_value = stats.ttest_rel(initial_df['rho'], final_df['rho'])
print(f"\nrho配对t检验:")
print(f"  t统计量: {rho_t_stat:.4f}")
print(f"  p值: {rho_p_value:.6f}")
print(f"  结论: {'显著差异' if rho_p_value < 0.05 else '无显著差异'} (α=0.05)")

# tau的t检验
tau_t_stat, tau_p_value = stats.ttest_rel(initial_df['tau_mean'], final_df['tau_mean'])
print(f"\ntau配对t检验:")
print(f"  t统计量: {tau_t_stat:.4f}")
print(f"  p值: {tau_p_value:.6f}")
print(f"  结论: {'显著差异' if tau_p_value < 0.05 else '无显著差异'} (α=0.05)")

# rewards的t检验
rewards_t_stat, rewards_p_value = stats.ttest_rel(initial_df['rewards_mean'], final_df['rewards_mean'])
print(f"\nrewards配对t检验:")
print(f"  t统计量: {rewards_t_stat:.4f}")
print(f"  p值: {rewards_p_value:.6f}")
print(f"  结论: {'显著差异' if rewards_p_value < 0.05 else '无显著差异'} (α=0.05)")

# 4. 分布分析
print("\n" + "=" * 100)
print("4. 分布特征分析")
print("=" * 100)

print("\nrho分布偏度和峰度:")
print(f"  初始 - 偏度: {stats.skew(initial_df['rho']):.4f}, 峰度: {stats.kurtosis(initial_df['rho']):.4f}")
print(f"  进化后 - 偏度: {stats.skew(final_df['rho']):.4f}, 峰度: {stats.kurtosis(final_df['rho']):.4f}")

print("\ntau分布偏度和峰度:")
print(f"  初始 - 偏度: {stats.skew(initial_df['tau_mean']):.4f}, 峰度: {stats.kurtosis(initial_df['tau_mean']):.4f}")
print(f"  进化后 - 偏度: {stats.skew(final_df['tau_mean']):.4f}, 峰度: {stats.kurtosis(final_df['tau_mean']):.4f}")

print("\nrewards分布偏度和峰度:")
print(f"  初始 - 偏度: {stats.skew(initial_df['rewards_mean']):.4f}, 峰度: {stats.kurtosis(initial_df['rewards_mean']):.4f}")
print(f"  进化后 - 偏度: {stats.skew(final_df['rewards_mean']):.4f}, 峰度: {stats.kurtosis(final_df['rewards_mean']):.4f}")

# 5. 聚类分析
print("\n" + "=" * 100)
print("5. 个体聚类分析")
print("=" * 100)

# 对进化后数据进行聚类
features = final_df[['rho', 'tau_mean', 'rewards_mean']].values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# K-means聚类
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(features_scaled)

comparison_df['cluster'] = clusters

print("\n聚类结果:")
for i in range(3):
    cluster_data = comparison_df[comparison_df['cluster'] == i]
    print(f"\n聚类 {i} (包含 {len(cluster_data)} 个个体):")
    print(f"  rho平均值: {cluster_data['rho_final'].mean():.6f} ± {cluster_data['rho_final'].std():.6f}")
    print(f"  tau平均值: {cluster_data['tau_mean_final'].mean():.4f} ± {cluster_data['tau_mean_final'].std():.4f}")
    print(f"  rewards平均值: {cluster_data['rewards_mean_final'].mean():.6f} ± {cluster_data['rewards_mean_final'].std():.6f}")
    print(f"  平均提升: rho {cluster_data['rho_change_pct'].mean():.2f}%, tau {cluster_data['tau_change_pct'].mean():.2f}%, rewards {cluster_data['rewards_change_pct'].mean():.2f}%")

# 6. 创建可视化
print("\n" + "=" * 100)
print("6. 生成可视化图表")
print("=" * 100)

fig, axes = plt.subplots(3, 3, figsize=(18, 15))

# rho分布对比
axes[0, 0].hist(initial_df['rho'], bins=20, alpha=0.7, label='初始', color='blue')
axes[0, 0].hist(final_df['rho'], bins=20, alpha=0.7, label='进化后', color='red')
axes[0, 0].set_xlabel('rho')
axes[0, 0].set_ylabel('频数')
axes[0, 0].set_title('rho分布对比')
axes[0, 0].legend()

# tau分布对比
axes[0, 1].hist(initial_df['tau_mean'], bins=20, alpha=0.7, label='初始', color='blue')
axes[0, 1].hist(final_df['tau_mean'], bins=20, alpha=0.7, label='进化后', color='red')
axes[0, 1].set_xlabel('tau_mean')
axes[0, 1].set_ylabel('频数')
axes[0, 1].set_title('tau分布对比')
axes[0, 1].legend()

# rewards分布对比
axes[0, 2].hist(initial_df['rewards_mean'], bins=20, alpha=0.7, label='初始', color='blue')
axes[0, 2].hist(final_df['rewards_mean'], bins=20, alpha=0.7, label='进化后', color='red')
axes[0, 2].set_xlabel('rewards_mean')
axes[0, 2].set_ylabel('频数')
axes[0, 2].set_title('rewards分布对比')
axes[0, 2].legend()

# 散点图：rho vs rewards
axes[1, 0].scatter(initial_df['rho'], initial_df['rewards_mean'], alpha=0.6, label='初始', color='blue')
axes[1, 0].scatter(final_df['rho'], final_df['rewards_mean'], alpha=0.6, label='进化后', color='red')
axes[1, 0].set_xlabel('rho')
axes[1, 0].set_ylabel('rewards_mean')
axes[1, 0].set_title('rho vs rewards')
axes[1, 0].legend()

# 散点图：tau vs rewards
axes[1, 1].scatter(initial_df['tau_mean'], initial_df['rewards_mean'], alpha=0.6, label='初始', color='blue')
axes[1, 1].scatter(final_df['tau_mean'], final_df['rewards_mean'], alpha=0.6, label='进化后', color='red')
axes[1, 1].set_xlabel('tau_mean')
axes[1, 1].set_ylabel('rewards_mean')
axes[1, 1].set_title('tau vs rewards')
axes[1, 1].legend()

# 变化量散点图
axes[1, 2].scatter(comparison_df['rho_change'], comparison_df['rewards_change'], alpha=0.6)
axes[1, 2].set_xlabel('rho_change')
axes[1, 2].set_ylabel('rewards_change')
axes[1, 2].set_title('rho变化 vs rewards变化')
axes[1, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 2].axvline(x=0, color='k', linestyle='--', alpha=0.3)

# 箱线图：rho
axes[2, 0].boxplot([initial_df['rho'], final_df['rho']], labels=['初始', '进化后'])
axes[2, 0].set_ylabel('rho')
axes[2, 0].set_title('rho箱线图')

# 箱线图：tau
axes[2, 1].boxplot([initial_df['tau_mean'], final_df['tau_mean']], labels=['初始', '进化后'])
axes[2, 1].set_ylabel('tau_mean')
axes[2, 1].set_title('tau箱线图')

# 箱线图：rewards
axes[2, 2].boxplot([initial_df['rewards_mean'], final_df['rewards_mean']], labels=['初始', '进化后'])
axes[2, 2].set_ylabel('rewards_mean')
axes[2, 2].set_title('rewards箱线图')

plt.tight_layout()
plt.savefig('deep_analysis_plots.png', dpi=300, bbox_inches='tight')
print("可视化图表已保存为 'deep_analysis_plots.png'")

# 7. 深入洞察分析
print("\n" + "=" * 100)
print("7. 深入洞察与模式识别")
print("=" * 100)

# 分析rho和tau的关系
print("\nrho与tau的关系分析:")
rho_tau_corr_initial = initial_df['rho'].corr(initial_df['tau_mean'])
rho_tau_corr_final = final_df['rho'].corr(final_df['tau_mean'])
print(f"  初始相关性: {rho_tau_corr_initial:.4f}")
print(f"  进化后相关性: {rho_tau_corr_final:.4f}")
print(f"  相关性变化: {rho_tau_corr_final - rho_tau_corr_initial:.4f}")

# 分析rho和rewards的关系
print("\nrho与rewards的关系分析:")
rho_rewards_corr_initial = initial_df['rho'].corr(initial_df['rewards_mean'])
rho_rewards_corr_final = final_df['rho'].corr(final_df['rewards_mean'])
print(f"  初始相关性: {rho_rewards_corr_initial:.4f}")
print(f"  进化后相关性: {rho_rewards_corr_final:.4f}")
print(f"  相关性变化: {rho_rewards_corr_final - rho_rewards_corr_initial:.4f}")

# 识别异常个体
print("\n异常个体识别:")
rho_z_scores = np.abs(stats.zscore(comparison_df['rho_change']))
tau_z_scores = np.abs(stats.zscore(comparison_df['tau_change']))
rewards_z_scores = np.abs(stats.zscore(comparison_df['rewards_change']))

outliers = comparison_df[(rho_z_scores > 2) | (tau_z_scores > 2) | (rewards_z_scores > 2)]
print(f"发现 {len(outliers)} 个异常个体 (|z-score| > 2):")
for idx, row in outliers.iterrows():
    print(f"  个体 {int(row['individual_id'])}: rho变化 {row['rho_change_pct']:.2f}%, tau变化 {row['tau_change_pct']:.2f}%, rewards变化 {row['rewards_change_pct']:.2f}%")

# 保存详细分析结果
comparison_df.to_csv('individual_comparison.csv', index=False, encoding='utf-8-sig')
print("\n个体对比数据已保存为 'individual_comparison.csv'")

print("\n" + "=" * 100)
print("深度分析完成")
print("=" * 100)
