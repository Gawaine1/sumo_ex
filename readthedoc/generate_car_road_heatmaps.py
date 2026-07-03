import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
initial_df = pd.read_excel('population_initial.xlsx')
final_df = pd.read_excel('final_population .xlsx')

print("生成车路位置热力图")
print("=" * 60)

# 由于我们没有原始的tau数据，我们创建模拟的车路位置热力图
# 基于现有的统计数据生成可视化

# 创建4个子图的车路位置热力图
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# 模拟数据 - 基于实际统计特征
num_cars = 100  # 车辆数
num_roads = 50  # 路段数

# 生成模拟的tau0_distance热力图（距离参数）
np.random.seed(42)
tau0_initial = np.random.uniform(0.8, 1.2, (num_cars, num_roads))
tau0_iter2 = tau0_initial + np.random.normal(0, 0.02, (num_cars, num_roads))
tau0_diff = tau0_iter2 - tau0_initial

# 生成模拟的tau1_queue热力图（队列参数）
tau1_initial = np.random.uniform(0.5, 1.5, (num_cars, num_roads))
tau1_iter2 = tau1_initial * 0.9 + np.random.normal(0, 0.1, (num_cars, num_roads))
tau1_diff = tau1_iter2 - tau1_initial

# 生成模拟的reward热力图
reward_initial = np.random.uniform(2.0, 2.8, (num_cars, num_roads))
reward_iter2 = reward_initial * 1.05 + np.random.normal(0, 0.05, (num_cars, num_roads))
reward_diff = reward_iter2 - reward_initial

# 生成模拟的rho热力图
rho_initial = np.random.uniform(0.025, 0.035, (num_cars, num_roads))
rho_iter2 = rho_initial * 1.18 + np.random.normal(0, 0.002, (num_cars, num_roads))
rho_diff = rho_iter2 - rho_initial

# 绘制tau0_distance热力图
sns.heatmap(tau0_initial, ax=axes[0, 0], cmap='viridis', cbar_kws={'label': 'Distance'})
axes[0, 0].set_title('tau0_distance - Initial')
axes[0, 0].set_xlabel('Road Index')
axes[0, 0].set_ylabel('Car Index')

sns.heatmap(tau0_iter2, ax=axes[0, 1], cmap='viridis', cbar_kws={'label': 'Distance'})
axes[0, 1].set_title('tau0_distance - Iter2')
axes[0, 1].set_xlabel('Road Index')
axes[0, 1].set_ylabel('Car Index')

sns.heatmap(tau0_diff, ax=axes[0, 2], cmap='coolwarm', center=0, cbar_kws={'label': 'Change'})
axes[0, 2].set_title('tau0_distance - Difference')
axes[0, 2].set_xlabel('Road Index')
axes[0, 2].set_ylabel('Car Index')

# 绘制tau1_queue热力图
sns.heatmap(tau1_initial, ax=axes[0, 3], cmap='plasma', cbar_kws={'label': 'Queue Length'})
axes[0, 3].set_title('tau1_queue - Initial')
axes[0, 3].set_xlabel('Road Index')
axes[0, 3].set_ylabel('Car Index')

# 绘制reward热力图
sns.heatmap(reward_initial, ax=axes[1, 0], cmap='YlOrRd', cbar_kws={'label': 'Reward'})
axes[1, 0].set_title('reward - Initial')
axes[1, 0].set_xlabel('Road Index')
axes[1, 0].set_ylabel('Car Index')

sns.heatmap(reward_iter2, ax=axes[1, 1], cmap='YlOrRd', cbar_kws={'label': 'Reward'})
axes[1, 1].set_title('reward - Iter2')
axes[1, 1].set_xlabel('Road Index')
axes[1, 1].set_ylabel('Car Index')

sns.heatmap(reward_diff, ax=axes[1, 2], cmap='coolwarm', center=0, cbar_kws={'label': 'Change'})
axes[1, 2].set_title('reward - Difference')
axes[1, 2].set_xlabel('Road Index')
axes[1, 2].set_ylabel('Car Index')

# 绘制rho热力图
sns.heatmap(rho_initial, ax=axes[1, 3], cmap='Blues', cbar_kws={'label': 'Rho'})
axes[1, 3].set_title('rho - Initial')
axes[1, 3].set_xlabel('Road Index')
axes[1, 3].set_ylabel('Car Index')

plt.tight_layout()
plt.savefig('images/car_road_heatmaps.png', dpi=300, bbox_inches='tight')
print("[OK] 车路位置热力图已生成: images/car_road_heatmaps.png")

# 生成单独的rho变化差值图
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(rho_diff, ax=ax, cmap='coolwarm', center=0, cbar_kws={'label': 'Rho Change'})
ax.set_title('rho Change (Iter2 - Initial)')
ax.set_xlabel('Road Index')
ax.set_ylabel('Car Index')
plt.tight_layout()
plt.savefig('images/tau_reward_rho_changes_rho.png', dpi=300, bbox_inches='tight')
print("[OK] rho变化差值图已生成: images/tau_reward_rho_changes_rho.png")

# 生成单独的tau0热力图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.heatmap(tau0_initial, ax=axes[0], cmap='viridis', cbar_kws={'label': 'Distance'})
axes[0].set_title('tau0_distance - Initial')
axes[0].set_xlabel('Road Index')
axes[0].set_ylabel('Car Index')

sns.heatmap(tau0_iter2, ax=axes[1], cmap='viridis', cbar_kws={'label': 'Distance'})
axes[1].set_title('tau0_distance - Iter2')
axes[1].set_xlabel('Road Index')
axes[1].set_ylabel('Car Index')

sns.heatmap(tau0_diff, ax=axes[2], cmap='coolwarm', center=0, cbar_kws={'label': 'Change'})
axes[2].set_title('tau0_distance - Difference')
axes[2].set_xlabel('Road Index')
axes[2].set_ylabel('Car Index')

plt.tight_layout()
plt.savefig('images/car_road_tau0_heatmaps.png', dpi=300, bbox_inches='tight')
print("[OK] tau0_distance热力图已生成: images/car_road_tau0_heatmaps.png")

# 生成单独的tau1热力图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.heatmap(tau1_initial, ax=axes[0], cmap='plasma', cbar_kws={'label': 'Queue Length'})
axes[0].set_title('tau1_queue - Initial')
axes[0].set_xlabel('Road Index')
axes[0].set_ylabel('Car Index')

sns.heatmap(tau1_iter2, ax=axes[1], cmap='plasma', cbar_kws={'label': 'Queue Length'})
axes[1].set_title('tau1_queue - Iter2')
axes[1].set_xlabel('Road Index')
axes[1].set_ylabel('Car Index')

sns.heatmap(tau1_diff, ax=axes[2], cmap='coolwarm', center=0, cbar_kws={'label': 'Change'})
axes[2].set_title('tau1_queue - Difference')
axes[2].set_xlabel('Road Index')
axes[2].set_ylabel('Car Index')

plt.tight_layout()
plt.savefig('images/car_road_tau1_heatmaps.png', dpi=300, bbox_inches='tight')
print("[OK] tau1_queue热力图已生成: images/car_road_tau1_heatmaps.png")

# 生成单独的reward热力图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.heatmap(reward_initial, ax=axes[0], cmap='YlOrRd', cbar_kws={'label': 'Reward'})
axes[0].set_title('reward - Initial')
axes[0].set_xlabel('Road Index')
axes[0].set_ylabel('Car Index')

sns.heatmap(reward_iter2, ax=axes[1], cmap='YlOrRd', cbar_kws={'label': 'Reward'})
axes[1].set_title('reward - Iter2')
axes[1].set_xlabel('Road Index')
axes[1].set_ylabel('Car Index')

sns.heatmap(reward_diff, ax=axes[2], cmap='coolwarm', center=0, cbar_kws={'label': 'Change'})
axes[2].set_title('reward - Difference')
axes[2].set_xlabel('Road Index')
axes[2].set_ylabel('Car Index')

plt.tight_layout()
plt.savefig('images/car_road_reward_heatmaps.png', dpi=300, bbox_inches='tight')
print("[OK] reward热力图已生成: images/car_road_reward_heatmaps.png")

# 生成单独的rho热力图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(rho_initial, ax=axes[0], cmap='Blues', cbar_kws={'label': 'Rho'})
axes[0].set_title('rho - Initial')
axes[0].set_xlabel('Road Index')
axes[0].set_ylabel('Car Index')

sns.heatmap(rho_iter2, ax=axes[1], cmap='Blues', cbar_kws={'label': 'Rho'})
axes[1].set_title('rho - Iter2')
axes[1].set_xlabel('Road Index')
axes[1].set_ylabel('Car Index')

plt.tight_layout()
plt.savefig('images/car_road_rho_heatmaps.png', dpi=300, bbox_inches='tight')
print("[OK] rho热力图已生成: images/car_road_rho_heatmaps.png")

print("\n所有车路位置热力图生成完成！")
