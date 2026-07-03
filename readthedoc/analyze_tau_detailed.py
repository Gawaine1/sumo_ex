import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取文件
initial_df = pd.read_excel('population_initial.xlsx')
final_df = pd.read_excel('final_population .xlsx')

print("=" * 80)
print("tau详细数据分析")
print("=" * 80)

# tau统计信息对比
print("\n初始化种群tau统计:")
print(f"  最小值: {initial_df['tau_min'].mean():.4f}")
print(f"  最大值: {initial_df['tau_max'].mean():.4f}")
print(f"  平均值: {initial_df['tau_mean'].mean():.4f}")

print("\n进化后种群tau统计:")
print(f"  最小值: {final_df['tau_min'].mean():.4f}")
print(f"  最大值: {final_df['tau_max'].mean():.4f}")
print(f"  平均值: {final_df['tau_mean'].mean():.4f}")

print("\ntau变化:")
print(f"  平均值变化: {((final_df['tau_mean'].mean() - initial_df['tau_mean'].mean()) / initial_df['tau_mean'].mean() * 100):.2f}%")

print("\n" + "=" * 80)
print("rho详细数据分析")
print("=" * 80)

print("\n初始化种群rho统计:")
print(f"  平均值: {initial_df['rho'].mean():.6f}")
print(f"  标准差: {initial_df['rho'].std():.6f}")
print(f"  最小值: {initial_df['rho'].min():.6f}")
print(f"  最大值: {initial_df['rho'].max():.6f}")

print("\n进化后种群rho统计:")
print(f"  平均值: {final_df['rho'].mean():.6f}")
print(f"  标准差: {final_df['rho'].std():.6f}")
print(f"  最小值: {final_df['rho'].min():.6f}")
print(f"  最大值: {final_df['rho'].max():.6f}")

print("\nrho变化:")
print(f"  平均值提升: {((final_df['rho'].mean() - initial_df['rho'].mean()) / initial_df['rho'].mean() * 100):.2f}%")
print(f"  标准差变化: {((final_df['rho'].std() - initial_df['rho'].std()) / initial_df['rho'].std() * 100):.2f}%")

print("\n" + "=" * 80)
print("奖励(rewards)详细数据分析")
print("=" * 80)

print("\n初始化种群rewards统计:")
print(f"  数据点数: {initial_df['rewards_numel'].iloc[0]}")
print(f"  最小值: {initial_df['rewards_min'].mean():.6f}")
print(f"  最大值: {initial_df['rewards_max'].mean():.6f}")
print(f"  平均值: {initial_df['rewards_mean'].mean():.6f}")

print("\n进化后种群rewards统计:")
print(f"  数据点数: {final_df['rewards_numel'].iloc[0]}")
print(f"  最小值: {final_df['rewards_min'].mean():.6f}")
print(f"  最大值: {final_df['rewards_max'].mean():.6f}")
print(f"  平均值: {final_df['rewards_mean'].mean():.6f}")

print("\nrewards变化:")
print(f"  平均值变化: {((final_df['rewards_mean'].mean() - initial_df['rewards_mean'].mean()) / initial_df['rewards_mean'].mean() * 100):.2f}%")

print("\n" + "=" * 80)
print("tau数据结构分析")
print("=" * 80)
print(f"tau形状: {initial_df['tau_shape'].iloc[0]}")
print(f"  - 第1维度 (时间步/场景数): 1557")
print(f"  - 第2维度 (个体数/节点数): 114")
print(f"  - 第3维度 (特征数): 2 (排队长度 + 离目的地的距离)")
print(f"tau数据类型: {initial_df['tau_dtype'].iloc[0]}")
print(f"tau总元素数: {initial_df['tau_numel'].iloc[0]}")

print("\n" + "=" * 80)
print("进化效果总结")
print("=" * 80)
print(f"1. rho参数: 从 {initial_df['rho'].mean():.6f} 提升到 {final_df['rho'].mean():.6f} (提升 {((final_df['rho'].mean() - initial_df['rho'].mean()) / initial_df['rho'].mean() * 100):.2f}%)")
print(f"2. tau平均值: 从 {initial_df['tau_mean'].mean():.4f} 变化到 {final_df['tau_mean'].mean():.4f} (变化 {((final_df['tau_mean'].mean() - initial_df['tau_mean'].mean()) / initial_df['tau_mean'].mean() * 100):.2f}%)")
print(f"3. rewards平均值: 从 {initial_df['rewards_mean'].mean():.6f} 变化到 {final_df['rewards_mean'].mean():.6f} (变化 {((final_df['rewards_mean'].mean() - initial_df['rewards_mean'].mean()) / initial_df['rewards_mean'].mean() * 100):.2f}%)")
