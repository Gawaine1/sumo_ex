import pandas as pd
import numpy as np

# 读取数据
initial_df = pd.read_excel('population_initial.xlsx')
final_df = pd.read_excel('final_population .xlsx')

print("初始种群数据概览：")
print("=" * 60)
print(initial_df.head())
print(f"\n初始种群形状: {initial_df.shape}")
print(f"初始种群列名: {initial_df.columns.tolist()}")

print("\n\n进化后种群数据概览：")
print("=" * 60)
print(final_df.head())
print(f"\n进化后种群形状: {final_df.shape}")
print(f"进化后种群列名: {final_df.columns.tolist()}")

# 提取关键统计信息
print("\n\n关键参数统计对比：")
print("=" * 60)

# 假设列名，根据实际情况调整
if 'rho' in initial_df.columns:
    print(f"初始种群 rho: 均值={initial_df['rho'].mean():.6f}, 标准差={initial_df['rho'].std():.6f}")
    print(f"进化后种群 rho: 均值={final_df['rho'].mean():.6f}, 标准差={final_df['rho'].std():.6f}")
    print(f"rho 变化: {(final_df['rho'].mean() - initial_df['rho'].mean()) / initial_df['rho'].mean() * 100:.2f}%")

if 'tau_mean' in initial_df.columns:
    print(f"初始种群 tau_mean: 均值={initial_df['tau_mean'].mean():.6f}, 标准差={initial_df['tau_mean'].std():.6f}")
    print(f"进化后种群 tau_mean: 均值={final_df['tau_mean'].mean():.6f}, 标准差={final_df['tau_mean'].std():.6f}")
    print(f"tau_mean 变化: {(final_df['tau_mean'].mean() - initial_df['tau_mean'].mean()) / initial_df['tau_mean'].mean() * 100:.2f}%")

if 'rewards_mean' in initial_df.columns:
    print(f"初始种群 rewards_mean: 均值={initial_df['rewards_mean'].mean():.6f}, 标准差={initial_df['rewards_mean'].std():.6f}")
    print(f"进化后种群 rewards_mean: 均值={final_df['rewards_mean'].mean():.6f}, 标准差={final_df['rewards_mean'].std():.6f}")
    print(f"rewards_mean 变化: {(final_df['rewards_mean'].mean() - initial_df['rewards_mean'].mean()) / initial_df['rewards_mean'].mean() * 100:.2f}%")

# 显示前10个个体详细数据
print("\n\n初始种群前10个个体详细数据：")
print("=" * 60)
print(initial_df.head(10).to_string())

print("\n\n进化后种群前10个个体详细数据：")
print("=" * 60)
print(final_df.head(10).to_string())

# 个体级别变化分析
print("\n\n个体级别变化分析：")
print("=" * 60)

if len(initial_df) == len(final_df):
    comparison_df = pd.DataFrame()
    comparison_df['个体ID'] = range(len(initial_df))
    
    if 'rho' in initial_df.columns:
        comparison_df['rho_initial'] = initial_df['rho'].values
        comparison_df['rho_final'] = final_df['rho'].values
        comparison_df['rho_change'] = comparison_df['rho_final'] - comparison_df['rho_initial']
        comparison_df['rho_change_pct'] = (comparison_df['rho_change'] / comparison_df['rho_initial']) * 100
    
    if 'tau_mean' in initial_df.columns:
        comparison_df['tau_initial'] = initial_df['tau_mean'].values
        comparison_df['tau_final'] = final_df['tau_mean'].values
        comparison_df['tau_change'] = comparison_df['tau_final'] - comparison_df['tau_initial']
        comparison_df['tau_change_pct'] = (comparison_df['tau_change'] / comparison_df['tau_initial']) * 100
    
    if 'rewards_mean' in initial_df.columns:
        comparison_df['reward_initial'] = initial_df['rewards_mean'].values
        comparison_df['reward_final'] = final_df['rewards_mean'].values
        comparison_df['reward_change'] = comparison_df['reward_final'] - comparison_df['reward_initial']
        comparison_df['reward_change_pct'] = (comparison_df['reward_change'] / comparison_df['reward_initial']) * 100
    
    print(comparison_df.head(10).to_string())
    
    # 保存详细对比数据
    comparison_df.to_csv('detailed_comparison.csv', index=False)
    print("\n详细对比数据已保存到: detailed_comparison.csv")
