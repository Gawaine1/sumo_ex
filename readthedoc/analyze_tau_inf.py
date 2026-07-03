import pandas as pd
import numpy as np

# 读取数据
initial_df = pd.read_excel('population_initial.xlsx')
final_df = pd.read_excel('final_population .xlsx')

print("检查tau数据中的无限值情况")
print("=" * 60)

# 检查初始种群的tau数据
print("\n初始种群tau数据检查:")
print(f"tau_mean统计信息:")
print(initial_df['tau_mean'].describe())

# 检查是否有无限值
initial_has_inf = np.isinf(initial_df['tau_mean']).sum()
initial_has_nan = np.isnan(initial_df['tau_mean']).sum()
print(f"无限值数量: {initial_has_inf}")
print(f"NaN值数量: {initial_has_nan}")

# 检查进化后种群的tau数据
print("\n进化后种群tau数据检查:")
print(f"tau_mean统计信息:")
print(final_df['tau_mean'].describe())

# 检查是否有无限值
final_has_inf = np.isinf(final_df['tau_mean']).sum()
final_has_nan = np.isnan(final_df['tau_mean']).sum()
print(f"无限值数量: {final_has_inf}")
print(f"NaN值数量: {final_has_nan}")

# 如果有无限值，分析其分布
if final_has_inf > 0:
    print(f"\n进化后种群存在{final_has_inf}个无限值")
    print("无限值索引:", final_df[np.isinf(final_df['tau_mean'])].index.tolist())
    
    # 分析无限值的特征
    inf_indices = final_df[np.isinf(final_df['tau_mean'])].index
    print("\n无限值个体的其他参数:")
    print(final_df.loc[inf_indices, ['rho', 'rewards_mean']].describe())

# 尝试访问tau的原始数据（如果有的话）
print("\n" + "=" * 60)
print("尝试分析tau的两个参数（排队长度和离目的地距离）")

# 检查是否有tau的详细数据
if 'tau' in initial_df.columns:
    print("\n初始种群tau列存在")
    print(f"tau数据类型: {type(initial_df['tau'].iloc[0])}")
    print(f"tau数据形状: {initial_df['tau'].iloc[0].shape if hasattr(initial_df['tau'].iloc[0], 'shape') else 'N/A'}")
else:
    print("\n初始种群没有tau列，只有tau_mean")

# 检查是否有tau的详细数据
if 'tau' in final_df.columns:
    print("\n进化后种群tau列存在")
    print(f"tau数据类型: {type(final_df['tau'].iloc[0])}")
    tau_shape = final_df['tau'].iloc[0].shape if hasattr(final_df['tau'].iloc[0], 'shape') else 'N/A'
    print(f"tau数据形状: {tau_shape}")
else:
    print("\n进化后种群没有tau列，只有tau_mean")

# 重新计算排除无限值后的统计量
print("\n" + "=" * 60)
print("重新计算排除无限值后的tau统计量")

# 初始种群
initial_tau_valid = initial_df['tau_mean'][~np.isinf(initial_df['tau_mean']) & ~np.isnan(initial_df['tau_mean'])]
print(f"\n初始种群有效tau_mean数量: {len(initial_tau_valid)}")
print(f"初始种群有效tau_mean统计:")
print(initial_tau_valid.describe())

# 进化后种群
final_tau_valid = final_df['tau_mean'][~np.isinf(final_df['tau_mean']) & ~np.isnan(final_df['tau_mean'])]
print(f"\n进化后种群有效tau_mean数量: {len(final_tau_valid)}")
print(f"进化后种群有效tau_mean统计:")
print(final_tau_valid.describe())

# 对比分析
print("\n" + "=" * 60)
print("排除无限值后的tau对比分析")
initial_mean = initial_tau_valid.mean()
final_mean = final_tau_valid.mean()
change_pct = (final_mean - initial_mean) / initial_mean * 100

print(f"初始种群tau_mean均值: {initial_mean:.6f}")
print(f"进化后种群tau_mean均值: {final_mean:.6f}")
print(f"变化幅度: {change_pct:.2f}%")

if final_has_inf > 0:
    print(f"\n注意: 进化后种群有{final_has_inf}个无限值被排除")
    print(f"这些无限值可能是导致tau_mean显著降低的原因")
