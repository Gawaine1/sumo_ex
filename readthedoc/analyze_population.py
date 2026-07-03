import pandas as pd
import numpy as np

# 读取两个文件
initial_df = pd.read_excel('population_initial.xlsx')
final_df = pd.read_excel('final_population .xlsx')

print("=" * 80)
print("初始化种群文件 (population_initial.xyz)")
print("=" * 80)
print(f"数据形状: {initial_df.shape}")
print(f"\n列名: {initial_df.columns.tolist()}")
print(f"\n前5行数据:")
print(initial_df.head())
print(f"\n数据类型:")
print(initial_df.dtypes)
print(f"\n基本统计信息:")
print(initial_df.describe())

print("\n" + "=" * 80)
print("进化后文件 (final_population.xlsx)")
print("=" * 80)
print(f"数据形状: {final_df.shape}")
print(f"\n列名: {final_df.columns.tolist()}")
print(f"\n前5行数据:")
print(final_df.head())
print(f"\n数据类型:")
print(final_df.dtypes)
print(f"\n基本统计信息:")
print(final_df.describe())

# 分析tau数据（如果存在）
print("\n" + "=" * 80)
print("tau数据分析")
print("=" * 80)

# 检查tau相关的列
tau_cols_initial = [col for col in initial_df.columns if 'tau' in col.lower()]
tau_cols_final = [col for col in final_df.columns if 'tau' in col.lower()]

print(f"初始化文件中的tau相关列: {tau_cols_initial}")
print(f"进化后文件中的tau相关列: {tau_cols_final}")

if tau_cols_initial:
    print(f"\n初始化文件tau数据示例:")
    print(initial_df[tau_cols_initial].head())
    
if tau_cols_final:
    print(f"\n进化后文件tau数据示例:")
    print(final_df[tau_cols_final].head())

# 分析rho数据
print("\n" + "=" * 80)
print("rho数据分析")
print("=" * 80)

rho_cols_initial = [col for col in initial_df.columns if 'rho' in col.lower()]
rho_cols_final = [col for col in final_df.columns if 'rho' in col.lower()]

print(f"初始化文件中的rho相关列: {rho_cols_initial}")
print(f"进化后文件中的rho相关列: {rho_cols_final}")

if rho_cols_initial:
    print(f"\n初始化文件rho数据统计:")
    print(initial_df[rho_cols_initial].describe())
    
if rho_cols_final:
    print(f"\n进化后文件rho数据统计:")
    print(final_df[rho_cols_final].describe())

# 分析r数据（奖励）
print("\n" + "=" * 80)
print("r（奖励）数据分析")
print("=" * 80)

r_cols_initial = [col for col in initial_df.columns if col == 'r' or 'reward' in col.lower()]
r_cols_final = [col for col in final_df.columns if col == 'r' or 'reward' in col.lower()]

print(f"初始化文件中的r相关列: {r_cols_initial}")
print(f"进化后文件中的r相关列: {r_cols_final}")

if r_cols_initial:
    print(f"\n初始化文件r数据统计:")
    print(initial_df[r_cols_initial].describe())
    
if r_cols_final:
    print(f"\n进化后文件r数据统计:")
    print(final_df[r_cols_final].describe())

# 对比分析
print("\n" + "=" * 80)
print("进化前后对比分析")
print("=" * 80)

if 'r' in initial_df.columns and 'r' in final_df.columns:
    print(f"\n奖励r的对比:")
    print(f"初始化种群 - 平均值: {initial_df['r'].mean():.4f}, 标准差: {initial_df['r'].std():.4f}")
    print(f"进化后种群 - 平均值: {final_df['r'].mean():.4f}, 标准差: {final_df['r'].std():.4f}")
    print(f"改进幅度: {((final_df['r'].mean() - initial_df['r'].mean()) / initial_df['r'].mean() * 100):.2f}%")
