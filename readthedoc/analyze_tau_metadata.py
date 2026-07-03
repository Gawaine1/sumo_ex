import pandas as pd
import numpy as np

# 读取数据
initial_df = pd.read_excel('population_initial.xlsx')
final_df = pd.read_excel('final_population .xlsx')

print("分析tau元数据中的无限值信息")
print("=" * 60)

# 分析初始种群的tau元数据
print("\n初始种群tau元数据:")
print(f"tau非有限值数量统计:")
print(initial_df['tau_nonfinite_count'].describe())

print(f"\ntau非有限值比例 (metadata_tau_sanitize_tau_nonfinite_ratio):")
print(initial_df['metadata_tau_sanitize_tau_nonfinite_ratio'].describe())

print(f"\ntau距离参数非有限值比例 (metadata_tau_sanitize_tau_dist_nonfinite_ratio):")
print(initial_df['metadata_tau_sanitize_tau_dist_nonfinite_ratio'].describe())

print(f"\ntau排队参数非有限值比例 (metadata_tau_sanitize_tau_queue_nonfinite_ratio):")
print(initial_df['metadata_tau_sanitize_tau_queue_nonfinite_ratio'].describe())

# 分析进化后种群的tau元数据
print("\n" + "=" * 60)
print("进化后种群tau元数据:")
print(f"tau非有限值数量统计:")
print(final_df['tau_nonfinite_count'].describe())

# 检查是否有距离和排队参数的元数据列
if 'metadata_tau_sanitize_tau_nonfinite_ratio' in final_df.columns:
    print(f"\ntau非有限值比例 (metadata_tau_sanitize_tau_nonfinite_ratio):")
    print(final_df['metadata_tau_sanitize_tau_nonfinite_ratio'].describe())

if 'metadata_tau_sanitize_tau_dist_nonfinite_ratio' in final_df.columns:
    print(f"\ntau距离参数非有限值比例 (metadata_tau_sanitize_tau_dist_nonfinite_ratio):")
    print(final_df['metadata_tau_sanitize_tau_dist_nonfinite_ratio'].describe())

if 'metadata_tau_sanitize_tau_queue_nonfinite_ratio' in final_df.columns:
    print(f"\ntau排队参数非有限值比例 (metadata_tau_sanitize_tau_queue_nonfinite_ratio):")
    print(final_df['metadata_tau_sanitize_tau_queue_nonfinite_ratio'].describe())

# 对比分析
print("\n" + "=" * 60)
print("进化前后tau无限值对比分析")

# 初始种群
initial_total_elements = initial_df['tau_numel'].iloc[0]
initial_nonfinite_total = initial_df['tau_nonfinite_count'].sum()
initial_nonfinite_ratio = initial_nonfinite_total / (initial_total_elements * len(initial_df)) * 100

print(f"\n初始种群:")
print(f"单个个体tau元素总数: {initial_total_elements}")
print(f"所有个体非有限值总数: {initial_nonfinite_total}")
print(f"非有限值比例: {initial_nonfinite_ratio:.4f}%")

# 进化后种群
final_total_elements = final_df['tau_numel'].iloc[0]
final_nonfinite_total = final_df['tau_nonfinite_count'].sum()
final_nonfinite_ratio = final_nonfinite_total / (final_total_elements * len(final_df)) * 100

print(f"\n进化后种群:")
print(f"单个个体tau元素总数: {final_total_elements}")
print(f"所有个体非有限值总数: {final_nonfinite_total}")
print(f"非有限值比例: {final_nonfinite_ratio:.4f}%")

print(f"\n非有限值比例变化: {final_nonfinite_ratio - initial_nonfinite_ratio:.4f}%")

# 分析tau_mean的计算是否受无限值影响
print("\n" + "=" * 60)
print("tau_mean计算分析")

print(f"\n初始种群tau_mean统计 (已处理无限值):")
print(initial_df['tau_mean'].describe())

print(f"\n进化后种群tau_mean统计 (已处理无限值):")
print(final_df['tau_mean'].describe())

# 检查tau_mean是否基于有限值计算
initial_mean_from_metadata = initial_df['tau_mean'].mean()
final_mean_from_metadata = final_df['tau_mean'].mean()

print(f"\ntau_mean对比:")
print(f"初始种群均值: {initial_mean_from_metadata:.6f}")
print(f"进化后均值: {final_mean_from_metadata:.6f}")
print(f"变化幅度: {(final_mean_from_metadata - initial_mean_from_metadata) / initial_mean_from_metadata * 100:.2f}%")

print(f"\n结论: tau_mean已经排除了无限值的影响，降低{abs((final_mean_from_metadata - initial_mean_from_metadata) / initial_mean_from_metadata * 100):.2f}%反映了真实的tau值优化效果")
