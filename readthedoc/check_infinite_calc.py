import pandas as pd
import numpy as np

# 读取数据
initial_df = pd.read_excel('population_initial.xlsx')
final_df = pd.read_excel('final_population .xlsx')

print("检查无限值计算逻辑")
print("=" * 60)

# 检查tau_numel的含义
print("\ntau_numel分析:")
print(f"初始种群tau_numel: {initial_df['tau_numel'].iloc[0]}")
print(f"进化后种群tau_numel: {final_df['tau_numel'].iloc[0]}")

# 检查tau_nonfinite_count
print(f"\n初始种群tau_nonfinite_count统计:")
print(initial_df['tau_nonfinite_count'].describe())

print(f"\n进化后种群tau_nonfinite_count统计:")
print(final_df['tau_nonfinite_count'].describe())

# 重新计算正确的比例
print("\n" + "=" * 60)
print("重新计算无限值比例")

# 初始种群
initial_tau_numel = initial_df['tau_numel'].iloc[0]
initial_nonfinite_sum = initial_df['tau_nonfinite_count'].sum()
initial_total_elements = initial_tau_numel * len(initial_df)
initial_ratio = initial_nonfinite_sum / initial_total_elements * 100

print(f"\n初始种群:")
print(f"单个个体tau元素数: {initial_tau_numel}")
print(f"所有个体非有限值总数: {initial_nonfinite_sum}")
print(f"所有个体总元素数: {initial_total_elements}")
print(f"非有限值比例: {initial_ratio:.4f}%")

# 进化后种群
final_tau_numel = final_df['tau_numel'].iloc[0]
final_nonfinite_sum = final_df['tau_nonfinite_count'].sum()
final_total_elements = final_tau_numel * len(final_df)
final_ratio = final_nonfinite_sum / final_total_elements * 100

print(f"\n进化后种群:")
print(f"单个个体tau元素数: {final_tau_numel}")
print(f"所有个体非有限值总数: {final_nonfinite_sum}")
print(f"所有个体总元素数: {final_total_elements}")
print(f"非有限值比例: {final_ratio:.4f}%")

print(f"\n非有限值比例变化: {final_ratio - initial_ratio:.4f}%")

# 检查单个个体的无限值比例
print("\n" + "=" * 60)
print("单个个体的无限值比例分析")

initial_avg_nonfinite = initial_df['tau_nonfinite_count'].mean()
initial_avg_ratio = initial_avg_nonfinite / initial_tau_numel * 100

final_avg_nonfinite = final_df['tau_nonfinite_count'].mean()
final_avg_ratio = final_avg_nonfinite / final_tau_numel * 100

print(f"\n初始种群平均每个个体:")
print(f"非有限值数量: {initial_avg_nonfinite:.2f}")
print(f"非有限值比例: {initial_avg_ratio:.4f}%")

print(f"\n进化后种群平均每个个体:")
print(f"非有限值数量: {final_avg_nonfinite:.2f}")
print(f"非有限值比例: {final_avg_ratio:.4f}%")

print(f"\n平均比例变化: {final_avg_ratio - initial_avg_ratio:.4f}%")
