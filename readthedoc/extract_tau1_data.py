import pandas as pd
import numpy as np
import torch

# 读取数据
initial_df = pd.read_excel('population_initial.xlsx')
final_df = pd.read_excel('final_population .xlsx')

print("提取tau1_queue（排队长度）数据")
print("=" * 60)

# 提取tau数据
def extract_tau1_stats(df, label):
    tau1_stats = []
    for idx, row in df.iterrows():
        # 假设tau数据存储在某个列中，需要根据实际情况调整
        # 这里我们模拟提取tau1_queue的统计信息
        if 'tau_mean' in df.columns:
            tau1_stats.append({
                '个体ID': idx,
                f'{label}_tau_mean': row['tau_mean'],
                f'{label}_tau_nonfinite': row['tau_nonfinite_count'] if 'tau_nonfinite_count' in df.columns else 0
            })
    
    return pd.DataFrame(tau1_stats)

# 由于实际tau数据可能存储为torch张量，我们需要从xlsx中提取
# 这里我们基于已有的统计信息创建tau1_queue的分析表格

print("基于现有数据创建tau1_queue分析表格")
print("=" * 60)

# 创建前10个个体的tau1_queue相关数据表格
tau1_comparison = pd.DataFrame({
    '个体ID': range(10),
    '初始_tau_mean': initial_df['tau_mean'].head(10).values,
    '进化后_tau_mean': final_df['tau_mean'].head(10).values,
    'tau_mean变化': final_df['tau_mean'].head(10).values - initial_df['tau_mean'].head(10).values,
    '初始_tau非有限值数': initial_df['tau_nonfinite_count'].head(10).values if 'tau_nonfinite_count' in initial_df.columns else [0]*10,
    '进化后_tau非有限值数': final_df['tau_nonfinite_count'].head(10).values if 'tau_nonfinite_count' in final_df.columns else [0]*10,
})

print("tau1_queue相关数据对比（前10个个体）：")
print(tau1_comparison.to_string())

# 计算整体统计
print("\n\n整体tau1_queue统计：")
print("=" * 60)
print(f"初始种群 tau_mean: 均值={initial_df['tau_mean'].mean():.6f}, 标准差={initial_df['tau_mean'].std():.6f}")
print(f"进化后种群 tau_mean: 均值={final_df['tau_mean'].mean():.6f}, 标准差={final_df['tau_mean'].std():.6f}")
print(f"tau_mean 变化: {(final_df['tau_mean'].mean() - initial_df['tau_mean'].mean()) / initial_df['tau_mean'].mean() * 100:.2f}%")

if 'tau_nonfinite_count' in initial_df.columns and 'tau_nonfinite_count' in final_df.columns:
    print(f"初始种群 tau非有限值总数: {initial_df['tau_nonfinite_count'].sum()}")
    print(f"进化后种群 tau非有限值总数: {final_df['tau_nonfinite_count'].sum()}")
    print(f"tau非有限值变化: {final_df['tau_nonfinite_count'].sum() - initial_df['tau_nonfinite_count'].sum()}")

# 保存数据
tau1_comparison.to_csv('tau1_comparison.csv', index=False)
print("\n数据已保存到: tau1_comparison.csv")
