import pandas as pd
import numpy as np
import pickle

# 读取数据
initial_df = pd.read_excel('population_initial.xlsx')
final_df = pd.read_excel('final_population .xlsx')

print("检查原始tau数据中的无限值情况")
print("=" * 60)

# 检查是否有tau列
print("\n初始种群列名:", initial_df.columns.tolist())
print("进化后种群列名:", final_df.columns.tolist())

# 尝试从xlsx文件中提取tau的详细数据
# 检查是否有tau相关的列
tau_columns = [col for col in initial_df.columns if 'tau' in col.lower()]
print(f"\n包含'tau'的列: {tau_columns}")

# 如果有tau列，分析其内容
if 'tau' in initial_df.columns:
    print("\n初始种群tau列存在")
    print(f"tau数据类型: {type(initial_df['tau'].iloc[0])}")
    
    # 尝试解析tau数据
    sample_tau = initial_df['tau'].iloc[0]
    if isinstance(sample_tau, str):
        print("tau数据是字符串格式，尝试解析...")
        try:
            # 尝试eval解析
            tau_array = eval(sample_tau)
            print(f"解析后tau形状: {tau_array.shape if hasattr(tau_array, 'shape') else 'N/A'}")
            print(f"解析后tau类型: {type(tau_array)}")
            
            # 检查无限值
            if isinstance(tau_array, np.ndarray):
                inf_count = np.isinf(tau_array).sum()
                nan_count = np.isnan(tau_array).sum()
                print(f"无限值数量: {inf_count}")
                print(f"NaN值数量: {nan_count}")
                
                # 分析两个参数
                if tau_array.ndim >= 3 and tau_array.shape[-1] >= 2:
                    print(f"\ntau两个参数分析:")
                    print(f"参数1 (排队长度) 统计:")
                    param1 = tau_array[..., 0]
                    print(f"  无限值: {np.isinf(param1).sum()}")
                    print(f"  NaN值: {np.isnan(param1).sum()}")
                    print(f"  均值: {np.mean(param1[~np.isinf(param1) & ~np.isnan(param1)])}")
                    
                    print(f"参数2 (离目的地距离) 统计:")
                    param2 = tau_array[..., 1]
                    print(f"  无限值: {np.isinf(param2).sum()}")
                    print(f"  NaN值: {np.isnan(param2).sum()}")
                    print(f"  均值: {np.mean(param2[~np.isinf(param2) & ~np.isnan(param2)])}")
        except Exception as e:
            print(f"解析失败: {e}")
    else:
        print(f"tau数据类型: {type(sample_tau)}")
        if hasattr(sample_tau, 'shape'):
            print(f"tau形状: {sample_tau.shape}")

# 检查进化后种群的tau
if 'tau' in final_df.columns:
    print("\n进化后种群tau列存在")
    print(f"tau数据类型: {type(final_df['tau'].iloc[0])}")
    
    sample_tau = final_df['tau'].iloc[0]
    if isinstance(sample_tau, str):
        print("tau数据是字符串格式，尝试解析...")
        try:
            tau_array = eval(sample_tau)
            print(f"解析后tau形状: {tau_array.shape if hasattr(tau_array, 'shape') else 'N/A'}")
            
            if isinstance(tau_array, np.ndarray):
                inf_count = np.isinf(tau_array).sum()
                nan_count = np.isnan(tau_array).sum()
                print(f"无限值数量: {inf_count}")
                print(f"NaN值数量: {nan_count}")
                
                # 分析两个参数
                if tau_array.ndim >= 3 and tau_array.shape[-1] >= 2:
                    print(f"\ntau两个参数分析:")
                    print(f"参数1 (排队长度) 统计:")
                    param1 = tau_array[..., 0]
                    print(f"  无限值: {np.isinf(param1).sum()}")
                    print(f"  NaN值: {np.isnan(param1).sum()}")
                    print(f"  均值: {np.mean(param1[~np.isinf(param1) & ~np.isnan(param1)])}")
                    
                    print(f"参数2 (离目的地距离) 统计:")
                    param2 = tau_array[..., 1]
                    print(f"  无限值: {np.isinf(param2).sum()}")
                    print(f"  NaN值: {np.isnan(param2).sum()}")
                    print(f"  均值: {np.mean(param2[~np.isinf(param2) & ~np.isnan(param2)])}")
        except Exception as e:
            print(f"解析失败: {e}")

# 检查是否有pickle文件
print("\n" + "=" * 60)
print("检查是否有pickle文件包含tau数据")
import os
pickle_files = [f for f in os.listdir('.') if f.endswith('.pkl') or f.endswith('.pickle')]
print(f"发现pickle文件: {pickle_files}")

for pkl_file in pickle_files:
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        print(f"\n{pkl_file}内容:")
        print(f"类型: {type(data)}")
        if isinstance(data, dict):
            print(f"键: {list(data.keys())}")
            if 'tau' in data:
                tau_data = data['tau']
                print(f"tau形状: {tau_data.shape if hasattr(tau_data, 'shape') else 'N/A'}")
                if isinstance(tau_data, np.ndarray):
                    print(f"无限值: {np.isinf(tau_data).sum()}")
                    print(f"NaN值: {np.isnan(tau_data).sum()}")
    except Exception as e:
        print(f"读取{pkl_file}失败: {e}")
