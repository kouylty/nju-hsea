import os
import json
import numpy as np
import pandas as pd

def decode_soln(soln):
    soln = np.array(soln, dtype=int)
    a_vals = soln // 2 + 1
    b_vals = soln % 2 + 1
    return np.stack((a_vals, b_vals), axis=1)

# 文件路径
# json_path = 'earth_904_ea_output.json'
dims = 902
folder_path = f'data/earth_{dims}'
seed = 2024
alg_name = 'sa'
file_name = f'{alg_name}_output_{seed}.json'
json_path = os.path.join(folder_path, file_name)
print(json_path)

with open(json_path, 'r') as f:
    data = json.load(f)

x_best = data['x_best']  # 获取编码的一维解
# print(x_best[:10])

# 解码
decoded_soln = decode_soln(x_best)

# 打印结果
print("解码后的解：")
print(decoded_soln[:10])


df_data = pd.read_csv('1aZeroMean_new.csv')  # columns: Event, Type, Level-Placed, Fossil name

# 为快速查找建立索引
df_data_keyed = df_data.set_index(['Event', 'Type'])

# 遍历 decoded_soln，并查找映射
results = []
for event, type_ in decoded_soln:
    try:
        row = df_data_keyed.loc[(event, type_)]
        results.append({
            'Event': event,
            'Type': type_,
            'Level-Placed': row['Level-Placed'],
            'Fossil name': row['Fossil name']
        })
    except KeyError:
        print(f"警告：在 data.csv 中未找到 (Event={event}, Type={type_}) 的映射")
        results.append({
            'Event': event,
            'Type': type_,
            'Level-Placed': None,
            'Fossil name': None
        })

# 保存为新的 CSV 文件
df_result = pd.DataFrame(results)
output_filename = f'earth_{dims}_{alg_name}_output.csv'
df_result.to_csv(output_filename, index=False)
print(f"映射并保存完成：{output_filename}")