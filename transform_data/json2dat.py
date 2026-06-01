import pandas as pd
import numpy as np
import json
import os
from algorithms._utils import decode_soln

def save_json2dat(input_file_path, output_file_name, output_file_dir=None):
    with open(input_file_path, 'r') as file:
        data = json.load(file)

    x_best = np.array(data['x_best'])
    y_best = data['y_best']

    decoded_soln = decode_soln(x_best)


    # 索引列
    n = decoded_soln.shape[0]
    index_col = np.arange(1, n + 1).reshape(-1, 1)

    # 拼接数据：[a_vals, b_vals, index]
    final_data = np.hstack((decoded_soln, index_col))

    # 转换为DataFrame以便保存为.dat文件
    df = pd.DataFrame(final_data, columns=["a_val", "b_val", "index"])

    if output_file_dir != None:
        # 保存为.dat文件（空格分隔）
        output_dat_dir_path = os.path.join(output_file_dir, 'dat')
        if not os.path.exists(output_dat_dir_path):
            os.makedirs(output_dat_dir_path)
        output_dat_path = os.path.join(output_dat_dir_path, output_file_name)

        df.to_csv(output_dat_path, sep=' ', header=False, index=False)
    else:
        df.to_csv(output_file_name, sep=' ', header=False, index=False)


dim = 2574
# seed_list = [i for i in range(2022, 2026)]

input_file_dir = 'data/earth_2574/tmp_best'
output_file_dir = 'data/earth_2574/tmp_best'
for root, dirs, files in os.walk(input_file_dir):
    for file in files:
        
        input_file_path = os.path.join(root, file)
        penalty_str = file.rsplit('.', 1)[0].split('_')[-1][1:]
        if eval(penalty_str) > 33636:
            continue
        output_file_name = 'soln-' + penalty_str + '.dat'
        save_json2dat(input_file_path, output_file_name, output_file_dir)

# for seed in seed_list:
#     try:    
#         dir_path = f'data/earth_{dim}'
#         file_path = os.path.join(dir_path, f'ea_output_{seed}.json')
#         print(file_path)

#         # 读取 JSON 文件
#         with open(file_path, 'r') as file:
#             data = json.load(file)

#         x_best = np.array(data['x_best'])
#         y_best = data['y_best']

#         decoded_soln = decode_soln(x_best)


#         # 索引列
#         n = decoded_soln.shape[0]
#         index_col = np.arange(1, n + 1).reshape(-1, 1)

#         # 拼接数据：[a_vals, b_vals, index]
#         final_data = np.hstack((decoded_soln, index_col))

#         # 转换为DataFrame以便保存为.dat文件
#         df = pd.DataFrame(final_data, columns=["a_val", "b_val", "index"])

#         # 保存为.dat文件（空格分隔）
#         output_dat_dir_path = os.path.join(dir_path, 'dat')
#         if not os.path.exists(output_dat_dir_path):
#             os.makedirs(output_dat_dir_path)
#         output_dat_path = os.path.join(output_dat_dir_path, f'ea_output_{seed}.dat')

#         df.to_csv(output_dat_path, sep=' ', header=False, index=False)

#         print(f"Saved to {output_dat_path}")
#     except FileNotFoundError:
#         print(f"File {file_path} not found. Skipping seed {seed}.")
#     except:
#         print(f"An error occurred while processing seed {seed}.")
#         continue
