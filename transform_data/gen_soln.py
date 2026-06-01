import os
import json
import numpy as np
import pandas as pd
from algorithms._utils import encode_soln, decode_soln
from algorithms._ea_operator import *

dir_path = "/data/trx/gzx/EarthBench/data/conop_ea_soln_4w"

file_list = os.listdir(dir_path)

for i, file_name1 in enumerate(file_list):
    parent1_name = file_name1.split('.')[0]
    parent1_path = os.path.join(dir_path, file_name1)
    data = np.loadtxt(parent1_path).astype(int)

    # print(data.shape)
    parent1 = encode_soln(data)


    seed_list = list(range(42, 43))
    for seed in seed_list:
        np.random.seed(seed)

        for j, file_name2 in enumerate(file_list):
            if i == j:
                continue
            
            output_dir_path = f"./data/offsprings_2574_crossover_all_0711/{parent1_name}"
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)
            parent2_name = file_name2.split('.')[0]
            parent_path = os.path.join(dir_path, file_name2)
            data = np.loadtxt(parent_path).astype(int)
            parent2 = encode_soln(data)

            offspring = pmx_crossover(parent1, parent2)
            # offspring = order_crossover(parent1, parent2)
            # offspring = cycle_crossover(parent1, parent2)

            # Mutation
            offspring = insert_mutation(offspring, repeats=1)

            offspring_list = offspring.tolist() if isinstance(offspring, np.ndarray) else offspring

            decoded_soln = decode_soln(offspring_list)

            n = decoded_soln.shape[0]
            index_col = np.arange(1, n + 1).reshape(-1, 1)

            # 拼接数据：[a_vals, b_vals, index]
            final_data = np.hstack((decoded_soln, index_col))

            # 转换为DataFrame以便保存为.dat文件
            df = pd.DataFrame(final_data, columns=["a_val", "b_val", "index"])

            output_dat_path = os.path.join(output_dir_path, f'{parent1_name}x{parent2_name}_seed{seed}.dat')

            df.to_csv(output_dat_path, sep=' ', header=False, index=False)

            print(f"Saved to {output_dat_path}")

# import os
# import json
# import numpy as np
# import pandas as pd
# from algorithms._utils import encode_soln, decode_soln
# from algorithms._ea_operator import *

# dir_path = "/data/trx/gzx/EarthBench/benchmarks/CONOPLib_2574"

# for base_idx in range(1,21):
#     parent1_path = os.path.join(dir_path, f"{base_idx}-soln.dat")
#     data = np.loadtxt(parent1_path).astype(int)

#     # print(data.shape)
#     parent1 = encode_soln(data)


#     seed_list = list(range(42, 43))
#     for seed in seed_list:
#         np.random.seed(seed)

#         for idx in range(1, 21):
#             if idx == base_idx:
#                 continue
#             output_dir_path = f"./data/offsprings_2574_crossover_all/{base_idx}"
#             if not os.path.exists(output_dir_path):
#                 os.makedirs(output_dir_path)
#             parent_path = os.path.join(dir_path, f"{idx}-soln.dat")
#             data = np.loadtxt(parent_path).astype(int)
#             parent2 = encode_soln(data)

#             offspring = pmx_crossover(parent1, parent2)
#             # offspring = order_crossover(parent1, parent2)
#             # offspring = cycle_crossover(parent1, parent2)

#             # Mutation
#             offspring = insert_mutation(offspring, repeats=1)

#             offspring_list = offspring.tolist() if isinstance(offspring, np.ndarray) else offspring

#             decoded_soln = decode_soln(offspring_list)

#             # json_data = {"x_best": offspring_list}

#             n = decoded_soln.shape[0]
#             index_col = np.arange(1, n + 1).reshape(-1, 1)

#             # 拼接数据：[a_vals, b_vals, index]
#             final_data = np.hstack((decoded_soln, index_col))

#             # 转换为DataFrame以便保存为.dat文件
#             df = pd.DataFrame(final_data, columns=["a_val", "b_val", "index"])

#             output_dat_path = os.path.join(output_dir_path, f'{base_idx}x{idx}_seed{seed}.dat')

#             df.to_csv(output_dat_path, sep=' ', header=False, index=False)

#             print(f"Saved to {output_dat_path}")