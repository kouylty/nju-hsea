import numpy as np 
import pandas as pd
import os

def encode_soln(data: np.ndarray) -> np.ndarray:
    result = []
    for i in range(data.shape[0]):
        a_val = data[i, 0]
        b_val = data[i, 1]
        encoded_val = (a_val - 1) * 2 + (b_val - 1)
        result.append(encoded_val)
    return np.array(result)

np.random.seed(42)
from algorithms._utils import init_v3_sampler, permutation_sampler
# init_sample = init_v3_sampler(n=10, dims=2574)
# print(init_sample.shape)

# init_permutation = permutation_sampler(n=20, dims=2538)



# # 2538 数据集上测试
# dir_path = './benchmarks/CONOPLib_2538/'
# file_path = os.path.join(dir_path, '28-1-soln.dat')
# data = np.loadtxt(file_path).astype(int)
# # print(data)

# res = encode_soln(data)

# from benchmarks.CONOPLib_2538.call import EarthBenchmark as EarthBenchmark_2538
# benchmark_2538 = EarthBenchmark_2538(file_path=None)
# # print(benchmark_2538(init_sample))
# total_pen = 0
# for permutation in init_sample:
#     tmp_pen = benchmark_2538(permutation)
#     total_pen += tmp_pen
#     print(f"penalty: {tmp_pen}")

# print(f"Avg: {total_pen / len(init_sample)}")

# total_pen = 0
# for permutation in init_permutation:
#     total_pen += benchmark_2538(permutation)

# print(f"Avg: {total_pen / len(init_permutation)}")





# 2574 数据集上测试
dir_path = './benchmarks/CONOPLib_2574/'
file_path = os.path.join(dir_path, '1-soln.dat')
data = np.loadtxt(file_path).astype(int)
print(data)

res = encode_soln(data)

from benchmarks.CONOPLib_2574.call import EarthBenchmark as EB
eb = EB()
print(eb(res))

# from benchmarks.CONOPLib_2574.call import EarthBenchmark as EarthBenchmark_2574
# benchmark_2574 = EarthBenchmark_2574(file_path=None)
# # print(benchmark_2538(init_sample))
# total_pen = 0
# for permutation in init_sample:
#     tmp_pen = benchmark_2574(permutation)
#     total_pen += tmp_pen
#     print(f"penalty: {tmp_pen}")

# print(f"Avg: {total_pen / len(init_sample)}")