# import os
import pandas as pd
import torch

print('hello world')
# os.makedirs(os.path.join('/Users/hellomax/Desktop/pytoch2', 'data'), exist_ok=True)
# data_file = os.path.join('/Users/hellomax/Desktop/pytoch2', 'data', 'house_tiny.csv')
# with open(data_file, 'w') as f:
#     f.write('NumRooms,Alley,Price\n')  # 列名
#     f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
#     f.write('2,NA,106000\n')
#     f.write('4,NA,178100\n')
#     f.write('NA,NA,140000\n')
data = pd.read_csv('/Users/hellomax/Desktop/pytoch2/data/house_tiny.csv')
print("**************************")
print(data)
print("**************************")
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
print(inputs)
# print("**************************")
# inputs = pd.get_dummies(inputs, dummy_na=True)
# print(inputs)
# # print(outputs)
#
# print("**************************")
# inputs2 = inputs.iloc[:, 0:1]
# print(inputs2)
# print("**************************")
# inputs2 = inputs2.fillna(inputs2.mean())
# print(inputs2)
# print("**************************")
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
# print("**************************")
# # print(inputs.select_dtypes(include = ['float64']))
inputs = inputs.fillna(inputs.select_dtypes(include=['float64']).mean())
print(inputs)
X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(X, y)
