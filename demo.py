import numpy as np

# 假设 output_groundtruth 是一个不规则的序列，例如：
output_groundtruth = [[1, 2], [3, 4, 5]]

# 在创建数组时指定 dtype=object
output_groundtruth_array = np.array(output_groundtruth, dtype=object).reshape(-1, 1)

# 打印转换后的数组
print(output_groundtruth_array)





# 假设有一个Python列表
my_list = [1, 2, 3, 4, 5]

# 将列表转化为NumPy数组
my_array = np.array(my_list)

# 打印转换后的数组
print(my_array)


trueLabel = [1,2,3,4,5,6]
# trueLabel1 = trueLabel.tolist()
print(trueLabel)
# print(trueLabel1)




# 创建一个NumPy数组
arr = np.array([[3, 1, 2],[5,3,6]])

# 使用np.argsort()获取排序后的索引数组
sorted_indices = np.argsort(-arr,axis=1)

print("排序后的索引：", sorted_indices)
print("使用索引数组排序：", arr[sorted_indices[0]],arr[sorted_indices[1]])

