
import json
import numpy as np
import re
import ast

#加载保存的exercises_data
exercises_data = np.load('./data/exercises_data_order.npy', allow_pickle=True).item()

# 加载保存的Q矩阵
Q_matrix_loaded = np.load('./data/Q_matrix_order.npy')


def read_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    data = []
    for i in range(0, len(lines), 5):
        student_data = tuple(lines[i:i + 3])  # 保持前3行不变
        exer_ids = [int(x) for x in lines[i + 2].strip().split(',')]  # 解析第三行的数据
        q_matrix_rows = []  # 存储查询到的 q_matrix_row
        for exer_id in exer_ids:
            if exer_id == 0:
                q_matrix_row = np.zeros_like(Q_matrix_loaded[0])  # 创建与 Q_matrix_loaded[0] 相同形状的全为 0 的 ndarray
            else:
                exer_index = list(exercises_data.keys()).index(exer_id)
                q_matrix_row = Q_matrix_loaded[exer_index]  # 根据第三行的数据查询Q_matrix_loaded
            q_matrix_rows.append(q_matrix_row)  # 将查询到的 q_matrix_row 添加到列表中

        # 解析数据  把原始格式的知识点只保留第一个进行转化
        data_concept = lines[i + 3]
        #解析数据
        # 使用正则表达式提取每个列表中的第一个数字
        numbers = re.findall(r'\[(\d+)', data_concept)

        # 将提取的数字组合成一个新的字符串
        result_concept = ','.join(numbers)
        data.append(student_data + (tuple(q_matrix_rows), result_concept,lines[i + 4]))  # 添加查询结果到第四行，保持第五行不变

    return data


if __name__ == '__main__':
    data=read_file("./data/test_data_order.txt")
    pass