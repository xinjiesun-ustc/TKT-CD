import numpy as np
import re

# 加载保存的exercises_data
exercises_data = np.load('./data/exercises_data_order.npy', allow_pickle=True).item()

# 加载保存的Q矩阵
Q_matrix_loaded = np.load('./data/Q_matrix_order.npy')

def process_single_data(lines, student_index, data_index):
    student_data = []
    for line_number in range(5):
        line = lines[student_index * 5 + line_number]
        line_data = line.strip().split(',')
        if line_number == 0:
            # 处理学生数据的第一行
            student_data_part = line_data[0] if line_data else ''
        elif line_number == 1:
            # 处理学生数据的第二行
            student_data_part = line_data[0] if line_data else ''
        elif line_number == 2:
            # 处理学生数据的第五行
            student_data_orginal= line_data[data_index] if data_index < len(line_data) else ''
            student_data.append(student_data_orginal)
        # elif line_number == 2:
            # 处理学生数据的第三行
            if data_index < len(line_data):
                exer_id = int(line_data[data_index])
                if exer_id == 0:
                    q_matrix_row = np.zeros_like(Q_matrix_loaded[0])
                else:
                    exer_index = list(exercises_data.keys()).index(exer_id)
                    q_matrix_row = Q_matrix_loaded[exer_index]
                student_data_part = q_matrix_row
            else:
                student_data_part = np.zeros_like(Q_matrix_loaded[0])
        elif line_number == 3:
            # 处理学生数据的第四行
            if data_index < len(line_data):
                data_concept = line_data[data_index]
                # 使用正则表达式提取每个列表中的第一个数字
                numbers = re.findall(r'\[(\d+)', data_concept)
                result_concept = ','.join(numbers)
                student_data_part = result_concept
            else:
                student_data_part = ''
        elif line_number == 4:
            # 处理学生数据的第五行
            student_data_part = line_data[data_index] if data_index < len(line_data) else ''
        student_data.append(student_data_part)

    return student_data

def read_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines() #ines = 30080  #30080 = 64*94*5  目的是避免舍弃最后一个批次19个学生的数据
        lines  = lines[0:30080]

    data = []
    num_students = len(lines) // 5  # 每个学生有5行数据
    num_data_points = len(lines[2].strip().split(','))  # 第三行的数据点数

    for data_index in range(num_data_points):
        for student_index in range(num_students):
            single_data = process_single_data(lines, student_index, data_index)
            data.append(single_data)

    return data

if __name__ == '__main__':
    data = read_file("./data/test_data_order.txt")
    for d in data:
        print(d)
