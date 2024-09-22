###数据处理，分3步
#分别为：
#1.divide_data对数据进行分割，将学生练习记录低于min_log的记录删除，并对把每个记录分割成max_log_per_student长度，分割后的长度再次低于min_log的话，同样进行删除处理。 结果为：new_stus.json
#2. split_train_test_data 对每个学生的学习记录进行分割，前80%作为训练集（平时练习），后20%作为测试集（考试）。结果为：train_data.json
#3. 数据集对齐，并对数据集进行填0操作 。结果为：train_data.txt

import json

def divide_data(min_log=15, max_log_per_student=50):  # 09:50  12:100
    with open('data/log_data_order.json', encoding='utf8') as i_f:
        stus = json.load(i_f)

    new_stus = []
    exercise_ids = set()
    skill_ids = set()

    student_count = 1

    for stu in stus:
        if stu['log_num'] < min_log:
            continue

        num_slices = stu['log_num'] // max_log_per_student
        remaining_logs = stu['log_num'] % max_log_per_student

        if stu['log_num'] < max_log_per_student:
            continue

        for i in range(num_slices):
            new_stu = {
                'user_id': f"{student_count}",
                'log_num': max_log_per_student,
                'logs': stu['logs'][i * max_log_per_student:(i + 1) * max_log_per_student]
            }
            new_stus.append(new_stu)

            for log in new_stu['logs']:
                exercise_ids.add(log['problem_id'])
                skill_ids.update(log['skill_id'])  # 将元素添加到集合中

            student_count += 1

        if remaining_logs >= min_log:
            new_stu = {
                'user_id': f"{student_count}",
                'log_num': remaining_logs,
                'logs': stu['logs'][num_slices * max_log_per_student:]
            }
            new_stus.append(new_stu)

            for log in new_stu['logs']:
                exercise_ids.add(log['problem_id'])
                skill_ids.update(log['skill_id'])  # 将元素添加到集合中

            student_count += 1

    with open('data/new_stus_order.json', 'w', encoding='utf8') as output_file:
        json.dump(new_stus, output_file, indent=4, ensure_ascii=False)

    return student_count - 1, len(exercise_ids), len(skill_ids)


# 获取最终保留的学生数量、problem_id 的个数和 skill_id 中不重复数据的个数
final_student_count, problem_id_count, skill_id_count = divide_data(min_log=15, max_log_per_student=50)
print(f"最终保留了 {final_student_count} 个学生。")


import math


def split_train_test_data(input_file, train_file, test_file, train_ratio=0.8):
    with open(input_file, 'r', encoding='utf8') as f:
        data = json.load(f)

    train_data = []
    test_data = []

    for student_data in data:
        user_id = student_data['user_id']
        total_logs = student_data['log_num']
        split_index = math.floor(total_logs * train_ratio)

        train_student_data = {
            'user_id': user_id,
            'log_num': split_index,
            'logs': student_data['logs'][:split_index]
        }
        test_student_data = {
            'user_id': user_id,
            'log_num': total_logs - split_index,
            'logs': student_data['logs'][split_index:]
        }

        train_data.append(train_student_data)
        test_data.append(test_student_data)

    with open(train_file, 'w', encoding='utf8') as train_f:
        json.dump(train_data, train_f, indent=4, ensure_ascii=False)

    with open(test_file, 'w', encoding='utf8') as test_f:
        json.dump(test_data, test_f, indent=4, ensure_ascii=False)


# 划分训练集和测试集，每个学生数据的80%作为训练集，20%作为测试集
split_train_test_data('data/new_stus_order.json', 'data/train_data_order.json', 'data/test_data_order.json', train_ratio=0.8)

# import json
#
#
# def save_data_as_txt(input_file, output_file):
#     with open(input_file, 'r', encoding='utf8') as f:
#         data = json.load(f)
#
#     with open(output_file, 'w', encoding='utf8') as txt_file:
#         for i, student_data in enumerate(data):
#             txt_file.write(f"{student_data['log_num']}\n")
#             txt_file.write(f"{student_data['user_id']}\n")
#
#             problem_ids = []
#             skill_ids = []
#             corrects = []
#
#             for log in student_data['logs']:
#                 problem_ids.append(str(log['problem_id']))
#                 skill_ids.append(str(log['skill_id']))
#                 corrects.append(str(log['correct']))
#
#             txt_file.write(','.join(problem_ids) + '\n')
#             txt_file.write(','.join(skill_ids) + '\n')
#             txt_file.write(','.join(corrects) + '\n')
#
#
# # 将训练集数据保存为txt
# save_data_as_txt('data/train_data.json', 'data/train_data.txt')
#
# # 将测试集数据保存为txt
# save_data_as_txt('data/test_data.json', 'data/test_data.txt')

import json


def save_data_as_txt(input_file, output_file):
    with open(input_file, 'r', encoding='utf8') as f:
        data = json.load(f)

    with open(output_file, 'w', encoding='utf8') as txt_file:
        for student_data in data:
            log_num = student_data['log_num']
            txt_file.write(f"{log_num}\n")
            txt_file.write(f"{student_data['user_id']}\n")

            problem_ids = []
            skill_ids = []
            corrects = []

            for log in student_data['logs']:
                problem_ids.append(str(log['problem_id']))
                skill_ids.append(str(log['skill_id']))  # 不使用列表格式，直接添加字符串
                corrects.append(str(int(log['correct'])))  # 将'correct'字段改为整数


            ##下面这段代码是为了打乱测试集中数据的顺序
            # 将三个列表组合成一个列表的元组
            if "rand" in output_file:
                combined = list(zip(problem_ids, skill_ids, corrects))
                import  random
                # 打乱组合列表的顺序
                random.shuffle(combined)

                # 将打乱后的数据拆分回原来的三个列表
                problem_ids, skill_ids, corrects = zip(*combined)
                # 将结果转换回列表（因为 zip 返回的是元组）
                problem_ids = list(problem_ids)
                skill_ids = list(skill_ids)
                corrects = list(corrects)

            # 如果log_num小于40，对每个学生的记录进行填充0，直到达到40个数据为止
            max_step = 40  #09:50*0.8=40  12：100*0.8=80
            pad_length = max_step - log_num
            problem_ids += ['0'] * pad_length
            corrects += ['0'] * pad_length

            if pad_length > 0:
                skill_ids += [[0]] * pad_length

            txt_file.write(','.join([str(item) for item in problem_ids]) + '\n')
            txt_file.write(','.join([str(item) for item in skill_ids]) + '\n')
            txt_file.write(','.join([str(item) for item in corrects]) + '\n')


# 处理训练集
save_data_as_txt('data/train_data_order.json', 'data/train_data_order.txt')

# 处理测试集
save_data_as_txt('data/test_data_order.json', 'data/test_data_order.txt')

# save_data_as_txt('data/test_data_order.json', 'data/test_data_random.txt')  #测试打乱测试集










