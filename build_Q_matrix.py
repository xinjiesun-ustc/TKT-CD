import json
import numpy as np

# 读取log_data.json文件
with open('./data/log_data_order.json', 'r') as file:
    log_data = json.load(file)




# 提取知识点和练习题数据
knowledge_points = set()
exercises_data = {}

for entry in log_data:
    logs = entry['logs']
    for log in logs:
        exer_id = log['problem_id']
        knowledge_code = log['skill_id']

        # 将知识点添加到集合中
        knowledge_points.update(knowledge_code)

        # 将练习题数据转换为知识点列表
        knowledge_list = [int(k) for k in knowledge_code]
        exercises_data[exer_id] = knowledge_list

# 构建Q矩阵
num_exercises = len(exercises_data)
num_knowledge_points = len(knowledge_points)

Q_matrix = np.zeros((num_exercises, num_knowledge_points))   #初始化为0
# # 使用 np.full 创建一个形状为 (num_exercises, num_knowledge_points) 的数组，并将所有值初始化为 0.03
# Q_matrix = np.full((num_exercises, num_knowledge_points), 0.03)  #让每一个知识点都不为0 也就是知识点之间 总有千丝万缕的联系

# 将新行插入到Q矩阵的开头

knowledge_points_map = {k: i for i, k in enumerate(sorted(knowledge_points))}




for i, (exer_id, knowledge_list) in enumerate(exercises_data.items()):
    for k in knowledge_list:
        Q_matrix[i, knowledge_points_map[k]] = 1


# 保存Q矩阵到文件
np.save('./data/Q_matrix_order.npy', Q_matrix)
np.save('./data/exercises_data_order.npy', exercises_data)


# 加载保存的Q矩阵
# Q_matrix_loaded = np.load('./data/Q_matrix.npy')

# 查询题目编号为5、7、10和100的Q矩阵
# exer_ids_to_query = [145,17746]
# for exer_id in exer_ids_to_query:
#     if exer_id in exercises_data:
#         exer_index = list(exercises_data.keys()).index(exer_id)
#         q_matrix_row = Q_matrix_loaded[exer_index]
#         print(f"题目编号为{exer_id}的Q矩阵：")
#         print(q_matrix_row)
#     else:
#         print(f"题目编号为{exer_id}的Q矩阵不存在。")
