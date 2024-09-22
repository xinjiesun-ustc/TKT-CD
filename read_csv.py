import pandas as pd
import json
from tqdm import tqdm

# 读取数据
data = pd.read_csv('./data/2009_skill_builder_data_corrected/skill_builder_data_corrected.csv',
                   usecols=['order_id', 'user_id', 'problem_id', 'skill_id', 'correct'],
                   encoding='ISO-8859-1').dropna(subset=['skill_id'])

# 将 skill_id 转换为整数
data['skill_id'] = data['skill_id'].astype(int)

# 对 problem_id、user_id 和 skill_id 重新进行编号
data['problem_id'] = pd.factorize(data['problem_id'])[0] + 1
data['user_id'] = pd.factorize(data['user_id'])[0] + 1
data['skill_id'] = pd.factorize(data['skill_id'])[0] + 1

# 按照 order_id 进行升序排列
data = data.sort_values(by='order_id')

# 按照 problem_id 进行分组，并将每个 problem_id 下的 unique skill_id 用逗号分隔开并放在一个列表中
grouped_data = data.groupby('problem_id')['skill_id'].unique().apply(
    lambda x: '[' + ','.join(map(str, x)) + ']').reset_index()

# 将数据按照 user_id 进行分组
grouped_user = data.groupby('user_id')

result = []

# 使用 tqdm 创建处理进度条
for user_id, user_data in tqdm(grouped_user, desc="Processing data"):
    user_logs = []
    log_num = len(user_data)

    # 遍历用户的每条记录
    for index, row in user_data.iterrows():
        problem_id = int(row['problem_id'])  # 确保 problem_id 是整数
        correct = int(row['correct'])  # 确保 correct 是整数

        # 在 grouped_data 中查找对应 problem_id 的 skill_id
        skill_id_str = grouped_data[grouped_data['problem_id'] == problem_id]['skill_id'].values[0]

        # 将 skill_id 字符串转换为整数列表
        skill_id = [int(skill) for skill in skill_id_str.strip('[]').split(',')]

        # 构建 log 字典
        log = {
            "problem_id": problem_id,
            "correct": correct,
            "skill_id": skill_id
        }

        user_logs.append(log)

    # 构建用户字典
    user_dict = {
        "user_id": int(user_id),  # 确保 user_id 是整数
        "log_num": log_num,
        "logs": user_logs
    }

    result.append(user_dict)

# 将结果以 JSON 格式输出
json_result = json.dumps(result, indent=4)

# 保存结果到文件
with open('./data/log_data_order.json', 'w') as file:
    file.write(json_result)

print("Data saved to log_data_order.json file.")