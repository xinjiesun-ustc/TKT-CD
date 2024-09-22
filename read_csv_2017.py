import pandas as pd
import json
from tqdm import tqdm

# 读取数据
data = pd.read_csv('./data/anonymized_full_release_competition_dataset.csv',
                   usecols=['startTime', 'studentId', 'problemId', 'skill', 'correct'],
                   encoding='ISO-8859-1').dropna(subset=['skill'])

# 将 skill 转换为整数
# 过滤掉包含特定字符串的行


data['skill'] = data['skill'].astype(int)

# 对 problemId、studentId 和 skill 重新进行编号
data['problemId'] = pd.factorize(data['problemId'])[0] + 1
data['studentId'] = pd.factorize(data['studentId'])[0] + 1
data['skill'] = pd.factorize(data['skill'])[0] + 1

# 按照 startTime 进行升序排列
data = data.sort_values(by='startTime')

# 按照 problemId 进行分组，并将每个 problemId 下的 unique skill 用逗号分隔开并放在一个列表中
grouped_data = data.groupby('problemId')['skill'].unique().apply(
    lambda x: '[' + ','.join(map(str, x)) + ']').reset_index()

# 将数据按照 studentId 进行分组
grouped_user = data.groupby('studentId')

result = []

# 使用 tqdm 创建处理进度条
for studentId, user_data in tqdm(grouped_user, desc="Processing data"):
    user_logs = []
    log_num = len(user_data)

    # 遍历用户的每条记录
    for index, row in user_data.iterrows():
        problemId = int(row['problemId'])  # 确保 problemId 是整数
        correct = int(row['correct'])  # 确保 correct 是整数

        # 在 grouped_data 中查找对应 problemId 的 skill
        skill_id_str = grouped_data[grouped_data['problemId'] == problemId]['skill'].values[0]

        # 将 skill 字符串转换为整数列表
        skill = [int(skill) for skill in skill_id_str.strip('[]').split(',')]

        # 构建 log 字典
        log = {
            "problemId": problemId,
            "correct": correct,
            "skill": skill
        }

        user_logs.append(log)

    # 构建用户字典
    user_dict = {
        "studentId": int(studentId),  # 确保 studentId 是整数
        "log_num": log_num,
        "logs": user_logs
    }

    result.append(user_dict)

# 将结果以 JSON 格式输出
json_result = json.dumps(result, indent=4)

# 保存结果到文件
with open('./data/log_data_2017_order.json', 'w') as file:
    file.write(json_result)

print("Data saved to log_data_order.json file.")