# DKT使用了embedding进行了复现

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,1,3"
from EduKTM import KTM
import logging
import torch
import torch.nn as nn
from torch.nn import Module, LSTM, Linear, Dropout
import tqdm
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)

class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


#新增 直接预测下一个知识点
def process_raw_pred_one(question, true_answer, answer):  #question, true_answer是一个学生所有的知识点和对应的答案，answer从第二个知识点开始的预测值
    mask = torch.zeros_like(question, dtype=torch.bool)
    mask[question != 0] = True  #找出一个学生所有知识点中非填充的知识点，为了下面的找真正知识点的真实值和预测值做准备
    count = torch.sum(mask)     #统计一个学生所有知识点中非填充的知识点的个数
    final_true_answer = torch.masked_select(true_answer[1:count], mask[1:count]).to(device) #[1:count]从第二个知识点对应的答案开始找
    final_answer = torch.masked_select(torch.flatten(answer)[0:count-1], mask[0:count-1]).to(device)#[0:count-1] 从第一个预测值开始使用，count-1是因为本来answer就是少一位 你可以用count=seqlen来举例，马上理解
    return final_answer, final_true_answer


#新增 直接预测当前知识点
def process_raw_pred_one_testset(question, true_answer, answer):  #question, true_answer是一个学生所有的知识点和对应的答案，answer从第二个知识点开始的预测值
    mask = torch.zeros_like(question, dtype=torch.bool)
    mask[question != 0] = True  #找出一个学生所有知识点中非填充的知识点，为了下面的找真正知识点的真实值和预测值做准备
    count = torch.sum(mask)     #统计一个学生所有知识点中非填充的知识点的个数
    final_true_answer = torch.masked_select(true_answer[0:count-1], mask[0:count-1]).to(device) #[1:count]从第二个知识点对应的答案开始找
    final_answer = torch.masked_select(torch.flatten(answer)[0:count-1], mask[0:count-1]).to(device)#[0:count-1] 从第一个预测值开始使用，count-1是因为本来answer就是少一位 你可以用count=seqlen来举例，马上理解
    return final_answer, final_true_answer

class myKT_DKT(KTM):
    def __init__(self, num_concepts,num_student,num_exercises, emb_size):
        super(myKT_DKT, self).__init__()
        self.num_cuestions = num_concepts
        self.num_student = num_student
        self.num_exercises = num_exercises
        self.dkt_model = DKT(num_concepts, num_student, num_exercises, emb_size).to(device)

        self.h1_list = []  # 用于保存每个批次的 h1

    def train(self, train_data, test_data=None, *, epoch: int, lr=0.002) -> ...:
        self.dkt_model.train()
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(self.dkt_model.parameters(), lr)
        auc_max = 0
        count_e =0
        for e in range(epoch):
            self.dkt_model.train()
            all_pred, all_target = torch.Tensor([]).to(device), torch.Tensor([]).to(device)
            for batch in tqdm.tqdm(train_data, "Epoch %s" % e):
                true_length,batch_stu_id,batch_p, batch_q_maritx,batch_concept,batch_a = batch
                # true_length, batch_p, batch_concept, batch_a = batch

                # 重新组织数据结构，使其符合要求 bs*seqlen*dim
                new_batch_q_maritx = []
                for i in range(len(batch_a)):
                    data = [item[i] for item in batch_q_maritx]
                    new_batch_q_maritx.append(data)

                # # 将 new_batch_c 中的数据转换为一个整体的 tensor，并发送到设备
                # new_batch_c_tensor = torch.stack([torch.stack(data) for data in new_batch_c]).to(device)

                # 将每个字符串转换为整数列表
                batch_stu_id = [list(map(int, si.split(','))) for si in batch_stu_id]
                batch_p = [list(map(int, kp.split(','))) for kp in batch_p]
                batch_concept = [list(map(int, c.split(','))) for c in batch_concept]
                batch_a = [list(map(int, answer.split(','))) for answer in batch_a]

                # 将 new_batch_c 转换为一个整体的 tensor，并发送到设备
                # new_batch_q_maritx= torch.stack([torch.stack(data) for data in new_batch_q_maritx]).to(device)
                # new_batch_c = [list(map(int, c.split(','))) for c in new_batch_c]


                # 将列表转换为张量（tensor）

                batch_stu_id = torch.tensor(batch_stu_id).to(device)
                batch_a = torch.tensor(batch_a).to(device)
                batch_p = torch.tensor(batch_p).to(device)
                batch_concept = torch.tensor(batch_concept).to(device)
                # new_batch_q_maritx = torch.tensor(new_batch_q_maritx).to(device)
                # new_batch_q_maritx = new_batch_q_maritx.to(device)
                # 将列表转换为张量
                new_batch_q_maritx = [torch.stack(sublist) for sublist in new_batch_q_maritx]
                new_batch_q_maritx = torch.stack(new_batch_q_maritx)

                # 将张量发送到设备
                new_batch_q_maritx = new_batch_q_maritx.to(device)

                # 新增
                batch_p_next = batch_p[:, 1:batch_p.shape[1]].to(device)
                new_batch_q_maritx_next = new_batch_q_maritx[:, 1:new_batch_q_maritx.shape[1]].to(device)
                batch_concept_next = batch_concept[:, 1:batch_concept.shape[1]].to(device)
                pred_y = self.dkt_model(batch_concept[:,0:batch_concept.shape[1]-1],new_batch_q_maritx[:,0:new_batch_q_maritx.shape[1]-1], batch_stu_id,batch_p[:,0:batch_p.shape[1]-1], batch_a[:,0:batch_a.shape[1]-1],batch_p_next, new_batch_q_maritx_next,batch_concept_next)  #forward
                batch_size = batch_concept.shape[0]
                for student in range(batch_size):
                    pred, truth = process_raw_pred_one(batch_p[student].to(device), batch_a[student].to(device),
                                                   pred_y[student].to(device))
                    all_pred = torch.cat([all_pred, pred])
                    all_target = torch.cat([all_target, truth.float().to(device)])


            print(f"预测长度{all_pred.size()}")
            loss = loss_function(all_pred, all_target)
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # self.dkt_model.apply_clipper()  # 保持单调性增加

            print("[Epoch %d] LogisticLoss: %.6f" % (e, loss))
            torch.save(self.dkt_model.h1_list, f'./temp_result/h1_epoch_{e}.pt')

            if test_data is not None:
                auc,acc = self.eval(test_data,epoch=e)
                print("[Epoch %d] auc acc: %.6f, %.6f" % (e, auc,acc))
                if auc > auc_max:
                    auc_max = auc
                    count_e = e + 1
        print(f"最大的auc是在第{count_e}轮出现的：{auc_max}")
        # 构造要写入文件的字符串
        output_string = f"最大的auc是在第{count_e}轮出现的：{auc_max}\n"

        # 指定文件路径
        file_path = "output.txt"

        # 将字符串写入文件
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(output_string)
            print("结果保存成功")

    def eval(self, test_data,epoch) -> float:
        self.dkt_model.eval()
        y_pred = torch.Tensor([]).to(device)
        y_truth = torch.Tensor([]).to(device)

        h1_list_loaded = torch.load(f'./temp_result/h1_epoch_{epoch}.pt')  # 加载保存的 h1 列表

        if not h1_list_loaded:
            raise ValueError(
                f"Loaded h1_list for epoch {epoch} is empty. Ensure that it is correctly saved during training.")
        print(f"Epoch {epoch}: Loaded h1_list with {len(h1_list_loaded)} items.")
        # 处理 batch 数据
        import copy
        self.dkt_model.h1_list = copy.deepcopy(h1_list_loaded)  # 深拷贝 h1_list
        if not self.dkt_model.h1_list:
            raise ValueError("h1_list is empty after deepcopy. Ensure that it is correctly loaded.")

        for batch in tqdm.tqdm(test_data):
            true_length, batch_stu_id, batch_p, batch_q_maritx, batch_concept, batch_a = batch
            # true_length, batch_p, batch_concept, batch_a = batch

            # 重新组织数据结构，使其符合要求 bs*seqlen*dim
            new_batch_q_maritx = []
            for i in range(len(batch_a)):
                data = [item[i] for item in batch_q_maritx]
                new_batch_q_maritx.append(data)

            # # 将 new_batch_c 中的数据转换为一个整体的 tensor，并发送到设备
            # new_batch_c_tensor = torch.stack([torch.stack(data) for data in new_batch_c]).to(device)

            # 将每个字符串转换为整数列表
            batch_stu_id = [list(map(int, si.split(','))) for si in batch_stu_id]
            batch_p = [list(map(int, kp.split(','))) for kp in batch_p]
            batch_concept = [list(map(int, c.split(','))) for c in batch_concept]
            batch_a = [list(map(int, answer.split(','))) for answer in batch_a]

            # 将 new_batch_c 转换为一个整体的 tensor，并发送到设备
            # new_batch_q_maritx= torch.stack([torch.stack(data) for data in new_batch_q_maritx]).to(device)
            # new_batch_c = [list(map(int, c.split(','))) for c in new_batch_c]

            # 将列表转换为张量（tensor）

            batch_stu_id = torch.tensor(batch_stu_id).to(device)
            batch_a = torch.tensor(batch_a).to(device)
            batch_p = torch.tensor(batch_p).to(device)
            batch_concept = torch.tensor(batch_concept).to(device)
            # new_batch_q_maritx = torch.tensor(new_batch_q_maritx).to(device)
            # new_batch_q_maritx = new_batch_q_maritx.to(device)
            # 将列表转换为张量
            new_batch_q_maritx = [torch.stack(sublist) for sublist in new_batch_q_maritx]
            new_batch_q_maritx = torch.stack(new_batch_q_maritx)

            # 将张量发送到设备
            new_batch_q_maritx = new_batch_q_maritx.to(device)

            # 新增
            batch_p_next = batch_p[:, 1:batch_p.shape[1]].to(device)
            new_batch_q_maritx_next = new_batch_q_maritx[:, 1:new_batch_q_maritx.shape[1]].to(device)
            batch_concept_next = batch_concept[:, 1:batch_concept.shape[1]].to(device)

            pred_y = self.dkt_model(batch_concept[:,0:batch_concept.shape[1]-1],new_batch_q_maritx[:,0:new_batch_q_maritx.shape[1]-1], batch_stu_id,batch_p[:,0:batch_p.shape[1]-1], batch_a[:,0:batch_a.shape[1]-1],batch_p_next, new_batch_q_maritx_next,batch_concept_next)  #forward
            batch_size = batch_p.shape[0]
            for student in range(batch_size):
                pred, truth = process_raw_pred_one(batch_p[student].to(device), batch_a[student].to(device),
                                                   pred_y[student].to(device))

                y_pred = torch.cat([y_pred, pred])
                y_truth = torch.cat([y_truth, truth])

        from sklearn.metrics import accuracy_score

        # 将PyTorch张量转换为NumPy数组
        y_truth_np = y_truth.cpu().detach().numpy()
        y_pred_np = y_pred.cpu().detach().numpy()
        y_pred_labels = (y_pred_np >= 0.5).astype(int)
        # 计算准确率
        accuracy = accuracy_score(y_truth_np, y_pred_labels)
        # print(f'Accuracy: {accuracy}')

        return roc_auc_score(y_truth.cpu().detach().numpy(), y_pred.cpu().detach().numpy()),accuracy

    def save(self, filepath):
        torch.save(self.dkt_model.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.dkt_model.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)





