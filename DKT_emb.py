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

class DKT(Module):
    def __init__(self,  num_c, num_stu, num_e, emb_size, dropout=0.1):
        super().__init__()
        self.num_c = num_c
        self.num_stu = num_stu
        self.num_e = num_e
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.prednet_input_len = 502
        self.prednet_len1, self.prednet_len2 = 256, 123  # changeable

        self.lstm_layer = LSTM(379, 123, batch_first=True)
        self.lstm_layer1 = LSTM(379, 123, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        self.out_layer = Linear(502, 1)
        self.concepts_emb = nn.Embedding(num_c + 1, self.emb_size)
        self.answer_emb = nn.Embedding(2, self.emb_size)

        ####加入NeuralCD相关内容

        # 这里的三个全连接层的预测 在测试集的时候使用
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(123, 1)

        self.student_emb = nn.Embedding( self.num_stu+1, self.num_c)  # 给每一个学生进行嵌入
        self.k_difficulty_emb = nn.Embedding(self.num_e+1, self.num_c)  # 给每个题目的每个知识点都进行难度嵌入
        self.e_discrimination_emb = nn.Embedding(self.num_e+1, 1)  #每个题目一个区分度

        self.embedding_problem = nn.Embedding(self.num_e+1,self.hidden_size)
        self.embedding_concept = nn.Embedding(self.num_c + 1, 256)

        self.zengzhang_abality = nn.Linear(123, 123)

        self.h1_list = []  # 用于保存每个批次的 h1
        self.h2_list = []  # 用于保存每个批次的 h1
        self.count =1
        self.d_t = nn.Parameter(torch.randn(1))
        self.d_e = nn.Parameter(torch.randn(1))


        #做题状态门
        # 初始化权重和偏置
        self.W_z = torch.nn.Parameter(torch.randn(123, 502))
        self.W_r = torch.nn.Parameter(torch.randn(123, 502))
        self.W_h = torch.nn.Parameter(torch.randn(123, 502))
        self.b_z = torch.nn.Parameter(torch.zeros(123))
        self.b_r = torch.nn.Parameter(torch.zeros(123))
        self.b_h = torch.nn.Parameter(torch.zeros(123))

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def forward(self,concept, q_maritx,s_id,e_id,r,q_next,q_maritx_next,concept_next):
        e_next_emb = self.embedding_problem(q_next)
        e_emb = self.embedding_problem(e_id)
        concept_emb = self.embedding_concept(concept)
        concept_next_emb= self.embedding_concept(concept_next)

        s_proficiency =torch.sigmoid(self.student_emb(s_id))
        # 或者使用 torch.repeat() 方法将维度为1重复40次  为了和习题难度的维度进行匹配
        s_proficiency = s_proficiency.repeat(1, 39, 1)
        k_difficulty = torch.sigmoid(self.k_difficulty_emb(e_id))
        e_discrimination = torch.sigmoid(self.e_discrimination_emb(e_id))*self.d_e
        r_emb = self.answer_emb(r)
        # 打开一个 txt 文件来写入数据  观察q_e的数据
        # with open("./data/q_e_data.txt", "w") as file:
        #     for i in range(q_e.size(0)):
        #         for j in range(q_e.size(1)):
        #             for k in range(q_e.size(2)):
        #                 file.write(str(q_e[i, j, k].item()) + "\n")

        #Personalized    Dynamic  Growth   Gate（PDGG） 中文：个性化动态增长门
        q_maritx = q_maritx.float()
        # concatenated_embeddings_concept = torch.cat((q_maritx, e_emb), dim=-1)
        concept_abality = self.zengzhang_abality(q_maritx)  #知识运用能力
        concatenated_embeddings_concept = torch.cat((concept_abality*s_proficiency, r_emb), dim=-1) #concept_abality*s_proficiency  个性化的知识应用能力
        concatenated_embeddings_concept = concatenated_embeddings_concept.float()
        # h1, (h_n,c_n) = self.lstm_layer(concatenated_embeddings_concept)bing

        # # 假设 concatenated_embeddings_concept 是你的输入序列
        batch_size, seq_len, input_dim = concept_abality.size()
        hidden_size = 123  # 隐藏层维度可以根据需要调整

        # 初始化隐藏状态
        hidden = torch.zeros(batch_size, hidden_size).to(concatenated_embeddings_concept.device)
        outputs_zengzhang = []

        # 定义线性变换
        input_to_hidden = nn.Linear(379, hidden_size).to(concatenated_embeddings_concept.device)
        hidden_to_hidden = nn.Linear(hidden_size, hidden_size).to(concatenated_embeddings_concept.device)
        time_gate = nn.Linear(1, hidden_size).to(concatenated_embeddings_concept.device)

        # 处理每个时间步长
        for t in range(seq_len):
            time_step = torch.tensor([[t]], dtype=torch.float).to(concatenated_embeddings_concept.device)
            combined = input_to_hidden(concatenated_embeddings_concept[:, t, :]) + hidden_to_hidden(hidden)
            gate = torch.sigmoid(time_gate(time_step))
            hidden = torch.tanh(combined) * gate
            outputs_zengzhang.append(hidden.unsqueeze(1))

        # 拼接所有时间步长的输出
        outputs_zengzhang = torch.cat(outputs_zengzhang, dim=1)

        # 假设 concatenated_embeddings_concept 是你的输入张量



        if self.training:
            #保存序列学习的最后一个个性化的动态增长能力值
            h_n =  outputs_zengzhang[:, -1:, :]
            average_pooled_output1 = torch.mean(outputs_zengzhang, dim=1).unsqueeze(1)  # torch.mean or torch.max  [0]
            # last_hidden_state = h_n.repeat(1, 39, 1)
            # last_hidden_state = h_n.permute(1, 0, 2)  # 形状: (batch_size, 1, hidden_size)
            self.h1_list.append(average_pooled_output1)  # 保存每个批次的 h1
            #认知诊断核心功能
            input_x1 = e_discrimination * (self.d_t *s_proficiency +(1-self.d_t )*outputs_zengzhang - k_difficulty) * q_maritx  # 最大的auc是在第89轮出现的：0.811998077612642
            # input_x1 = (s_proficiency) * q_maritx
            desired_dtype = outputs_zengzhang.dtype
            concatenated_embeddings_concept2 = torch.cat((input_x1, r_emb), dim=-1)
            concatenated_embeddings_concept2 = concatenated_embeddings_concept2.float()

            #Adaptive   Cognitive   Control    Gate(ACCG)    中文：自适应认知控制门
            # 我们可以设计一个简单的门控机制，包含两个门：更新门和重置门。更新门控制新信息和旧信息的混合比例，重置门控制旧信息保留多少。
            # \begin  {align *}
            # z_t &= \sigma(W_z \cdot[x_t, h_ {t - 1}]),  \ \
            # r_t &= \sigma(W_r \cdot[x_t, h_ {t - 1}]),  \ \
            # \tilde {h}_t &= \tanh(W_h \cdot[x_t, r_t \odot  h_{t - 1}]),  \ \
            #  h_t &= (1 - z_t) \odot  h_{t - 1} + z_t \odot \tilde {h}_t
            # \end {align *}
            # 其中，$\sigma$表示sigmoid函数，$\tanh$表示tanh函数，$\odot$表示逐元素乘法。

            # 使用自定义的门控机制处理输入
            # 初始化隐藏状态
            batch_size = concatenated_embeddings_concept2.size(0)
            h = torch.zeros(batch_size, 123).to(concatenated_embeddings_concept2.device)
            seq_len = concatenated_embeddings_concept2.size(1)

            outputs = []
            # for t in range(seq_len):  #GRU方式实现
            #     x_t = concatenated_embeddings_concept2[:, t, :]
            #     combined = torch.cat((x_t, h), dim=-1)
            #
            #     z_t = torch.sigmoid(F.linear(combined, self.W_z, self.b_z))
            #     r_t = torch.sigmoid(F.linear(combined, self.W_r, self.b_r))
            #
            #     combined_reset = torch.cat((x_t, r_t * h), dim=-1)
            #     h_tilde = torch.tanh(F.linear(combined_reset, self.W_h, self.b_h))
            #
            #     h = (1 - z_t) * h + z_t * h_tilde
            #     outputs.append(h.unsqueeze(1))
            for t in range(seq_len):  # GRU方式实现
                x_t = concatenated_embeddings_concept2[:, t, :]
                combined = torch.cat((x_t, h), dim=-1)

                z_t = torch.sigmoid(F.linear(combined, self.W_z, self.b_z))
                h_tilde = torch.tanh(F.linear(combined, self.W_h, self.b_h))

                h = (1 - z_t) * h + z_t * h_tilde
                outputs.append(h.unsqueeze(1))


            # 将 e_next_emb 和 q_maritx_next 转换为期望的数据类型
            e_next_emb = e_next_emb.to(desired_dtype)
            q_maritx_next = q_maritx_next.to(desired_dtype)

            # 将 h、e_next_emb 和 q_maritx_next 拼接在一起
            outputs = torch.cat(outputs, dim=1)  # 形状: (batch_size, seq_len, hidden_size)

            h_n_2 = outputs[:, -1:, :]
            # 对时间步维度进行平均池化，得到形状为 (batch_size, hidden_size) 的张量
            average_pooled_output = torch.mean(outputs, dim=1)  #torch.mean or torch.max  [0]

            # 增加一个维度，得到形状为 (batch_size, 1, hidden_size) 的张量
            # h_n_2 = average_pooled_output.unsqueeze(1)
            # last_hidden_state = h_n.repeat(1, 39, 1)
            # last_hidden_state = h_n.permute(1, 0, 2)  # 形状: (batch_size, 1, hidden_size)
            self.h2_list.append(average_pooled_output)  # 保存每个批次的 h1

            h = torch.cat((outputs, e_next_emb, q_maritx_next), dim=-1)

            # 确保 self.out_layer 的权重矩阵的数据类型与 h 相同
            # 假设 desired_dtype 是您希望的数据类型

            # 将 self.out_layer 的权重矩阵和偏置项转换为期望的数据类型
            self.out_layer.weight.data = self.out_layer.weight.data.to(desired_dtype)

            if self.out_layer.bias is not None:
                self.out_layer.bias.data = self.out_layer.bias.data.to(desired_dtype)

            input_jiaohu1 = self.drop_1(torch.sigmoid(self.prednet_full1(h)))
            input_jiaohu2 = self.drop_2(torch.sigmoid(self.prednet_full2(input_jiaohu1)))
            output = torch.sigmoid(self.prednet_full3(input_jiaohu2))

            # output = output.float()  # 确保 h 是 Float 类型
            # y = self.out_layer(h)
            # y = torch.sigmoid(y)
        else:
            h1 = self.h1_list.pop(0).to(concept.device)  # 从列表中加载 h1
            # 将最后一个数据点沿着第二个维度复制 seqlen-1 次，形状变为 (64, 39, 123)
            h2 = h1.repeat(1, 39, 1)
            input_x2 = e_discrimination * (self.d_t  +(1-self.d_t )*h2-k_difficulty) * q_maritx  # 最大的a
            # input_x2 = (s_proficiency) * q_maritx
            concatenated_embeddings_concept1 = torch.cat((input_x2*s_proficiency, r_emb), dim=-1)
            concatenated_embeddings_concept1 = concatenated_embeddings_concept1.float()

            # 使用自定义的门控机制处理输入
            # 初始化隐藏状态
            batch_size = concatenated_embeddings_concept1.size(0)
            h = torch.zeros(batch_size, 123).to(concatenated_embeddings_concept1.device)
            seq_len = concatenated_embeddings_concept1.size(1)
            outputs = []
            # for t in range(seq_len):
            #     x_t = concatenated_embeddings_concept1[:, t, :]
            #     combined = torch.cat((x_t, h), dim=-1)
            #
            #     z_t = torch.sigmoid(F.linear(combined, self.W_z, self.b_z))
            #     r_t = torch.sigmoid(F.linear(combined, self.W_r, self.b_r))
            #
            #     combined_reset = torch.cat((x_t, r_t * h), dim=-1)
            #     h_tilde = torch.tanh(F.linear(combined_reset, self.W_h, self.b_h))
            #
            #     h = (1 - z_t) * h + z_t * h_tilde
            #     outputs.append(h.unsqueeze(1))


            for t in range(seq_len):  # GRU方式实现
                x_t = concatenated_embeddings_concept1[:, t, :]
                if self.h2_list:
                    h2 = self.h2_list.pop(0).to(x_t.device)  # 从列表中加载 h2
                else:
                    # 处理 h2_list 为空的情况，使用默认值
                    h2 = torch.zeros(x_t.size(0), 123).to(x_t.device)
                # 获取 x_t 和 h2 的批次大小
                x_t_batch_size = x_t.size(0)
                h2_batch_size = h2.size(0)
                if x_t_batch_size > h2_batch_size:
                    # 填充 h2 使其批次大小与 x_t 一致
                    padding_size = x_t_batch_size - h2_batch_size
                    h2_padded = torch.cat([h2, h2.new_zeros(padding_size, h2.size(1))], dim=0)
                    h2 = h2_padded
                    combined = torch.cat((x_t, h2), dim=-1)
                elif x_t_batch_size < h2_batch_size:
                    # 裁剪 h2 使其批次大小与 x_t 一致
                    h2_trimmed = h2[:x_t_batch_size, :]
                    h2 = h2_trimmed
                    combined = torch.cat((x_t, h2), dim=-1)
                else:
                    # 大小一致，直接拼接
                    combined = torch.cat((x_t, h2), dim=-1)

                # combined = torch.cat((x_t, h2), dim=-1)  #正常 h2 是h
                z_t = torch.sigmoid(F.linear(combined, self.W_z, self.b_z))
                h_tilde = torch.tanh(F.linear(combined, self.W_h, self.b_h))

                h = (1 - z_t) * h + z_t * h_tilde
                outputs.append(h.unsqueeze(1))
            desired_dtype = h1.dtype

            # 将 e_next_emb 和 q_maritx_next 转换为期望的数据类型
            e_next_emb = e_next_emb.to(desired_dtype)
            q_maritx_next = q_maritx_next.to(desired_dtype)

            # 将 h、e_next_emb 和 q_maritx_next 拼接在一起
            outputs = torch.cat(outputs, dim=1)  # 形状: (batch_size, seq_len, hidden_size)

            # h2=outputs[:, -1:, :]
            # h2 = self.h2_list.pop(0).to(concept.device)  # 从列表中加载 h1
            # # # 将最后一个数据点沿着第二个维度复制 seqlen-1 次，形状变为 (64, 39, 123)
            # h2 = h2.repeat(1, 39, 1)


            h = torch.cat((outputs, e_next_emb, q_maritx_next), dim=-1)

            # 确保 self.out_layer 的权重矩阵的数据类型与 h 相同
            # 假设 desired_dtype 是您希望的数据类型

            # 将 self.out_layer 的权重矩阵和偏置项转换为期望的数据类型
            self.out_layer.weight.data = self.out_layer.weight.data.to(desired_dtype)

            if self.out_layer.bias is not None:
                self.out_layer.bias.data = self.out_layer.bias.data.to(desired_dtype)

            input_jiaohu1 = self.drop_1(torch.sigmoid(self.prednet_full1(h)))
            input_jiaohu2 = self.drop_2(torch.sigmoid(self.prednet_full2(input_jiaohu1)))
            output = torch.sigmoid(self.prednet_full3(input_jiaohu2))

            # output = output.float()  # 确保 h 是 Float 类型

            # h = h.float()  # 确保 h 是 Float 类型
            # y = self.out_layer(h)
            # y = torch.sigmoid(y)


        return output


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





