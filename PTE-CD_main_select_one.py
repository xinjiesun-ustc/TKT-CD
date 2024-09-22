from torch.utils.data import DataLoader,SequentialSampler
from data_load_select_one import read_file

from DKT_emb_select_one import myKT_DKT
import torch
import numpy as np
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(1025)  # 你可以选择你想要的任何种子值

# Number of Students, Number of Exercises, Number of Knowledge Concepts
NUM_CONCEPTS = 123
NUM_STUDENT  =  6035
NUM_EXERCISES = 20000
EMBED_SIZE = 256
BATCH_SIZE = 64


# #读取数据
train_students = read_file("./data/train_data_order.txt")
test_students = read_file("./data/test_data_order.txt")

#按batch加载数据u
# 创建数据加载器，保持顺序不变
sequential_sampler = SequentialSampler(train_students)
train_data_loader = DataLoader(train_students, batch_size=BATCH_SIZE, drop_last=True,sampler=sequential_sampler)

sequential_sampler = SequentialSampler(test_students)
test_data_loader = DataLoader(test_students, batch_size=BATCH_SIZE, drop_last=True,sampler=sequential_sampler)

# train_data_loader = DataLoader(train_students, batch_size=BATCH_SIZE, shuffle=True)  # 创建数据加载器
# test_data_loader = DataLoader(test_students, batch_size=BATCH_SIZE, shuffle=True)  # 创建数据加载器

dkt = myKT_DKT(NUM_CONCEPTS, NUM_STUDENT, NUM_EXERCISES, EMBED_SIZE)
# dkt = myKT_DKT(NUM_CONCEPTS, EMBED_SIZE)
dkt.train(train_data_loader,test_data_loader, epoch=20)
dkt.save("dkt.params")
dkt.load("dkt.params")
# auc = dkt.eval(test_data_loader)
# print("auc: %.6f" % auc)