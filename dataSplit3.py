# -*- coding:utf-8 -*-
# 划分数据集
# new已被分好为train、valid、test,做 N way K shot

import os
import random
import shutil
from shutil import copy2


N = 5
K = 5
v = 5
t = 10


ways = os.listdir("train")
# 打乱文件顺序
ways_num = len(ways)
way1 = list(range(ways_num))
random.seed()
random.shuffle(way1)


way = way1[:N] # 取N个做训练
trainDir = "NWKS/train/"
os.makedirs(trainDir)
for i in way:
    os.mkdir(trainDir + ways[i])
validDir = 'NWKS/valid/'
os.makedirs(validDir)
for i in way:
    os.mkdir(validDir + ways[i])
testDir = 'NWKS/test/'
os.makedirs(testDir)
for i in way:
    os.mkdir(testDir + ways[i])

for i in way:
    new_f1 = trainDir + ways[i]    # 新目录路径
    new_f2 = validDir + ways[i]
    new_f3 = testDir + ways[i]

    f1 = "train/" + ways[i] # 图片来源
    f2 = "valid/" + ways[i]
    f3 = "test/" + ways[i]

    # K shot
    path_file = os.listdir(f1)         # 该路径下的各图片名
    path_file_num = len(path_file)
    index_list = list(range(path_file_num))
    random.shuffle(index_list)
    for i in index_list[:K]:
        fileName = os.path.join(f1, path_file[i])
        copy2(fileName, new_f1)


    # 验证集挑选 v 个用作验证
    path_file = os.listdir(f2)  # 该路径下的各图片名
    path_file_num = len(path_file)
    index_list = list(range(path_file_num))
    random.shuffle(index_list)
    for i in index_list[:v]:
        fileName = os.path.join(f2, path_file[i])
        copy2(fileName, new_f2)


    # 测试集挑选 t 个用作测试
    path_file = os.listdir(f3)  # 该路径下的各图片名
    path_file_num = len(path_file)
    index_list = list(range(path_file_num))
    random.shuffle(index_list)
    for i in index_list[:t]:
        fileName = os.path.join(f3, path_file[i])
        copy2(fileName, new_f3)
