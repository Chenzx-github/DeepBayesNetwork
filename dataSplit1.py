# -*- coding:utf-8 -*-
# 划分数据集类别为基类和新类

import os
import random
from shutil import copy2

trainClassNumber = 80 # 基类数，剩余为新类
datadir_normal = "./cifar/"
ways = os.listdir(datadir_normal)  # 类名(目录名)，该数据库结构 datasetname / classnames / pictures

ways_num = len(ways)
way1 = list(range(ways_num))
random.seed()


random.shuffle(way1) # 打乱顺序划分基（新）类
waytrain = way1[:trainClassNumber]
waytest = way1[trainClassNumber:]


trainDir = "./base/"  #训练集
os.makedirs(trainDir)
for i in waytrain:
    os.mkdir(trainDir + ways[i])


testDir = './new/'  # 测试集
os.makedirs(testDir)
for i in waytest:
    os.mkdir(testDir + ways[i])


for i in waytrain:
    new_f1 = trainDir + ways[i]    # 新目录路径
    f1 = datadir_normal + ways[i]  # 源图片来源
    # 训练集挑选 K shot
    path_file = os.listdir(f1)  # 源图片该类下的各图片名
    path_file_num = len(path_file)
    for i in range(path_file_num):
        fileName = os.path.join(f1, path_file[i])
        copy2(fileName, new_f1)

for i in waytest:
    new_f2 = testDir + ways[i]
    f2 = datadir_normal + ways[i]
    path_file = os.listdir(f2)  # 该路径下的各图片名
    path_file_num = len(path_file)
    for i in range(path_file_num):
        fileName = os.path.join(f2, path_file[i])
        copy2(fileName, new_f2)

print("done!")
