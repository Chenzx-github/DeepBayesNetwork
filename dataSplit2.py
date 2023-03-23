# -*- coding:utf-8 -*-
# new下划分为train、valid、test


import os
import random
import shutil
from shutil import copy2

all_data = os.listdir("./new/")  # 类名（目录名）

trainDir = "./train/"
os.mkdir(trainDir)
for i in range(len(all_data)):
    os.mkdir(trainDir + all_data[i])
validDir = './valid/'
os.mkdir(validDir)
for i in range(len(all_data)):
    os.mkdir(validDir + all_data[i])
testDir = './test/'
os.mkdir(testDir)
for i in range(len(all_data)):
    os.mkdir(testDir + all_data[i])


for i in range(len(all_data)): # 对每类进行划分样本
    cnum = 0
    f1 = trainDir + all_data[i]
    f2 = validDir + all_data[i]
    f3 = testDir +  all_data[i]

    curpath = "./new/" + all_data[i]  # new下各类文件路径
    path_file = os.listdir(curpath)

    # 打乱图片顺序，随机划分样本到训练、验证以及测试集
    path_file_num = len(path_file)
    index_list = list(range(path_file_num))
    random.shuffle(index_list)

    for i in index_list:
        fileName = os.path.join(curpath, path_file[i])
        if cnum < path_file_num * 0.6: # train 0.6
            copy2(fileName, f1)
        elif cnum < path_file_num * 0.8: # valid 0.2 , test 0.2
            copy2(fileName, f2)
        else:
            copy2(fileName, f3)
        cnum += 1

print("done!")