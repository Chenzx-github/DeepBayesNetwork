import os
import random
import shutil
from shutil import copy2

# there are train set, vaid set and test set on new classes in each datasets
# build data set for N way K shot problem on miniImageNet, CIFAR-FS, Omniglot datasets

# N way K shot, N*K samples in train dataset
N = 5 
K = 1
# T1 samples in valid set, T2 samples in test set
T1 = 5
T2 = 10

datadir_normal = "./miniImageNet/new/"    # split data from new classes
ways = os.listdir(datadir_normal + "train/")

ways_num = len(ways)
way1 = list(range(ways_num))
random.seed()

epoch = 20  
for it in range(epoch):
    random.shuffle(way1)
    way = way1[:N]
    print(way)

    trainDir = datadir_normal  + str(it) + "/train/"
    os.makedirs(trainDir)
    for i in way:
        os.mkdir(trainDir + ways[i])
    validDir = datadir_normal + str(it) + '/valid/'
    os.makedirs(validDir)
    for i in way:
        os.mkdir(validDir + ways[i])
    testDir = datadir_normal + str(it) + '/test/'
    os.makedirs(testDir)
    for i in way:
        os.mkdir(testDir + ways[i])



    for i in way:
        new_f1 = trainDir + ways[i]
        new_f2 = validDir + ways[i]
        new_f3 = testDir + ways[i]

        f1 = datadir_normal + "train/" + ways[i]
        f2 = datadir_normal + "valid/" + ways[i]
        f3 = datadir_normal + "test/" + ways[i]

        path_file = os.listdir(f1)
        path_file_num = len(path_file)
        index_list = list(range(path_file_num))
        random.shuffle(index_list)
        for i in index_list[:K]:
            fileName = os.path.join(f1, path_file[i])
            copy2(fileName, new_f1)


        path_file = os.listdir(f2)
        path_file_num = len(path_file)
        index_list = list(range(path_file_num))
        random.shuffle(index_list)
        for i in index_list[:T1]:
            fileName = os.path.join(f2, path_file[i])
            copy2(fileName, new_f2)


        path_file = os.listdir(f3)
        path_file_num = len(path_file)
        index_list = list(range(path_file_num))
        random.shuffle(index_list)

        for i in index_list[:T2]:
            fileName = os.path.join(f3, path_file[i])
            copy2(fileName, new_f3)