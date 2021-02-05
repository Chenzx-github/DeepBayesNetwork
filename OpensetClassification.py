import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import  train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import cv2

# open-set classification on miniImageNet and tinyImageNet

positive = []
negative = []

image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}


datadir = "tinyImageNet/osc/"    # miniImageNet and tinyImageNet
ways = os.listdir(datadir)
ways_num = len(ways)

for i in range(ways_num):

    dataset = datadir + str(i)    # each group data set
    new_dir = dataset + '/models/'
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    train_directory = os.path.join(dataset, 'train') # konwn classes
    valid_directory = os.path.join(dataset, 'valid') # konwn classes
    test_directory = os.path.join(dataset, 'test')# all classes in dataset (konwn classes and unkonwn classes )

    batch_size = 32


    data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
    'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])

    }
    train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
    valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)
    test_data = DataLoader(data['test'], batch_size=batch_size, shuffle=True)

    train_data_size = len(data['train'])
    valid_data_size = len(data['valid'])
    test_data_size = len(data['test'])
    print(train_data_size, valid_data_size, test_data_size)

    # the model first is trained on known classes
    resnet18 = models.resnet18(pretrained=True)

    num_classes = 20
    fc_inputs = resnet18.fc.in_features
    resnet18.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
        nn.LogSoftmax(dim=1)
    )

    resnet18 = resnet18.to('cuda:0')
    loss_func = nn.NLLLoss()
    optimizer = optim.Adam(resnet18.parameters())

    def train_and_valid(model, loss_function, optimizer, epochs=20):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        history = []
        best_acc = 0.0
        best_epoch = 0

        for epoch in range(epochs):
            epoch_start = time.time()
            print("Epoch: {}/{}".format(epoch + 1, epochs))
            model.train()
            train_loss = 0.0
            train_acc = 0.0
            valid_loss = 0.0
            valid_acc = 0.0

            for i, (inputs, labels) in enumerate(train_data):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                train_acc += acc.item() * inputs.size(0)

            with torch.no_grad():
                model.eval()
                for j, (inputs, labels) in enumerate(valid_data):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                    valid_loss += loss.item() * inputs.size(0)
                    ret, predictions = torch.max(outputs.data, 1)
                    correct_counts = predictions.eq(labels.data.view_as(predictions))
                    acc = torch.mean(correct_counts.type(torch.FloatTensor))
                    valid_acc += acc.item() * inputs.size(0)

            avg_train_loss = train_loss / train_data_size
            avg_train_acc = train_acc / train_data_size

            avg_valid_loss = valid_loss / valid_data_size
            avg_valid_acc = valid_acc / valid_data_size

            history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
            if best_acc < avg_valid_acc:
                best_acc = avg_valid_acc
                best_epoch = epoch + 1
            epoch_end = time.time()

            print(
                "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                    epoch + 1, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                    epoch_end - epoch_start
                ))
            print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

        return model, history


    # 1 3 进行已预训练ResNet上的再训练
    num_epochs = 10
    trained_model, history = train_and_valid(resnet18, loss_func, optimizer, num_epochs) # trained_model 为训练后的模型

    def forward(model, x):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def percentage(Y_predict):
        p = [[] for i in range(len(Y_predict))]
        for i in range(len(Y_predict)):
            maxn = max(Y_predict[i])
            minn = min(Y_predict[i])
            for j in Y_predict[i]:
                if maxn - minn:
                    tmp = (j - minn) / (maxn - minn)
                    p[i].append(math.exp(tmp))
                else:
                    p[i].append(1.0 / len(Y_predict[i]))
            p[i] = [x / sum(p[i]) for x in p[i]]

        return p


    def confidence(X1, X2, shold):
        # calculate p(X2), where X2 is feature vectors test samples and X1 is feature vectors training samples
        X1 = binarize(X1, threshold=shold)
        X2 = binarize(X2, threshold=shold)

        x1 = []
        for item in X1.tolist():
            item = item[:]
            x1.append(item)
        x1 = np.array(x1)
        a = np.sum(x1, axis=0)
        # a = a / len(x1)
        # plt.plot(a,c='darkblue')
        # plt.show()
        feature = {}
        for i in range(len(a)):
            if a[i] >= 0.2: feature[i] = a[i]
            feature[i] = a[i]

        x2 = []
        for item in X2.tolist():
            item = item[:]
            x2.append(item)
        x2 = np.array(x2)

        res = []
        for item in x2:
            p = 1
            for j in range(len(item)):
                if j in feature:
                    if item[j] == 1:
                        p = p * feature[j]
                    else:
                        if feature[j] >= 0.99:
                            p = p * 0.01
                        else:
                            p = p * (1 - feature[j])
            res.append(p)
        return res


    from sklearn.preprocessing import binarize
    from sklearn.preprocessing import LabelBinarizer
    import math
    class ML:
        def predict(self, x):
            X = binarize(x, threshold=self.threshold)

            X1 = []
            for item in X.tolist():
                item = item[:]
                X1.append(item)
            X1 = np.array(X1)
            Y_predict = np.dot(X1, np.log(self.prob).T) - np.dot(X1, np.log(1 - self.prob).T) + np.log(
                1 - self.prob).sum(axis=1)
            return self.classes[np.argmax(Y_predict, axis=1)], Y_predict.tolist()
    class Bayes(ML):
        def __init__(self, threshold):
            self.threshold = threshold
            self.classes = []
            self.prob = 0.0

        def fit(self, X, y):
            labelbin = LabelBinarizer()
            Y = labelbin.fit_transform(y)
            self.classes = labelbin.classes_
            Y = Y.astype(np.float64)

            X = binarize(X, threshold=self.threshold)
            X1 = []
            for item in X.tolist():
                item = item[:]
                X1.append(item)
            X1 = np.array(X1)

            feature_count = np.dot(Y.T, X1)
            class_count = Y.sum(axis=0)

            # Laplace smoothing, solving the zero probability problem
            alpha = 1.0
            smoothed_fc = feature_count + alpha
            smoothed_cc = class_count + alpha * 2
            self.prob = smoothed_fc / smoothed_cc.reshape(-1, 1)

            return self



    import datetime
    starttime = datetime.datetime.now()
    import numpy as np
    from sklearn.model_selection import  train_test_split
    from sklearn.metrics import confusion_matrix, classification_report
    import os
    import cv2

    print("______________________test____________________")

    X_train=[]
    y_train=[]
    traindir = './' + dataset + '/train/'
    labellist = os.listdir(traindir)
    for i in range(0, len(labellist)):
        lists = os.listdir(traindir + "%s" % labellist[i])
        for f in os.listdir(traindir + "%s" % labellist[i]):
            Img = cv2.imread(traindir + "%s/%s" % (labellist[i], f))
            img = cv2.resize(Img, (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            img = transform(img).cuda()
            img = img.unsqueeze(0)
            X_train.append(forward(trained_model,img).data.cpu().numpy()[0])
            y_train.append(labellist[i])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    featurel.append(np.mean(featureSum_train))

    X_test = []
    num_nei = 0
    num_wai = 0
    testdir = './' + dataset + '/test/'
    list = os.listdir(testdir)
    for i in range(0, len(list)):
        for f in os.listdir(testdir + "%s" % list[i]):
            if list[i] == "konwn":num_nei += 1
            if list[i] == "unkonwn":num_wai += 1
            Img = cv2.imread(testdir + "%s/%s" % (list[i], f))
            img = cv2.resize(Img, (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            img = transform(img).cuda()
            img = img.unsqueeze(0)
            X_test.append(forward(trained_model, img).data.cpu().numpy()[0])
            featureSum_test.append(sum(forward(trained_model, img).data.cpu().numpy()[0]))

    X_test = np.array(X_test)

    fs = 1.1158 # threshold parameter in yes determined though performance of known classes valid test
    confidence111(X_test[:num_nei], fs)
    confidence111(X_test[num_nei:], fs)
    clf0 = Bayes(fs).fit(X_train, y_train)
    predictions_labels, pre1 = clf0.predict(X_test)

    pre1 = percentage(pre1)
    distribu_nei = [max(i) for i in pre1]

    fun = {}  # stores the probability of each sample with its feature in the training set
    for i in range(len(labellist)):
        fun[labellist[i]] = []
    # calculate the probability of each sample with its feature in the training set
    trainRes = confidence(X_train, X_train,fs)
    for i in range(len(y_train)):
        fun[y_train[i]].append(trainRes[i])
    # calculate the probability of each sample with its feature in the test set
    testRes = confidence(X_train, X_test,fs)


    finalLabel = predictions_labels
    predictions_distribution = pre1
    predictions_maxdistribution = [max(x) for x in predictions_distribution]
    testConfidence = []
    for i in range(len(testRes)):
        # P(Xy) = P(X)P(y), X is feature vector, y is label
        testConfidence.append(testRes[i] * predictions_maxdistribution[i])
    print(max(testConfidence), min(testConfidence))


    print(len(X_test),num_wai,num_nei)
    print("----------------------------------------------------------------")
    print("konwn class")
    Rcount = 0
    Wcount = 0
    oldClassn = 1
    w1 = 0
    w2 = 0
    w3 = 0
    for i in range(len(testConfidence))[:num_nei]:
        if min(fun[finalLabel[i]]) <= testConfidence[i] <= max(fun[finalLabel[i]]):
            moreCount = sum(1 for item in fun[finalLabel[i]] if item >= testConfidence[i])
            pro = max(moreCount / len(fun[finalLabel[i]]), 1 - moreCount / len(fun[finalLabel[i]]))
        else:
            pro = 1

        flag = True
        dis = predictions_distribution[i]
        dis.sort(reverse=True)

        labels = [x for x in dis if x > 1 / num_classes]
        if pro >= 0.8 and dis[0] <= np.mean(distribu_nei):
            flag = False
            w1 += 1
        if (dis[0] - dis[1]) / dis[1] <= 0.05:
            flag = False
            w2 += 1
        if len(labels) > (num_classes+1) / 2:
            flag = False
            w3 += 1
        if flag:
            Rcount += 1
        else:
            Wcount += 1
    print("test 类别内判断 TP 正确率", Rcount / num_nei)
    positive.append(Rcount / num_nei)

    # 类别外数据集判断
    print("----------------------------------------------------------------")
    print("unknown class")

    Wcount = 0
    Rcount = 0
    oldClassn = 0
    w1 = 0
    w2 = 0
    w3 = 0
    for i in range(len(testConfidence))[num_nei:]:
        if min(fun[finalLabel[i]]) <= testConfidence[i] <= max(fun[finalLabel[i]]):
            moreCount = sum(1 for item in fun[finalLabel[i]] if item >= testConfidence[i])
            pro = max(moreCount / len(fun[finalLabel[i]]), 1 - moreCount / len(fun[finalLabel[i]]))
        else:
            pro = 1

        dis = predictions_distribution[i]
        dis.sort(reverse=True)
        flag = True

        labels = [x for x in dis if x > 1 / num_classes]
        if pro >= 0.8 and dis[0] <= np.mean(distribu_nei):
            flag = False
            w1 += 1
        if (dis[0] - dis[1]) / dis[1] <= 0.05:
            flag = False
            w2 += 1
        if len(labels) > (num_classes+1) / 2 :
            flag = False
            w3 += 1

        if flag:
            Rcount += 1
        else:
            Wcount += 1
    print("test 类别外判断 TN 正确率", Wcount / num_wai)
    negative.append(Wcount / num_wai)

print(positive)
print(negative)






