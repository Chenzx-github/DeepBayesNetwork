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

# N way K shot on miniImageNet, CIFAR-FS, Omniglot datasets

res = []
bys = []


image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.Resize(224),
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


# # on base classes
# datadir = "miniImageNet/base"    # miniImageNet, CIFAR-FS, Omniglot

# on new classes
datadir = "miniImageNet/new/5way1shot/"    # miniImageNet, CIFAR-FS, Omniglot
# datadir = "miniImageNet/new/5way5shot/"    # miniImageNet, CIFAR-FS, Omniglot



ways = os.listdir(datadir)
ways_num = len(ways)

x = []
y = []
z = []
count = 0
for i in range(ways_num):

    dataset = datadir + str(i)   # each group data set
    new_dir = dataset + '/models/'
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    train_directory = os.path.join(dataset, 'train')
    valid_directory = os.path.join(dataset, 'valid')
    test_directory = os.path.join(dataset, 'test')

    batch_size = 32

    data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
    'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])

    }
    train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
    valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)
    test_data = DataLoader(data['test'], batch_size=batch_size, shuffle=True)
    print(len(train_data))

    train_data_size = len(data['train'])
    valid_data_size = len(data['valid'])
    test_data_size = len(data['test'])
    print(train_data_size, valid_data_size, test_data_size)



    # # on base classes
    # # the model first is trained on base classes, the paraments and the model structure be saved in file(xxxx.pt)
    # resnet18 = models.resnet18(pretrained=True)

    # # on new classes
    # transfer learning
    resnet18=torch.load("models/miniImageNet/base/_model_15.pt") # 调用
    resnet18.eval()
    for param in resnet18.parameters():
        param.requires_grad = False



    # # on base classes
    # # Train in 80 base classes
    # num_classes = 80
    # fc_inputs = resnet18.fc.in_features          # (fc): Linear(in_features=512, out_features=1000, bias=True)
    # resnet18.fc = nn.Sequential(
    #     nn.Linear(fc_inputs, 256),
    #     nn.ReLU(),
    #     nn.Dropout(0.4),
    #     nn.Linear(256, num_classes),
    #     nn.LogSoftmax(dim=1)
    # )

    # on new classes
    # Train in 5 way 1 or 5 shot
    num_classes = 5
    resnet18.fc = nn.Sequential(
        nn.Linear(512, 64),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(64, num_classes),
        nn.LogSoftmax(dim=1)

    )
    resnet18 = resnet18.to('cuda:0')
    loss_func = nn.NLLLoss()
    optimizer = optim.Adam(resnet18.parameters())

    def train_and_valid(model, loss_function, optimizer, epochs=5):
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
                    epoch + 1, avg_valid_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                    epoch_end - epoch_start
                ))
            print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

            # # on base classes, save the paraments and the structure after training on base classes
            # torch.save(model, 'models/' + "mini-imagenet/base/" +'_model_' + str(epoch + 1) + '.pt')

        return model, history

    num_epochs = 5
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
        # Each class of probabilities is converted to a percentage
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
            # f(x) = x^p(1-x)^(1-p)-----> lnf(x)=xlnp+(1-x)ln(1-p)
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
    list = os.listdir(traindir)
    for i in range(0, len(list)):
        he = 0
        lists = os.listdir(traindir + "%s" % list[i])
        for f in os.listdir(traindir + "%s" % list[i]):
            Img = cv2.imread(traindir + "%s/%s" % (list[i], f))
            img = cv2.resize(Img, (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            img = transform(img).cuda()
            img = img.unsqueeze(0)
            X_train.append(forward(trained_model,img).data.cpu().numpy()[0])
            y_train.append(i)
    X_train = np.array(X_train)
    y_train = np.array(y_train)


    X_test = []
    y_test = []
    testdir = './' + dataset + '/test/'

    # result in pure resnet
    pre2 = []
    res_test = []

    for i in range(0, len(list)):
        for f in os.listdir(testdir + "%s" % list[i]):
            Img = cv2.imread(testdir + "%s/%s" % (list[i], f))
            img = cv2.resize(Img, (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            img = transform(img).cuda()
            img = img.unsqueeze(0)

            X_test.append(forward(trained_model, img).data.cpu().numpy()[0])
            y_test.append(i)

            # resnet test result
            out = trained_model(img)
            pre2.append(out.cpu().detach().numpy().tolist()[0])
            ret, predictions = torch.max(out.data, 1)
            res_test.append(predictions.cpu().numpy()[0])


    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # result of bys
    clf0 = Bayes(0.05).fit(X_train, y_train)
    predictions_labels, pre1 = clf0.predict(X_test)

    # be changed to percent
    pre1 = percentage(pre1)
    pre2 = percentage(pre2)

    import sklearn
    res.append(sklearn.metrics.accuracy_score(y_test, res_test))
    bys.append(sklearn.metrics.accuracy_score(y_test, predictions_labels))

x.apend(np.mean(res))
y.append(np.mean(bys))

plt.plot(x)
plt.plot(y)
plt.legend(['ResNet Accurary', 'Bys Accurary'])
plt.xlabel('Epoch Number')
plt.ylabel('Accurary')
plt.savefig("epoch2accurary.png")
plt.show()

