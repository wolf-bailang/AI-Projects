"""Train CIFAR10 with PyTorch."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from models import VGG
from utils import progress_bar


# Data
# 获取数据集，并先进行预处理
def data_Load(batch_size):
    print('==> Preparing data..')
    # 图像预处理和增强
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    '''下载训练集 CIFAR-10 10分类训练集'''
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)  # batch_size=128
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)  # batch_size=100
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # specify the names of the classes
    classes_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
                    8: 'ship', 9: 'truck'}
    return trainset, testset, trainloader, testloader, classes_dict


# Model
# 继续训练模型或新建一个模型
def model_build(resume):
    print('==> Building model..')
    net = VGG('VGG16')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(device)
    # use_cuda = torch.cuda.is_available()
    net = net.to(device)
    # 如果GPU可用，使用GPU
    if device == 'cuda':

        # parallel use GPU
        net = torch.nn.DataParallel(net)
        # speed up slightly
        cudnn.benchmark = True
    #else:
        #net = VGG('VGG16')
    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        # .pth格式模型加载
        #checkpoint = torch.load('./checkpoint/ckpt.pth', map_location=torch.device('cpu'))
        #net.load_state_dict(checkpoint['net'])
        #best_acc = checkpoint['acc']
        #start_epoch = checkpoint['epoch']

        # .pkl格式模型加载
        #net.load_state_dict(torch.load('./checkpoint/ckpt.pkl', map_location=torch.device('cpu')))

        net_dict = torch.load('./checkpoint/ckpt.pkl', map_location=torch.device('cpu'))
        # 如果提示module.出错放开下面的代码
        new_state_dict = OrderedDict()
        for k, v in net_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        net.load_state_dict(new_state_dict)

    return net, device


# 定义度量和优化
def model_process(net, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    return criterion, optimizer


# Training
# 训练阶段
def train(net, device, epoch, trainloader, optimizer, criterion, trainset, Loss_list, Train_Accuracy_list):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # batch 数据
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # 将数据移到GPU上
        inputs, targets = inputs.to(device), targets.to(device)
        # 先将optimizer梯度先置为0
        optimizer.zero_grad()
        # 模型输出
        outputs = net(inputs)
        # 计算loss，图的终点处
        loss = criterion(outputs, targets)
        # 反向传播，计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()

        train_loss += loss.item()
        # 数据统计
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        if total == len(trainset):
            print('Finish {} epoch, train_Loss: {:.6f}, train_Acc: {:.6f}, train_correct: {:.6f}, total: {:.6f}'.format(
                epoch + 1, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            Loss_list.append(train_loss / (batch_idx + 1))
            Train_Accuracy_list.append(100. * correct / total)
    return Loss_list, Train_Accuracy_list


# 测试阶段
def test(device, epoch, net, testloader, criterion, testset, Test_Accuracy_list):
    global best_acc
    # 先切到测试模型
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            if total == len(testset):
                print(
                    'Finish {} epoch, test_Loss: {:.6f}, test_Acc: {:.6f}, test_correct: {:.6f}, total: {:.6f}'.format(
                        epoch + 1, test_loss / total, 100. * correct / total, correct, total))
                Test_Accuracy_list.append(100. * correct / total)

    # Save checkpoint.
    # 保存模型
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        torch.save(net.state_dict(), './checkpoint/ckpt.pkl')
        best_acc = acc
    return Test_Accuracy_list


# 运行模型
def Run_process(Epochs, start_epoch, net, device, trainloader, testloader, optimizer,
                criterion, trainset, testset, Loss_list, Train_Accuracy_list, Test_Accuracy_list):
    for epoch in range(start_epoch, start_epoch + Epochs):
        Loss_list, Train_Accuracy_list = train(net, device, epoch, trainloader, optimizer,
                                               criterion, trainset, Loss_list, Train_Accuracy_list)
        Test_Accuracy_list = test(device, epoch, net, testloader, criterion, testset, Test_Accuracy_list)
        # 清除部分无用变量
        # torch.cuda.empty_cache()
    print(best_acc)
    Draw_Accuracy(Train_Accuracy_list, Test_Accuracy_list, Loss_list)


def Draw_Accuracy(Train_Accuracy_list, Test_Accuracy_list, Loss_list):
    x1 = range(0, 51)
    x2 = range(0, 51)
    y1 = Train_Accuracy_list
    y2 = Test_Accuracy_list

    fig = plt.figure(figsize=(8, 10))
    # ax = plt.axes()
    plt.subplot(2, 1, 1)
    # plt.plot(x1, y1, '', y2, '')
    plt.plot(x1, y1, '', label='Training')
    plt.plot(x1, y2, '', label='Testing')
    plt.title('Accuracy', fontsize=20)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('%', fontsize=16)
    plt.legend(fontsize=14)  # 将样例显示出来

    y3 = Loss_list
    plt.subplot(2, 1, 2)
    plt.plot(x2, y3, '')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.savefig("./Accuracy_Loss.png")
    plt.show()


def view_classify(img, ps, title, classes_dict):
    """
    Function for viewing an image and it's predicted classes
    with matplotlib.

    INPUT:
        img - (tensor) image file
        ps - (tensor) predicted probabilities for each class
        title - (str) string with true label
    """
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(10, 10), nrows=2, ncols=1)
    image = img.permute(1, 2, 0)
    ax1.imshow(image.numpy())
    ax1.axis('off')

    ax2.bar(np.arange(10), ps)
    ax2.set_aspect(10)
    ax2.set_xticks(np.arange(10))
    ax2.set_xticklabels(list(classes_dict.values()), size='small', rotation=45, fontsize=13)
    ax2.set_title(title, fontsize=20)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("./predict.png")
    #plt.show()


def predict(net, testloader, device, classes_dict, index):
    net.eval()
    for batch_idx, (inputs, labels) in enumerate(testloader):
        index = np.random.randint(0, 100)
        inputs, labels = inputs.to(device), labels.to(device)
        img = inputs[index]
        label_true = labels[index]
        pred = net(inputs)
        # print(pred[index])
        # print(torch.softmax(pred[index].cpu(), dim=0))
        view_classify(img.cpu(), torch.softmax(pred[index].cpu(), dim=0), classes_dict[int(label_true.cpu().numpy())],
                      classes_dict)
        break


def Hyperparameters():
    """
       parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
       parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
       parser.add_argument('--resume', '-r', action='store_true',
                           help='resume from checkpoint')
       args = parser.parse_args()
    """
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    Epochs = 50
    learning_rate = 0.01
    batch_size = 32
    # 定义两个数组
    Loss_list = []
    Train_Accuracy_list = []
    Test_Accuracy_list = []
    Loss_list.append(3)
    Test_Accuracy_list.append(0)
    Train_Accuracy_list.append(0)
    return best_acc, start_epoch, Epochs, learning_rate, batch_size, Loss_list, Train_Accuracy_list, Test_Accuracy_list


def VGG16_Classifier(train_model, resume):
    best_acc, start_epoch, Epochs, learning_rate, batch_size, Loss_list, Train_Accuracy_list, Test_Accuracy_list = Hyperparameters()
    trainset, testset, trainloader, testloader, classes_dict = data_Load(batch_size)
    net, device = model_build(resume)
    criterion, optimizer = model_process(net, learning_rate)
    if train_model:
        Run_process(Epochs, start_epoch, net, device, trainloader, testloader, optimizer,
                    criterion, trainset, testset, Loss_list, Train_Accuracy_list, Test_Accuracy_list)
    # else:
    # predicte(net, testloader, device, classes_dict)