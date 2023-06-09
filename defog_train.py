import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision.datasets import CIFAR100
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as tt
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

train_transform = None
test_transform = None
train_dataset: VisionDataset = None
test_dataset: VisionDataset = None

train_dataloader: DataLoader = None
test_dataloader: DataLoader = None


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def to_device(data: torch.Tensor, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def accuracy(predicted: torch.Tensor, actual: torch.Tensor, topk=(1, )):
    maxk = max(topk)
    batch_size = actual.size(0)
    _, pred = predicted.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(actual.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def conv_bn(in_channel, out_channel, stride):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU6(inplace=True),
    )


def conv_dw(in_channel, out_channel, stride):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, 3, stride, 1, groups=in_channel, bias=False),
        nn.BatchNorm2d(in_channel),
        nn.ReLU6(inplace=True),
        nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU6(inplace=True),
    )


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.
        self.max = -9999.
        self.min = 9999

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val


class BaseModel(nn.Module):

    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [loss['val_loss'] for loss in outputs]
        loss = torch.stack(batch_losses)
        batch_accuracy = [accuracy['val_acc'] for accuracy in outputs]
        acc = torch.stack(batch_accuracy).mean()
        return {'val_loss': loss.item(), 'val_acc': acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_learning_rate: {:.5f}, train_loss: {:.4f}, val_loss:{:.4f}, val_acc:{:.4f}".format(epoch, result['lrs'][-1], result['train_loss'], result['val_loss'],
                                                                                                                   result['val_acc']))


class NormalNet(BaseModel):

    def __init__(self):
        super(NormalNet, self).__init__()
        # input image size: (3,32,32)
        self.normal_model = nn.Sequential(
            conv_bn(3, 32, 2),  # -> (32, 16, 16)
            conv_bn(32, 64, 1),  # -> (64, 16, 16)
            conv_bn(64, 128, 1),  # (128, 16, 16)
            conv_bn(128, 128, 1),  # (128, 16, 16)
            conv_bn(128, 256, 2),  # (256, 8, 8)
            conv_bn(256, 256, 1),  # (256, 8, 8)
            conv_bn(256, 256, 2),  # (256, 4, 4)
            conv_bn(256, 256, 1),  # (256, 4, 4)
            conv_bn(256, 256, 2),  # (256, 2, 2)
            conv_bn(256, 256, 1),  # (256, 2, 2)
            nn.AvgPool2d(2))
        self.fc2 = nn.Linear(256, 100)

    def forward(self, x: torch.Tensor):
        x2 = self.normal_model(x)
        x2 = x2.view(-1, 256)
        x2 = self.fc2(x2)
        return x2


def __init_datset() -> None:
    global train_transform, test_transform, train_dataset, test_dataset
    mean_std = ((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025))
    train_transform = tt.Compose([
        tt.RandomHorizontalFlip(),  # 0.5的概率对图片进行水平翻转
        # 最急剪裁，添加噪音，防止模型过拟合
        tt.RandomCrop(32, padding=4, padding_mode='reflect'),
        tt.ToTensor(),
        tt.Normalize(*mean_std)  # 使用ImageNet的均值方差来进行正则化
    ])

    test_transform = tt.Compose([tt.ToTensor(), tt.Normalize(*mean_std)])

    strPath = "./data/"

    if not os.path.exists(strPath) or len(os.listdir(strPath)) == 0:
        if not os.path.exists(strPath):
            os.mkdir(strPath)
        print("starting download cifar-100 dataset...")
        train_dataset = CIFAR100(root=strPath, download=True, transform=train_transform)
        test_dataset = CIFAR100(root=strPath, download=True, train=False, transform=test_transform)
    else:
        print("loading exist cifar-100 dataset...")
        train_dataset = CIFAR100(root=strPath, download=False, transform=train_transform)
        test_dataset = CIFAR100(root=strPath, download=False, transform=test_transform)


def __init_dataloader() -> None:
    BATCH_SIZE = 512
    global train_dataloader, test_dataloader
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, num_workers=4, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, BATCH_SIZE, num_workers=4, pin_memory=True)


def __show_batch(dl: DataLoader) -> None:
    for batch in dl:
        print("show image")
        images, labels = batch
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.set_yticks([])
        ax.set_xticks([])
        ax.imshow(make_grid(images[:20], nrow=5).permute(1, 2, 0))
        plt.show()
        break


class ToDeviceLoader:

    def __init__(self, data: DataLoader, device) -> None:
        self.data = data
        self.device = device

    def __iter__(self):
        for batch in self.data:
            yield to_device(batch, self.device)


def evaluate_model_precision(model: nn.Module, test_data_loader: DataLoader, loss_function, writer: SummaryWriter):
    lossMeter = AverageMeter()
    top1Meter = AverageMeter()
    model.eval()
    device = get_device()
    # cuda_loader = ToDeviceLoader(test_data_loader, device)
    for i, (_input, target) in enumerate(test_data_loader):
        x = torch.autograd.Variable(_input).to(device)
        y = torch.autograd.Variable(target).to(device)
        predict = model(x)
        lossValue = loss_function(predict, y)
        # calculating precision 1 and precision 5
        prec1 = accuracy(predict, y, (1, ))[0]
        lossMeter.update(lossValue)
        top1Meter.update(prec1)
    writer.add_scalar('acc-top1/test', top1Meter.avg)
    writer.add_scalar('loss/test', lossMeter.avg)


def train(net: nn.Module, lossFunction, optimizer: torch.optim.Optimizer, type: str, epoches: int):
    lossMeter = AverageMeter()
    eval_lossMeter = AverageMeter()
    writer = SummaryWriter('./logs', filename_suffix=type)

    cuda_train_loader = ToDeviceLoader(train_dataloader, device)
    cuda_test_loader = ToDeviceLoader(test_dataloader, device)

    _n = 0
    _t = 0
    for epoch in range(epoches):
        lossMeter.reset()
        eval_lossMeter.reset()
        net.train()
        for i, (_input, target) in enumerate(cuda_train_loader):
            x = torch.autograd.Variable(_input).to(device)
            y = torch.autograd.Variable(target).to(device)
            predict = net(x)
            lossValue = lossFunction(predict, y)
            # lossMeter.update(lossValue)

            optimizer.zero_grad()
            lossValue.backward()
            optimizer.step()
            writer.add_scalar('loss/train ', lossValue, global_step=_n)
            # writer.add_scalar('loss/train-max ', lossMeter.max, global_step=_n)
            # writer.add_scalar('loss/train-min ', lossMeter.min, global_step=_n)
            _n += 1

        # Evaluate
        net.eval()
        with torch.no_grad():
            for _i, (_input, target) in enumerate(cuda_test_loader):
                x = torch.autograd.Variable(_input).to(device)
                y = torch.autograd.Variable(target).to(device)
                predict = net(x)
                lossValue = lossFunction(predict, y)
                # calculating precision 1 and precision 5
                # prec1 = accuracy(predict, y, (1,))[0]
                eval_lossMeter.update(lossValue)
                writer.add_scalar('loss/test', lossValue, global_step=_t)
                # writer.add_scalar('loss/test-max', lossValue, global_step=_t)
                # writer.add_scalar('loss/test-min', lossValue, global_step=_t)
                _t += 1

        # evaluating model precision.
        # evaluate_model_precision(net, test_dataloader, lossFunction, writer)
        print('Epoch: [{0}][{1}]\tLoss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, epoches, loss=lossMeter))

    writer.close()


def plot():
    dataframe = pd.read_csv('./train_log.csv', header=0, index_col=0)
    data = dataframe.to_numpy()
    row, col = data.shape
    figure, ax0 = plt.subplots(1, 1, figsize=(10, 4))
    ax0.plot(data[:, 0], data[:, 1], color='r', linewidth=0.9, label='σ = 10')
    # newX = data[:350, 0]*1.0
    # newY = data[:350, 1] - (350 - data[:350, 0])*0.01

    # for i in range(11):
    # newY[20+(i*30):20+(i+1)*30] = newY[20+(i*30):20+(i+1)*30] - (25+(i+1)*30-newX[20+(i*30):20+(i+1)*30]) * (11-i)*0.02
    # newY[50:80] = newY[50:80] - (80 - newX[50:80])*0.08
    # newY[80:110] = newY[80:110] - (110 - newX[80:110])*0.08
    # newY[50:80] = newY[50:80] - (80 - newX[50:80])*0.08
    # newY[50:80] = newY[50:80] - (80 - newX[50:80])*0.08
    # newY[50:80] = newY[50:80] - (80 - newX[50:80])*0.08
    # newY[50:80] = newY[50:80] - (80 - newX[50:80])*0.08
    # print(testX)
    # ax0.plot(newX / 10. + 0.5, newY, color='b', linewidth=0.9, label='σ = 5')

    # newX = data[:650, 0]*1.0
    # newY = data[:650, 1] - (650 - data[:650, 0])*0.02
    # ax0.plot(newX / 10. + 1.1, newY, color='g', linewidth=0.9, label='σ = 1')

    # newX = data[:750, 0]*1.0
    # newY = data[:750, 1] - (720 - data[:750, 0])*0.01
    # ax0.plot(newX / 10. + 1.1, newY, color='black')

    ax0.grid(True, 'both', 'both', alpha=0.3, linewidth=0.9)
    # ax0.set_xlim(-10, 600)
    # ax0.set_ylim(50, 100)
    ax0.set_ylabel('Loss Value', loc='top')
    ax0.set_xlabel('Train Epoches', loc='right')
    ax0.minorticks_on()

    plt.tight_layout()

    figure.savefig('exp8.svg', format='svg')
    plt.show()


if __name__ == "__main__":
    plot()
    # __init_datset()
    # __init_dataloader()
    # epoches = 300

    # print(get_device())

    # end = time.time()
    # device = get_device()

    # net1 = NormalNet().to(device)
    # net1 = nn.DataParallel(net1)
    # lossFunction1 = torch.nn.CrossEntropyLoss()
    # optimizer1 = torch.optim.Adam(net1.parameters())
    # train(net1, lossFunction1, optimizer1, 'defog', epoches)
