'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

from resnet import *
from resnet_no_batch import *
from utils import progress_bar
#from prettytable import PrettyTable


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--cpu_gpu', '-c', default='cpu', help='CPU or GPU')
parser.add_argument('--optimizer', '-o', default='sgd', help='optimizer')
parser.add_argument('--workers', '-w', default=0, type=int, help='data loader workers')
parser.add_argument('--batchNorm', '-b', default='yes', help='Batch Norm yes or no?')
parser.add_argument('--parameters', '-p', default='hide', help='Batch Norm hide or show?')
args = parser.parse_args()

# needed for C5
device = 'cuda' if (torch.cuda.is_available() and args.cpu_gpu=='gpu') else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)


testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=32, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

net = ResNet18_no_batch() if args.batchNorm == 'no' else ResNet18()
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

# needed for C6
if args.optimizer =='sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
elif args.optimizer =='nesterov':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov = True)
elif args.optimizer =='adagrad':
    optimizer = optim.Adagrad(net.parameters(), lr=args.lr,  weight_decay=5e-4)
elif args.optimizer =='adadelta':
    optimizer = optim.Adadelta(net.parameters(), lr=args.lr)
elif args.optimizer =='adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Training
def train(epoch, trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    total_batchLoad_time = 0.0 
    total_training_time = 0.0
    epoch_time =0.0
    
    training_loss = 0.0
    top_1 = 0.0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        start_batchLoad=time.perf_counter()
        inputs, targets = inputs.to(device), targets.to(device)
        stop_batchLoad=time.perf_counter()
        total_batchLoad_time = total_batchLoad_time + (stop_batchLoad - start_batchLoad) # needed for C2

        start_optimizer=time.perf_counter()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        stop_optimizer=time.perf_counter()
        total_training_time = total_training_time + (stop_optimizer-start_optimizer) # needed for C2

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        training_loss = train_loss/(batch_idx+1)
        top_1 = 100.*correct/total

    progress_bar(len(trainloader)-1, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (training_loss, top_1, correct, total)) # needed for C2
    print("Data Loader Time: %.3lf sec, Training Time: %.3lf sec"% (total_batchLoad_time, total_training_time))
    print("Average Data Loader Time per batch: %.6lf sec, Average Training Time per batch: %.6lf sec"% ((total_batchLoad_time/len(trainloader)), total_training_time/len(trainloader))) # needed for C7
    print("Average Training Loss %0.3f, Top 1 percent accuracy %0.3f%%"%(training_loss, top_1))
    return total_training_time, training_loss, top_1

# def count_parameters(model):
#     table = PrettyTable(["Modules", "Parameters"])
#     total_params = 0
#     for name, parameter in model.named_parameters():
#         if not parameter.requires_grad: continue
#         params = parameter.numel()
#         table.add_row([name, params])
#         total_params+=params
#     print(table)
#     print(f"Total Trainable Params: {total_params}")
#     return total_params
    
def main():
    # needed for C
    if args.parameters =='show': 
        count_parameters(net)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=args.workers)
    
    avg_total_training_time = 0.0
    epoch_total = 0.0
    for epoch in range(5):
        epoch_start_time = time.perf_counter()
        training_time, training_loss, top_1 = train(epoch, trainloader)
        epoch_stop_time = time.perf_counter()
        epoch_total = epoch_total + (epoch_stop_time - epoch_start_time)
        
        avg_total_training_time = avg_total_training_time + training_time
        scheduler.step()

    print("Average Running Time per epoch: %.3lf sec, Average Training Time per epoch: %.3lf sec \n\n"% ((epoch_total/5), (avg_total_training_time/5)))
 
if __name__ == '__main__':
    main()