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

import numpy as np

from models import *
from utils import progress_bar

torch.manual_seed(0)

num_epochs = 50

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--filename', '-f', default='resnet.csv', help='name of file to write radings too?')
parser.add_argument('--model', '-m', default='resnet18', help='Resnet Model Name?')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=False, num_workers=4)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18()
if args.model == 'ResNet34':
    net = ResNet34()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=1e-4)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Training
def train(epoch):
    #print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    total_batchLoad_time = 0.0 
    total_training_time = 0.0
    total_inference_time = 0.0
    epoch_time =0.0
    training_loss = 0.0
    top_1 = 0.0
    t_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        start_batchLoad=time.perf_counter()
        inputs, targets = inputs.to(device), targets.to(device)
        stop_batchLoad=time.perf_counter()
        total_batchLoad_time = total_batchLoad_time + (stop_batchLoad - start_batchLoad) # needed for C2

        start_optimizer=time.perf_counter()
        optimizer.zero_grad()
        inference_time_start = time.perf_counter()
        outputs = net(inputs)
        inference_time_stop = time.perf_counter()
        total_inference_time = total_inference_time + (inference_time_stop - inference_time_start)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        stop_optimizer=time.perf_counter()
        total_training_time = total_training_time + (stop_optimizer-start_optimizer) # needed for C2

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        t_loss = train_loss/(batch_idx+1)

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #                       % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))        
        
        top_1 = 100.*correct/total

    # print("Data Loader Time: %.3lf sec, Training Time: %.3lf sec"% (total_batchLoad_time, total_training_time))
    # print("Average Data Loader Time per batch: %.6lf sec, Average Training Time per batch: %.6lf sec"% ((total_batchLoad_time/len(trainloader)), total_training_time/len(trainloader))) # needed for C7
    # print("Average Training Loss %0.3f, Top 1 percent accuracy %0.3f%%"%(training_loss, top_1))
    #return total_training_time, total_batchLoad_time, total_inference_time, t_loss, top_1
    return t_loss, top_1

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    total_batchLoad_time = 0.0 
    total_inference_time = 0.0
    top_1 = 0.0
    t_loss = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            start_batchLoad=time.perf_counter()
            inputs, targets = inputs.to(device), targets.to(device)
            stop_batchLoad=time.perf_counter()
            total_batchLoad_time = total_batchLoad_time + (stop_batchLoad - start_batchLoad) # needed for C2

            inference_time_start = time.perf_counter()
            outputs = net(inputs)
            inference_time_stop = time.perf_counter()
            total_inference_time = total_inference_time + (inference_time_stop - inference_time_start)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            t_loss = test_loss/(batch_idx+1)

            top_1 = 100.*correct/total
            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    #return total_batchLoad_time, total_inference_time, t_loss, top_1
    return t_loss, top_1


#https://deci.ai/blog/measure-inference-time-deep-neural-networks/
#Latency function
def measure_inference_latency():
    dummy_input = torch.ones(1, 3,32,32, dtype=torch.float).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 100
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for _ in range(10):
        _ = net(dummy_input)

    with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = net(dummy_input)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    #std_syn = np.std(timings)
    #print("Mean Latency " + str(mean_syn) + " [ms/image]")
    return mean_syn

avg_total_training_time = 0.0
training_epoch_total = 0.0
everthing_time_start = time.perf_counter()

f = open(args.filename, 'w', encoding="utf-8")
f.write("Epoch, Wall Time, Epoch time, Train_loss, Train_top_1, Test_loss, Test_top_1\r\n")

previous_epoch_time = 0

for epoch in range(start_epoch, start_epoch+num_epochs):
    training_epoch_start_time = time.perf_counter()
    tr = train(epoch)
    training_epoch_stop_time = time.perf_counter()
    training_epoch_total = training_epoch_total + (training_epoch_stop_time - training_epoch_start_time)
    avg_total_training_time = avg_total_training_time + training_epoch_total
    
    te = test(epoch)
    wall_time = time.perf_counter()-everthing_time_start
    epoch_time = wall_time - previous_epoch_time
    f.write(str(epoch+1)+","+str(wall_time)+","+str(epoch_time)+","+str(tr+te)[1:-1]+"\r\n")
    #scheduler.step()
    print("Epoch: %d Wall Time: %.6lf s Epoch time  %.6lf s Train Loss: %.6lf Train Acc: %.6lf Eval Loss: %.6lf Eval Acc: %.6lf"%(epoch+1, epoch_time, wall_time, tr[0], tr[1], te[0], te[1]))
    previous_epoch_time = wall_time

everthing_time_stop = time.perf_counter()
everthing_time = everthing_time_stop - everthing_time_start

f.write("\r\n")
#f.write("Total Time to finish the program, " + str(everthing_time) + " sec\r\n" + ",Average Running Time per epoch:, %.3lf sec, Average Training Time per epoch:, %.3lf sec \r\n"% ((avg_total_training_time/num_epochs), (avg_total_training_time/num_epochs)) )
latency = measure_inference_latency()
f.write("latency," + str(latency)+ ",in ms\r\n")
f.close()
print("Inference Latency (BS = 1): " +str(latency) + " [ms / image]")
#print("Total Time to finish the program " + str(everthing_time) + " sec\r\n")
#print("Average Running Time per epoch: %.3lf sec, Average Training Time per epoch: %.3lf sec \r\n"% ((avg_total_training_time/num_epochs), (avg_total_training_time/num_epochs)))
