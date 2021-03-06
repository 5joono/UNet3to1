import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model_SASA import UNet
from dataset_SASA import *
from util import *
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import time

t1 = time.time()
parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=3e-4, type=float, dest="lr")
parser.add_argument("--channel", default=64, type=int, dest="channel")
parser.add_argument("--batch_size", default=1, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=30, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./dataset", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint_SASA", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log_SASA", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result_SASA", type=str, dest="result_dir")

parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

args = parser.parse_args()

## 트레이닝 파라메터 설정하기
lr = args.lr
channel = args.channel
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("learning rate: %.4e" % lr)
print("channel: %d" % channel)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("result dir: %s" % result_dir)
print("mode: %s" % mode)

## 디렉토리 생성하기
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

## 네트워크 학습하기
if mode == 'train':
    transform = transforms.Compose([Normalization(min = -100, max = 400),ToTensor()])

    dataset_train = Dataset(dir_data=os.path.join(data_dir, 'train'), transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

    dataset_val = Dataset(dir_data=os.path.join(data_dir, 'val'), transform=transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

    # 그밖에 부수적인 variables 설정하기
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)
else:
    transform = transforms.Compose([Normalization(min = -100, max = 400),ToTensor()])

    dataset_test = Dataset(dir_data=os.path.join(data_dir, 'test'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

    # 그밖에 부수적인 variables 설정하기
    num_data_test = len(dataset_test)

    num_batch_test = np.ceil(num_data_test / batch_size)

## 네트워크 생성하기
net = UNet(channel).to(device)

## 손실함수 정의하기
fn_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.78, 0.65, 8.57])).to(device)

## Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(), lr=lr)

## 그밖에 부수적인 functions 설정하기
fn_tonumpy = lambda x: x.to('cpu').detach().numpy()
fn_class = lambda x: torch.argmax(x, dim=1)

## Tensorboard 를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

## 네트워크 학습시키기
st_epoch = 0

# TRAIN MODE
if mode == 'train':
    if train_continue == "on":
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):
    
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)
            input_prev = data['input_prev'].to(device)
            input_next = data['input_next'].to(device)
            output = net(input_prev,input,input_next)

            # backward pass
            optim.zero_grad()
            
            loss = fn_loss(output, label)
            loss.backward()

            optim.step()

            # 손실함수 계산
            loss_arr += [loss.item()]

            print("TRAIN: EPOCH %05d / %05d | BATCH %05d / %05d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            # Tensorboard 저장하기
            '''label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            '''
            t2 = time.time();print("%04f sec" % (t2-t1));t1=t2;

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        with torch.no_grad():
            net.eval()
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)
                input_prev = data['input_prev'].to(device)
                input_next = data['input_next'].to(device)
                output = net(input_prev,input,input_next)

                # 손실함수 계산하기
                loss = fn_loss(output, label)

                loss_arr += [loss.item()]

                print("VALID: EPOCH %05d / %05d | BATCH %05d / %05d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

                # Tensorboard 저장하기
                '''label = fn_tonumpy(label)
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_class(output))

                writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                '''
        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        if epoch % 5 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()
    writer_val.close()

# TEST MODE
else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            # forward pass
            name = data['name']
            label = data['label'].to(device)
            input = data['input'].to(device)
            input_prev = data['input_prev'].to(device)
            input_next = data['input_next'].to(device)
            output = net(input_prev,input,input_next)

            # 손실함수 계산하기
            loss = fn_loss(output, label)

            loss_arr += [loss.item()]

            print("TEST: BATCH %05d / %05d | LOSS %.4f" %
                  (batch, num_batch_test, np.mean(loss_arr)))

            # Tensorboard 저장하기
            label = fn_tonumpy(label.unsqueeze(dim=1))
            input = fn_tonumpy(input.squeeze(dim=1))
            output = fn_tonumpy(fn_class(output))
            for j in range(label.shape[0]):
                plt.imsave(os.path.join(result_dir, 'png', f'label_{name[j]}.png'), label, cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', f'input_{name[j]}.png'), input, cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', f'output_{name[j]}.png'), output, cmap='gray')
                np.save(os.path.join(result_dir, 'numpy', f'label_{name[j]}.npy'), label)
                np.save(os.path.join(result_dir, 'numpy', f'input_{name[j]}.npy'), input)
                np.save(os.path.join(result_dir, 'numpy', f'output_{name[j]}.npy'), output)

    print("AVERAGE TEST: BATCH %05d / %05d | LOSS %.4f" %
          (batch, num_batch_test, np.mean(loss_arr)))

