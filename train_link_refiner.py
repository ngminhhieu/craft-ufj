import os
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import scipy.io as scio
import argparse
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import random
import h5py
import re
from refinenet import RefineNet
from data_loader import Data_LinkRefiner,showmat
from math import exp
from lossforlink import Maploss
from collections import OrderedDict
from craft import CRAFT
from torch.autograd import Variable
from multiprocessing import Pool
from datetime import datetime
from imgproc import denormalizeMeanVariance

# 3.2768e-5
random.seed(42)
PATH_SAVED = "/home/aimenext2/cuongdx/Models"

# class SynAnnotationTransform(object):
#     def __init__(self):
#         pass
#     def __call__(self, gt):
#         image_name = gt['imnames'][0]
parser = argparse.ArgumentParser(description='CRAFT reimplementation')

parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--batch_size', default=128, type=int,
                    help='batch size of training')
# parser.add_argument('--cdua', default=True, type=str2bool,
# help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--num_workers', default=32, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--region_thr', default=0.3, type=float,
                    help='region_thr')
parser.add_argument('--affinity_thr', default=0.5, type=int,
                    help='Number of workers used in dataloading')


args = parser.parse_args()
print(args)


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (0.8 ** step)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def freeze_craft(net):
    for layer in net.children():
        for param in layer.parameters():
            param.requires_grad = False
    return net

def load_net_gpu(net):
    net = net.cuda()
    net = torch.nn.DataParallel(net, device_ids=[0]).cuda()
    return  net

if __name__ == '__main__':

    craft_net = CRAFT()
    pre_trained_craft = "/home/aimenext2/cuongdx/Models/2910/432_2.0854.pth"
    craft_net.load_state_dict(copyStateDict(torch.load(pre_trained_craft)))
    craft_net = load_net_gpu(craft_net)
    craft_net=freeze_craft(craft_net)
    craft_net.eval()

    link_net=RefineNet()
    pre_trained="/home/aimenext2/cuongdx/Models/11_07_2019_1e-05/787_9.6003.pth"
    link_net.load_state_dict(copyStateDict(torch.load(pre_trained)))
    link_net=load_net_gpu(link_net)
    model_name = os.path.basename(pre_trained)

    cudnn.benchmark = False
    data = Data_LinkRefiner("/home/aimenext2/cuongdx/data/0411")
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=5,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True)

    # net.train()
    optimizer = optim.Adam(link_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = Maploss()
    # criterion = torch.nn.MSELoss(reduce=True, size_average=True)
    # net.train()

    step_index = 0
    now = datetime.now()
    format_time = now.strftime("%m_%d_%Y")
    format_time = format_time + "_%s" % (args.lr)
    print("format: ", format_time)
    saved = os.path.join(PATH_SAVED, format_time)
    if not os.path.exists(saved):
        os.mkdir(saved)
    loss_time = 0
    loss_value = 0
    compare_loss = 1
    init_epoch = int(model_name.split("_")[0])
    link_net.eval()
    for epoch in range(1000):
        if epoch < init_epoch:
            if epoch % 20 == 0 and epoch != 0:
                step_index = epoch // 20
                print("change lr")
                adjust_learning_rate(optimizer, args.gamma, step_index)
            continue
        train_time_st = time.time()
        loss_value = 0
        total_loss = 0
        if epoch % 20 == 0 and epoch != 0:
            step_index = epoch // 20
            print("change lr")
            adjust_learning_rate(optimizer, args.gamma, step_index)

        st = time.time()
        print("lr: ", optimizer.param_groups[0]['lr'])
        for index, (images, mask) in enumerate(data_loader):
            # imgs=images.permute(0,2,3,1).cpu().numpy()
            # print(imgs[0].shape)
            # msk=mask.cpu().numpy()
            images = Variable(images.type(torch.FloatTensor)).cuda()
            print(images.shape)
            mask = mask.type(torch.FloatTensor).cuda()
            mask = Variable(mask).cuda()
            with torch.no_grad():
                feature, upconv4 = craft_net(images)
            out=link_net(feature,upconv4)
            # for i in range(5):
            #     showmat("img",denormalizeMeanVariance(imgs[i]))
            #     print(images[i].shape)
            #     showmat("lb",np.uint8(msk[i]*255))
            #     showmat("pre",np.uint8(out[i].cpu().detach().numpy()*255))
            optimizer.zero_grad()
            out= out[:, :, :, 0].cuda()
            # print(out)
            loss = criterion(mask, out)
            loss.backward()
            optimizer.step()
            loss_value += loss.item()
            total_loss += loss_value / 2
            if index % 2 == 0 and index > 0:
                et = time.time()
                print(
                    'epoch {}:({}/{}) batch || training time for batch {} || training loss {} ||'.format(epoch, index,
                                                                                                           len(
                                                                                                               data_loader),
                                                                                                           et - st,
                                                                                                           loss_value ))
                loss_time = 0
                loss_value = 0
                st = time.time()

        torch.save(link_net.state_dict(), os.path.join(saved, repr(epoch) + "_" + "{:.6s}".format(
            str(total_loss)) + '.pth'))
        # test('/home/aimenext2/cuongdx/Models/2409/' + repr(epoch) + '.pth')
        # test('/data/CRAFT-pytorch/300_1.00000.pth')
        torch.cuda.empty_cache()
