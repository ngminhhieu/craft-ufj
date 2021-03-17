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
from log_train import LogTrain

from data_loader import Synth80k, WordLevelDataset,CharLevelDataset
from math import exp
from mseloss import Maploss
from collections import OrderedDict
from PIL import Image
from torchvision.transforms import transforms
from craft import CRAFT
from torch.autograd import Variable
from multiprocessing import Pool
from datetime import datetime
from torchutil import save_train_info
import config
from shutil import  copyfile

# 3.2768e-5
random.seed(42)

parser = argparse.ArgumentParser(description='CRAFT')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--num_workers', default=32, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument("--path_save", default=config.path_saved, type=str)
parser.add_argument("--syn_data", default=config.synth_data, type=str)
parser.add_argument("--pretrained", type=str, required=True)
parser.add_argument("--word_data", default=config.word_data, nargs='+')
parser.add_argument("--char_data",default=config.char_data, nargs='+')
args = parser.parse_args()


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


if __name__ == '__main__':

    batch_size_char = config.batch_size_synthtext
    target_size=config.target_size
    data_syn = Synth80k(args.syn_data, target_size=target_size)
    train_loader_syn = torch.utils.data.DataLoader(
        data_syn,
        batch_size=batch_size_char,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True)
    batch_syn = iter(train_loader_syn)
    dict_infor = vars(args)
    dict_infor['batch_size_char'] = batch_size_char

    data_char = CharLevelDataset(args.char_data)
    train_loader_char = torch.utils.data.DataLoader(
        data_char,
        batch_size=batch_size_char,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True)
    batch_char = iter(train_loader_char)

    net = CRAFT()

    pre_trained = args.pretrained
    net.load_state_dict(copyStateDict(torch.load(pre_trained)))
    model_name = os.path.basename(pre_trained)

    net = net.cuda()
    net = torch.nn.DataParallel(net, device_ids=[0]).cuda()
    # net.train()

    cudnn.benchmark = False
    batch_size_word = config.batch_size_word
    realdata = WordLevelDataset(net, args.word_data, target_size=target_size, viz=False)
    real_data_loader = torch.utils.data.DataLoader(
        realdata,
        batch_size=batch_size_word,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True)

    dict_infor['batch_size_real'] = batch_size_word

    # net.train()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # criterion = torch.nn.MSELoss(reduce=True, size_average=True)
    # net.train()

    step_index = 0
    now = datetime.now()
    format_time = now.strftime("%Y_%m_%d_%H_%M")
    format_time = format_time + "_%s" % (args.lr)
    print("format: ", format_time)
    path_save=args.path_save
    saved = os.path.join(path_save, format_time)
    dict_infor["saved"] = saved
    if not os.path.exists(saved):
        os.mkdir(saved)
    # save_info(dict_infor)
    # copyfile("config.py",os.path.join(saved,"config.py"))
    save_train_info(dict_infor,saved,"config.py")

    loss_value = 0
    init_epoch = int(model_name.split("_")[0])
    step_index = init_epoch // 20
    adjust_learning_rate(optimizer, args.gamma, step_index)
    log_file = LogTrain(saved)
    criterion = Maploss(log_file)
    epochs_end=config.epochs_end
    nb_epoch_change_lr=config.nb_epochs_change_lr
    for epoch in range(init_epoch+1, epochs_end):
        train_time_st = time.time()
        loss_value = 0
        total_loss = 0
        if epoch % nb_epoch_change_lr == 0 and epoch != 0:
            step_index = epoch // nb_epoch_change_lr
            print("change lr")
            adjust_learning_rate(optimizer, args.gamma, step_index)

        st = time.time()
        print("lr: ", optimizer.param_groups[0]['lr'])
        net.eval()
        for index, (real_images, real_gh_label, real_gah_label, real_mask, _) in enumerate(real_data_loader):
            # net.train()
            # real_images, real_gh_label, real_gah_label, real_mask = next(batch_real)
            if random.random()<config.prob_syn:
                syn_images, syn_gh_label, syn_gah_label, syn_mask, __ = next(batch_syn)
            else:
                try:
                    syn_images, syn_gh_label, syn_gah_label, syn_mask, __ = next(batch_char)
                except:
                    batch_char = iter(train_loader_char)
                    syn_images, syn_gh_label, syn_gah_label, syn_mask, __ = next(batch_char)
                        
            net.train()
            images = torch.cat((syn_images, real_images), 0)
            gh_label = torch.cat((syn_gh_label, real_gh_label), 0)
            gah_label = torch.cat((syn_gah_label, real_gah_label), 0)
            mask = torch.cat((syn_mask, real_mask), 0)
            # affinity_mask = torch.cat((syn_mask, real_affinity_mask), 0)

            images = Variable(images.type(torch.FloatTensor)).cuda()
            # print(images.shape)
            gh_label = gh_label.type(torch.FloatTensor)
            gah_label = gah_label.type(torch.FloatTensor)
            gh_label = Variable(gh_label).cuda()
            gah_label = Variable(gah_label).cuda()
            mask = mask.type(torch.FloatTensor)
            mask = Variable(mask).cuda()

            out, _ = net(images)
            optimizer.zero_grad()

            out1 = out[:, :, :, 0].cuda()
            out2 = out[:, :, :, 1].cuda()
            loss = criterion(gh_label, gah_label, out1, out2, mask)
            # print("loss: ", loss)
            loss.backward()
            optimizer.step()
            loss_value += loss.item()
            total_loss += loss_value / 2
            et = time.time()
            log_info = 'epoch {}:({}/{}) || time {:.5f} || loss {:5f} || total loss {:5f} '.format(epoch, index,
                                                                                                           len(
                                                                                                               real_data_loader),
                                                                                                           et - st,
                                                                                                           loss_value / 2, total_loss)
            log_file.write(log_info + "\n")
            print(log_info)
            loss_value = 0
            st = time.time()

            net.eval()

        torch.save(net.state_dict(), os.path.join(saved, repr(epoch) + "_" + "{:.6s}".format(
            str(total_loss)) + '.pth'))
        torch.cuda.empty_cache()
