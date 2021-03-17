# coding=utf-8
from collections import OrderedDict
import torch.nn as nn
import torch.nn.init as init
import json
import os
from shutil import copyfile
import glob


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


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

def save_info(dict_data):
    path_save=dict_data["saved"]
    with open(os.path.join(path_save,"infor.json"),"w") as file:
        json.dump(dict_data,file)

def save_train_info(dict_data,path_save,config_file="config.py"):
    save_info(dict_data)
    copyfile(config_file,os.path.join(path_save,config_file))

def get_valid_gt(path,save):
    gt_files = glob.glob(path + "/*")
    for file in gt_files:
        fname = os.path.basename(file)
        print(fname)
        rs = []
        with open(file, "r") as gt_file:
            data = gt_file.readlines()
        with open(os.path.join(save,fname),"w+") as new_gt_file:
            for line in data:
                box = {}
                line = line.rstrip().replace("\n", "")
                line = line.split(",")
                lb=line[-1]
                line[-1]="Latin"
                if lb!="###":
                    len_lb=int(lb)
                    line.append('A'*len_lb)
                else:
                    line.append("###")
                new_gt_file.write(','.join(str(l) for l in line)+'\n')

if __name__ == '__main__':
    get_valid_gt("/home/aimenext/cuongdx/ufj/data/split/train/997/error/done_0119/gt_tool","/home/aimenext/cuongdx/ufj/data/split/train/997/error/done_0119/gt")