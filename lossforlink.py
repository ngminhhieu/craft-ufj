import numpy as np
import torch
import torch.nn as nn

THRESH_PIXEL=1

class Maploss(nn.Module):

    def __init__(self, use_gpu = True):

        super(Maploss,self).__init__()


    def single_image_loss(self, pre_loss, loss_label):
        batch_size = pre_loss.shape[0]
        sum_loss = torch.mean(pre_loss.view(-1))*0
        pre_loss = pre_loss.view(batch_size, -1)
        loss_label = loss_label.view(batch_size, -1)
        for i in range(batch_size):
            # print("shape: ",pre_loss[i].shape)
            # print("min=%s, max=%s"%(torch.min(pre_loss[i],dim=0),torch.max(pre_loss[i],dim=0)))
            average_number = 0
            positive_pixel = len(pre_loss[i][(loss_label[i] >= THRESH_PIXEL)])
            average_number += positive_pixel
            if positive_pixel != 0:
                posi_loss = torch.mean(pre_loss[i][(loss_label[i] >= THRESH_PIXEL)])
                sum_loss += posi_loss
                if len(pre_loss[i][(loss_label[i] <THRESH_PIXEL)]) < 3*positive_pixel:
                    nega_loss = torch.mean(pre_loss[i][(loss_label[i] < THRESH_PIXEL)])
                    average_number += len(pre_loss[i][(loss_label[i] < THRESH_PIXEL)])
                else:
                    nega_loss = torch.mean(torch.topk(pre_loss[i][(loss_label[i] <THRESH_PIXEL)], 3*positive_pixel)[0])
                    average_number += 3*positive_pixel
                sum_loss += nega_loss
            else:
                nega_loss = torch.mean(torch.topk(pre_loss[i], 500)[0])
                average_number += 500
                sum_loss += nega_loss

        return sum_loss

    def forward(self, ground_truth, predict):
        ground_truth = ground_truth
        predict = predict
        loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)
        assert predict.size() == ground_truth.size()
        loss = loss_fn(predict, ground_truth)
        link_loss = self.single_image_loss(loss, ground_truth)
        return link_loss/loss.shape[0]