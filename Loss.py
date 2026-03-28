# -*- coding: utf-8 -*

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb
import gc

def MI(outputs_target):
    batch_size = outputs_target.size(0)
    softmax_outs_t = nn.Softmax(dim=1)(outputs_target)  # 32,31
    avg_softmax_outs_t = torch.sum(softmax_outs_t, dim=0) / float(batch_size)  # 31
    log_avg_softmax_outs_t = torch.log(avg_softmax_outs_t)
    item1 = -torch.sum(avg_softmax_outs_t * log_avg_softmax_outs_t)
    item2 = -torch.sum(softmax_outs_t * torch.log(softmax_outs_t)) / float(batch_size)
    return item1 - item2

def CalculateMean(features, labels, class_num):
    N = features.size(0)   # 795
    C = class_num          # 31
    A = features.size(1)   # 256

    avg_CxA = torch.zeros(C, A).cuda()   # 31, 256
    NxCxFeatures = features.view(N, 1, A).expand(N, C, A)  # 795,31,256

    onehot = torch.zeros(N, C).cuda()   # 795,31
    onehot.scatter_(1, labels.view(-1, 1), 1)      # 标签值变为onehot向量
    NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)  # 795,31,256

    Amount_CxA = NxCxA_onehot.sum(0)   # 每类有多少个样本
    Amount_CxA[Amount_CxA == 0] = 1.0

    del onehot
    gc.collect()
    for c in range(class_num):
        c_temp = NxCxFeatures[:, c, :].mul(NxCxA_onehot[:, c, :])
        c_temp = torch.sum(c_temp, dim=0)
        avg_CxA[c] = c_temp / Amount_CxA[c]
    return avg_CxA.detach()

def Calculate_CV(features, labels, ave_CxA, class_num):
    N = features.size(0)
    C = class_num
    A = features.size(1)

    var_temp = torch.zeros(C, A, A).cuda()
    NxCxFeatures = features.view(N, 1, A).expand(N, C, A)

    onehot = torch.zeros(N, C).cuda()
    onehot.scatter_(1, labels.view(-1, 1), 1)
    NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

    Amount_CxA = NxCxA_onehot.sum(0)
    Amount_CxA[Amount_CxA == 0] = 1
    Amount_CxAxA = Amount_CxA.view(C, A, 1).expand(C, A, A)
    del Amount_CxA, onehot
    gc.collect()

    avg_NxCxA = ave_CxA.expand(N, C, A)
    for c in range(C):
        features_by_sort_c = NxCxFeatures[:, c, :].mul(NxCxA_onehot[:, c, :])
        avg_by_sort_c = avg_NxCxA[:, c, :].mul(NxCxA_onehot[:, c, :])
        var_temp_c = features_by_sort_c - avg_by_sort_c
        var_temp[c] = torch.mm(var_temp_c.permute(1,0), var_temp_c).div(Amount_CxAxA[c])
    return var_temp.detach()

class Cls_Loss(nn.Module):
    def __init__(self, class_num):
        super(Cls_Loss, self).__init__()
        self.class_num = class_num
        self.cross_entropy = nn.CrossEntropyLoss()

    def aug(self, s_mean_matrix, t_mean_matrix, fc, features, y_s, labels_s, t_cv_matrix, Lambda):
        N = features.size(0)  # 32
        C = self.class_num    # 31
        A = features.size(1)  # 256

        weight_m = list(fc.parameters())[0]   # 31,256
        NxW_ij = weight_m.expand(N, C, A)     # 32,31,256
        NxW_kj = torch.gather(NxW_ij, 1, labels_s.view(N, 1, 1).expand(N, C, A))

        t_CV_temp = t_cv_matrix[labels_s]  # 32,256,256

        sigma2 = Lambda * torch.bmm(torch.bmm(NxW_ij - NxW_kj, t_CV_temp), (NxW_ij - NxW_kj).permute(0, 2, 1)) # 32,31,31
        sigma2 = sigma2.mul(torch.eye(C).cuda().expand(N, C, C)).sum(2).view(N, C) # 32,31

        sourceMean_NxA = s_mean_matrix[labels_s]
        targetMean_NxA = t_mean_matrix[labels_s]
        dataMean_NxA = (targetMean_NxA - sourceMean_NxA)
        dataMean_NxAx1 = dataMean_NxA.expand(1, N, A).permute(1, 2, 0)  # 32,256,1

        del t_CV_temp, sourceMean_NxA, targetMean_NxA, dataMean_NxA
        gc.collect()

        dataW_NxCxA = NxW_ij - NxW_kj
        dataW_x_detaMean_NxCx1 = torch.bmm(dataW_NxCxA, dataMean_NxAx1)
        datW_x_detaMean_NxC = dataW_x_detaMean_NxCx1.view(N, C)

        aug_result = y_s + 0.5 * sigma2 + Lambda * datW_x_detaMean_NxC  # 32,31
        return aug_result

    def forward(self, fc, features_source: torch.Tensor, y_s, labels_source, Lambda, mean_source, mean_target, covariance_target):
        aug_y = self.aug(mean_source, mean_target, fc, features_source, y_s, labels_source, covariance_target, Lambda)
        loss = self.cross_entropy(aug_y, labels_source)
        return loss


# BSP


def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()   # 64,31
    feature = input_list[0]    # 64,256
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))  # 64,31,256
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))  #
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 2)
    N, C = size[0], size[1]
    feat_var = feat.view(N, C).var(dim=1) + eps
    feat_std = feat_var.sqrt().view(N, 1)
    feat_mean = feat.view(N, C).mean(dim=1).view(N, 1)
    return feat_mean, feat_std

def adain(content_feat, style_feat, idx, fs, ft):
    assert (content_feat.size()[1] == style_feat.size()[1])

    content = content_feat[idx]
    style = style_feat[idx]
    size = content.size()   # 32,256

    content_mean, content_std = calc_mean_std(content)
    style_mean, style_std = calc_mean_std(style)

    normalized_f_s = (fs - content_mean.expand(size)) / content_std.expand(size)
    f_s_sty = normalized_f_s * style_std.expand(size) + style_mean.expand(size)
    normalized_f_t = (ft - style_mean.expand(size)) / style_std.expand(size)
    f_t_ctt = normalized_f_t * content_std.expand(size) + content_mean.expand(size)

    return normalized_f_s, normalized_f_t, f_s_sty, f_t_ctt