import copy
import torch
from torch import nn
import numpy as np
from utils.dp_mechanism import cal_sensitivity, cal_sensitivity_MA, Laplace, Gaussian_Simple, Gaussian_MA


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedWeightAvg(w, size):
    totalSize = sum(size)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w[0][k]*size[0]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * size[i]
        # print(w_avg[k])
        w_avg[k] = torch.div(w_avg[k], totalSize)
    return w_avg

def FedDiff(w, w0, p=0):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] - w0[k]
        w_avg[k] = torch.div(w_avg[k], len(w)/(1+p))
    for k in w_avg.keys():
        w_avg[k] += w0[k]
    return w_avg

def FedAvg_serial(w):
    w_avg = copy.deepcopy(w[-1])
    for k in w_avg.keys():
        w_avg[k] = torch.div(w_avg[k]-w[0][k], len(w)-1)
    return w_avg

def FedAvg_Secagg(args, w, clients, reconstruct_secret, reconstruct_bu, total):
    print(f"模型聚合个数：{total}")
    w_avg = copy.deepcopy(w)
    for k in w_avg.keys():
        for id,seed in reconstruct_bu.items():
            local_random = np.random.default_rng(seed)
            sensitivity = clients[id].args.local_ep * cal_sensitivity_MA(clients[id].args.lr, clients[id].args.dp_clip, len(clients[id].idxs_sample))
            noise = -1 * local_random.normal(
                    loc=0,
                    scale=sensitivity * clients[id].noise_scale,  # 需要从args获取或计算
                    size=w_avg[k].shape
                )
            w_avg[k] += torch.from_numpy(noise).to(args.device)
        for (id, id1), seed in reconstruct_secret.items():
            local_random = np.random.default_rng(seed)
            sensitivity = clients[id].args.local_ep * cal_sensitivity_MA(clients[id].args.lr, clients[id].args.dp_clip, len(clients[id].idxs_sample))
            sign = 1 if id < id1 else -1
            noise = sign * local_random.normal(
                    loc=0,
                    scale=sensitivity * clients[id].noise_scale,  # 需要从args获取或计算
                    size=w_avg[k].shape
                )
            w_avg[k] += torch.from_numpy(noise).to(args.device)
        w_avg[k] = torch.div(w_avg[k], total)
    return w_avg