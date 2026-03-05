import torch
from torch import nn, autograd
from utils.dp_mechanism import cal_sensitivity, cal_sensitivity_MA, Laplace, Gaussian_Simple, Gaussian_MA
from cryptography.hazmat.primitives.asymmetric import dh
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from random import Random
from sklearn import metrics
import cProfile
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import torch.nn.functional as F
import secrets
from tqdm import tqdm
import time
from random import randint
from typing import Union, Dict
import math

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def calculate_noise_scale(args, times):
    if args.dp_mechanism == 'Laplace':
        epsilon_single_query = args.dp_epsilon / times
        return Laplace(epsilon=epsilon_single_query)
    elif args.dp_mechanism == 'Gaussian':
        epsilon_single_query = args.dp_epsilon / times
        delta_single_query = args.dp_delta / times
        return Gaussian_Simple(epsilon=epsilon_single_query, delta=delta_single_query)
    elif args.dp_mechanism == 'MA':
        return Gaussian_MA(epsilon=args.dp_epsilon, delta=args.dp_delta, q=args.dp_sample, epoch=times)

class LocalUpdateDP(object):
    def __init__(self, args, noise_scale, dh_param, id, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.idxs_sample = np.random.choice(list(idxs), int(self.args.dp_sample * len(idxs)), replace=False)
        self.ldr_train = DataLoader(DatasetSplit(dataset, self.idxs_sample), batch_size=min(args.local_bs, len(self.idxs_sample)),
                                    shuffle=True)
        self.idxs = idxs
        self.dataset = dataset
        self.times = self.args.epochs * self.args.frac
        self.lr = args.lr
        self.noise_scale = noise_scale
        self.dh_param = dh_param
        self.id = id
        self.shared_keys = dict()
        self.personal_mask_seeds = dict()
        self.share_bu = dict()
        self.received_shares = dict()
        
        self.noise_sum = 0
        self.round = 0

        self.sigma_k = 0.0
        self.neg_sigma = 0.0
        self.noise_dict = {id: {} for id in range(self.args.num_users)}


    def generate_keys(self):
        self.private_key = self.dh_param.generate_private_key()
        self.public_key = self.private_key.public_key()
        return self.private_key, self.public_key

    def exchange_keys(self, keys_dict):
        self.shared_keys.clear()
        for id, public_key in keys_dict.items():
            if id == self.id:
                continue
            self.shared_keys[id] = int.from_bytes(self.private_key.exchange(public_key)[-2:], byteorder="big")
        return self.shared_keys

    def generate_mask_seed(self):
        return secrets.randbelow(65536)  # 2B
    
    def store_share(self, from_id, target_id, with_id, share):
        key = (from_id, target_id, with_id)
        self.received_shares[key] = share
    
    def shamir_share(self, secret: Union[bytes, int], t: int, n: int, prime: int = 65537) -> Dict[int, int]:
        if isinstance(secret, bytes):
            secret_int = int.from_bytes(secret, byteorder="big")
        elif isinstance(secret, int):
            secret_int = secret
        else:
            raise TypeError("Secret must be bytes or int")

        coeffs = [secret_int] + [randint(0, prime - 1) for _ in range(t - 1)]
        shares = {}
        for i in range(1, n + 1):  
            fx = sum([coeffs[j] * pow(i, j, prime) for j in range(t)]) % prime
            shares[i] = fx
        return shares

    def share_mask_seeds_and_shared_keys(self, clients):
        n = len(clients)
        assert 0 <= self.args.d < 1, "self.args.d 应该是一个合理的掉线比例(0~1 之间）"
        t = math.floor((1 - self.args.d) * n)
        t = max(t, 1)

        # 分割 bu
        shares = self.shamir_share(self.personal_mask_seeds[self.id], t=t, n=n)
        self.share_bu = shares
        #print(f"pu_shares{self.share_bu}")

        communication_bytes = 0  # 初始化通信开销统计

        # 每个 bu share 大小约为 127 位
        for share in shares.values():
            communication_bytes += (share.bit_length() + 7) // 8

        # 分割 shared_keys
        for with_id in self.shared_keys:
            key_shares = self.shamir_share(self.shared_keys[with_id], t=t, n=n)
            for id, share in key_shares.items():
                self.store_share(self.id, id-1 , with_id, share)
                communication_bytes += (share.bit_length() + 7) // 8
        
        if self.args.account:
            return communication_bytes
    def shamir_reconstruct(self, shares, prime: int = 65537) -> int:
        if isinstance(shares, dict):
            shares = list(shares.items())  
            
        if isinstance(shares, list) and all(isinstance(s, int) for s in shares):
            shares = list(enumerate(shares, start=1))
        
        def lagrange_basis(j, x_values):
            num = 1
            denom = 1
            xj = x_values[j]
            for m in range(len(x_values)):
                if m != j:
                    xm = x_values[m]
                    num = (num * (-xm)) % prime
                    denom = (denom * (xj - xm)) % prime
            return num * pow(denom, -1, prime) % prime

        x_vals = [x for x, _ in shares]
        y_vals = [y for _, y in shares]

        secret = 0
        for j in range(len(shares)):
            lj = lagrange_basis(j, x_vals)
            secret = (secret + y_vals[j] * lj) % prime
        return secret

    def get_sensitivity(self, id, users_num):
        assert(self.args.dp_mechanism == 'MA' or self.args.dp_mechanism == 'Gaussian')
        sensitivity = self.args.local_ep * cal_sensitivity_MA(self.args.lr, self.args.dp_clip, len(self.idxs_sample))
        return sensitivity
            


    def train(self, net):
        self.idxs_sample = np.random.choice(list(self.idxs), int(self.args.dp_sample * len(self.idxs)), replace=False)
        self.ldr_train = DataLoader(DatasetSplit(self.dataset, self.idxs_sample), batch_size=min(self.args.local_bs, len(self.idxs_sample)),
                                    shuffle=True)
        net.train()
        self.round += 1
        # if self.round == 20:
        #     self.lr /= 10
        if self.args.optim == 'Adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        elif self.args.optim == 'SGD':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.lr)
        else:
            print('optim error')
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=self.args.lr_decay)
        self.loss_func = nn.CrossEntropyLoss()
        privacy_engine = PrivacyEngine(accountant='rdp')
        net, optimizer, self.loss_func, self.ldr_train = privacy_engine.make_private(
            module=net,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            data_loader=self.ldr_train,
            # target_delta=self.args.dp_delta,
            # target_epsilon=self.args.dp_epsilon,
            # epochs=self.args.epochs,
            noise_multiplier=0,
            max_grad_norm=self.args.dp_clip,
            grad_sample_mode='ghost',
        )
        total = 0
        correct = 0
        epoch_loss = []
        total = []
        correct = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for images, labels in self.ldr_train:
                if images.size(0) == 0:
                    continue
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                if labels.dim() > 1 and labels.size(1) == 1:
                    labels = labels.squeeze(1)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                # self.clip_gradients(net)
                optimizer.step()
                scheduler.step()
                # add noises to parameters
                # if self.args.dp_mechanism != 'no_dp' and self.args.algorithm == 'DP-SGD':
                #     self.add_noise(net)

                batch_loss.append(loss.item())

                # Compute training accuracy
                _, predicted = torch.max(log_probs, 1)  # Get the class with highest probability

                total.append(labels.size(0))  # Number of labels in the batch
                correct.append((predicted == labels).sum().item())  # Count correct predictions
                predicted, log_probs, images, labels = None, None, None, None
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        s = time.time()
        self.add_noise(net)
        self.lr = scheduler.get_last_lr()[0]
        state_dict = net.state_dict()
        self.loss_func = None
        optimizer = None
        net = None
        loss_ret = sum(epoch_loss) / len(epoch_loss)
        acc_ret = sum(correct) / sum(total)
        epoch_loss = None
        return state_dict, loss_ret, acc_ret

    def clip_gradients(self, net):
        if self.args.dp_mechanism == 'Laplace':
            # Laplace use 1 norm
            self.per_sample_clip(net, self.args.dp_clip, norm=1)
        elif self.args.dp_mechanism == 'Gaussian' or self.args.dp_mechanism == 'MA':
            # Gaussian use 2 norm
            self.per_sample_clip(net, self.args.dp_clip, norm=2)
        elif self.args.dp_mechanism == 'no_dp':
            self.per_sample_clip(net, self.args.dp_clip, norm=2)

    def per_sample_clip(self, net, clipping, norm):
        grad_samples = [x.grad_sample for x in net.parameters()]
        per_param_norms = [
            g.reshape(len(g), -1).norm(norm, dim=-1) for g in grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(norm, dim=1)
        per_sample_clip_factor = (
            torch.div(clipping, (per_sample_norms + 1e-6))
        ).clamp(max=1.0)
        for grad in grad_samples:
            factor = per_sample_clip_factor.reshape(per_sample_clip_factor.shape + (1,) * (grad.dim() - 1))
            grad.detach().mul_(factor.to(grad.device))
        # average per sample gradient after clipping and set back gradient
        for param in net.parameters():
            param.grad = param.grad_sample.detach().mean(dim=0)

    def add_noise(self, net):
        sensitivity = cal_sensitivity(self.lr, self.args.dp_clip, len(self.idxs_sample))
        state_dict = net.state_dict()
        if self.args.algorithm == "DP-SGD":
            if self.args.dp_mechanism == 'Laplace':
                for k, v in state_dict.items():
                    state_dict[k] += torch.from_numpy(np.random.laplace(loc=0, scale=sensitivity * self.noise_scale,
                                                                        size=v.shape)).to(self.args.device)
            elif self.args.dp_mechanism == 'Gaussian':
                for k, v in state_dict.items():
                    state_dict[k] += torch.from_numpy(np.random.normal(loc=0, scale=sensitivity * self.noise_scale,
                                                                    size=v.shape)).to(self.args.device)
            elif self.args.dp_mechanism == 'MA':
                sensitivity = cal_sensitivity_MA(self.args.lr, self.args.dp_clip, len(self.idxs_sample))
                for k, v in state_dict.items():
                    noise = np.random.normal(loc=0, scale=sensitivity * self.noise_scale, size=v.shape)
                    if self.args.debug_mode:
                        self.noise_sum += np.sum(noise)
                    state_dict[k] += torch.from_numpy(noise).to(self.args.device)
        elif self.args.algorithm == "DP-NSA":
            assert(self.args.dp_mechanism == 'MA' or self.args.dp_mechanism == 'Gaussian')
            sensitivity = self.args.local_ep * cal_sensitivity_MA(self.args.lr, self.args.dp_clip, len(self.idxs_sample))
            # 预初始化随机数生成器
            torch_generators = {
                other_id: torch.Generator(device=self.args.device).manual_seed(seed)
                for other_id, seed in self.shared_keys.items()
                if other_id != self.id
            }

            if self.args.debug_mode:
                self.noise_sum = 0.0

            num_peers = len(self.shared_keys) - 1  # 除了自己以外的节点数
            scale_base = (1 / (1 - self.args.k)) * sensitivity * self.noise_scale / num_peers

            # 遍历模型参数
            for k, v in state_dict.items():
                total_noise = torch.zeros_like(v, device=self.args.device)

                for other_id, gen in torch_generators.items():
                    sign = 1 if other_id > self.id else -1

                    noise = torch.normal(
                        mean=0.0,
                        std=scale_base,
                        size=v.shape,
                        generator=gen,
                        device=self.args.device
                    ) * sign

                    if self.args.debug_mode:
                        noise_sum_current = noise.sum().item()
                        self.noise_sum += noise_sum_current
                        print(f'id = {other_id}, self_id = {self.id}, seed = {gen.initial_seed()}, noise_sum = {noise_sum_current}')

                    total_noise += noise

                # 累加最终噪声
                state_dict[k] += total_noise

                if self.args.debug_mode:
                    print(f'{k} total_noise_sum = {total_noise.sum().item()}')

        elif self.args.algorithm == "NISS":
            assert(self.args.dp_mechanism == 'MA' or self.args.dp_mechanism == 'Gaussian')
            bs = min(self.args.local_bs,len(self.idxs_sample))
            sensitivity = self.args.local_ep * cal_sensitivity_MA(self.args.lr, self.args.dp_clip, bs) * (len(self.idxs_sample) // bs)
            # 初始化总噪声记录
            if self.args.debug_mode:
                self.noise_sum = 0.0

            # 生成扰动因子（常数乘法）
            mod_scale = 2 * self.args.k - 1 if self.args.k > 0.5 else 0.0
            modifiers = torch.normal(mean=1.0, std=mod_scale, size=[self.args.num_users], device=self.args.device)
            # 遍历模型参数
            for k, v in state_dict.items():
                total_noise = torch.zeros_like(v, device=self.args.device)

                for other_id in range(self.args.num_users):
                    if other_id == self.id:
                        continue

                    # 初始化 PyTorch 生成器
                    self_gen = torch.Generator(device=self.args.device)
                    self_gen.manual_seed(other_id * self.id + self.id)
                    #print(f"self.id,other_id,self_seed = {self.id},{other_id},{other_id * self.id + self.id}")

                    other_gen = torch.Generator(device=self.args.device)
                    other_gen.manual_seed(other_id * self.id + other_id)
                    #print(f"self.id,other_id,other_seed = {self.id},{other_id},{other_id * self.id + other_id}")

                    scale = sensitivity * self.noise_scale / (self.args.num_users - 1)
                    # print("scale",sensitivity, self.noise_scale, scale)
                    # 生成噪声张量（直接在目标设备上）
                    noise_self = torch.normal(
                        mean=0.0,
                        std=scale,
                        size=v.shape,
                        generator=self_gen,
                        device=self.args.device
                    )

                    noise_other = torch.normal(
                        mean=0.0,
                        std=scale,
                        size=v.shape,
                        generator=other_gen,
                        device=self.args.device
                    )

                    noise = noise_self - noise_other * modifiers[other_id]

                    if self.args.debug_mode:
                        noise_sum_current = noise.sum().item()
                        self.noise_sum += noise_sum_current
                        # print(f'id = {other_id}, self_id = {self.id}, noise_sum = {noise_sum_current}')

                    total_noise += noise

                # 一次性加总噪声到参数
                state_dict[k] += total_noise

                if self.args.debug_mode:
                    print(f'{k} total_noise_sum = {total_noise.sum().item()}')
        elif self.args.algorithm == "Secagg":
            for id, seed in self.shared_keys.items():
                if id == self.id:
                    continue
                local_random = np.random.default_rng(seed)
                sign = 1 if id > self.id else -1
                if self.args.debug_mode:
                    print(f'self_id = {self.id}, id = {id}, seed = {seed}')
                for k, v in state_dict.items():
                    mask = sign * local_random.uniform(low=-1, high=1, size=v.shape).astype(np.float32)
                    if self.args.debug_mode:
                        self.noise_sum += np.sum(mask)
                        print('client.self_id = {}, id = {}, seed = {}, noise_sum = {}'.format(self.id, id, seed, np.sum(mask)))
                    state_dict[k] += torch.from_numpy(mask).to(self.args.device)
                    
            if self.args.d > 0: #if exit dropout clients need online upload personal mask
                bu = self.personal_mask_seeds[self.id]
                local_random = np.random.default_rng(bu)
                #print("add personal mask")
                for k, v in state_dict.items():
                    noise = sign * local_random.normal(loc=0, scale=sensitivity * self.noise_scale / (len(self.shared_keys) - 1), size=v.shape)
                    if self.args.debug_mode:
                        self.noise_sum += np.sum(noise)
                        print('self_id = {}, personal_seed = {}, noise_sum = {}'.format(self.id, self.personal_mask_seeds[self.id], np.sum(noise)))
                    state_dict[k] += torch.from_numpy(noise).to(self.args.device)


        net.load_state_dict(state_dict)


#=============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================


class LocalUpdateDPMIA(object):
    def __init__(self, args, noise_scale, dh_param, id, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.idxs_sample = np.random.choice(list(idxs), int(self.args.dp_sample * len(idxs)), replace=False)
        self.ldr_train = None

        self.idxs = idxs
        self.dataset = dataset
        self.times = self.args.epochs * self.args.frac
        self.lr = args.lr
        self.noise_scale = noise_scale
        self.dh_param = dh_param
        self.id = id
        self.shared_keys = dict()
        self.personal_mask_seeds = dict()
        self.share_bu = dict()
        self.received_shares = dict()
        
        self.noise_sum = 0
        self.round = 0

        self.sigma_k = 0.0
        self.neg_sigma = 0.0
        self.noise_dict = {id: {} for id in range(self.args.num_users)}

    def generate_keys(self):
        self.private_key = self.dh_param.generate_private_key()
        self.public_key = self.private_key.public_key()
        return self.private_key, self.public_key

    def exchange_keys(self, keys_dict):
        self.shared_keys.clear()
        for id, public_key in keys_dict.items():
            if id == self.id:
                continue
            self.shared_keys[id] = int.from_bytes(self.private_key.exchange(public_key)[-2:], byteorder="big")
        return self.shared_keys

    def generate_mask_seed(self):
        return secrets.randbelow(65536)  # 2B
    
    def store_share(self, from_id, target_id, with_id, share):
        key = (from_id, target_id, with_id)
        self.received_shares[key] = share
    
    def shamir_share(self, secret: Union[bytes, int], t: int, n: int, prime: int = 65537) -> Dict[int, int]:
        if isinstance(secret, bytes):
            secret_int = int.from_bytes(secret, byteorder="big")
        elif isinstance(secret, int):
            secret_int = secret
        else:
            raise TypeError("Secret must be bytes or int")

        coeffs = [secret_int] + [randint(0, prime - 1) for _ in range(t - 1)]
        shares = {}
        for i in range(1, n + 1):  
            fx = sum([coeffs[j] * pow(i, j, prime) for j in range(t)]) % prime
            shares[i] = fx
        return shares

    def share_mask_seeds_and_shared_keys(self, clients):
        n = len(clients)
        assert 0 <= self.args.d < 1, "self.args.d 应该是一个合理的掉线比例(0~1 之间）"
        t = math.floor((1 - self.args.d) * n)
        t = max(t, 1)

        # 分割 bu
        shares = self.shamir_share(self.personal_mask_seeds[self.id], t=t, n=n)
        self.share_bu = shares
        #print(f"pu_shares{self.share_bu}")

        communication_bytes = 0  # 初始化通信开销统计

        # 每个 bu share 大小约为 127 位
        for share in shares.values():
            communication_bytes += (share.bit_length() + 7) // 8

        # 分割 shared_keys
        for with_id in self.shared_keys:
            key_shares = self.shamir_share(self.shared_keys[with_id], t=t, n=n)
            for id, share in key_shares.items():
                self.store_share(self.id, id-1 , with_id, share)
                communication_bytes += (share.bit_length() + 7) // 8
        
        if self.args.account:
            return communication_bytes
    def shamir_reconstruct(self, shares, prime: int = 65537) -> int:
        if isinstance(shares, dict):
            shares = list(shares.items())  
            
        if isinstance(shares, list) and all(isinstance(s, int) for s in shares):
            shares = list(enumerate(shares, start=1))
        
        def lagrange_basis(j, x_values):
            num = 1
            denom = 1
            xj = x_values[j]
            for m in range(len(x_values)):
                if m != j:
                    xm = x_values[m]
                    num = (num * (-xm)) % prime
                    denom = (denom * (xj - xm)) % prime
            return num * pow(denom, -1, prime) % prime

        x_vals = [x for x, _ in shares]
        y_vals = [y for _, y in shares]

        secret = 0
        for j in range(len(shares)):
            lj = lagrange_basis(j, x_vals)
            secret = (secret + y_vals[j] * lj) % prime
        return secret

    def get_sensitivity(self, id, users_num):
        assert(self.args.dp_mechanism == 'MA' or self.args.dp_mechanism == 'Gaussian')
        sensitivity = self.args.local_ep * cal_sensitivity_MA(self.args.lr, self.args.dp_clip, len(self.idxs_sample))
        return sensitivity
            

    def train(self, net, dataloader, lr):
        net.train()
        if self.args.optim == 'Adam':
            optimizer = torch.optim.AdamW(net.parameters(), lr=lr,
                                weight_decay=0.0005)
        elif self.args.optim == 'SGD':
            optimizer = torch.optim.SGD(net.parameters(), lr=lr,momentum=0.9,
                                weight_decay=0.0005)
        else:
            print('optim error')
        train_ldr = dataloader 
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=self.args.lr_decay)
        self.loss_func = nn.CrossEntropyLoss()
        privacy_engine = PrivacyEngine(accountant='rdp')
        net, optimizer, self.loss_func,  self.ldr_train= privacy_engine.make_private(
            module=net,
            optimizer=optimizer,
            criterion=self.loss_func,
            data_loader=train_ldr,
            # target_delta=self.args.dp_delta,
            # target_epsilon=self.args.dp_epsilon,
            # epochs=self.args.epochs,
            noise_multiplier=0,
            max_grad_norm=self.args.dp_clip,
            grad_sample_mode='ghost',
        )

        total = []
        correct = []
        epoch_loss = []

        for epoch in range(self.args.local_ep):
            batch_loss = []
            for images, labels in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                # scheduler.step()
                batch_loss.append(loss.item())
                _, predicted = torch.max(log_probs, 1)  # Get the class with highest probability

                total.append(labels.size(0))  # Number of labels in the batch
                correct.append((predicted == labels).sum().item())  # Count correct predictions
                predicted, log_probs, images, labels = None, None, None, None
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        if self.args.dp_mechanism != 'no_dp':
            self.add_noise(net)
        # self.lr = self.args.lr = scheduler.get_last_lr()[0]
        state_dict = net.state_dict()
        self.loss_func = None
        optimizer = None
        net = None
        loss_ret = sum(epoch_loss) / len(epoch_loss)
        acc_ret = sum(correct) / sum(total)
        epoch_loss = None
        epoch_acc = None
        return state_dict, loss_ret, acc_ret
    def clip_gradients(self, net):
        if self.args.dp_mechanism == 'Laplace':
            # Laplace use 1 norm
            self.per_sample_clip(net, self.args.dp_clip, norm=1)
        elif self.args.dp_mechanism == 'Gaussian' or self.args.dp_mechanism == 'MA':
            # Gaussian use 2 norm
            self.per_sample_clip(net, self.args.dp_clip, norm=2)
        elif self.args.dp_mechanism == 'no_dp':
            self.per_sample_clip(net, self.args.dp_clip, norm=2)

    def per_sample_clip(self, net, clipping, norm):
        grad_samples = [x.grad_sample for x in net.parameters()]
        per_param_norms = [
            g.reshape(len(g), -1).norm(norm, dim=-1) for g in grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(norm, dim=1)
        per_sample_clip_factor = (
            torch.div(clipping, (per_sample_norms + 1e-6))
        ).clamp(max=1.0)
        for grad in grad_samples:
            factor = per_sample_clip_factor.reshape(per_sample_clip_factor.shape + (1,) * (grad.dim() - 1))
            grad.detach().mul_(factor.to(grad.device))
        # average per sample gradient after clipping and set back gradient
        for param in net.parameters():
            param.grad = param.grad_sample.detach().mean(dim=0)

    def add_noise(self, net):
        sensitivity = cal_sensitivity(self.lr, self.args.dp_clip, len(self.idxs_sample))
        state_dict = net.state_dict()
        if self.args.algorithm == "DP-SGD":
            if self.args.dp_mechanism == 'Laplace':
                for k, v in state_dict.items():
                    state_dict[k] += torch.from_numpy(np.random.laplace(loc=0, scale=sensitivity * self.noise_scale,
                                                                        size=v.shape)).to(self.args.device)
            elif self.args.dp_mechanism == 'Gaussian':
                for k, v in state_dict.items():
                    state_dict[k] += torch.from_numpy(np.random.normal(loc=0, scale=sensitivity * self.noise_scale,
                                                                    size=v.shape)).to(self.args.device)
            elif self.args.dp_mechanism == 'MA':
                sensitivity = cal_sensitivity_MA(self.args.lr, self.args.dp_clip, len(self.idxs_sample))
                for k, v in state_dict.items():
                    noise = np.random.normal(loc=0, scale=sensitivity * self.noise_scale, size=v.shape)
                    if self.args.debug_mode:
                        self.noise_sum += np.sum(noise)
                    state_dict[k] += torch.from_numpy(noise).to(self.args.device)
        elif self.args.algorithm == "DP-NSA":
            assert(self.args.dp_mechanism == 'MA' or self.args.dp_mechanism == 'Gaussian')
            sensitivity = self.args.local_ep * cal_sensitivity_MA(self.args.lr, self.args.dp_clip, len(self.idxs_sample))
            # 预初始化随机数生成器
            torch_generators = {
                other_id: torch.Generator(device=self.args.device).manual_seed(seed)
                for other_id, seed in self.shared_keys.items()
                if other_id != self.id
            }

            if self.args.debug_mode:
                self.noise_sum = 0.0

            num_peers = len(self.shared_keys) - 1  # 除了自己以外的节点数
            scale_base = (1 / (1 - self.args.k)) * sensitivity * self.noise_scale / num_peers

            # 遍历模型参数
            for k, v in state_dict.items():
                total_noise = torch.zeros_like(v, device=self.args.device)

                for other_id, gen in torch_generators.items():
                    sign = 1 if other_id > self.id else -1

                    noise = torch.normal(
                        mean=0.0,
                        std=scale_base,
                        size=v.shape,
                        generator=gen,
                        device=self.args.device
                    ) * sign

                    if self.args.debug_mode:
                        noise_sum_current = noise.sum().item()
                        self.noise_sum += noise_sum_current
                        print(f'id = {other_id}, self_id = {self.id}, seed = {gen.initial_seed()}, noise_sum = {noise_sum_current}')

                    total_noise += noise

                # 累加最终噪声
                state_dict[k] += total_noise

                if self.args.debug_mode:
                    print(f'{k} total_noise_sum = {total_noise.sum().item()}')

        elif self.args.algorithm == "NISS":
            assert(self.args.dp_mechanism == 'MA' or self.args.dp_mechanism == 'Gaussian')
            bs = min(self.args.local_bs,len(self.idxs_sample))
            sensitivity = self.args.local_ep * cal_sensitivity_MA(self.args.lr, self.args.dp_clip, bs) * (len(self.idxs_sample) // bs)
            # 初始化总噪声记录
            if self.args.debug_mode:
                self.noise_sum = 0.0

            # 生成扰动因子（常数乘法）
            mod_scale = 2 * self.args.k - 1 if self.args.k > 0.5 else 0.0
            modifiers = torch.normal(mean=1.0, std=mod_scale, size=[self.args.num_users], device=self.args.device)
            # 遍历模型参数
            for k, v in state_dict.items():
                total_noise = torch.zeros_like(v, device=self.args.device)

                for other_id in range(self.args.num_users):
                    if other_id == self.id:
                        continue

                    # 初始化 PyTorch 生成器
                    self_gen = torch.Generator(device=self.args.device)
                    self_gen.manual_seed(other_id * self.id + self.id)
                    #print(f"self.id,other_id,self_seed = {self.id},{other_id},{other_id * self.id + self.id}")

                    other_gen = torch.Generator(device=self.args.device)
                    other_gen.manual_seed(other_id * self.id + other_id)
                    #print(f"self.id,other_id,other_seed = {self.id},{other_id},{other_id * self.id + other_id}")

                    scale = sensitivity * self.noise_scale / (self.args.num_users - 1)
                    # print("scale",sensitivity, self.noise_scale, scale)
                    # 生成噪声张量（直接在目标设备上）
                    noise_self = torch.normal(
                        mean=0.0,
                        std=scale,
                        size=v.shape,
                        generator=self_gen,
                        device=self.args.device
                    )

                    noise_other = torch.normal(
                        mean=0.0,
                        std=scale,
                        size=v.shape,
                        generator=other_gen,
                        device=self.args.device
                    )

                    noise = noise_self - noise_other * modifiers[other_id]

                    #if self.args.debug_mode:
                    noise_sum_current = noise.sum().item()
                    self.noise_sum += noise_sum_current
                        # print(f'id = {other_id}, self_id = {self.id}, noise_sum = {noise_sum_current}')

                    total_noise += noise

                # 一次性加总噪声到参数
                state_dict[k] += total_noise

                if self.args.debug_mode:
                    print(f'{k} total_noise_sum = {total_noise.sum().item()}')
        elif self.args.algorithm == "Secagg":
            for id, seed in self.shared_keys.items():
                if id == self.id:
                    continue
                local_random = np.random.default_rng(seed)
                sign = 1 if id > self.id else -1
                if self.args.debug_mode:
                    print(f'self_id = {self.id}, id = {id}, seed = {seed}')
                for k, v in state_dict.items():
                    mask = sign * local_random.uniform(low=-1, high=1, size=v.shape).astype(np.float32)
                    if self.args.debug_mode:
                        self.noise_sum += np.sum(mask)
                        print('client.self_id = {}, id = {}, seed = {}, noise_sum = {}'.format(self.id, id, seed, np.sum(mask)))
                    state_dict[k] += torch.from_numpy(mask).to(self.args.device)
                    
            if self.args.d > 0: #if exit dropout clients need online upload personal mask
                bu = self.personal_mask_seeds[self.id]
                local_random = np.random.default_rng(bu)
                #print("add personal mask")
                for k, v in state_dict.items():
                    noise = sign * local_random.normal(loc=0, scale=sensitivity * self.noise_scale / (len(self.shared_keys) - 1), size=v.shape)
                    if self.args.debug_mode:
                        self.noise_sum += np.sum(noise)
                        print('self_id = {}, personal_seed = {}, noise_sum = {}'.format(self.id, self.personal_mask_seeds[self.id], np.sum(noise)))
                    state_dict[k] += torch.from_numpy(noise).to(self.args.device)


        net.load_state_dict(state_dict)


class LocalUpdateDPSerial(LocalUpdateDP):
    def __init__(self, args, noise_scale, dh_param, id, dataset=None, idxs=None):
        super().__init__(args, noise_scale, dh_param, id, dataset, idxs)

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=self.args.lr_decay)
        total = 0
        correct = 0
        epoch_loss = []
        total = []
        correct = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for images, labels in self.ldr_train:
                net.zero_grad()
                index = int(len(images) / self.args.serial_bs)
                total_grads = [torch.zeros(size=param.shape).to(self.args.device) for param in net.parameters()]
                for i in range(0, index + 1):
                    net.zero_grad()
                    start = i * self.args.serial_bs
                    end = (i+1) * self.args.serial_bs if (i+1) * self.args.serial_bs < len(images) else len(images)
                    # print(end - start)
                    if start == end:
                        break
                    image_serial_batch, labels_serial_batch \
                        = images[start:end].to(self.args.device), labels[start:end].to(self.args.device)
                    log_probs = net(image_serial_batch)
                    loss = self.loss_func(log_probs, labels_serial_batch)
                    loss.backward()
                    self.clip_gradients(net)
                    grads = [param.grad.detach().clone() for param in net.parameters()]
                    for idx, grad in enumerate(grads):
                        total_grads[idx] += torch.mul(torch.div((end - start), len(images)), grad)

                    batch_loss.append(loss.item() * (end - start))
                    # Compute training accuracy
                    _, predicted = torch.max(log_probs, 1)  # Get the class with highest probability

                    total.append(labels_serial_batch.size(0))  # Number of labels in the batch
                    correct.append((predicted == labels_serial_batch).sum().item())  # Count correct predictions

                for i, param in enumerate(net.parameters()):
                    param.grad = total_grads[i]
                optimizer.step()
                scheduler.step()
                # add noises to parameters
                if self.args.dp_mechanism != 'no_dp':
                    self.add_noise(net)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        if self.args.dp_mechanism != 'no_dp' and self.args.algorithm == 'DP-NSA':
            self.add_noise(net)

        self.lr = scheduler.get_last_lr()[0]
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(correct) / sum(total)
