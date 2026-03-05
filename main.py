from datetime import datetime
import random
import time
import cProfile
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os

from cryptography.hazmat.primitives.asymmetric import dh

from utils.sampling import fashion_iid, medmnist_iid, mnist_iid, mnist_noniid, cifar_10_iid, cifar_10_noniid, cifar_100_iid, cifar_100_noniid, sample_dataset_by_dirichlet
from utils.options import args_parser
from models.Update import LocalUpdateDP, LocalUpdateDPSerial, calculate_noise_scale
from models.Nets import MLP, MnistCNN, CifarCNN, SqueezeNet, ShuffleNetV2_LN,  LeNet5, MobileNetV2_GN, ResNet18_GN, ResNet9, ResNet18_LN
from models.Fed import FedAvg, FedWeightAvg, FedDiff, FedAvg_Secagg, FedAvg_serial
from models.test import test_img
from utils.dataset import FEMNIST, ShakeSpeare
from opacus.grad_sample import GradSampleModule
from tqdm import tqdm
import medmnist
from medmnist import INFO

class TorchLabelWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

        # -------- 统一 labels / targets --------
        if hasattr(dataset, "targets"):
            labels = dataset.targets
        elif hasattr(dataset, "labels"):
            labels = dataset.labels
        else:
            # 兜底：逐个取（慢，但只在 init 跑一次）
            labels = [dataset[i][1] for i in range(len(dataset))]

        # 转成 torch.LongTensor（关键）
        self.targets = torch.as_tensor(labels, dtype=torch.long)
        self.labels = self.targets  # 双保险，两个名字都给

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]

        # image 一般已经是 tensor（MedMNIST 是）
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.long)

        return x, y

def remove_prefix_from_dict_keys(input_dict):
    result_dict = {}
    for key in input_dict.keys():
        if key.startswith("_module."):
            new_key = key[8:]  # 去掉 "_module." 前缀，长度为 6
            result_dict[new_key] = input_dict[key]
        else:
            result_dict[key] = input_dict[key]
    return result_dict

def simulate_dropouts(idxs_users, d):
    num_total = len(idxs_users)
    num_dropouts = math.ceil(d * num_total)  # 向上取整以保证至少掉一个
    dropouts = random.sample(idxs_users, num_dropouts)
    return dropouts

if __name__ == '__main__':
    # parse args
    

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    dict_users = {}
    dataset_train, dataset_test = None, None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.algorithm == 'DP-NSA':
        args.dp_mechanism = 'MA'
    elif args.algorithm == 'Fed-AVG':
        args.dp_mechanism = 'no_dp'

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        args.num_channels = 1
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = sample_dataset_by_dirichlet(dataset_train, args.num_users, args.alpha)
    elif args.dataset == 'cifar_10':
        #trans_cifar_10 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        args.num_channels = 3
        trans_cifar_10_train = transforms.Compose([
            transforms.Resize((224, 224)),          # 添加这一行
            transforms.RandomCrop(224, padding=4), # 这里crop改成224，保持一致
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        trans_cifar_10_test = transforms.Compose([
            transforms.Resize((224, 224)),          # 添加这一行
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        dataset_train = datasets.CIFAR10('./data/cifar_10', train=True, download=True, transform=trans_cifar_10_train)
        dataset_test = datasets.CIFAR10('./data/cifar_10', train=False, download=True, transform=trans_cifar_10_test)
        if args.iid:
            dict_users = cifar_10_iid(dataset_train, args.num_users)
        else:
            dict_users = sample_dataset_by_dirichlet(dataset_train, args.num_users, args.alpha)
    elif args.dataset == 'cifar_100':
        args.num_channels = 3
        
        # 数据增强和归一化操作与 CIFAR-10 类似
        trans_cifar_100_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 随机裁剪
            transforms.RandomHorizontalFlip(),    # 随机水平翻转
            transforms.ToTensor(),                # 转换为 Tensor
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),  # CIFAR-100 的均值和标准差
        ])
        
        trans_cifar_100_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),  # CIFAR-100 的均值和标准差
        ])
        
        # 加载 CIFAR-100 数据集
        dataset_train = datasets.CIFAR100('./data/cifar_100', train=True, download=True, transform=trans_cifar_100_train)
        dataset_test = datasets.CIFAR100('./data/cifar_100', train=False, download=True, transform=trans_cifar_100_test)

        # 数据划分逻辑：IID 或非 IID
        if args.iid:
            dict_users = cifar_100_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_100_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fashion-mnist':
        args.num_channels = 1
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test  = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                              transform=trans_fashion_mnist)
        if args.iid:
            dict_users = fashion_iid(dataset_train, args.num_users)
        else:
            dict_users = sample_dataset_by_dirichlet(dataset_train, args.num_users, args.alpha)

    elif args.dataset == 'pathmnist':  # 判断dataset是否存在于medmnist信息中集
        info = INFO[args.dataset]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])

        DataClass = getattr(medmnist, info['python_class'])
        
        # 定义数据集预处理
        if n_channels == 1:  # 如果是单通道数据（灰度图像）
            data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        else:  # 如果是RGB图像
            data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # load the data
        dataset_train = DataClass(split='train', transform=data_transform, download=True,
            root="./data/medmnist/pathmnist"
        )
        dataset_test = DataClass(split='test', transform=data_transform, download=True,
            root="./data/medmnist/pathmnist"
        )
        dataset_train = TorchLabelWrapper(dataset_train)
        dataset_test  = TorchLabelWrapper(dataset_test)
        # 根据是否IID加载用户数据
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = sample_dataset_by_dirichlet(dataset_train, args.num_users, args.alpha)

    elif args.dataset == 'chestmnist':  # 判断dataset是否存在于medmnist信息中集
        info = INFO[args.dataset]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])

        DataClass = getattr(medmnist, info['python_class'])
        
        # 定义数据集预处理
        if n_channels == 1:  # 如果是单通道数据（灰度图像）
            data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        else:  # 如果是RGB图像
            data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # load the data
        dataset_train = DataClass(split='train', transform=data_transform, download=True,
            root="./data/medmnist/chestmnist"
        )
        dataset_test = DataClass(split='test', transform=data_transform, download=True,
            root="./data/medmnist/chestmnist"
        )
        # 根据是否IID加载用户数据
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = sample_dataset_by_dirichlet(dataset_train, args.num_users, args.alpha)
    elif args.dataset == 'femnist':
        args.num_channels = 1
        dataset_train = FEMNIST(train=True)
        dataset_test = FEMNIST(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: femnist dataset is naturally non-iid')
        else:
            print("Warning: The femnist dataset is naturally non-iid, you do not need to specify iid or non-iid")
    elif args.dataset == 'shakespeare':
        dataset_train = ShakeSpeare(train=True)
        dataset_test = ShakeSpeare(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: ShakeSpeare dataset is naturally non-iid')
        else:
            print("Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
    else:
        exit('Error: unrecognized dataset')
        
    img_size = dataset_train[0][0].shape

    net_glob = None
    # build model
    # if args.model == 'cnn' and args.dataset == 'cifar_10':
    #     net_glob = CNNCifar(args=args).to(args.device)
    if args.model == 'mobilenet_v2':
        net_glob = MobileNetV2_GN(num_classes=10).to(args.device)
    elif args.model == 'resnet18' and args.dataset == 'cifar_10':
        net_glob = ResNet18_LN(num_classes=10).to(args.device)
    elif args.model == 'squeezenet' and args.dataset == 'cifar_10':
        net_glob = SqueezeNet(num_classes=10).to(args.device)
    elif args.model == 'cifarcnn' and args.dataset == 'cifar_10':
        net_glob =  CifarCNN(num_classes=10).to(args.device)
    elif args.model == 'squeezenet' and args.dataset == 'mnist':
        net_glob = SqueezeNet(num_classes=10).to(args.device)
    elif args.model == 'shufflenet':
        net_glob = ShuffleNetV2_LN(num_classes=10).to(args.device)
    elif args.model == 'resnet18' and args.dataset == 'cifar_100':
        net_glob = ResNet18_GN(num_classes=100).to(args.device)
    elif args.model == 'mnistcnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net_glob = MnistCNN(num_classes=10).to(args.device)
    elif args.model == 'lenet5' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net_glob = LeNet5(num_classes=10).to(args.device)
    elif args.model == 'lenet5' and args.dataset == 'pathmnist':
        net_glob = LeNet5(num_classes=9,in_channels=3).to(args.device)
    elif args.model == 'mnistcnn' and args.dataset == 'pathmnist':
        net_glob = MnistCNN(num_classes=9,in_channels=3).to(args.device)
    # elif args.dataset == 'femnist' and args.model == 'cnn':
    #     net_glob = CNNFemnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    if args.account:
        total_comm_overhead = 0

    # use opacus to wrap model to clip per sample gradient
    # if args.dp_mechanism != 'no_dp':
    # net_glob = GradSampleModule(net_glob)
    #print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    all_clients = list(range(args.num_users))

    if args.algorithm == 'Chain-PPFL':
        w_noise = copy.deepcopy(w_glob)

    # training
    acc_test = []
    acc_train = []
    loss_test = []
    noise_scale = 0.00001
    dh_param = dh.generate_parameters(generator=2, key_size=1024)
    
    account_overhead = []

    if args.dp_mechanism != 'no_dp':
        noise_scale = calculate_noise_scale(args=args, times=args.epochs * args.frac)
        #print("noise_scale = ", noise_scale)
    if args.serial:
        clients = [LocalUpdateDPSerial(args=args, noise_scale=noise_scale, dh_param=dh_param, dataset=dataset_train, id=i, idxs=dict_users[i]) for i in range(args.num_users)]
    else:
        clients = [LocalUpdateDP(args=args, noise_scale=noise_scale, dh_param=dh_param, dataset=dataset_train, id=i, idxs=dict_users[i]) for i in range(args.num_users)]
    m, loop_index = max(int(args.frac * args.num_users), 1), int(1 / args.frac)
    
    if args.account or args.account2:
        model_size = 0
        for param in w_glob.values():
            model_size += param.nelement() * param.element_size()  # -noise
        if args.debug_mode:
            print(f"model_size:{model_size}B")
    

    start_time = datetime.now()
    print('FL start! | algorithm({}) | task({}) | model({})'.format(args.algorithm, args.dataset, args.model))
    for iter in range(args.epochs):
        if args.account:
            train_time = 0
            communication_time = 0
            aggregation_time = 0
            total_time = 0

        t_start = time.time()
        w_locals, loss_locals, weight_locols = [], [], []
        # round-robin selection
        begin_index = (iter % loop_index) * m
        end_index = begin_index + m
        idxs_users = all_clients[begin_index:end_index]

        if args.d > 0:
            dropout_clients = simulate_dropouts(idxs_users, args.d)
            online_clients = list(set(idxs_users) - set(dropout_clients))
        key_exchange_comm = 0
        if args.algorithm == "DP-NSA":
            pb_keys = dict()
            for idx in idxs_users:
                _, pb_key = clients[idx].generate_keys()
                pb_keys[idx] = pb_key
            for idx in idxs_users:
                shared_keys = clients[idx].exchange_keys(pb_keys)
                if args.debug_mode:
                    print('user_{}\'s shared keys is {}'.format(idx, shared_keys))
        
            if args.account:
                num_clients = len(idxs_users)
                for pb_key in pb_keys:
                    key_exchange_comm += 128
                if args.debug_mode:
                    print(f"[key_exchange_comm]:{key_exchange_comm :.2f} B")
                shared_keys_exchange_comm = 0
                for shared_key in shared_keys:
                    shared_keys_exchange_comm += 128
                if args.debug_mode:
                    print(f"[shared_keys_exchange_comm]:{shared_keys_exchange_comm :.2f} B")
                key_exchange_comm += shared_keys_exchange_comm * num_clients
                if args.debug_mode:
                    print(f"[key_exchange_comm]:{key_exchange_comm :.2f} B")

            if args.account1:
                total_bytes = key_exchange_comm
                # 网络带宽：100 Mbps = 12,500,000 Bytes/sec
                network_bandwidth_bytes_per_sec = 100 * 1e6 / 8
                upload_time_sec = total_bytes / network_bandwidth_bytes_per_sec
                communication_time += upload_time_sec

            if args.account2:
                total_bytes = model_size
                # 网络带宽：100 Mbps = 12,500,000 Bytes/sec
                network_bandwidth_bytes_per_sec = 100 * 1e6 / 8
                upload_time_sec = total_bytes / network_bandwidth_bytes_per_sec
                network_latency = upload_time_sec
                print(f"network_latency:{network_latency}s,{network_latency * 1000}ms") 
                break
        elif args.algorithm == "NISS":
            if args.account:
                for idx in idxs_users:
                    clients[idx].nosie = 0.0
                for idx in idxs_users:
                    candidate_list = idxs_users.copy()  
                    if idx in candidate_list:
                        candidate_list.remove(idx)
                    size = np.random.randint(1, len(candidate_list)+1)
                    if args.debug_mode:
                        print('choice v:',size)
                    random_list = sorted(np.random.choice(candidate_list, size=size, replace=False))
                    sensitivity = clients[idx].get_sensitivity(idx, len(candidate_list)) 
                    sigma_k = (sensitivity * clients[idx].noise_scale / (len(candidate_list)))
                    clients[idx].sigma_k = sigma_k

                    for id in random_list:
                        clients[id].neg_sigma += sigma_k / (len(random_list))   # 1/v shares sigma_k
                        if args.debug_mode:
                            print('add user_{}\'s neg_sigma is {}'.format(id, clients[id].neg_sigma))
                        key_exchange_comm += model_size  # bytes
                    if args.debug_mode:
                        print(f"random_size:{len(random_list)},key_comm:{key_exchange_comm}B")
            if args.account1:
                total_bytes = key_exchange_comm
                # 网络带宽：100 Mbps = 12,500,000 Bytes/sec
                network_bandwidth_bytes_per_sec = 100 * 1e6 / 8
                upload_time_sec = total_bytes / network_bandwidth_bytes_per_sec
                communication_time += upload_time_sec

            if args.account2:
                total_bytes = model_size
                # 网络带宽：100 Mbps = 12,500,000 Bytes/sec
                network_bandwidth_bytes_per_sec = 100 * 1e6 / 8
                upload_time_sec = total_bytes / network_bandwidth_bytes_per_sec
                network_latency = upload_time_sec
                network_latency *= 2.0
                print(f"network_latency:{network_latency}s,{network_latency * 1000}ms") 
                break

            if args.debug_mode:
                for idx in idxs_users:
                    print('user_{}\'s noise is {}, neg_noise is {}'.format(idx, clients[idx].sigma_k,  clients[idx].neg_sigma))
                
                if args.account:    
                    print(f"[key_exchange_comm]:{key_exchange_comm :.2f} B")


        elif args.algorithm == 'Chain-PPFL':
            w_serial = []
            for lk in w_glob.keys():
                w_noise[lk] = torch.rand(w_noise[lk].size()).to(args.device)
            w_serial.append(copy.deepcopy(w_noise))
            w_noise_plussed = copy.deepcopy(w_noise)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

            if args.account:
                key_exchange_comm = model_size
                if args.debug_mode:
                    print("key_exchange_comm:",key_exchange_comm)
            if args.account1:
                total_bytes = key_exchange_comm
                # 网络带宽：100 Mbps = 12,500,000 Bytes/sec
                network_bandwidth_bytes_per_sec = 100 * 1e6 / 8
                upload_time_sec = total_bytes / network_bandwidth_bytes_per_sec
                communication_time += upload_time_sec

            if args.account2:
                total_bytes = model_size
                # 网络带宽：100 Mbps = 12,500,000 Bytes/sec
                network_bandwidth_bytes_per_sec = 100 * 1e6 / 8
                upload_time_sec = total_bytes / network_bandwidth_bytes_per_sec
                network_latency = upload_time_sec
                network_latency *= len(idxs_users)
                print(f"network_latency:{network_latency}s,{network_latency * 1000}ms") 
                break

        elif args.algorithm == "Secagg":
            pb_keys = dict()
            if args.debug_mode:
                for idx in idxs_users:
                    clients[idx].noise_sum = 0.0
            for idx in idxs_users:
                _, pb_key = clients[idx].generate_keys()
                pb_keys[idx] = pb_key
            for idx in idxs_users:
                shared_keys = clients[idx].exchange_keys(pb_keys)
                if args.debug_mode:
                    print('user_{}\'s shared keys is {}'.format(idx, shared_keys))
            if args.account:
                num_clients = len(idxs_users)
                for pb_key in pb_keys:
                    key_exchange_comm += 128
                if args.debug_mode:
                    print(f"[key_exchange_comm]:{key_exchange_comm :.2f} B")
                shared_keys_exchange_comm = 0
                for shared_key in shared_keys:
                    shared_keys_exchange_comm += 128
                if args.debug_mode:
                    print(f"[shared_keys_exchange_comm]:{shared_keys_exchange_comm :.2f} B")
                key_exchange_comm += shared_keys_exchange_comm * num_clients
                if args.debug_mode:
                    print(f"[key_exchange_comm]:{key_exchange_comm :.2f} B")

            
            for idx in idxs_users:
                bu = clients[idx].generate_mask_seed()  
                clients[idx].personal_mask_seeds[idx] = bu
                if args.debug_mode:
                    print("user_{}'s mask seed bu: {}".format(idx, bu))
                if args.account:
                    key_exchange_comm += clients[idx].share_mask_seeds_and_shared_keys(clients)  
                    if args.debug_mode:
                        print(f"[key_exchange_comm]:{key_exchange_comm :.2f} B")
                else:
                    clients[idx].share_mask_seeds_and_shared_keys(clients) 
                # print(f"bu_shares:{clients[idx].share_bu}")
                # print(f"received_shares:{clients[idx].received_shares}")
    
            
            if args.account1:
                total_bytes = key_exchange_comm
                # 网络带宽：100 Mbps = 12,500,000 Bytes/sec
                network_bandwidth_bytes_per_sec = 100 * 1e6 / 8
                upload_time_sec = total_bytes / network_bandwidth_bytes_per_sec
                communication_time += upload_time_sec

            if args.account2:
                total_bytes = model_size
                # 网络带宽：100 Mbps = 12,500,000 Bytes/sec
                network_bandwidth_bytes_per_sec = 100 * 1e6 / 8
                upload_time_sec = total_bytes / network_bandwidth_bytes_per_sec
                network_latency = upload_time_sec
                print(f"network_latency:{network_latency}s,{network_latency * 1000}ms")
                break
        
        train_acc = 0
        w_sum = None
        total = 0

        down_comm_overhead = 0 
        upload_comm_overhead = 0  
        reconstruct_secret = dict()
        reconstruct_bu = dict()


        for idx in tqdm(idxs_users):
            local = clients[idx]
            if args.d > 0:
                if idx in dropout_clients:
                    if args.debug_mode:
                        print(f"用户:{idx}掉线")
                    for id in clients[idx].shared_keys:
                        s = clients[idx].shared_keys[id]
                        if args.debug_mode:
                            print(f"第:{id}个对等掩码")
                            print(f"clients{idx}.received_shares:{clients[idx].received_shares}")
                        result = [
                            (receiver+1, v)  # (x, y)
                            for (sender, receiver, target), v in clients[idx].received_shares.items()
                            if sender == idx and receiver not in dropout_clients and target == id
                        ]
                        if args.debug_mode:
                            print(f"result:{result}")
                        reconstruct_secret[(idx,id)] = clients[idx].shamir_reconstruct(result)
                        if args.debug_mode:
                            print(f"s:{s}")
                            print(f"reconstruct s:{reconstruct_secret[(idx,id)]}")
                            print(f"reconstruct s equal with s?:{reconstruct_secret[(idx,id)]==s}")

                    
                else:
                    
                    if args.debug_mode:
                        print(f"在线客户端 ID : {idx}")
                    valid_share_bu = {
                        k: v for k, v in clients[idx].share_bu.items() if k-1 in online_clients
                    }
                    if args.debug_mode:
                        print(f"valid_share_bu:{valid_share_bu}")
                    reconstruct_bu[idx]= clients[idx].shamir_reconstruct(valid_share_bu)
                    if args.debug_mode:
                        print(f"reconstruct_bu:{reconstruct_bu[idx]}")
                        print(f"bu:{clients[idx].personal_mask_seeds[idx]}")
                        print(f"reconstruct bu equal with bu?:{reconstruct_bu[idx]==clients[idx].personal_mask_seeds[idx]}")
                        
            if args.account1:
                train_start = time.time()
            w, loss, acc = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.account1:
                train_end = time.time()
                if args.algorithm == 'Chain-PPFL':
                    train_time += train_end - train_start
                else:
                    train_time = max(train_time, train_end - train_start)
                #print(f"one round trian time:{train_end - train_start}")
            
            total += 1
            if args.algorithm == 'Chain-PPFL': 
                w_noise_plus = copy.deepcopy(w)
                w_noise_plus = remove_prefix_from_dict_keys(w_noise_plus)
                for lk in w_noise_plus.keys():
                    w_noise_plussed[lk] += w_noise_plus[lk].to(args.device)
            
                w_serial.append(copy.deepcopy(w_noise_plussed))
                
            if args.account:
                 # 统计下传权重的通信量（以字节为单位）
                down_comm_overhead += model_size
                if args.debug_mode:
                    print(f"model down_comm_overhead:{down_comm_overhead}")
            if args.account1:
                if args.algorithm != 'Chain-PPFL':
                    total_bytes = model_size
                else:
                    total_bytes = down_comm_overhead
                # 网络带宽：100 Mbps = 12,500,000 Bytes/sec
                network_bandwidth_bytes_per_sec = 100 * 1e6 / 8
                upload_time_sec = total_bytes / network_bandwidth_bytes_per_sec
                communication_time += upload_time_sec


            if w_sum == None:
                w_sum = copy.deepcopy(w)
            else:
                for k in w_sum.keys():
                    w_sum[k] += w[k]
            loss_locals.append(copy.deepcopy(loss))
            weight_locols.append(len(dict_users[idx]))
            train_acc += acc
            w = None
        
        if args.algorithm == "Secagg":
            if args.d > 0:
                if args.account:
                    total_bu = len(reconstruct_bu) * 2
                    total_secret = len(reconstruct_secret) * 2
                    key_exchange_comm += total_bu
                    key_exchange_comm += total_secret


        train_acc /= len(idxs_users)

        if args.debug_mode:
            noise_sum = 0
            for idx in idxs_users:
                noise_sum += clients[idx].noise_sum
            print('({}, {}) total noise: {}'.format(args.algorithm, args.dp_mechanism, noise_sum))

        # update global weights
        # w_glob = FedAvg(w_locals)
        if args.algorithm == 'Chain-PPFL': 
            if args.account1:
                aggregation_start = time.time()
            w_glob = FedAvg_serial(w_serial)
            if args.account1:
                aggregation_end = time.time()
                aggregation_time += aggregation_end - aggregation_start
        elif args.algorithm == 'Secagg' and args.d > 0:
            if args.account1:
                aggregation_start = time.time()
            w_glob = FedAvg_Secagg(args, w_sum, clients, reconstruct_secret, reconstruct_bu, total)
            if args.account1:
                aggregation_end = time.time()
                aggregation_time += aggregation_end - aggregation_start
        else:
            if args.account1:
                aggregation_start = time.time()
            for k in w_sum.keys():
                w_glob[k] = torch.div(w_sum[k], total)
            if args.account1:
                aggregation_end = time.time()
                aggregation_time += aggregation_end - aggregation_start
        # copy weight to net_glob
        # net_glob.load_state_dict(w_glob)
        net_glob.load_state_dict(remove_prefix_from_dict_keys(w_glob))

        if args.account:
            if args.d > 0:
                # 每一轮：计算上传模型通信量
                upload_comm_overhead = model_size
                upload_comm_overhead = upload_comm_overhead * (len(idxs_users) - len(dropout_clients))
                if args.debug_mode:
                    print(f"model upload_comm_overhead:{upload_comm_overhead}")
            else:
                 # 每一轮：计算上传模型通信量
                upload_comm_overhead = model_size
                upload_comm_overhead = upload_comm_overhead * len(idxs_users)
                if args.debug_mode:
                    print(f"model upload_comm_overhead:{upload_comm_overhead}")
        if args.account1:  
            if args.algorithm != 'Chain-PPFL':
                total_bytes = model_size
            else:  
                total_bytes = upload_comm_overhead
            # 网络带宽：100 Mbps = 12,500,000 Bytes/sec
            network_bandwidth_bytes_per_sec = 100 * 1e6 / 8
            upload_time_sec = total_bytes / network_bandwidth_bytes_per_sec
            communication_time += upload_time_sec

        if args.account:
            total_comm_overhead += down_comm_overhead
            total_comm_overhead += upload_comm_overhead 
            total_comm_overhead += key_exchange_comm

            account_overhead.append(down_comm_overhead + upload_comm_overhead + key_exchange_comm)
            print(f"Round {iter}: round {(down_comm_overhead + upload_comm_overhead + key_exchange_comm)/1024/1024/1024:.2f} GB, {(down_comm_overhead + upload_comm_overhead + key_exchange_comm):.2f} B")
            
        if args.account1:
            total_time = train_time + communication_time + aggregation_time
            print(f"algorithm:{args.algorithm}")
            print(f"train_time: {train_time}s,communication_time: {communication_time}s,aggregation_time: {aggregation_time}s,total_time: {total_time}s")
            break
            

        # print accuracy
        net_glob.eval()
        acc_t, loss_t = test_img(net_glob, dataset_test, args)
        t_end = time.time()
        print("Round {:3d},Train accuracy: {:.2f},Testing accuracy: {:.2f},Testing loss: {:.2f},Time:  {:.2f}s".format(iter, 100*train_acc, acc_t, loss_t, t_end - t_start))

        acc_test.append(acc_t)
        acc_train.append(100*train_acc)
        loss_test.append(loss_t)

        if args.account:
            if (acc_t >= args.acc or (iter+1)==args.epochs):
                print("Target accuracy reached, stopping training.")
                print(f"algorithm:{args.algorithm},dataset:{args.dataset}")
                print(f"Total communication: {total_comm_overhead/1024/1024/1024:.2f} GB,{total_comm_overhead/1024/1024:.2f} MB,{total_comm_overhead:.2f} B")
                
                rootpath = './log' + '/' + args.dataset + '/account' + '/'
                if not os.path.exists(rootpath):
                    os.makedirs(rootpath)
                accfile = open(rootpath + '({})account_fed_{}_{}_{}_{}_iid{}_alpha_{}_dp_{}_epsilon_{}_dp_sample_{}_dp_clip_{}_lr_{}_local_bs_{}_local_ep_{}.dat'.
                    format(args.algorithm, args.dataset, args.model, args.num_users, args.epochs, args.iid, args.alpha,
                        args.dp_mechanism, args.dp_epsilon, args.dp_sample, args.dp_clip, args.lr, args.local_bs, args.local_ep), "w")

                for i in range(len(account_overhead)):
                    accfile.write("Round {:3d},overhead: {:3d},train acc:{:.2f},test acc:{:.2f},target acc: {:.2f}".format(i, account_overhead[i],acc_train[i],acc_test[i] ,args.acc))
                    accfile.write('\n')
                accfile.write(f"Total communication: {total_comm_overhead/1024/1024/1024:.2f} GB,{total_comm_overhead/1024/1024:.2f} MB,{total_comm_overhead:.2f} B")
                accfile.close()
                break

    rootpath = './log' + '/' + args.dataset
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    if args.algorithm=='NISS': 
        accfile = open(rootpath + '/({})accfile_fed_{}_{}_{}_{}_iid{}_alpha_{}_dp_{}_epsilon_{}_dp_sample_{}_dp_clip_{}_lr_{}_local_bs_{}_local_ep_{}_k_{}.dat'.
                format(args.algorithm, args.dataset, args.model, args.num_users, args.epochs, args.iid, args.alpha,
                        args.dp_mechanism, args.dp_epsilon, args.dp_sample, args.dp_clip, args.lr, args.local_bs, args.local_ep, args.k), "w")
    else:
        accfile = open(rootpath + '/({})accfile_fed_{}_{}_{}_{}_iid{}_alpha_{}_dp_{}_epsilon_{}_dp_sample_{}_dp_clip_{}_lr_{}_local_bs_{}_local_ep_{}.dat'.
                    format(args.algorithm, args.dataset, args.model, args.num_users, args.epochs, args.iid, args.alpha,
                        args.dp_mechanism, args.dp_epsilon, args.dp_sample, args.dp_clip, args.lr, args.local_bs, args.local_ep), "w")

    
    # ===== 记录开始时间 =====
    accfile.write("=" * 60 + "\n")
    accfile.write(f"Start Time     : {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    accfile.write("=" * 60 + "\n\n")

    # ===== 写每一轮结果 =====
    for i in range(len(acc_test)):
        accfile.write(
            "Round {:3d}, Train Acc: {:.2f}, Test Acc: {:.2f}, Test Loss: {:.2f}\n".format(
                i, acc_train[i], acc_test[i], loss_test[i]
            )
        )

    # ===== 记录结束时间 =====
    end_time = datetime.now()
    elapsed_time = end_time - start_time

    accfile.write("\n" + "=" * 60 + "\n")
    accfile.write(f"End Time       : {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    accfile.write(f"Elapsed Time   : {elapsed_time}\n")
    accfile.write("=" * 60 + "\n")

    accfile.close()

    # # plot acc curve
    # plt.figure()
    # plt.plot(range(len(acc_test)), acc_test)
    # plt.ylabel('test accuracy')
    # plt.savefig(rootpath + '/figure' + '/{}_{}_{}_{}_C{}_iid{}_alpha{}_dp_{}_epsilon_{}_acc.png'.format(
    #     args.algorithm, args.dataset, args.model, args.epochs, args.frac, args.iid, args.alpha, args.dp_mechanism, args.dp_epsilon))




