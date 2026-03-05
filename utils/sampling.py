import torch
import numpy as np
from torchvision import datasets, transforms

# def sample_dataset_by_dirichlet(dataset, n, alpha):
#     # 获取数据集的标签
#     targets = np.array(dataset.targets)
#     num_classes = len(np.unique(targets))
#     num_samples_per_split = len(targets) // n
    
#     # 按类别统计样本索引
#     class_indices = [np.where(targets == i)[0] for i in range(num_classes)]
    
#     # 存储每次采样的结果
#     sampled_indices = []
    
#     for _ in range(n):
#         # 根据 Dirichlet 分布生成类别分布
#         class_proportions = np.random.dirichlet([alpha] * num_classes)
        
#         # 按类别分配样本数
#         sampled_split = []
#         for class_id, indices in enumerate(class_indices):
#             np.random.shuffle(indices)
#             num_samples = int(class_proportions[class_id] * num_samples_per_split)
            
#             # 确保分配的总数等于 num_samples_per_split
#             if len(sampled_split) + num_samples > num_samples_per_split:
#                 num_samples = num_samples_per_split - len(sampled_split)
#             sampled_split.extend(indices[:num_samples])
        
#         # 如果还有剩余空位，随机补充
#         while len(sampled_split) < num_samples_per_split:
#             extra_sample = np.random.choice(targets.size)
#             if extra_sample not in sampled_split:
#                 sampled_split.append(extra_sample)
        
#         sampled_indices.append(sampled_split)
    
#     return sampled_indices

def sample_dataset_by_dirichlet(dataset, n, alpha):
    # 获取数据集的标签（targets 或 labels）
    if hasattr(dataset, 'targets'):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        targets = np.array(dataset.labels)
    else:
        raise AttributeError('Dataset has neither targets nor labels')

    targets = targets.squeeze()   # 防止 (N,1)
    num_classes = len(np.unique(targets))
    num_samples_per_split = len(targets) // n
    
    # 按类别统计样本索引
    class_indices = [np.where(targets == i)[0] for i in range(num_classes)]
    
    sampled_indices = []
    
    for _ in range(n):
        class_proportions = np.random.dirichlet([alpha] * num_classes)
        
        sampled_split = []
        for class_id, indices in enumerate(class_indices):
            np.random.shuffle(indices)
            num_samples = int(class_proportions[class_id] * num_samples_per_split)
            
            if len(sampled_split) + num_samples > num_samples_per_split:
                num_samples = num_samples_per_split - len(sampled_split)
            sampled_split.extend(indices[:num_samples])
        
        while len(sampled_split) < num_samples_per_split:
            extra_sample = np.random.choice(len(targets))
            if extra_sample not in sampled_split:
                sampled_split.append(extra_sample)
        
        sampled_indices.append(sampled_split)
    
    return sampled_indices


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users = {}
    num_items = int(len(dataset) / num_users)
    all_idxs = [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {}
    num_shards, num_imgs = num_users * 2, int(len(dataset) / (num_users * 2))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users

def fashion_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users = {}
    num_items = int(len(dataset) / num_users)
    all_idxs = [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def fashion_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = num_users * 2, int(len(dataset) / (num_users * 2))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users

def medmnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MEDMNIST dataset
    :param dataset: The dataset object (e.g., MedMNIST dataset)
    :param num_users: Number of clients (users) to split the data among
    :return: dict of image index for each user
    """
    # 检查 dataset 是否有效
    if dataset is None:
        raise ValueError("The dataset is None. Please ensure the dataset is loaded correctly.")
    
    # 如果 dataset 有 __len__ 方法，则使用它获取数据集的大小
    try:
        dataset_len = len(dataset)
    except TypeError:
        # 如果没有 len() 方法，尝试通过遍历计算大小
        dataset_len = sum(1 for _ in dataset)
    
    # 打印数据集大小
    print(f"Dataset size: {dataset_len}")

    dict_users = {}
    num_items = int(dataset_len / num_users)  # 每个用户分配的数据量
    all_idxs = list(range(dataset_len))  # 所有样本的索引
    
    # 遍历每个用户，随机分配数据
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))  # 随机选择样本
        all_idxs = list(set(all_idxs) - dict_users[i])  # 移除已经分配的样本
    
    return dict_users


def cifar_10_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users = {}
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_10_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = num_users * 2, int(len(dataset) / (num_users * 2))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users

def cifar_100_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR100 dataset
    :param dataset: CIFAR100 dataset
    :param num_users: Number of users
    :return: dict of image indices for each user
    """
    dict_users = {}
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_100_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR100 dataset
    :param dataset: CIFAR100 dataset
    :param num_users: Number of users
    :return: dict of image indices for each user
    """
    num_shards, num_imgs = num_users * 2, int(len(dataset) / (num_users * 2))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)  # Use CIFAR100 labels
    # Sort indices by labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # Divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users

if __name__ == '__main__':
    trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset_train = datasets.FashionMNIST('../data/fashion-mnist', train=True, download=True,
                                          transform=trans_fashion_mnist)
    # num = 100
    # d = mnist_iid(dataset_train, num)
    # path = '../data/fashion_iid_100clients.dat'
    # file = open(path, 'w')
    # for idx in range(num):
    #     for i in d[idx]:
    #         file.write(str(i))
    #         file.write(',')
    #     file.write('\n')
    # file.close()
    # trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    print(fashion_iid(dataset_train, 1000)[0])


