import json
import os
from collections import defaultdict
import numpy as np
import random
from torch.utils.data import Dataset
import torch
from utils.language_utils import word_to_indices, letter_to_vec
import torchvision.transforms as transforms
import torchvision
from torchvision import datasets
from torch.utils.data import Subset
class FEMNIST(Dataset):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """

    def __init__(self, train=True, transform=None, target_transform=None, ):
        super(FEMNIST, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        train_clients, train_groups, train_data_temp, test_data_temp = read_data("./data/femnist/train",
                                                                                 "./data/femnist/test")
        if self.train:
            self.dic_users = {}
            train_data_x = []
            train_data_y = []
            for i in range(len(train_clients)):
                # if i == 100:
                #     break
                self.dic_users[i] = set()
                l = len(train_data_x)
                cur_x = train_data_temp[train_clients[i]]['x']
                cur_y = train_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    self.dic_users[i].add(j + l)
                    train_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                    train_data_y.append(cur_y[j])
            self.data = train_data_x
            self.label = train_data_y
        else:
            test_data_x = []
            test_data_y = []
            for i in range(len(train_clients)):
                cur_x = test_data_temp[train_clients[i]]['x']
                cur_y = test_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    test_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                    test_data_y.append(cur_y[j])
            self.data = test_data_x
            self.label = test_data_y

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]
        img = np.array([img])
        # img = Image.fromarray(img, mode='L')
        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return torch.from_numpy((0.5-img)/0.5).float(), target

    def __len__(self):
        return len(self.data)

    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            exit("The test dataset do not have dic_users!")


class ShakeSpeare(Dataset):
    def __init__(self, train=True):
        super(ShakeSpeare, self).__init__()
        train_clients, train_groups, train_data_temp, test_data_temp = read_data("./data/shakespeare/train",
                                                                                 "./data/shakespeare/test")
        self.train = train

        if self.train:
            self.dic_users = {}
            train_data_x = []
            train_data_y = []
            for i in range(len(train_clients)):
                # if i == 100:
                #     break
                self.dic_users[i] = set()
                l = len(train_data_x)
                cur_x = train_data_temp[train_clients[i]]['x']
                cur_y = train_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    self.dic_users[i].add(j + l)
                    train_data_x.append(cur_x[j])
                    train_data_y.append(cur_y[j])
            self.data = train_data_x
            self.label = train_data_y
        else:
            test_data_x = []
            test_data_y = []
            for i in range(len(train_clients)):
                cur_x = test_data_temp[train_clients[i]]['x']
                cur_y = test_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    test_data_x.append(cur_x[j])
                    test_data_y.append(cur_y[j])
            self.data = test_data_x
            self.label = test_data_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index], self.label[index]
        indices = word_to_indices(sentence)
        target = letter_to_vec(target)
        # y = indices[1:].append(target)
        # target = indices[1:].append(target)
        indices = torch.LongTensor(np.array(indices))
        # y = torch.Tensor(np.array(y))
        # target = torch.LongTensor(np.array(target))
        return indices, target

    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            exit("The test dataset do not have dic_users!")


def batch_data(data, batch_size, seed):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(seed)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        yield (batched_x, batched_y)


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data



def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def get_data(dataset, data_root, iid, num_users,data_aug, noniid_beta):
    ds = dataset 
    
    if ds == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        train_set_mia = datasets.MNIST('./data/mnist/', 
                                   train=True, 
                                   download=True, 
                                   transform=trans_mnist)
        
        train_set_mia = DatasetSplit(train_set_mia, np.arange(0, 60000))

        test_set_mia = datasets.MNIST('./data/mnist/', 
                                  train=False, 
                                  download=True, 
                                  transform=trans_mnist)

        train_set = datasets.MNIST('./data/mnist/', 
                                   train=True, 
                                   download=True, 
                                   transform=trans_mnist)
        
        train_set = DatasetSplit(train_set, np.arange(0, 60000))

        test_set = datasets.MNIST('./data/mnist/', 
                                  train=False, 
                                  download=True, 
                                  transform=trans_mnist)

    

    if ds == 'cifar10':
    
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ColorJitter(brightness=0.25, contrast=0.8),
                                              transforms.ToTensor(),
                                              normalize,
                                              ])  
        transform_test = transforms.Compose([transforms.CenterCrop(32),
                                             transforms.ToTensor(),
                                             normalize,
                                             ])
        
        transform_train_mia=transform_train
        transform_test_mia=transform_test

        train_set_mia = datasets.CIFAR10('./data/cifar_10',
                                               train=True,
                                               download=True,
                                               transform=transform_train_mia
                                               )
        train_set_mia = DatasetSplit(train_set_mia, np.arange(0, 50000))

        test_set_mia = datasets.CIFAR10('./data/cifar_10',
                                                train=False,
                                                download=True,
                                                transform=transform_test_mia
                                                )


        train_set = torchvision.datasets.CIFAR10('./data/cifar_10',
                                               train=True,
                                               download=True,
                                               transform=transform_train
                                               )

        train_set = DatasetSplit(train_set, np.arange(0, 50000))

        test_set = torchvision.datasets.CIFAR10('./data/cifar_100',
                                                train=False,
                                                download=True,
                                                transform=transform_test
                                                )
    
    if ds == 'cifar100':
        if data_aug :
            print("data_aug:",data_aug)
            normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),#
                                                transforms.RandomVerticalFlip(),
                                                transforms.RandomRotation(45),
                                                transforms.ColorJitter(brightness=0.25, contrast=0.8),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010))
                                                ])  
            transform_test = transforms.Compose([transforms.CenterCrop(32),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010))
                                                ])
            
            transform_train_mia = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010))])

            transform_test_mia = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        else:
            transform_train = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010))])

            transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            
            transform_train_mia=transform_train
            transform_test_mia=transform_test

        train_set_mia = datasets.CIFAR100('./data/cifar_100',
                                               train=True,
                                               download=True,
                                               transform=transform_train_mia
                                               )
        train_set_mia = DatasetSplit(train_set_mia, np.arange(0, 50000))

        test_set_mia = datasets.CIFAR100('./data/cifar_100',
                                                train=False,
                                                download=False,
                                                transform=transform_test_mia
                                                )

        train_set = torchvision.datasets.CIFAR100('./data/cifar_100',
                                               train=True,
                                               download=True,
                                               transform=transform_train
                                               )

        train_set = DatasetSplit(train_set, np.arange(0, 50000))

        test_set = torchvision.datasets.CIFAR100('./data/cifar_100',
                                                train=False,
                                                download=False,
                                                transform=transform_test
                                                )

    if iid:
        dict_users, train_idxs, val_idxs = data_iid_MIA(train_set, num_users)
    else:
        dict_users, train_idxs, val_idxs = data_beta(train_set, noniid_beta, num_users)

    return train_set, test_set, train_set_mia, test_set_mia, dict_users, train_idxs, val_idxs

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        if isinstance(item, list):
            return self.dataset[[self.idxs[i] for i in item]]

        image, label = self.dataset[self.idxs[item]]
        return image, label

def data_iid_MIA(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    all_idx0=all_idxs
    train_idxs=[]
    val_idxs=[]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        train_idxs.append(list(dict_users[i] ))
        all_idxs = list(set(all_idxs) - dict_users[i])
        val_idxs.append(list(set(all_idx0)-dict_users[i]))
    return dict_users, train_idxs, val_idxs

def data_beta(dataset, beta, n_clients):  
     #beta = 0.1, n_clients = 10
    print("The dataset is splited with non-iid param ", beta)
    label_distributions = []
    for y in range(len(dataset.dataset.classes)):
    #for y in range(dataset.__len__):
        label_distributions.append(np.random.dirichlet(np.repeat(beta, n_clients)))  
    
    labels = np.array(dataset.dataset.targets).astype(np.int32)
    #print("labels:",labels)
    client_idx_map = {i:{} for i in range(n_clients)}
    client_size_map = {i:{} for i in range(n_clients)}
    #print("classes:",dataset.dataset.classes)
    for y in range(len(dataset.dataset.classes)):
    #for y in range(dataset.__len__):
        label_y_idx = np.where(labels == y)[0] # [93   107   199   554   633   639 ... 54222]
        label_y_size = len(label_y_idx)
        #print(label_y_idx[0:100])
        
        sample_size = (label_distributions[y]*label_y_size).astype(np.int32)
        #print(sample_size)
        sample_size[n_clients-1] += label_y_size - np.sum(sample_size)
        #print(sample_size)
        for i in range(n_clients):
            client_size_map[i][y] = sample_size[i]

        np.random.shuffle(label_y_idx)
        sample_interval = np.cumsum(sample_size)
        for i in range(n_clients):
            client_idx_map[i][y] = label_y_idx[(sample_interval[i-1] if i>0 else 0):sample_interval[i]]

    train_idxs=[]
    val_idxs=[]    
    client_datasets = []
    all_idxs=[i for i in range(len(dataset))]
    for i in range(n_clients):
        client_i_idx = np.concatenate(list(client_idx_map[i].values()))
        np.random.shuffle(client_i_idx)
        subset = Subset(dataset.dataset, client_i_idx)
        client_datasets.append(subset)
        # save the idxs for attack
        train_idxs.append(client_i_idx)
        val_idxs.append(list(set(all_idxs)-set(client_i_idx)))

    return client_datasets, train_idxs, val_idxs



# if __name__ == '__main__':
#     test = ShakeSpeare(train=True)
#     x = test.get_client_dic()
#     print(len(x))
#     for i in range(100):
#         print(len(x[i]))
