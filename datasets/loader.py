
import numpy as np
from .dataset_mini import *
from .dataset_tiered import *
from .dataset_cifar100 import *
from .dataset_multi import *

DATASET_NUM_DICT = {'Aircraft':100, 'Fungi':150, 'Bird':60, 'Car':80}

def create_loader(cfg):
    if cfg.train:
        if cfg.dataset == 'miniimagenet':
            loader_train = dataset_mini(600, 100, 'train', {'x_dim': '84,84,3', 'seed': 0})
            loader_val = dataset_mini(600, 100, 'val', {'x_dim': '84,84,3', 'seed': 0})
        elif cfg.dataset == 'tiered':
            loader_train = dataset_tiered(600, 100, 'train', {'x_dim': '84,84,3', 'seed': 0})
            loader_val = dataset_tiered(600, 100, 'val', {'x_dim': '84,84,3', 'seed': 0})
        elif cfg.dataset == 'cifar100':
            loader_train = dataset_cifar100('train', {'x_dim': '32,32,3', 'seed': 0})
            loader_val = dataset_cifar100('val', {'x_dim': '32,32,3', 'seed': 0})
        # elif cfg.dataset == 'multiple':
        #     loader_train = [dataset_multi('train', set_name, DATASET_NUM_DICT[set_name], cfg) for set_name in cfg.subsets]
        #     loader_val = [dataset_multi('val', set_name, DATASET_NUM_DICT[set_name], cfg) for set_name in cfg.subsets]
        
        return loader_train, loader_val
    
    else:
        if cfg.dataset=='miniimagenet':
            loader_test = dataset_mini(600, 100, 'test', {'x_dim': '84,84,3', 'seed': 1000})
        elif cfg.dataset=='tiered':
            loader_test = dataset_tiered(600, 100, 'test', {'x_dim': '84,84,3', 'seed': 1000})
        elif cfg.dataset=='cifar100':
            loader_test = dataset_cifar100('test', {'x_dim': '32,32,3', 'seed': 1000})
        # elif cfg.dataset == 'multiple':
        #     loader_test = [dataset_multi('test', set_name, DATASET_NUM_DICT[set_name], cfg) for set_name in cfg.subsets]
        
        return loader_test


def load_batch_data(loader, n_way, n_shot, n_query, batch_size=1):
    inputa = []
    labela = []
    inputb = []
    labelb = []
    
    for b in range(batch_size):#max(FLAGS.meta_batch_size, batch_size)):
        s, s_labels, q, q_labels = loader.next_data(n_way, n_shot, n_query)
        s = np.reshape(s,(n_way*n_shot,-1))
        q = np.reshape(q,(n_way*n_query,-1))
        s_labels = np.reshape(s_labels, (-1))
        q_labels = np.reshape(q_labels, (-1))
        inputa.append(s)
        inputb.append(q)
        
        s_onehot = np.zeros((n_shot*n_way,n_way))
        s_onehot[np.arange(n_shot*n_way),s_labels] = 1
        labela.append(s_onehot)
        q_onehot = np.zeros((n_query*n_way,n_way))
        q_onehot[np.arange(n_query*n_way),q_labels] = 1
        labelb.append(q_onehot)


    inputa = np.array(inputa)
    labela = np.array(labela)
    inputb = np.array(inputb)
    labelb = np.array(labelb)

    return inputa, labela, inputb, labelb


# def load_batch_data(loaders, n_way, n_shot, n_query, batch_size=1):
#     inputa = []
#     labela = []
#     inputb = []
#     labelb = []
#     for loader in loaders: 
#         # import pdb; pdb.set_trace()
#         assert batch_size % len(loaders) == 0, 'Batch size should be the multiple of loader numbers'
#         for b in range(batch_size//len(loaders)):
#             s, s_labels, q, q_labels = loader.next_data(n_way, n_shot, n_query, flip=False)
#             s = np.reshape(s,(n_way*n_shot,-1))
#             q = np.reshape(q,(n_way*n_query,-1))
#             s_labels = np.reshape(s_labels, (-1))
#             q_labels = np.reshape(q_labels, (-1))
#             inputa.append(s)
#             inputb.append(q)
            
#             s_onehot = np.zeros((n_shot*n_way,n_way))
#             s_onehot[np.arange(n_shot*n_way),s_labels] = 1
#             labela.append(s_onehot)
#             q_onehot = np.zeros((n_query*n_way,n_way))
#             q_onehot[np.arange(n_query*n_way),q_labels] = 1
#             labelb.append(q_onehot)


#         inputa = np.array(inputa)
#         labela = np.array(labela)
#         inputb = np.array(inputb)
#         labelb = np.array(labelb)

#         return inputa, labela, inputb, labelb