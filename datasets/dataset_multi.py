import numpy as np
import pickle as pkl
import os

class dataset_multi(object):
    def __init__(self, split, name, n_examples, args):
        self.im_width, self.im_height, self.channels = list(map(int, args['x_dim'].split(',')))
        self.n_examples = n_examples
        self.n_episodes = 100
        self.split = split
        self.ratio = args['ratio']
        self.seed = args['seed']
        self.dataset_name =  name
        self.root_dir = './data/meta-dataset'

        self.dim_input = self.im_width*self.im_height*self.channels
    
    
    def next_data(self, n_way, n_shot, n_query):
        """
            get support,query,unlabel data from n_way
            get unlabel data from n_distractor
        """
        support = np.zeros([n_way, n_shot, self.im_height, self.im_width, self.channels], dtype=np.float32)
        query = np.zeros([n_way, n_query, self.im_height, self.im_width, self.channels], dtype=np.float32)

        selected_classes = np.random.permutation(self.n_classes)[:n_way]
        for i, cls in enumerate(selected_classes[0:n_way]): # train way
            # labled data
            idx1 = np.random.permutation(self.n_examples)[:n_shot + n_query]
            support[i] = self.dataset_l[cls, idx1[:n_shot]]
            query[i] = self.dataset_l[cls, idx1[n_shot:]]

        support_labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_shot)).astype(np.uint8)
        query_labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8)
        
        return support, support_labels, query, query_labels
    
    def load_data_pkl(self):
        """
            load the pkl processed mini-imagenet into label,unlabel
        """
        pkl_name = '{}/{}-{}.pkl'.format(self.root_dir, self.dataset_name, self.split)
        print('Loading pkl dataset: {} '.format(pkl_name))

        try:
          with open(pkl_name, "rb") as f:
            data = pkl.load(f, encoding='latin1')
            image_data = np.array(data['data'])
            labels_array = np.array(data['labels'])
        except:
          with open(pkl_name, "rb") as f:
            data = pkl.load(f)
            image_data = data['data']
            labels_array = data['labels']

        data_classes = np.unique(labels_array)
        # cls_indexes = np.unique(labels_array, return_index=True)[1]
        # data_classes = 
        print(data.keys(), image_data.shape, data_classes)
        # data_classes = sorted(classes) # sorted to keep the order

        n_classes = len(data_classes)
        print('n_classes:{}, n_examples:{}'.format(n_classes, self.n_examples))
        # import pdb; pdb.set_trace()
        assert n_classes * self.n_examples == labels_array.shape[0]
        dataset_l = np.zeros([n_classes, self.n_examples, self.im_height, self.im_width, self.channels], dtype=np.float32)

        # for i, cls in enumerate(data_classes):
        #     idxs = class_dict[cls] 
        # import pdb; pdb.set_trace()
        for i in range(n_classes):
            idxs = np.arange(i * self.n_examples, i * self.n_examples + self.n_examples) 
            np.random.RandomState(self.seed).shuffle(idxs) # fix the seed to keep label,unlabel fixed
            dataset_l[i] = image_data[idxs]
    
        self.dataset_l = dataset_l
        self.n_classes = n_classes

        del image_data