from __future__ import print_function
import numpy as np
from PIL import Image
import pickle as pkl
import os
import glob
import csv

class dataset_mini(object):
    def __init__(self, n_example, n_episode, split, args):
        self.im_width, self.im_height, self.channels = list(map(int, args['x_dim'].split(',')))
        self.n_examples = 600
        self.n_episodes = 100
        self.split = split
        self.seed = args['seed']
        self.root_dir = './data/miniimagenet'

        self.dataset_l = []

        self.dim_input = self.im_width*self.im_height*self.channels

    def load_data(self):
        """
            Load data into memory and partition into label,unlabel
        """
        print('Loading {} dataset'.format(self.split))
        data_split_path = os.path.join(self.root_dir, '{}.csv'.format(self.split))
        with open(data_split_path,'r') as f:
            reader = csv.reader(f, delimiter=',')
            data_classes = {}
            for i,row in enumerate(reader):
                if i==0:
                    continue
                data_classes[row[1]] = 1
            data_classes = data_classes.keys()
        print(data_classes)

        n_classes = len(data_classes)
        print('n_classes:{}, n_label:{}, n_unlabel:{}'.format(n_classes,self.n_label,self.n_unlabel))
        dataset_l = np.zeros([n_classes, self.n_label, self.im_height, self.im_width, self.channels],
                                 dtype=np.float32)

        for i, cls in enumerate(data_classes):
            im_dir = os.path.join(self.root_dir, '{}/'.format(self.split), cls)
            im_files = sorted(glob.glob(os.path.join(im_dir, '*.jpg')))
            np.random.RandomState(self.seed).shuffle(im_files) # fix the seed to keep label,unlabel fixed
            for j, im_file in enumerate(im_files):
                im = np.array(Image.open(im_file).resize((self.im_width, self.im_height)), 
                              np.float32, copy=False)
                im = im/255.0
                dataset_l[i, j] = im

        print('labeled data:', np.shape(dataset_l))
    
        self.dataset_l = dataset_l
        self.n_classes = n_classes
    
    
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
            idx1 = np.random.permutation(self.n_label)[:n_shot + n_query]
            support[i] = self.dataset_l[cls, idx1[:n_shot]]
            query[i] = self.dataset_l[cls, idx1[n_shot:]]

        support_labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_shot)).astype(np.uint8)
        query_labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8)
        
        return support, support_labels, query, query_labels
    
    def load_data_pkl(self):
        """
            load the pkl processed mini-imagenet into label,unlabel
        """
        pkl_name = '{}/mini-imagenet-cache-{}.pkl'.format(self.root_dir, self.split)
        print('Loading pkl dataset: {} '.format(pkl_name))

        try:
          with open(pkl_name, "rb") as f:
            data = pkl.load(f, encoding='bytes')
            image_data = data[b'image_data']
            class_dict = data[b'class_dict']
        except:
          with open(pkl_name, "rb") as f:
            data = pkl.load(f)
            image_data = data['image_data']
            class_dict = data['class_dict']

        print(data.keys(), image_data.shape, class_dict.keys())
        data_classes = sorted(class_dict.keys()) # sorted to keep the order

        n_classes = len(data_classes)
        print('n_classes:{}, n_label:{}, n_unlabel:{}'.format(n_classes,self.n_label,self.n_unlabel))
        dataset_l = np.zeros([n_classes, self.n_label, self.im_height, self.im_width, self.channels], dtype=np.float32)


        for i, cls in enumerate(data_classes):
            idxs = class_dict[cls] 
            np.random.RandomState(self.seed).shuffle(idxs) # fix the seed to keep label,unlabel fixed
            dataset_l[i] = image_data[idxs[0:self.n_label]]

        print('labeled data:', np.shape(dataset_l))

    
        self.dataset_l = dataset_l
        self.n_classes = n_classes

        del image_data
