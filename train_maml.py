"""
  This is the code for main entrance to train or test MAML
"""

import os
import numpy as np
import pickle
import random
import tensorflow as tf
import re
# from dataset_mini import *
# from dataset_tiered import *
# from dataset_cifar100 import *
from datasets.loader import create_loader, load_batch_data
from maml import MAML
# from maml_ori import MAML as MAML_Ori
from tensorflow.python.platform import flags

import pdb
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans

from tqdm import tqdm

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('dataset', 'sinusoid', 'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
flags.DEFINE_integer('num_groups', 5, 'number of clusters.')
flags.DEFINE_integer('img_size', 32, 'image size')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') #0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')

## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 32, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')
flags.DEFINE_bool('cosann', False, 'if True, cosine lr decay')

## Logging, saving, and testing options
flags.DEFINE_string('logdir', './logs/maml', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot

# added flags
flags.DEFINE_integer('lr_mode', 0, 'inner lr mode (default 0), 1: all variables share one lr 2: each variable has one lr  3: each variable has one lr with the same shape')
flags.DEFINE_integer('num_test_tasks', 600, 'number to tasks for testing')


def train(model, saver, sess, resume_itr=0):

    n_query = 15
    loader_train, loader_val = create_loader(FLAGS)

    loader_train.load_data_pkl()
    loader_val.load_data_pkl()

    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 2000
    PRINT_INTERVAL = 100
    TEST_PRINT_INTERVAL = 2000 #PRINT_INTERVAL*5

    print('Done initializing, starting training.')
    pre_accs = []
    post_accs = []

    num_classes = FLAGS.num_classes # for classification, 1 otherwise


    NUM_TRAIN_POINTS = 10000
    solutions, metaval_accuracies = [], []
    best_val_acc = 0

    for iteration in range(FLAGS.metatrain_iterations):
        inputa, labela, inputb, labelb = load_batch_data(loader_train, num_classes, FLAGS.update_batch_size, n_query, batch_size=FLAGS.meta_batch_size)
        feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb}
        input_tensors = [model.metatrain_op]

        if (iteration % 20 ==0):
            input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])

        result = sess.run(input_tensors, feed_dict)

        if (iteration % 20 ==0):
            pre_accs.append(result[-2]/ FLAGS.meta_batch_size)
            post_accs.append(result[-1]/ FLAGS.meta_batch_size)

        #if (iteration!=0) and iteration % PRINT_INTERVAL == 0:
        if iteration % 200 == 0:
            print_str = 'Iteration ' + str(iteration)+'\t'
            print_str += 'Pre: ' + str(np.mean(pre_accs)) + ', Post:' + str(np.mean(post_accs))
            print(print_str)
            pre_accs, post_accs = [], []

        if iteration % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/model' + str(iteration))
            

        # # sinusoid is infinite data, so no need to test on meta-validation set.
        
        if  iteration % TEST_PRINT_INTERVAL == 0:
            task_i = 0
            pre_acc_val, post_acc_val = [],[]
            for task_i in range(600):
                inputa, labela, inputb, labelb = load_batch_data(loader_val, num_classes, FLAGS.update_batch_size, n_query, batch_size=1)

                feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb}
                result = sess.run([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]], feed_dict)
                pre_acc_val.append(result[0]/ FLAGS.meta_batch_size)
                post_acc_val.append(result[1]/ FLAGS.meta_batch_size)

            print('Validation results: ' + str(np.array(pre_acc_val).mean(0)) + ', ' + str(np.array(post_acc_val).mean(0)))

            with open(os.path.join(FLAGS.logdir,'validation_acc.txt'), 'a+') as ff:
                ff.write('Val Iter-{} results: {}\t {}\n'.format(iteration, str(np.array(pre_acc_val).mean(0)), str(np.array(post_acc_val).mean(0)) ))
                # ff.write('Clusters counting:{}'.format(clu_counting))
                ff.write('\n')
                ff.close()

            if np.array(post_acc_val).mean(0) > best_val_acc:
                best_val_acc = np.array(post_acc_val).mean(0)
                saver.save(sess, FLAGS.logdir + '/bestmodel')

    saver.save(sess, FLAGS.logdir + '/model'+str(iteration))



def test(model, saver, sess, test_num_updates=None):
    NUM_TEST_POINTS = FLAGS.num_test_tasks
    num_classes = FLAGS.num_classes # for classification, 1 otherwise

    np.random.seed(1000)
    random.seed(1000)

    metaval_accuracies = []

    n_query = 15
    loader_test = create_loader(FLAGS)
    
    loader_test.load_data_pkl()
    
    for task_i in tqdm(range(NUM_TEST_POINTS)):
        inputa, labela, inputb, labelb = load_batch_data(loader_test, num_classes, FLAGS.update_batch_size, n_query, batch_size=1)
        feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb}
        result = sess.run([model.total_accuracy1] + model.total_accuracies2, feed_dict)
        metaval_accuracies.append(result)
    
    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)

    print('-'*50+'Mean accuracy'+'-'*50)
    print(means)
    print('-'*50+'Standard deviations'+'-'*50)
    print(stds)
    print('-'*50+'Confidence intervals'+'-'*50)
    print(ci95)

def main():

    test_num_updates = 1 if FLAGS.train else 20

    if not FLAGS.train:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1


    dim_output = FLAGS.num_classes
    if FLAGS.dataset == 'cifar100':
        dim_input = 32*32*3
    else:
        dim_input = 84*84*3

    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
    if FLAGS.train:
        model.construct_model(input_tensors=None, prefix='metatrain_')
    else:
        model.construct_model(input_tensors=None, prefix='metaval_')


    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=40)
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        print(var.name)

    sess = tf.InteractiveSession()

    resume_itr = 0

    
    # Build graph before this..........
    tf.global_variables_initializer().run()

    
    if not FLAGS.train:
        model_file = os.path.join(FLAGS.logdir,'bestmodel')
        print("Loading model weights from " + model_file)
        saver.restore(sess, model_file)


    if FLAGS.train:
        train(model, saver, sess, resume_itr)
    else:
        test(model, saver, sess, test_num_updates)

if __name__ == "__main__":
    main()
