#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

caffe_root = '/work/personal/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe, os
from sklearn.neighbors import NearestNeighbors

def load_traindata():
    test_listfile = '/work/school/16-824/assignment01/list/trainlist_class.txt'
    test_list = np.loadtxt(test_listfile, str, comments=None, delimiter='\n')
    data_counts = len(test_list)

    counts = {}
    for i in xrange(data_counts):
        label = int(test_list[i].split(' ')[1])
        if label not in counts:
            counts[label] = 0
        counts[label] += 1

    return test_list, data_counts, counts

def initialize_caffe():
    caffe.set_device(0)
    caffe.set_mode_gpu()

def run_classification_network(data_counts):
    net = caffe.Net('classification.prototxt',
                    'models/classification_iter_90000.caffemodel',
                    caffe.TEST)
    return run_network(net, data_counts)

def run_localization_network(data_counts):
    net = caffe.Net('localize.prototxt',
                    'models/localize_iter_30000.caffemodel',
                    caffe.TEST)
    return run_network(net, data_counts)

def fwd_cls():
    net = caffe.Net('test.classification.prototxt',
                    'models/classification_iter_90000.caffemodel',
                    caffe.TEST)
    return run_network(net, 3)


def fwd_loc():
    net = caffe.Net('test.localize.prototxt',
                    'models/localize_iter_30000.caffemodel',
                    caffe.TEST)
    return run_network(net, 3)

def run_network(net, data_counts):
    classification_pool5 = []
    classification_fc7 = []

    batch_size = net.blobs['data'].data.shape[0]
    batch_count = int(np.ceil(float(data_counts)/batch_size))

    for i in xrange(batch_count):
        if i % 100 == 0:
            print('Batch %d/%d' % (i, batch_count))
        out = net.forward()
        
        pool5_current = np.reshape(net.blobs['pool5'].data, (50, 9216) )
        fc7_current   = net.blobs['fc7'].data

        for j in xrange(batch_size):
            id = i*batch_size + j
            if id >= data_counts:
                break

            classification_pool5.append(pool5_current[j])
            classification_fc7.append(fc7_current[j])

    pool5 = np.array(classification_pool5, dtype=np.float32)
    fc7 = np.array(classification_fc7, dtype=np.float32)
    return pool5, fc7

def find_closest_match_index(inp, features):
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='brute').fit(features.tolist())
    distances, indices = nbrs.kneighbors([inp.tolist()])
    return indices

def main():
    initialize_caffe()
    test_list, data_counts, counts = load_traindata()

    print('Generating input feature vectors')
    inp_pool5_c, inp_fc7_c = fwd_cls()
    inp_pool5_l, inp_fc7_l = fwd_loc()

    print('Generating classification features')
    pool5_c, fc7_c = run_classification_network(data_counts)


    print('Classification network')
    for i in xrange(3):
        print('Image %d ====================' % i)
        idx1 = find_closest_match_index(inp_pool5_c[i], pool5_c)
        idx2 = find_closest_match_index(inp_fc7_c[i], fc7_c)
        print('  classification_pool5: ')
        print('    %s' % idx1)
        print('  classification_fc7: ')
        print('    %s' % idx2)

    del pool5_c
    del fc7_c

    ####################################################
    print('Generating localization features')
    pool5_l, fc7_l = run_localization_network(data_counts)

    print('')
    print('')
    print('')
    print('')
    print('Localization network')
    for i in xrange(3):
        idx3 = find_closest_match_index(inp_pool5_l[i], pool5_l)
        idx4 = find_closest_match_index(inp_fc7_l[i], fc7_l)
        print('  localize_pool5: ')
        print('    %s' % idx3)
        print('  localize_fc7: ')
        print('    %s' % idx4)

    del pool5_l
    del fc7_l

if __name__ == "__main__":
    main()
