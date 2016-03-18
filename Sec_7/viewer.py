#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt

# TODO change this path
caffe_root = '/work/personal/caffe/'
sys.path.insert(0, caffe_root+'python')
import caffe

def load_testdata():
    test_listfile = '/work/school/16-824/assignment01/list/testlist_bbox2.txt'
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

    net = caffe.Net('test.prototxt', 'models/retrained_iter_90000.caffemodel', caffe.TEST)
    return net

def run_forward(net, test_list, data_counts, counts):
    batch_size = net.blobs['data'].data.shape[0]
    batch_count = int(np.ceil(float(data_counts)/ batch_size))

    correct = []
    wrong   = []


    confusion = np.zeros( (30, 30), np.float32)
    for i in xrange(batch_count):
        print('Batch %d/%d' % (i, batch_count))
        out = net.forward()
        for j in xrange(batch_size):
            id = i*batch_size + j
            if id >= data_counts:
                break

            label_correct = int(test_list[id].split(' ')[1])
            prop = out['softmax'][j]
            label_predicted = prop.argmax()
            fname = test_list[id].split(' ')[0]

            confusion[label_correct][label_predicted] += 1.0/counts[label_correct]

            if label_correct == label_predicted:
                correct.append(fname)
            else:
                wrong.append(fname)
    return correct, wrong, confusion

def write_confusion(confusion):
    plt.imshow(confusion, interpolation='nearest')
    plt.savefig('confusion.png')
    return

def write_correct_wrong(correct, wrong):
    fp_correct = open('./correct.txt', 'w')
    fp_wrong   = open('./wrong.txt', 'w')

    fp_correct.write("\n".join(correct))
    fp_wrong.write("\n".join(wrong))

    fp_correct.close()
    fp_wrong.close()

def print_summary(confusion):
    print('Correct predictions: ')
    for i in xrange(30):
        print('  %d\t%f' % (i, confusion[i][i]))

def main():
    test_list, data_counts, counts = load_testdata()
    net = initialize_caffe()
    correct, wrong, confusion = run_forward(net, test_list, data_counts, counts)
    write_correct_wrong(correct, wrong)
    write_confusion(confusion)

    print_summary(confusion)

if __name__ == "__main__":
    main()
