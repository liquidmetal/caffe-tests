#!/usr/bin/python

import numpy as np

# Make sure that caffe is on the python path:
caffe_root = '/work/personal/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

test_listfile = '/work/school/16-824/assignment01/list/testlist_class.txt'

caffe.set_device(0)
caffe.set_mode_gpu()
accuracy = {}
for number in xrange(10000, 90001, 10000):
    net = caffe.Net('test.prototxt',
                    'models/retrained_iter_%d.caffemodel' % number,
                    caffe.TEST)

    result_file = 'cls_results-%d.txt' % number
    test_list = np.loadtxt(test_listfile,  str, comments=None, delimiter='\n')
    data_counts = len(test_list)
    batch_size = net.blobs['data'].data.shape[0]
    batch_count = int(np.ceil(data_counts * 1.0 / batch_size))
    acc = 0

    f = open(result_file, 'w')
    print(batch_count)
    for i in range(batch_count):

        out = net.forward()
        print(i)
        for j in range(batch_size):
            id = i * batch_size + j
            if id >= data_counts:
                break

            lbl = int(test_list[id].split(' ')[1])
            fname = test_list[id].split(' ')[0]
            
            prop = out['softmax'][j] 
            pred_lbl = prop.argmax()
            if pred_lbl == lbl:
                acc = acc + 1

            f.write(fname)
            f.write('{0: d}'.format(pred_lbl))
            f.write('\n')

    f.close()

    accuracy[number] = acc * 1.0 / ( data_counts) 

print accuracy
