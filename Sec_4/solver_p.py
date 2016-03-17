#!/usr/bin/env python


#from __future__ import division
import sys

print("Trying to modify pythonpath")
#caffe_root = '/work/personal/caffe/'
#sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np

print("Imported numpy and caffe")

# init
caffe.set_mode_gpu()
caffe.set_device(0)

# caffe.set_mode_cpu()

print("Loading model into memory")
solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from('/work/personal/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')

niter = 100000
train_loss = np.zeros(niter)

f = open('log.txt', 'w')

print("Starting iterations")
for it in range(niter): 
    solver.step(1)
    train_loss[it] = solver.net.blobs['loss'].data
    f.write('{0: f}\n'.format(train_loss[it]))
f.close()

print("Done with iterations")

# solver.step(80000)


