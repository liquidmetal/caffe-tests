#!/usr/bin/env python

import shutil

trainlist = []
with open('../../list/trainlist_class.txt', 'r') as fp:
    trainlist = fp.readlines()

files_to_copy = []
with open('yoyo.log', 'r') as fp:
    files_to_copy = fp.readlines()

for line in files_to_copy:
    idx = int(line.strip())

    fname = '/work/datasets/vlr/hw1/crop_imgs/%s' % trainlist[idx].split()[0].strip()
    shutil.copyfile(fname, '../figures/%d.jpg' % idx)
