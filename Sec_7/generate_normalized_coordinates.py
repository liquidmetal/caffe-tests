#!/usr/bin/env python

from PIL import Image

fp = open('/work/school/16-824/assignment01/list/testlist_both.txt', 'r')

for line in fp.read().split('\n'):
    parts = line.split()

    img_file = '/work/datasets/vlr/hw1/crop_imgs/%s' % parts[0]
    img = Image.open(img_file)
    (w, h) = img.size


    img_class = int(parts[1])
    bbox_x1 = float(int(parts[2]))/w
    bbox_y1 = float(int(parts[3]))/h
    bbox_x2 = float(int(parts[4]))/w
    bbox_y2 = float(int(parts[5]))/h
    print "%s %s %f %f %f %f" % (parts[0], img_class, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
