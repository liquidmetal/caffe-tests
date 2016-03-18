#!/usr/bin/env python

import matplotlib.pyplot as plt

with open("./loss_values.txt", 'r') as fp:
    data = fp.read().split('\n')

df = [float(data[i]) for i in xrange(len(data)) if len(data[i]) > 0]
fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.set_title("Loss vs Iteration")    
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')

y = [20*i for i in xrange(len(data)) if len(data[i]) > 0]

st = 20
en = -1
ax1.plot(y[st:en], df[st:en], c='r', label='Loss')
leg = ax1.legend()

plt.savefig('loss-iterations.png')
