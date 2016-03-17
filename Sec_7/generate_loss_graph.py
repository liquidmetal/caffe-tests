#!/usr/bin/env python

import matplotlib.pyplot as plt

def generate_loss_graph(filename, output):
    with open(filename, 'r') as fp:
        data = fp.read().split('\n')

    df = [float(data[i]) for i in xrange(len(data)) if len(data[i]) > 0]
    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    ax1.set_title("Loss vs Iteration")    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')

    y = [20*i for i in xrange(len(data)) if len(data[i]) > 0]

    ax1.plot(y, df, c='r', label='Loss')
    leg = ax1.legend()

    plt.savefig(output)

def main():
    generate_loss_graph("./loss_bbox.txt", 'graph-loss-bbox.png')
    generate_loss_graph("./loss_class.txt", 'graph-loss-class.png')

if __name__ == "__main__":
    main()
