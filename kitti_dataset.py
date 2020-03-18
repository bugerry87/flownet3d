'''
    Provider for duck dataset from xingyu liu
'''

import os
import os.path
import json
import numpy as np
import sys
import pickle
import glob


class SceneflowDataset():
    def __init__(self, root='./', npoints=2048, train=False):
        self.npoints = npoints
        self.train = train
        self.root = root
        self.datapath = glob.glob(os.path.join(self.root, '*.npz'))
        self.cache = {}
        self.cache_size = 30000

        ###### deal with one bad datapoint with nan value
        #self.datapath = [d for d in self.datapath if 'TRAIN_C_0140_left_0006-0' not in d]
        ######

    def __getitem__(self, index):
        if index in self.cache:
            #pos1, pos2, color1, color2, flow, mask1 = self.cache[index]
            pos1, pos2, gt = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pos1 = data['pos1']
                pos2 = data['pos2']
                color1 = np.zeros(pos1.shape)
                color2 = np.zeros(pos2.shape)
                flow = data['gt']
                mask1 = np.ones(len(pos1), dtype = bool)

            if len(self.cache) < self.cache_size:
                self.cache[index] = (pos1, pos2, color1, color2, flow, mask1)

            if self.train:
                n1 = pos1.shape[0]
                sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
                n2 = pos2.shape[0]
                sample_idx2 = np.random.choice(n2, self.npoints, replace=False)

                pos1 = pos1[sample_idx1, :]
                pos2 = pos2[sample_idx2, :]
                color1 = color1[sample_idx1, :]
                color2 = color2[sample_idx2, :]
                flow = flow[sample_idx1, :]
                mask1 = mask1[sample_idx1]
            else:
                pos1 = pos1[:self.npoints, :]
                pos2 = pos2[:self.npoints, :]
                color1 = color1[:self.npoints, :]
                color2 = color2[:self.npoints, :]
                flow = flow[:self.npoints, :]
                mask1 = mask1[:self.npoints]

        return pos1, pos2, color1, color2, flow, mask1

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    d = SceneflowDataset(npoints=2048*4)
    print('len of SceneflowDataset: ', len(d))
    import time
    tic = time.time()
    for i in range(len(d)):
        print('i: ', i)
        pc1, pc2, c1, c2, flow, m1 = d[i]

        npz_file = 'trans/trans_' + str(i) + '.npz'
        np.savez(npz_file, points1 = pc1, points2 = pc2, color1 = c1, color2 = c2, flow = flow, valid_mask1 = m1)
        print(npz_file + ' DONE!')
