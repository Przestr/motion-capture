import os
import glob
import pandas as pd 
import numpy as np
import pickle


for path in glob.glob('C:/Users/Strum/Desktop/cvapr/Movement-classification/data/*/'):
    for bvh in glob.glob(path + '*.bvh'):
        bvh = bvh[bvh.index('data'):]
        # if(bvh[bvh.rindex('/')+1:-7] in arr):
        os.system("bvh-converter " + bvh)


for path in glob.glob('C:/Users/Strum/Desktop/cvapr/Movement-classification/data/*/'):
    for path2 in glob.glob(path + '*.csv'):

        if (path2 not in path):
            print(path2)
            dat = pd.read_csv(path2)
            dat = dat.drop(['Time'], axis=1)

            coors = []
            coors_final = []

            for col in dat.columns:
                if col.endswith('.Z'):
                    continue
                else:
                    coors.append(dat[col].values.tolist())

            ind = [27, 28, 29, 33, 34, 35, 5, 6, 7, 14, 15, 16, 23]

            for i in range(len(coors[0])):
                coord = []
                for j in range(len(coors) // 2):
                    if (j in ind):
                        coord.append([int(coors[j * 2][i] * 15), int(coors[j * 2 + 1][i] * 15)])
                coors_final.append(coord),

            path2 = path2.replace('\\', '/')
            # path2 = path2[0:len(path2) - 3] + 'pickle'
            # print("Path: " + path2)

            pickle_out = open(path2[0:len(path2) - 3] + 'pickle', "wb")
            # pickle_out = open('C:/Users/elson/Desktop/Inne projekty CVAPR/Movement-classification-master/data/01/01_01_worldpos.pickle',"wb")
            pickle.dump(coors_final, pickle_out)
            pickle_out.close()
    