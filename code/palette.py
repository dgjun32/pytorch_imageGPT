import numpy as np
import os
import sklearn 
import torch
import pickle
import joblib

from torch.utils.data import DataLoader
from sklearn.cluster import MiniBatchKMeans

from dataset import ImageDataset
from config import config as cfg

# k-means clustering on RGB pixels for 9-bit palette
if __name__ == '__main__':
    batch_size = 8
    dataset = ImageDataset(obj='train', cfg=cfg)
    dataloader = DataLoader(dataset, batch_size = batch_size)
    print(len(dataloader))
    # kmeans algorithm 
    algo = MiniBatchKMeans(n_clusters=512,
                            batch_size=batch_size*cfg.model.img_size**2,
                            max_iter=10000)
    print('Start Clustering')
    for step, batch in enumerate(dataloader):
        batch = batch.permute(0,2,3,1).numpy()
        batch = batch.reshape(-1, 3)
        algo.partial_fit(batch)
        if (step+1)%1000 == 0:
            print('{} percent done'.format(100*step/len(dataloader)))
    # save kmeans clustering model
    joblib.dump(algo, '../output/kmeans_palette.pkl')





