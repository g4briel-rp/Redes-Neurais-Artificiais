import numpy as np
import simpsom as sps
from sklearn.datasets import load_breast_cancer

from pylettes import *
pylette = Tundra(reverse=True).cmap

dataset = load_breast_cancer()
X = dataset.data

net = sps.SOMNet(50, 50, X, topology='hexagonal',
                PBC=True, init='PCA', metric='cosine',
                neighborhood_fun='gaussian',
                random_seed=32, GPU=False)

net.train(train_algo='batch', start_learning_rate=0.01,
          epochs=-1, batch_size=-1)
net.save_map("./trained_som_breast_cancer.npy")

_ = net.plot_map_by_difference(show=True, print_out=True,
    cmap=pylette)

_ = net.plot_map_by_feature(feature_ix=0, show=True, print_out=True,
    cmap=pylette)

from simpsom.plots import scatter_on_map

projection = net.project_onto_map(X)

scatter_on_map([projection[dataset.target==i] for i in range(2)],
               [[node.pos[0], node.pos[1]] for node in net.nodes_list],
               net.polygons, color_val=None,
               show=True, print_out=True, cmap=pylette)





















