
import numpy as np
import simpsom as sps

from pylettes import *
pylette = Tundra(reverse=True).cmap

from keras.datasets import mnist
(train_X, train_y), (test_X, test_y) = mnist.load_data()

flat_data = train_X.reshape(train_X.shape[0], -1)

net = sps.SOMNet(50, 50, flat_data[0:1000], topology='hexagonal',
                PBC=True, init='PCA', metric='cosine',
                neighborhood_fun='gaussian',
                random_seed=32, GPU=False)

net.train(train_algo='batch', start_learning_rate=0.01,
          epochs=-1, batch_size=-1)
net.save_map("./trained_som_MNIST.npy")

_ = net.plot_map_by_difference(show=True, print_out=True,
    cmap=pylette)

_ = net.plot_map_by_feature(feature_ix=100, show=True, print_out=True,
    cmap=pylette)

from simpsom.plots import scatter_on_map

projection = net.project_onto_map(flat_data[0:1000])

scatter_on_map([projection[train_y[0:1000]==i][:1000] for i in range(10)],
               [[node.pos[0], node.pos[1]] for node in net.nodes_list],
               net.polygons, color_val=None,
               show=True, print_out=True, cmap=pylette)





















