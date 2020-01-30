# !/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph

from autosp import predict_k


def consistent_labels(labels):
    """Achieve "some" consistency of color between true labels and pred labels.


    Parameters
    ----------
    labels : array of integers, shape: n_samples
        The labels of the clusters.

    Returns
    ----------
    color_map : dict object {integer: integer}
        The map of labels.
    """

    color_map = {}

    i = 0
    v = 0
    while v != max(labels) + 1:
        if labels[i] in color_map:
            pass
        else:
            color_map[labels[i]] = v
            v += 1
        i += 1

    return color_map

if __name__ == "__main__":

    # Generate artificial datasets.
    number_of_blobs = 5  # You can change this!!
    data, labels_true = make_blobs(n_samples=number_of_blobs * 10,
                                   centers=number_of_blobs)

    # Calculate affinity_matrix.
    connectivity = kneighbors_graph(data, n_neighbors=10)
    affinity_matrix = 0.5 * (connectivity + connectivity.T)

    # auto_spectral_clustering
    k = predict_k(affinity_matrix)
    sc = SpectralClustering(n_clusters=k,
                            affinity="precomputed",
                            assign_labels="kmeans").fit(affinity_matrix)

    labels_pred = sc.labels_

    print("%d blobs(artificial datasets)." % number_of_blobs)
    print("%d clusters(predicted)." % k)

    # Plot.
    from pylab import *
    t_map = consistent_labels(labels_true)
    t = [t_map[v] for v in labels_true]

    p_map = consistent_labels(labels_pred)
    p = [p_map[v] for v in labels_pred]

    subplot(211)
    title("%d blobs(artificial datasets)." % number_of_blobs)
    scatter(data[:, 0], data[:, 1], s=150, c=t)

    subplot(212)
    title("%d clusters(predicted)." % k)
    scatter(data[:, 0], data[:, 1], s=150, c=p)

    show()
