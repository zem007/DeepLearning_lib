""" Wrapper functions for TensorFlow layers.
Author: Charles R. Qi
Date: November 2016
"""

import numpy as np
import math
    
#########################################################################################
### those are for DeepGCN
def pairwise_distance(point_cloud):
    """Compute pairwise distance of a point cloud.
       Args:
           point_cloud: tensor (batch_size, num_points, num_dims)
       Returns:
           pairwise distance: (batch_size, num_points, num_points)
    """
    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])
    point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)
    point_cloud_inner = -2*point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True)
    point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
    return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose

def knn(adj_matrix, k=20):
    """Get KNN based on the pairwise distance.
       Args:
          pairwise distance: (batch_size, num_points, num_points)
          k: int
       Returns:
          nearest neighbors: (batch_size, num_points, k)
     """
    neg_adj = -adj_matrix
    _, nn_idx = tf.nn.top_k(neg_adj, k=k)
    return nn_idx

def knn_graph(vertex_features, k):
    '''Find the neighbors' indices based on knn'''
    dists = pairwise_distance(vertex_features)
    neigh_idx = knn(dists, k=k) # (batch, num_points, k)
    return neigh_idx

def get_edge_feature(point_cloud, nn_idx, k):
    """Construct edge feature for each point
       Args:
           point_cloud: (batch_size, num_points, 1, num_dims)
           nn_idx: (batch_size, num_points, k)
           k: int
       Returns:
           edge features: (batch_size, num_points, k, num_dims)  [8, 1024, 16, 3]
     """
    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_central = point_cloud

    point_cloud_shape = point_cloud.get_shape()
    batch_size = point_cloud_shape[0].value
    num_points = point_cloud_shape[1].value
    num_dims = point_cloud_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)
    point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

    edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors-point_cloud_central], axis=-1)
    return edge_feature

def edge_conv_layer(inputs,
              neigh_idx,
              nn,
              k,
              num_outputs,
              scope=None,
              is_training=None):
    
    '''
      EdgeConv layer:
        Wang, Y, Yongbin S, Ziwei L, Sanjay S, Michael B, Justin S.
        "Dynamic graph cnn for learning on point clouds."
        arXiv:1801.07829 (2018).
    '''
    edge_features = get_edge_feature(inputs, neigh_idx, k)
    out = nn.build(edge_features,
                 num_outputs,
                 scope=scope,
                 is_training=is_training)
    vertex_features = tf.reduce_max(out, axis=-2, keep_dims=True)

    return vertex_features
##########################################################################################


def shuffle_mini_batches(X, Y_cls, Y_seg, info, batch_size, adj = False):
    '''
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (n, nums_point, 3)
    Y -- true "label" vector (containing 0 if negative, 1 if positive), of shape (n, nums_point)
    batch_size - size of the mini-batches, integer

    Returns:
    mini_batches -- list of (mini_batch_X, mini_batch_Y)
    '''

    m = X.shape[0]                  # number of training examples
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :,:]
    shuffled_Y_cls = Y_cls[permutation,]
    shuffled_Y_seg = Y_seg[permutation, :]
    shuffled_info = info[permutation, :]
    if type(adj) == np.ndarray:
        shuffled_adj = adj[permutation, :,:]

    num_complete_minibatches = math.floor(m/batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * batch_size : k * batch_size + batch_size, :,:]
        mini_batch_Y_cls = shuffled_Y_cls[k * batch_size : k * batch_size + batch_size,]
        mini_batch_Y_seg = shuffled_Y_seg[k * batch_size : k * batch_size + batch_size, :]
        mini_batch_info = shuffled_info[k * batch_size : k * batch_size + batch_size, :]
        if type(adj) == np.ndarray:
            mini_batch_adj = shuffled_adj[k * batch_size : k * batch_size + batch_size, :, :]
        if type(adj) == np.ndarray:
            mini_batch = (mini_batch_X, mini_batch_Y_cls, mini_batch_Y_seg, mini_batch_info, mini_batch_adj)
        else: mini_batch = (mini_batch_X, mini_batch_Y_cls, mini_batch_Y_seg, mini_batch_info)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < batch_size)
    if m % batch_size != 0:
#         mini_batch_X = shuffled_X[num_complete_minibatches * batch_size : m, :,:]
#         mini_batch_Y = shuffled_Y[num_complete_minibatches * batch_size : m, :]
        mini_batch_X = shuffled_X[m - batch_size : m, :,:]
        mini_batch_Y_cls = shuffled_Y_cls[m - batch_size : m,]
        mini_batch_Y_seg = shuffled_Y_seg[m - batch_size : m, :]
        mini_batch_info = shuffled_info[m - batch_size : m, :]
        if type(adj) == np.ndarray:
            mini_batch_adj = shuffled_adj[m - batch_size : m, :,:]
        if type(adj) == np.ndarray:
            mini_batch = (mini_batch_X, mini_batch_Y_cls, mini_batch_Y_seg, mini_batch_info, mini_batch_adj)
        else: mini_batch = (mini_batch_X, mini_batch_Y_cls, mini_batch_Y_seg, mini_batch_info)
        mini_batches.append(mini_batch)

    return mini_batches


