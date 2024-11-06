import numpy as np

def train_hopfield_weights(img, weights):

    flatten = img.reshape((-1, 1, img.shape[0]*img.shape[0]))
    # w_ij weigths = to sum of product of a neuron with all other neurons so the outer product
    w_update = np.outer(flatten, flatten)

    weights += w_update
    # no self connection so diagonal is zero (w_ii = 0)
    np.fill_diagonal(weights, 0)

    return (weights.astype(np.float32))

def energy(x, w):

    energy = -x.dot(w).dot(x.T)
    return energy