from functools import partial

import numpy as np
import wquantiles as w


# todo follow up on https://github.com/numpy/numpy/pull/9211
# todo follow up on https://github.com/tinybike/weightedstats/issues/2 (should be fixed now)
# todo check if there is a performance boost with @tf.function:
# @tf.function
# def avg(clients: tf.RaggedTensor):
#   return tf.reduce_mean(clients, axis=0)
#
# a = [[1, 2, 3], [4]]
# b = [[3, 3, 3], [6]]
# c = [[2, 2, 3], [40]]
# z = avg_aggr(tf.ragged.constant(np.vstack((a, b))))

def mean(points, weights):
    return np.average(points, axis=0, weights=weights)#.astype(points.dtype)


def coordinatewise(fn, points, weights):
    points = np.asarray(points)
    if points.ndim == 1:
        return fn(points, weights)
    shape = points.shape
    res = np.empty_like(points, shape=shape[1:])
    for index in np.ndindex(*shape[1:]):
        coordinates = points[(..., *index)]
        res[index] = fn(coordinates, weights)
    return res


def quantile(points, weights, quantile):
    if weights is None:
        return np.quantile(points, quantile, axis=0).astype(np.float32)
    return coordinatewise(partial(w.quantile_1D, quantile=quantile), points, weights)


def median(points, weights):
    return quantile(points, weights, 0.5)
    # return np.median(points, axis=0) if weights is None \
    #     else np.apply_along_axis(weightedstats.numpy_weighted_median, 0,
    #                              points,
    #                              weights)


def trimmed_mean_1d(vector, weights, beta):
    if weights is None:
        lower_bound, upper_bound = np.quantile(vector, (beta, 1 - beta)).astype(np.float32)
        trimmed = [v for v in vector if lower_bound < v < upper_bound]
        if trimmed:
            return mean(trimmed, None)
        else:
            return (lower_bound + upper_bound) / 2
    else:
        lower_bound, upper_bound = w.quantile_1D(vector, weights, beta), w.quantile_1D(vector, weights, 1 - beta)

        trimmed = [(v, w) for v, w in zip(vector, weights) if lower_bound < v < upper_bound]
        if trimmed:
            trimmed_vector, trimmed_weights = zip(*trimmed)

            return mean(trimmed_vector, trimmed_weights)
        else:
            return (lower_bound + upper_bound) / 2


def trimmed_mean(points, weights, beta):
    return coordinatewise(partial(trimmed_mean_1d, beta=beta), points, weights)

# def quantile(points, weights, quantile):
#     if weights is None:
#         return np.median(points, quantile, axis=0)
#
#
#     weights_tiles = np.empty_like(points)
#     for i, w in enumerate(weights):
#         weights_tiles[i] = w
#
#     ind_sorted = np.argsort(points, axis=0)
#     sorted_data = np.take_along_axis(points, ind_sorted, axis=0)
#     sorted_weights = np.take_along_axis(weights_tiles, ind_sorted, axis=0)
#
#     Sn = np.cumsum(sorted_weights, axis=0)
#
#     Pn = (Sn-0.5*sorted_weights)/Sn[-1]
#
#     return np.apply_along_axis(lambda data_1d: np.interp(quantile, Pn, data_1d), 0,  sorted_data)
# def geometric_median(points, weights, max_steps=20, eps=1e-5, rel_tol=1e-6,
#                      dist=lambda points, median: np.linalg.norm(points - median, axis=1)):
#     ''' Weiszfeld's Algorithm for geometric median '''
#
#     # initial guess
#     median = centroid(points, weights)
#
#     distances_from_median = dist(points, median)
#     score = np.sum(weights * distances_from_median)
#
#     trace = [(median, score)]
#
#     # start
#     for _ in range(max_steps):
#         prev_median, prev_score = median, score
#
#         # new guess
#         beta_weights = weights / np.maximum(eps, distances_from_median)
#         median = np.average(points, axis=0, weights=beta_weights)
#
#         distances_from_median = dist(points, median)
#         score = np.sum(weights * distances_from_median)
#
#         trace.append((median, score))
#
#         if math.isclose(prev_score, score, rel_tol=rel_tol):
#             break
#     return median, trace
