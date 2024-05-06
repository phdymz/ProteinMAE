import numpy as np
from math import pi
import torch
# from pykeops.torch import LazyTensor
# from plyfile import PlyData, PlyElement
from utils.helper import *
import torch.nn as nn
import torch.nn.functional as F

# from matplotlib import pyplot as plt
# from pykeops.torch.cluster import grid_cluster, cluster_ranges_centroids, from_matrix
from math import pi, sqrt



# def subsample(x, batch=None, scale=1.0):
#     """Subsamples the point cloud using a grid (cubic) clustering scheme.
#
#     The function returns one average sample per cell, as described in Fig. 3.e)
#     of the paper.
#
#     Args:
#         x (Tensor): (N,3) point cloud.
#         batch (integer Tensor, optional): (N,) batch vector, as in PyTorch_geometric.
#             Defaults to None.
#         scale (float, optional): side length of the cubic grid cells. Defaults to 1 (Angstrom).
#
#     Returns:
#         (M,3): sub-sampled point cloud, with M <= N.
#     """
#
#     if batch is None:  # Single protein case:
#         if True:  # Use a fast scatter_add_ implementation
#             labels = grid_cluster(x, scale).long()
#             C = labels.max() + 1
#
#             # We append a "1" to the input vectors, in order to
#             # compute both the numerator and denominator of the "average"
#             #  fraction in one pass through the data.
#             x_1 = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
#             D = x_1.shape[1]
#             points = torch.zeros_like(x_1[:C])
#             points.scatter_add_(0, labels[:, None].repeat(1, D), x_1)
#             return (points[:, :-1] / points[:, -1:]).contiguous()
#
#         else:  # Older implementation;
#             points = scatter(points * weights[:, None], labels, dim=0)
#             weights = scatter(weights, labels, dim=0)
#             points = points / weights[:, None]
#
#     else:  # We process proteins using a for loop.
#         # This is probably sub-optimal, but I don't really know
#         # how to do more elegantly (this type of computation is
#         # not super well supported by PyTorch).
#         batch_size = torch.max(batch).item() + 1  # Typically, =32
#         points, batches = [], []
#         for b in range(batch_size):
#             p = subsample(x[batch == b], scale=scale)
#             points.append(p)
#             batches.append(b * torch.ones_like(batch[: len(p)]))
#
#     return torch.cat(points, dim=0), torch.cat(batches, dim=0)
#
#
# def soft_distances(x, y, batch_x, batch_y, smoothness=0.01, atomtypes=None):
#     """Computes a soft distance function to the atom centers of a protein.
#
#     Implements Eq. (1) of the paper in a fast and numerically stable way.
#
#     Args:
#         x (Tensor): (N,3) atom centers.
#         y (Tensor): (M,3) sampling locations.
#         batch_x (integer Tensor): (N,) batch vector for x, as in PyTorch_geometric.
#         batch_y (integer Tensor): (M,) batch vector for y, as in PyTorch_geometric.
#         smoothness (float, optional): atom radii if atom types are not provided. Defaults to .01.
#         atomtypes (integer Tensor, optional): (N,6) one-hot encoding of the atom chemical types. Defaults to None.
#
#     Returns:
#         Tensor: (M,) values of the soft distance function on the points `y`.
#     """
#     # Build the (N, M, 1) symbolic matrix of squared distances:
#     x_i = LazyTensor(x[:, None, :])  # (N, 1, 3) atoms
#     y_j = LazyTensor(y[None, :, :])  # (1, M, 3) sampling points
#     D_ij = ((x_i - y_j) ** 2).sum(-1)  # (N, M, 1) squared distances
#
#     # Use a block-diagonal sparsity mask to support heterogeneous batch processing:
#     D_ij.ranges = diagonal_ranges(batch_x, batch_y)
#
#     if atomtypes is not None:
#         # Turn the one-hot encoding "atomtypes" into a vector of diameters "smoothness_i":
#         # (N, 6)  -> (N, 1, 1)  (There are 6 atom types)
#         atomic_radii = torch.cuda.FloatTensor(
#             [170, 110, 152, 155, 180, 190], device=x.device
#         )
#         atomic_radii = atomic_radii / atomic_radii.min()
#         atomtype_radii = atomtypes * atomic_radii[None, :]  # n_atoms, n_atomtypes
#         # smoothness = atomtypes @ atomic_radii  # (N, 6) @ (6,) = (N,)
#         smoothness = torch.sum(
#             smoothness * atomtype_radii, dim=1, keepdim=False
#         )  # n_atoms, 1
#         smoothness_i = LazyTensor(smoothness[:, None, None])
#
#         # Compute an estimation of the mean smoothness in a neighborhood
#         # of each sampling point:
#         # density = (-D_ij.sqrt()).exp().sum(0).view(-1)  # (M,) local density of atoms
#         # smooth = (smoothness_i * (-D_ij.sqrt()).exp()).sum(0).view(-1)  # (M,)
#         # mean_smoothness = smooth / density  # (M,)
#
#         # soft_dists = -mean_smoothness * (
#         #    (-D_ij.sqrt() / smoothness_i).logsumexp(dim=0)
#         # ).view(-1)
#         mean_smoothness = (-D_ij.sqrt()).exp().sum(0)
#         mean_smoothness_j = LazyTensor(mean_smoothness[None, :, :])
#         mean_smoothness = (
#             smoothness_i * (-D_ij.sqrt()).exp() / mean_smoothness_j
#         )  # n_atoms, n_points, 1
#         mean_smoothness = mean_smoothness.sum(0).view(-1)
#         soft_dists = -mean_smoothness * (
#             (-D_ij.sqrt() / smoothness_i).logsumexp(dim=0)
#         ).view(-1)
#
#     else:
#         soft_dists = -smoothness * ((-D_ij.sqrt() / smoothness).logsumexp(dim=0)).view(
#             -1
#         )
#
#     return soft_dists




# Surface mesh -> Normals ======================================================


# def mesh_normals_areas(vertices, triangles=None, scale=[1.0], batch=None, normals=None):
#     """Returns a smooth field of normals, possibly at different scales.
#
#     points, triangles or normals, scale(s)  ->      normals
#     (N, 3),    (3, T) or (N,3),      (S,)   ->  (N, 3) or (N, S, 3)
#
#     Simply put - if `triangles` are provided:
#       1. Normals are first computed for every triangle using simple 3D geometry
#          and are weighted according to surface area.
#       2. The normal at any given vertex is then computed as the weighted average
#          of the normals of all triangles in a neighborhood specified
#          by Gaussian windows whose radii are given in the list of "scales".
#
#     If `normals` are provided instead, we simply smooth the discrete vector
#     field using Gaussian windows whose radii are given in the list of "scales".
#
#     If more than one scale is provided, normal fields are computed in parallel
#     and returned in a single 3D tensor.
#
#     Args:
#         vertices (Tensor): (N,3) coordinates of mesh vertices or 3D points.
#         triangles (integer Tensor, optional): (3,T) mesh connectivity. Defaults to None.
#         scale (list of floats, optional): (S,) radii of the Gaussian smoothing windows. Defaults to [1.].
#         batch (integer Tensor, optional): batch vector, as in PyTorch_geometric. Defaults to None.
#         normals (Tensor, optional): (N,3) raw normals vectors on the vertices. Defaults to None.
#
#     Returns:
#         (Tensor): (N,3) or (N,S,3) point normals.
#         (Tensor): (N,) point areas, if triangles were provided.
#     """
#
#     # Single- or Multi-scale mode:
#     if hasattr(scale, "__len__"):
#         scales, single_scale = scale, False
#     else:
#         scales, single_scale = [scale], True
#     scales = torch.Tensor(scales).type_as(vertices)  # (S,)
#
#     # Compute the "raw" field of normals:
#     if triangles is not None:
#         # Vertices of all triangles in the mesh:
#         A = vertices[triangles[0, :]]  # (N, 3)
#         B = vertices[triangles[1, :]]  # (N, 3)
#         C = vertices[triangles[2, :]]  # (N, 3)
#
#         # Triangle centers and normals (length = surface area):
#         centers = (A + B + C) / 3  # (N, 3)
#         V = (B - A).cross(C - A)  # (N, 3)
#
#         # Vertice areas:
#         S = (V ** 2).sum(-1).sqrt() / 6  # (N,) 1/3 of a triangle area
#         areas = torch.zeros(len(vertices)).type_as(vertices)  # (N,)
#         areas.scatter_add_(0, triangles[0, :], S)  # Aggregate from "A's"
#         areas.scatter_add_(0, triangles[1, :], S)  # Aggregate from "B's"
#         areas.scatter_add_(0, triangles[2, :], S)  # Aggregate from "C's"
#
#     else:  # Use "normals" instead
#         areas = None
#         V = normals
#         centers = vertices
#
#     # Normal of a vertex = average of all normals in a ball of size "scale":
#     x_i = LazyTensor(vertices[:, None, :])  # (N, 1, 3)
#     y_j = LazyTensor(centers[None, :, :])  # (1, M, 3)
#     v_j = LazyTensor(V[None, :, :])  # (1, M, 3)
#     s = LazyTensor(scales[None, None, :])  # (1, 1, S)
#
#     D_ij = ((x_i - y_j) ** 2).sum(-1)  #  (N, M, 1)
#     K_ij = (-D_ij / (2 * s ** 2)).exp()  # (N, M, S)
#
#     # Support for heterogeneous batch processing:
#     if batch is not None:
#         batch_vertices = batch
#         batch_centers = batch[triangles[0, :]] if triangles is not None else batch
#         K_ij.ranges = diagonal_ranges(batch_vertices, batch_centers)
#
#     if single_scale:
#         U = (K_ij * v_j).sum(dim=1)  # (N, 3)
#     else:
#         U = (K_ij.tensorprod(v_j)).sum(dim=1)  # (N, S*3)
#         U = U.view(-1, len(scales), 3)  # (N, S, 3)
#
#     normals = F.normalize(U, p=2, dim=-1)  # (N, 3) or (N, S, 3)
#
#     return normals, areas


# Compute tangent planes and curvatures ========================================


def tangent_vectors(normals):
    """Returns a pair of vector fields u and v to complete the orthonormal basis [n,u,v].

          normals        ->             uv
    (N, 3) or (N, S, 3)  ->  (N, 2, 3) or (N, S, 2, 3)

    This routine assumes that the 3D "normal" vectors are normalized.
    It is based on the 2017 paper from Pixar, "Building an orthonormal basis, revisited".

    Args:
        normals (Tensor): (N,3) or (N,S,3) normals `n_i`, i.e. unit-norm 3D vectors.

    Returns:
        (Tensor): (N,2,3) or (N,S,2,3) unit vectors `u_i` and `v_i` to complete
            the tangent coordinate systems `[n_i,u_i,v_i].
    """
    x, y, z = normals[..., 0], normals[..., 1], normals[..., 2]
    s = (2 * (z >= 0)) - 1.0  # = z.sign(), but =1. if z=0.
    a = -1 / (s + z)
    b = x * y * a
    uv = torch.stack((1 + s * x * x * a, s * b, -s * x, b, s + y * y * a, -y), dim=-1)
    uv = uv.view(uv.shape[:-1] + (2, 3))

    return uv


#  Fast tangent convolution layer ===============================================
class ContiguousBackward(torch.autograd.Function):
    """
    Function to ensure contiguous gradient in backward pass. To be applied after PyKeOps reduction.
    N.B.: This workaround fixes a bug that will be fixed in ulterior KeOp releases.
    """
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.contiguous()
