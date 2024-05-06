import time
import torch
import torch.nn as nn
from datasets.protein_dataset import Protein_ligand
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pytorch3d.ops.knn import knn_points
from config.Arguments import parser
import  numpy as np
from utils.geometry_processing import tangent_vectors
import torch.nn.functional as F
from math import pi, sqrt
from pytorch3d.ops.knn import knn_points
from pytorch3d.ops.ball_query import ball_query
from pytorch3d.ops.sample_farthest_points import sample_farthest_points
from torch.nn import (
    Sequential as Seq,
    Dropout,
    Linear as Lin,
    LeakyReLU,
    ReLU,
    BatchNorm1d as BN,
)

from .transformer_model import Transformer_cls



def get_graph_feature(x, k):
    _, idx, _ = knn_points(p1=x, p2=x, K=k)
    feature = index_points(x, idx)

    feature = torch.cat((feature - x.unsqueeze(-2), x.unsqueeze(-2).repeat(1,1,k,1)), dim=-1).contiguous()

    return feature


class DEConv(torch.nn.Module):
    """Set abstraction module."""

    def __init__(self, nn, k, aggr):
        super(DEConv, self).__init__()
        self.conv = nn
        self.aggr = aggr
        self.k = k

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)
        x = self.conv(x)
        x = x.max(dim=-2, keepdim=False)[0]

        return x


class DGCNN_seg(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, n_layers, k=40, aggr="max"
    ):
        super(DGCNN_seg, self).__init__()

        self.name = "DGCNN_seg"
        self.I, self.O = (
            in_channels + 3,
            out_channels,
        )  # Add coordinates to input channels
        self.n_layers = n_layers

        self.transform_1 = DEConv(MLP([2 * 3, 64, 128],batch_norm=False), k, aggr)
        self.transform_2 = MLP([128, 1024],batch_norm=False)
        self.transform_3 = MLP([1024, 512, 256], batch_norm=False)
        self.transform_4 = Lin(256, 3 * 3)

        self.conv_layers = nn.ModuleList(
            [DEConv(MLP([2 * self.I, self.O, self.O], batch_norm=False), k, aggr)]
            + [
                DEConv(MLP([2 * self.O, self.O, self.O], batch_norm=False), k, aggr)
                for i in range(n_layers - 1)
            ]
        )

        self.linear_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.O, self.O), nn.ReLU(), nn.Linear(self.O, self.O)
                )
                for i in range(n_layers)
            ]
        )

        self.linear_transform = nn.ModuleList(
            [nn.Linear(self.I, self.O)]
            + [nn.Linear(self.O, self.O) for i in range(n_layers - 1)]
        )
        self.use_TNet = True

    def forward(self, positions, features):
        # Lab: (B,), Pos: (N, 3), Batch: (N,)
        pos, feat = positions, features

        # TransformNet:
        x = pos  # Don't use the normals!

        if self.use_TNet:
            x = self.transform_1(x)  # (B, N, 3) -> (B, N, 128)
            x = self.transform_2(x)  # (B, N, 128) -> (B, N, 1024)
            x = x.max(dim = -2, keepdim=False)[0]  # (B, 1024)

            x = self.transform_3(x)  # (B, 256)
            x = self.transform_4(x)  # (B, 3*3)
            x = x.view(-1, 3, 3)  # (B, 3, 3)

            # Apply the transform:
            x0 = torch.einsum("nki,nij->nkj", pos, x)  # (N, 3)
        else:
            x0 = x

        # Add features to coordinates
        x = torch.cat([x0, feat], dim=-1).contiguous()

        for i in range(self.n_layers):
            x_i = self.conv_layers[i](x)
            x_i = self.linear_layers[i](x_i)
            x = self.linear_transform[i](x)
            x = x + x_i

        return x


def soft_dimension(features):
    """Continuous approximation of the rank of a (N, D) sample.

    Let "s" denote the (D,) vector of eigenvalues of Cov,
    the (D, D) covariance matrix of the sample "features".
    Then,
        R(features) = \sum_i sqrt(s_i) / \max_i sqrt(s_i)

    This quantity encodes the number of PCA components that would be
    required to describe the sample with a good precision.
    It is equal to D if the sample is isotropic, but is generally much lower.

    Up to the re-normalization by the largest eigenvalue,
    this continuous pseudo-rank is equal to the nuclear norm of the sample.
    """

    nfeat = features.shape[-1]
    # features = features.view(-1, nfeat)
    x = features - torch.mean(features, dim=1, keepdim=True)
    cov = x.permute(0,2,1) @ x
    try:
        u, s, v = torch.svd(cov)
        R = s.sqrt().sum() / s.sqrt().max()
    except:
        return -1
    return R.item()


def MLP(channels, batch_norm=True):
    """Multi-layer perceptron, with ReLU non-linearities and batch normalization."""
    return Seq(
        *[
            Seq(
                Lin(channels[i - 1], channels[i]),
                BN(channels[i]) if batch_norm else nn.Identity(),
                LeakyReLU(negative_slope=0.2),
            )
            for i in range(1, len(channels))
        ]
    )


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def sample_and_group(npoint, radius, nsample, xyz, points):
    B, N, C = xyz.shape
    S = int(npoint * N)
    if not S == N:
        new_xyz, _ = sample_farthest_points(xyz, K = S)
    else:
        new_xyz = xyz

    _, idx, _ = ball_query(p1 = new_xyz, p2 = xyz, K=nsample, radius=radius)
    mask = idx != -1
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_xyz, new_points, mask


class SAModule(torch.nn.Module):
    """Set abstraction module."""

    def __init__(self, ratio, r, nn, max_num_neighbors=64):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = nn
        self.max_num_neighbors = max_num_neighbors

    def forward(self, x, pos):

        xyz = pos
        points = x

        new_xyz, new_points, mask = sample_and_group(self.ratio, self.r, self.max_num_neighbors, xyz, points)

        new_points = self.conv(new_points)

        new_points = new_points * mask.unsqueeze(-1)

        new_points = torch.max(new_points, 2)[0]

        return new_points, new_xyz


class PointNet2_seg(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(PointNet2_seg, self).__init__()

        self.name = "PointNet2"
        self.I, self.O = in_channels, out_channels
        self.radius = args.radius
        self.k = 512  # We don't restrict the number of points in a patch
        self.n_layers = args.n_layers

        # self.sa1_module = SAModule(1.0, self.radius, MLP([self.I+3, self.O, self.O]),self.k)
        self.layers = nn.ModuleList(
            [SAModule(1.0, self.radius, MLP([self.I + 3, self.O, self.O], batch_norm=False), self.k)]
            + [
                SAModule(1.0, self.radius, MLP([self.O + 3, self.O, self.O], batch_norm=False), self.k)
                for i in range(self.n_layers - 1)
            ]
        )

        self.linear_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.O, self.O), nn.ReLU(), nn.Linear(self.O, self.O)
                )
                for i in range(self.n_layers)
            ]
        )

        self.linear_transform = nn.ModuleList(
            [nn.Linear(self.I, self.O)]
            + [nn.Linear(self.O, self.O) for i in range(self.n_layers - 1)]
        )

    def forward(self, positions, features):
        x = (features, positions)
        for i, layer in enumerate(self.layers):
            x_i, pos = layer(x[0], x[1])
            x_i = self.linear_layers[i](x_i)
            x = self.linear_transform[i](x[0])
            x = x + x_i
            x = (x, pos)

        return x[0]


class Atom_embedding_MP(nn.Module):
    def __init__(self, args):
        super(Atom_embedding_MP, self).__init__()
        self.D = args.atom_dims
        self.k = 16
        self.n_layers = 3
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * self.D + 1, 2 * self.D + 1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(2 * self.D + 1, self.D),
                )
                for i in range(self.n_layers)
            ]
        )
        self.norm = nn.ModuleList(
            [nn.GroupNorm(2, self.D) for i in range(self.n_layers)]
        )
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, dist, atomtypes):

        num_points = dist.shape[1]
        num_dims = atomtypes.shape[-1]

        point_emb = torch.ones_like(atomtypes[:,:,0,:])
        for i in range(self.n_layers):
            features = torch.cat([point_emb[:,:,None,:].repeat(1, 1, self.k, 1), atomtypes, dist], dim = -1)

            messages = self.mlp[i](features)  # N,8,6
            messages = messages.sum(-2)  # N,6
            point_emb = point_emb + self.relu(self.norm[i](messages.reshape(-1,6))).reshape(-1, num_points, num_dims)

        return point_emb


class AtomNet_MP(nn.Module):
    def __init__(self, args):
        super(AtomNet_MP, self).__init__()
        self.args = args

        self.transform_types = nn.Sequential(
            nn.Linear(args.atom_dims, args.atom_dims),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(args.atom_dims, args.atom_dims),
        )

        self.embed = Atom_embedding_MP(args)
        # self.atom_atom = Atom_Atom_embedding_MP(args)

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('atomnet'):
                    base_ckpt[k[len('atomnet.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                else:
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print('missing_keys')
                print(
                        print(incompatible.missing_keys)
                    )
            if incompatible.unexpected_keys:
                print('unexpected_keys')
                print(
                        print(incompatible.unexpected_keys)

                    )

            print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')

    def forward(self, curvature, dist, atom_type):
        # Run a DGCNN on the available information:
        atomtypes = self.transform_types(atom_type)
        # atomtypes = self.atom_atom(
        #     atom_xyz, atom_xyz, atomtypes, atom_batch, atom_batch
        # )
        atomtypes = self.embed(dist, atomtypes)


        return atomtypes


class dMaSIFConv(nn.Module):
    def __init__(
        self, in_channels=1, out_channels=1, radius=1.0, hidden_units=None, cheap=False
    ):
        """Creates the KeOps convolution layer.

        I = in_channels  is the dimension of the input features
        O = out_channels is the dimension of the output features
        H = hidden_units is the dimension of the intermediate representation
        radius is the size of the pseudo-geodesic Gaussian window w_ij = W(d_ij)


        This affordable layer implements an elementary "convolution" operator
        on a cloud of N points (x_i) in dimension 3 that we decompose in three steps:

          1. Apply the MLP "net_in" on the input features "f_i". (N, I) -> (N, H)

          2. Compute H interaction terms in parallel with:
                  f_i = sum_j [ w_ij * conv(P_ij) * f_j ]
            In the equation above:
              - w_ij is a pseudo-geodesic window with a set radius.
              - P_ij is a vector of dimension 3, equal to "x_j-x_i"
                in the local oriented basis at x_i.
              - "conv" is an MLP from R^3 to R^H:
                 - with 1 linear layer if "cheap" is True;
                 - with 2 linear layers and C=8 intermediate "cuts" otherwise.
              - "*" is coordinate-wise product.
              - f_j is the vector of transformed features.

          3. Apply the MLP "net_out" on the output features. (N, H) -> (N, O)


        A more general layer would have implemented conv(P_ij) as a full
        (H, H) matrix instead of a mere (H,) vector... At a much higher
        computational cost. The reasoning behind the code below is that
        a given time budget is better spent on using a larger architecture
        and more channels than on a very complex convolution operator.
        Interactions between channels happen at steps 1. and 3.,
        whereas the (costly) point-to-point interaction step 2.
        lets the network aggregate information in spatial neighborhoods.

        Args:
            in_channels (int, optional): numper of input features per point. Defaults to 1.
            out_channels (int, optional): number of output features per point. Defaults to 1.
            radius (float, optional): deviation of the Gaussian window on the
                quasi-geodesic distance `d_ij`. Defaults to 1..
            hidden_units (int, optional): number of hidden features per point.
                Defaults to out_channels.
            cheap (bool, optional): shall we use a 1-layer deep Filter,
                instead of a 2-layer deep MLP? Defaults to False.
        """

        super(dMaSIFConv, self).__init__()

        self.Input = in_channels
        self.Output = out_channels
        self.Radius = radius
        self.Hidden = self.Output if hidden_units is None else hidden_units
        self.Cuts = 8  # Number of hidden units for the 3D MLP Filter.
        self.cheap = cheap

        # For performance reasons, we cut our "hidden" vectors
        # in n_heads "independent heads" of dimension 8.
        self.heads_dim = 8  # 4 is probably too small; 16 is certainly too big

        # We accept "Hidden" dimensions of size 1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, ...
        if self.Hidden < self.heads_dim:
            self.heads_dim = self.Hidden

        if self.Hidden % self.heads_dim != 0:
            raise ValueError(f"The dimension of the hidden units ({self.Hidden})"\
                    + f"should be a multiple of the heads dimension ({self.heads_dim}).")
        else:
            self.n_heads = self.Hidden // self.heads_dim


        # Transformation of the input features:
        self.net_in = nn.Sequential(
            nn.Linear(self.Input, self.Hidden),  # (H, I) + (H,)
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.Hidden, self.Hidden),  # (H, H) + (H,)
            # nn.LayerNorm(self.Hidden),#nn.BatchNorm1d(self.Hidden),
            nn.LeakyReLU(negative_slope=0.2),
        )  #  (H,)
        self.norm_in = nn.GroupNorm(4, self.Hidden)
        # self.norm_in = nn.LayerNorm(self.Hidden)
        # self.norm_in = nn.Identity()

        # 3D convolution filters, encoded as an MLP:
        if cheap:
            self.conv = nn.Sequential(
                nn.Linear(3, self.Hidden), nn.ReLU()  # (H, 3) + (H,)
            )  # KeOps does not support well LeakyReLu
        else:
            self.conv = nn.Sequential(
                nn.Linear(3, self.Cuts),  # (C, 3) + (C,)
                nn.ReLU(),  # KeOps does not support well LeakyReLu
                nn.Linear(self.Cuts, self.Hidden),
            )  # (H, C) + (H,)

        # Transformation of the output features:
        self.net_out = nn.Sequential(
            nn.Linear(self.Hidden, self.Output),  # (O, H) + (O,)
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.Output, self.Output),  # (O, O) + (O,)
            # nn.LayerNorm(self.Output),#nn.BatchNorm1d(self.Output),
            nn.LeakyReLU(negative_slope=0.2),
        )  #  (O,)

        self.norm_out = nn.GroupNorm(4, self.Output)
        # self.norm_out = nn.LayerNorm(self.Output)
        # self.norm_out = nn.Identity()

        # Custom initialization for the MLP convolution filters:
        # we get interesting piecewise affine cuts on a normalized neighborhood.
        with torch.no_grad():
            nn.init.normal_(self.conv[0].weight)
            nn.init.uniform_(self.conv[0].bias)
            self.conv[0].bias *= 0.8 * (self.conv[0].weight ** 2).sum(-1).sqrt()

            if not cheap:
                nn.init.uniform_(
                    self.conv[2].weight,
                    a=-1 / np.sqrt(self.Cuts),
                    b=1 / np.sqrt(self.Cuts),
                )
                nn.init.normal_(self.conv[2].bias)
                self.conv[2].bias *= 0.5 * (self.conv[2].weight ** 2).sum(-1).sqrt()


    def forward(self, points, nuv, features):
        """Performs a quasi-geodesic interaction step.

        points, local basis, in features  ->  out features
        (N, 3),   (N, 3, 3),    (N, I)    ->    (N, O)

        This layer computes the interaction step of Eq. (7) in the paper,
        in-between the application of two MLP networks independently on all
        feature vectors.

        Args:
            points (Tensor): (N,3) point coordinates `x_i`.
            nuv (Tensor): (N,3,3) local coordinate systems `[n_i,u_i,v_i]`.
            features (Tensor): (N,I) input feature vectors `f_i`.
            ranges (6-uple of integer Tensors, optional): low-level format
                to support batch processing, as described in the KeOps documentation.
                In practice, this will be built by a higher-level object
                to encode the relevant "batch vectors" in a way that is convenient
                for the KeOps CUDA engine. Defaults to None.

        Returns:
            (Tensor): (N,O) output feature vectors `f'_i`.
        """

        # 1. Transform the input features: -------------------------------------
        B, N, _ = features.shape
        features = self.net_in(features)  # (B, N, I) -> (B, N, H)
        features = features.permute(0,2,1)  # (B,H,N)
        features = self.norm_in(features)
        features = features.permute(0,2,1)  # (B, H, N) -> (B, N, H)

        # 2. Compute the local "shape contexts": -------------------------------

        # 2.a Normalize the kernel radius:
        points = points / (sqrt(2.0) * self.Radius)  # (B, N, 3)

        # 2.b Encode the variables as KeOps LazyTensors

        # Vertices:
        x_i = points[:, :, None, :]  # (B, N, 1, 3)
        x_j = points[:, None, :, :]  # (B, 1, N, 3)

        # WARNING - Here, we assume that the normals are fixed:
        normals = (
            nuv[:, :, 0, :].contiguous().detach()
        )  # (B, N, 3) - remove the .detach() if needed

        # Local bases:
        nuv_i = nuv.unsqueeze(2) #.view(B, N, 1, 9)  # (N, 1, 9)
        # Normals:
        n_i = nuv_i[:,:,:,0]  # (B, N, 1, 3)

        n_j = normals[:, None, :, :]  # (B, 1, N, 3)

        # To avoid register spilling when using large embeddings, we perform our KeOps reduction
        # over the vector of length "self.Hidden = self.n_heads * self.heads_dim"
        # as self.n_heads reduction over vectors of length self.heads_dim (= "Hd" in the comments).
        head_out_features = []
        for head in range(self.n_heads):

            # Extract a slice of width Hd from the feature array
            head_start = head * self.heads_dim
            head_end = head_start + self.heads_dim
            head_features = features[:, :, head_start:head_end].contiguous()  # (B, N, H) -> (B, N, Hd)

            # Features:
            f_j = head_features[:, None, :, :]  # (B, 1, N, Hd)

            # Convolution parameters:
            # if self.cheap:
            #     # Extract a slice of Hd lines: (H, 3) -> (Hd, 3)
            #     A = self.conv[0].weight[head_start:head_end, :].contiguous()
            #     # Extract a slice of Hd coefficients: (H,) -> (Hd,)
            #     B = self.conv[0].bias[head_start:head_end].contiguous()
            #     AB = torch.cat((A, B[:, None]), dim=1)  # (Hd, 4)
            #     ab = AB.view(1, 1, -1)  # (1, 1, Hd*4)
            # else:
            #     A_1, B_1 = self.conv[0].weight, self.conv[0].bias  # (C, 3), (C,)
            #     # Extract a slice of Hd lines: (H, C) -> (Hd, C)
            #     A_2 = self.conv[2].weight[head_start:head_end, :].contiguous()
            #     # Extract a slice of Hd coefficients: (H,) -> (Hd,)
            #     B_2 = self.conv[2].bias[head_start:head_end].contiguous()
            #     a_1 = A_1.view(1, 1, -1)  # (1, 1, C*3)
            #     b_1 = B_1.view(1, 1, -1)  # (1, 1, C)
            #     a_2 = A_2.view(1, 1, -1)  # (1, 1, Hd*C)
            #     b_2 = B_2.view(1, 1, -1)  # (1, 1, Hd)

            # 2.c Pseudo-geodesic window:
            # Pseudo-geodesic squared distance:
            d2_ij = ((x_j - x_i) ** 2).sum(-1) * ((2 - ((n_i * n_j).sum(-1))) ** 2)  # (N, N, 1)
            # Gaussian window:
            window_ij = (-d2_ij).exp()  # (N, N, 1)

            # 2.d Local MLP:
            # Local coordinates:
            X_ij = torch.matmul(nuv_i , (x_j - x_i).unsqueeze(-1)).squeeze(-1)
            # X_ij = nuv_i.matvecmult(x_j - x_i)  # (N, N, 9) "@" (N, N, 3) = (N, N, 3)
            # MLP:
            if self.cheap:
                X_ij = self.conv[0](X_ij)
                X_ij = X_ij.relu()  # (N, N, Hd)
            else:
                X_ij = self.conv[0](X_ij)
                X_ij = X_ij.relu()  # (N, N, C)
                X_ij = self.conv[2](X_ij)
                X_ij = X_ij.relu()

            # 2.e Actual computation:
            F_ij = window_ij.unsqueeze(-1) * X_ij * f_j  # (B, N, N, Hd)

            head_out_features.append(F_ij.sum(dim=-2))  # (B, N, Hd)

        # Concatenate the result of our n_heads "attention heads":
        features = torch.cat(head_out_features, dim=-1)  # n_heads * (N, Hd) -> (N, H)

        # 3. Transform the output features: ------------------------------------
        features = self.net_out(features)  # (N, H) -> (N, O)
        features = features.permute(0,2,1)
        features = self.norm_out(features)
        features = features.permute(0,2,1)

        return features


class dMaSIFConv_seg(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels, n_layers, radius=9.0):
        super(dMaSIFConv_seg, self).__init__()

        self.name = "dMaSIFConv_seg_keops"
        self.radius = radius
        self.I, self.O = in_channels, out_channels

        self.layers = nn.ModuleList(
            [dMaSIFConv(self.I, self.O, radius, self.O)]
            + [dMaSIFConv(self.O, self.O, radius, self.O) for i in range(n_layers - 1)]
        )

        self.linear_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.O, self.O), nn.ReLU(), nn.Linear(self.O, self.O)
                )
                for i in range(n_layers)
            ]
        )

        self.linear_transform = nn.ModuleList(
            [nn.Linear(self.I, self.O)]
            + [nn.Linear(self.O, self.O) for i in range(n_layers - 1)]
        )

    def forward(self, features):
        # Lab: (B,), Pos: (N, 3), Batch: (N,)
        points, nuv = self.points, self.nuv
        x = features
        for i, layer in enumerate(self.layers):
            x_i = layer(points, nuv, x)
            x_i = self.linear_layers[i](x_i)
            x = self.linear_transform[i](x)
            x = x + x_i

        return x

    def load_mesh(self, xyz, normals=None, weights=None):

        # 1. Save the vertices for later use in the convolutions ---------------
        self.points = xyz
        self.normals = normals
        self.weights = weights

        # 2. Estimate the normals and tangent frame ----------------------------
        # Normalize the scale:
        points = xyz / self.radius

        # Normals and local areas:
        tangent_bases = tangent_vectors(normals)  # Tangent basis (B, N, 2, 3)
        B, N, _, _ = tangent_bases.shape

        # 3. Steer the tangent bases according to the gradient of "weights" ----

        # 3.a) Encoding as KeOps LazyTensors:
        # Orientation scores:
        weights_j = weights  # (1, N, 1)
        # Vertices:
        x_i = points[:, :, None, :]  # (N, 1, 3)
        x_j = points[:, None, :, :]  # (1, N, 3)
        # Normals:
        # n_i = normals[:, :, None, :]  # (N, 1, 3)
        # n_j = normals[:, None, :, :]  # (1, N, 3)
        # Tangent basis:
        # uv_i = tangent_bases.view(B, N, 6)  # (N, 1, 6)

        # 3.b) Pseudo-geodesic window:
        # Pseudo-geodesic squared distance:
        rho2_ij = ((x_j - x_i) ** 2).sum(-1) * ((2 - torch.matmul(normals, normals.permute(0,2,1))) ** 2)  # (N, N, 1)
        # Gaussian window:
        window_ij = (-rho2_ij).exp()  # (N, N, 1)

        # 3.c) Coordinates in the (u, v) basis - not oriented yet:
        # X_ij = uv_i.matvecmult(x_j - x_i)  # (N, N, 2)
        X_ij = torch.matmul((x_j - x_i).unsqueeze(-2), tangent_bases.unsqueeze(2).permute(0,1,2,4,3)).squeeze(-2)

        # 3.d) Local average in the tangent plane:
        orientation_weight_ij = (window_ij * weights_j).unsqueeze(-1)  # (N, N, 1)
        orientation_vector_ij = orientation_weight_ij * X_ij  # (N, N, 2)

        # Support for heterogeneous batch processing:
        # orientation_vector_ij.ranges = self.ranges  # Block-diagonal sparsity mask

        orientation_vector_i = orientation_vector_ij.sum(dim=-2)  # (N, 2)
        orientation_vector_i = (
            orientation_vector_i + 1e-5
        )  # Just in case someone's alone...

        # 3.e) Normalize stuff:
        orientation_vector_i = F.normalize(orientation_vector_i, p=2, dim=-1)  #  (N, 2)
        ex_i, ey_i = (
            orientation_vector_i[:, :, 0][:, :, None],
            orientation_vector_i[:, :, 1][:, :, None],
        )  # (N,1)

        # 3.f) Re-orient the (u,v) basis:
        uv_i = tangent_bases  # (B, N, 2, 3)
        u_i, v_i = uv_i[:, :, 0, :], uv_i[:, :, 1, :]  # (B, N, 3)
        tangent_bases = torch.cat(
            (ex_i * u_i + ey_i * v_i, -ey_i * u_i + ex_i * v_i), dim=-1
        ).contiguous()  # (B, N, 6)

        # 4. Store the local 3D frame as an attribute --------------------------
        self.nuv = torch.cat(
            (normals.view(B, N, 1, 3), tangent_bases.view(B, N, 2, 3)), dim=-2
        )


class dMaSIF(nn.Module):
    def __init__(self, args):
        super(dMaSIF, self).__init__()
        self.curvature_scales = args.curvature_scales
        self.args = args

        I = args.in_channels
        O = args.orientation_units
        E = args.emb_dims
        H = args.post_units

        # Computes chemical features
        self.atomnet = AtomNet_MP(args)
        if args.ckpt is not None:
            self.atomnet.load_model_from_ckpt(args.ckpt)

        self.dropout = nn.Dropout(args.dropout)

        if args.embedding_layer == "dMaSIF":
            # Post-processing, without batch norm:
            self.orientation_scores = nn.Sequential(
                nn.Linear(I, O),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(O, 1),
            )

            # Segmentation network:
            self.conv = dMaSIFConv_seg(
                args,
                in_channels=I,
                out_channels=E,
                n_layers=args.n_layers,
                radius=args.radius,
            )

        elif args.embedding_layer == "DGCNN":
            self.conv = DGCNN_seg(I + 3, E, self.args.n_layers, self.args.k)

        elif args.embedding_layer == "PointNet++":
            self.conv = PointNet2_seg(args, I, E)

        elif args.embedding_layer == "Transformer":
            self.conv = Transformer_cls(args, I, E)
            if args.ckpt is not None:
                self.conv.load_model_from_ckpt(args.ckpt)
            # if args.search:
            #     self.conv2 = Transformer_seg(args, I, E)





    def forward(self, xyz, normal, curvature, dist, atom_type):
        feats_chem = self.atomnet(curvature, dist.unsqueeze(-1), atom_type)
        feats_geo = curvature
        features = torch.cat([feats_geo, feats_chem], dim = -1)
        features = self.dropout(features)

        if self.args.embedding_layer == "dMaSIF":
            self.conv.load_mesh(
                xyz,
                normals=normal,
                weights=self.orientation_scores(features)
            )
            embedding = self.conv(features)

        elif self.args.embedding_layer == "DGCNN":
            features = torch.cat([features, xyz], dim=-1).contiguous()
            embedding = self.conv(xyz, features)

        elif self.args.embedding_layer == "PointNet++":
            embedding = self.conv(xyz, features)

        # Transformer
        elif self.args.embedding_layer == "Transformer":
            embedding, feature_tsne = self.conv(xyz, features)
            # if self.args.search:
            #     embedding = self.conv2(xyz, features)

        return embedding











if __name__ == "__main__":
    dataset = Protein(phase='train', sample_num = 2048, sample_type = 'knn')
    dataloader = DataLoader(
        dataset,
        batch_size=2,
    )

    args = parser.parse_args()
    model = dMaSIF(args).cuda()


    for xyz, normal, label, curvature, dist, atom_type in tqdm(dataloader):
        xyz = xyz.cuda()
        normal = normal.cuda()
        label = label.cuda()
        curvature = curvature.cuda()
        dist = dist.cuda()
        atom_type = atom_type.cuda()

        pred = model(xyz, normal, curvature, dist, atom_type)
        # loss, _, _ = site_loss(pred, label)






