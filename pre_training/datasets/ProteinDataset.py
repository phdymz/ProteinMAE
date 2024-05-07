import os
import torch
import pickle
import numpy as np
import torch.utils.data as data
from scipy.spatial.transform import Rotation
from .io import IO
from .build import DATASETS
from utils.logger import *
import open3d as o3d
from pytorch3d.ops.knn import knn_points


def load_obj(path):
    """
    read a dictionary from a pickle file
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def to_tsfm(rot, trans):
    tsfm = np.eye(4)
    tsfm[:3, :3] = rot
    tsfm[:3, 3] = trans.flatten()
    return tsfm


def to_array(tensor):
    """
    Conver tensor to array
    """
    if (not isinstance(tensor, np.ndarray)):
        if (tensor.device == torch.device('cpu')):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor


def to_o3d_pcd(xyz):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_array(xyz))
    return pcd


def get_correspondences(src_pcd, tgt_pcd, trans, search_voxel_size, K=None):
    src_pcd.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)

    correspondences = []
    for i, point in enumerate(src_pcd.points):
        [count, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            correspondences.append([i, j])

    correspondences = np.array(correspondences)
    correspondences = torch.from_numpy(correspondences)
    return correspondences

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


@DATASETS.register_module()
class ProteinPretraining(data.Dataset):
    def __init__(self, config, K = 16, sample_type = 'uniform'):
        self.config = config
        self.infos = os.listdir(config.DATA_PATH)
        self.subset = config.subset
        self.base_dir = config.DATA_PATH
        # self.max_points = config.N_POINTS
        self.data_augmentation = config.subset == 'train'
        self.sample_points_num = config.npoints

        self.K = K
        self.sample_type = sample_type

        self.rot_factor = 1.
        self.augment_noise = config.AUG_NOISE

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='Protein-Pretraining')
        print_log(f'[DATASET] Open file {config.DATA_PATH}', logger='Protein-Pretraining')

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        pc_number = np.arange(pc.shape[0])
        permutation_arr = np.random.choice(pc_number, num, p=None)
        pc = pc[permutation_arr, :]
        return pc

    def __getitem__(self, item):
        # get pointcloud
        file_name = os.path.join(self.base_dir, self.infos[item])
        data = np.load(file_name)

        xyz = data['xyz']
        normal = data['normal']
        curvature = data['curvature']

        atom = data['atom']
        atom_type = data['atom_type']
        atom_center = data['atom_center']

        label = np.zeros([len(xyz),1]).astype('float32')

        if self.sample_type == 'uniform':
            if len(xyz) > self.sample_points_num:
                idx = np.random.choice(len(xyz), self.sample_points_num, replace=False)
                xyz = xyz[idx]
                normal = normal[idx]
                label = label[idx]
                curvature = curvature[idx]
            else:
                idx = np.random.choice(len(xyz), self.sample_points_num, replace=True)
                xyz = xyz[idx]
                normal = normal[idx]
                label = label[idx]
                curvature = curvature[idx]

            xyz = torch.from_numpy(xyz)
            normal = torch.from_numpy(normal)
            label = torch.from_numpy(label)
            curvature = torch.from_numpy(curvature)
            atom = torch.from_numpy(atom)
            atom_type = torch.from_numpy(atom_type)

        dists, idx, _ = knn_points(xyz.unsqueeze(0), atom.unsqueeze(0), K=self.K)
        dists = dists.squeeze(0)
        idx = idx.squeeze(0)
        atom_type_sel = atom_type[idx]

        # files = os.path.join(self.base_dir, self.infos[item])
        # sample = np.load(files, allow_pickle=True)
        # sample_xyz = sample['xyz'] - sample['atom_xyz'].mean(0)
        # sample_chem = sample['feat'][:, :5]
        #
        # # if we get too many points, we do some downsampling
        # if (sample_xyz.shape[0] > self.max_points):
        #     idx = np.random.permutation(sample_xyz.shape[0])[:self.max_points]
        #     sample_xyz = sample_xyz[idx]
        #     sample_chem = sample_chem[idx]
        #
        # if self.data_augmentation:
        #     # rotate the point cloud
        #     euler_ab = np.random.rand(3) * np.pi * 2 / self.rot_factor  # anglez, angley, anglex
        #     rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
        #     sample_xyz = np.matmul(rot_ab, sample_xyz.T).T
        #     # add gaussian noise
        #     sample_xyz += (np.random.rand(sample_xyz.shape[0], 3) - 0.5) * self.augment_noise
        #
        # sample_xyz = torch.from_numpy(sample_xyz).float()
        # sample_chem = torch.from_numpy(sample_chem).float()
        # data = torch.cat([sample_xyz, sample_chem], dim=-1)
        # data = self.random_sample(data, self.sample_points_num)
        # return 0, 0, data

        return 0, 0, xyz, curvature, dists, atom_type_sel

    def __len__(self):
        return len(self.infos)


@DATASETS.register_module()
class ProteinTask1(data.Dataset):
    def __init__(self, config):
        self.infos = os.listdir(os.path.join(config.DATA_PATH, config.subset))
        self.config = config
        self.subset = config.subset
        self.base_dir = config.DATA_PATH
        self.max_points = config.N_POINTS
        self.data_augmentation = config.subset == 'train'
        self.sample_points_num = config.npoints
        self.num_category = config.NUM_CATEGORY

        self.rot_factor = 1.

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='Protein-Task1')
        print_log(f'[DATASET] Open file {config.DATA_PATH}', logger='Protein-Task1')

    def _get_item(self, index):

        # get pointcloud
        files = os.path.join(self.base_dir, self.subset, self.infos[index])
        sample = np.load(files, allow_pickle=True)
        sample_xyz = sample['xyz'] - sample['atom_xyz'].mean(0)
        sample_chem = sample['feat'][:, :5]
        label = sample['label']

        # if we get too many points, we do some downsampling
        if (sample_xyz.shape[0] > self.max_points):
            idx = np.random.permutation(sample_xyz.shape[0])[:self.max_points]
            sample_xyz = sample_xyz[idx]
            sample_chem = sample_chem[idx]

        if self.data_augmentation:
            # rotate the point cloud
            euler_ab = np.random.rand(3) * np.pi * 2 / self.rot_factor  # anglez, angley, anglex
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            sample_xyz = np.matmul(rot_ab, sample_xyz.T).T

        sample_xyz = torch.from_numpy(sample_xyz).float()
        sample_chem = torch.from_numpy(sample_chem).float()
        data = torch.cat([sample_xyz, sample_chem], dim=-1)
        data = farthest_point_sample(data, self.sample_points_num)

        data[:, 0:3] = pc_normalize(data[:, 0:3])

        return data, label

    def __getitem__(self, index):
        points, label = self._get_item(index)
        pt_idxs = np.arange(0, points.shape[0])   # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        return 'Task1', 'sample', (current_points, label)

    def __len__(self):
        return len(self.infos)
