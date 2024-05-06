import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pytorch3d.ops.knn import knn_points
from scipy.spatial.transform import Rotation
import open3d as o3d


class Protein_ligand(Dataset):
    def __init__(self, data_root = '/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/Ligand/processed', K = 16,
                 phase = 'train',  rot_aug = True, sample_type = 'knn', sample_num = 2048, pretrain = False, radius = 12,
                 over_sample = False, over_sample_num = 2048, balance_sampling = False):
        # dataset parameters
        assert phase in ['train', 'test', 'valid']
        assert sample_type in ['radius', 'knn', None]

        self.root = os.path.join(data_root, phase)
        self.phase = phase
        self.sample_num = sample_num
        self.sample_type = sample_type
        self.rot_aug = rot_aug
        self.factor = 1
        self.pretrain = pretrain
        self.K = K
        self.files = np.load('./split/local/'+phase+'.npy')
        self.radius = radius
        self.over_sample = over_sample
        self.over_sample_num = over_sample_num
        self.balance_sampling = balance_sampling
        if self.balance_sampling:
            self.init_files()

    def init_files(self):
        print('solve data imbalance in '+self.phase+'set')
        labels = np.zeros(7)
        for i, item in enumerate(tqdm(self.files)):
            file_name = os.path.join(self.root, item)
            data = np.load(file_name)
            label = int(data['label'])
            labels[label] += 1
        labels = labels/labels.sum()
        labels = np.round(labels.max()/labels)
        labels = labels.astype(np.int)
        files = []
        for i, item in enumerate(tqdm(self.files)):
            file_name = os.path.join(self.root, item)
            data = np.load(file_name)
            label = int(data['label'])
            for j in range(labels[label]):
                files.append(item)
        np.random.seed(0)
        np.random.shuffle(files)
        self.files = files


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_name = os.path.join(self.root, self.files[index])
        data = np.load(file_name)

        xyz = data['xyz']
        normal = data['normal']
        curvature = data['curvature']

        atom = data['atom']
        atom_type = data['atom_type']
        atom_center = data['atom_center']
        label = data['label']
        ligand_coord = data['ligand']


        if self.sample_type == 'knn':
            if len(xyz) <= self.sample_num:
                idx = np.random.choice(len(xyz), self.sample_num, replace=True)
                xyz = xyz[idx]
                normal = normal[idx]
                curvature = curvature[idx]

                xyz = torch.from_numpy(xyz)
                normal = torch.from_numpy(normal)
                label = torch.from_numpy(label)
                curvature = torch.from_numpy(curvature)
                atom = torch.from_numpy(atom)
                atom_type = torch.from_numpy(atom_type)

            else:
                kpts = torch.from_numpy(ligand_coord.mean(0)).reshape(1,3)
                xyz = torch.from_numpy(xyz)
                normal = torch.from_numpy(normal)
                label = torch.from_numpy(label)
                curvature = torch.from_numpy(curvature)
                atom = torch.from_numpy(atom)
                atom_type = torch.from_numpy(atom_type)

                _, idx, _ = knn_points(kpts.unsqueeze(0), xyz.unsqueeze(0), K=self.sample_num)
                idx = idx.squeeze()
                xyz = xyz[idx]
                normal = normal[idx]
                curvature = curvature[idx]

        #coming soon
        if self.sample_type == 'radius':
            if len(xyz) <= self.sample_num:
                idx = np.random.choice(len(xyz), self.sample_num, replace=True)
                xyz = xyz[idx]
                normal = normal[idx]
                curvature = curvature[idx]

                xyz = torch.from_numpy(xyz)
                normal = torch.from_numpy(normal)
                label = torch.from_numpy(label)
                curvature = torch.from_numpy(curvature)
                atom = torch.from_numpy(atom)
                atom_type = torch.from_numpy(atom_type)

            else:
                kpts = torch.from_numpy(ligand_coord.mean(0))
                xyz = torch.from_numpy(xyz)
                normal = torch.from_numpy(normal)
                label = torch.from_numpy(label)
                curvature = torch.from_numpy(curvature)
                atom = torch.from_numpy(atom)
                atom_type = torch.from_numpy(atom_type)

                _, idx, _ = knn_points(kpts.unsqueeze(0), xyz.unsqueeze(0), K=self.sample_num)
                idx = idx.squeeze()
                xyz = xyz[idx]
                normal = normal[idx]
                curvature = curvature[idx]

        elif self.sample_type == None:
            xyz = torch.from_numpy(xyz)
            normal = torch.from_numpy(normal)
            label = torch.from_numpy(label)
            curvature = torch.from_numpy(curvature)
            atom = torch.from_numpy(atom)
            atom_type = torch.from_numpy(atom_type)

        if self.rot_aug:
            rot = self.gen_rot()
            xyz += atom_center
            atom += atom_center
            ligand_coord += atom_center

            xyz = np.matmul(rot, xyz.T).T
            atom = np.matmul(rot, atom.T).T
            ligand_coord = np.matmul(rot, ligand_coord.T).T

            normal = np.matmul(rot, normal.T).T

            atom_center = atom.mean(0)
            xyz = xyz - atom_center
            atom = atom - atom_center
            ligand_coord = ligand_coord - atom_center


        dists, idx, _ = knn_points(xyz.unsqueeze(0), atom.unsqueeze(0), K=self.K)
        dists = dists.squeeze(0)
        idx = idx.squeeze(0)
        atom_type_sel = atom_type[idx]

        if self.over_sample and self.over_sample_num > self.sample_num:
            idx = np.random.choice(self.sample_num, self.over_sample_num, replace=True)
            xyz = xyz[idx]
            normal = normal[idx]
            curvature = curvature[idx]
            dists = dists[idx]
            atom_type_sel = atom_type_sel[idx]


        return xyz, normal, label, curvature, dists, atom_type_sel


    def gen_rot(self):
        R = torch.FloatTensor(Rotation.random().as_matrix())
        return R








if __name__ == "__main__":
    dataset = Protein_ligand(phase = 'train', sample_type = 'knn')
    dataloader = DataLoader(
        dataset,
        batch_size=4,
    )

    # for xyz, normal, label, curvature, dists, atom_type_sel in tqdm(dataset):
    #     print(type(xyz))
    #     break

    for xyz, normal, label, curvature, dists, atom_type_sel in tqdm(dataloader):
        print(type(xyz))


        # src_pcd = xyz.squeeze().numpy()
        # tgt_pcd = ligand_coord.squeeze().numpy()
        #
        # pcd0 = o3d.geometry.PointCloud()
        # pcd0.points = o3d.utility.Vector3dVector(src_pcd)
        # pcd0.paint_uniform_color([1, 0, 0])
        #
        # pcd1 = o3d.geometry.PointCloud()
        # pcd1.points = o3d.utility.Vector3dVector(tgt_pcd)
        # pcd1.paint_uniform_color([0, 0, 1])
        #
        # o3d.visualization.draw_geometries([pcd0, pcd1])
        # break