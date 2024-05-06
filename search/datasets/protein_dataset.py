import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pytorch3d.ops.knn import knn_points
from scipy.spatial.transform import Rotation
import open3d as o3d


class Protein_search(Dataset):
    def __init__(self, data_root = '/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/search_processed', K = 16,
                 phase = 'train',  rot_aug = True, sample_type = 'uniform', sample_num = 2048):
        # dataset parameters
        assert phase in ['train', 'test', 'val']
        assert sample_type in ['radius', 'knn', 'uniform']

        self.root = os.path.join(data_root, phase)
        self.phase = phase
        self.sample_points_num = sample_num
        self.sample_type = sample_type
        self.rot_aug = rot_aug
        self.factor = 1
        self.K = K
        for _, _, files in os.walk(self.root):
            break
        np.random.seed(0)
        np.random.shuffle(files)
        self.files = files


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_name = os.path.join(self.root, self.files[index])
        data = np.load(file_name)

        xyz1 = data['xyz1']
        normal1 = data['normal1']
        curvature1 = data['curvature1']

        atom1 = data['atom1']
        atom_type1 = data['atom_type1']

        xyz2 = data['xyz2']
        normal2 = data['normal2']
        curvature2 = data['curvature2']

        atom2 = data['atom2']
        atom_type2 = data['atom_type2']

        if self.sample_type == 'uniform':
            if len(xyz1) > self.sample_points_num:
                idx1 = np.random.choice(len(xyz1), self.sample_points_num, replace=False)
                xyz1 = xyz1[idx1]
                normal1 = normal1[idx1]
                curvature1 = curvature1[idx1]
            else:
                idx1 = np.random.choice(len(xyz1), self.sample_points_num, replace=True)
                xyz1 = xyz1[idx1]
                normal1 = normal1[idx1]
                curvature1 = curvature1[idx1]

            if len(xyz2) > self.sample_points_num:
                idx2 = np.random.choice(len(xyz2), self.sample_points_num, replace=False)
                xyz2 = xyz2[idx2]
                normal2 = normal2[idx2]
                curvature2 = curvature2[idx2]
            else:
                idx2 = np.random.choice(len(xyz2), self.sample_points_num, replace=True)
                xyz2 = xyz2[idx2]
                normal2 = normal2[idx2]
                curvature2 = curvature2[idx2]


        if self.rot_aug:
            rot = self.gen_rot()
        else:
            rot = torch.eye(3)

        xyz1 = np.matmul(rot, xyz1.T).T
        atom1 = np.matmul(rot, atom1.T).T
        normal1 = np.matmul(rot, normal1.T).T

        xyz2 = np.matmul(rot, xyz2.T).T
        atom2 = np.matmul(rot, atom2.T).T
        normal2 = np.matmul(rot, normal2.T).T

        atom_center1 = atom1.mean(0)
        xyz1 = xyz1 - atom_center1
        atom1 = atom1 - atom_center1

        atom_center2 = atom2.mean(0)
        xyz2 = xyz2 - atom_center2
        atom2 = atom2 - atom_center2



        dists1, idx1, _ = knn_points(xyz1.unsqueeze(0), atom1.unsqueeze(0), K=self.K)
        dists1 = dists1.squeeze(0)
        idx1 = idx1.squeeze(0)
        atom_type_sel1 = atom_type1[idx1]

        dists2, idx2, _ = knn_points(xyz2.unsqueeze(0), atom2.unsqueeze(0), K=self.K)
        dists2 = dists2.squeeze(0)
        idx2 = idx2.squeeze(0)
        atom_type_sel2 = atom_type2[idx2]

        #calculate label
        dists_xyz12, _, _ = knn_points(xyz1.unsqueeze(0), xyz2.unsqueeze(0), K=1)
        dists_xyz21, _, _ = knn_points(xyz2.unsqueeze(0), xyz1.unsqueeze(0), K=1)

        label1 = (dists_xyz12.squeeze() < 1.0) + 0.0
        label2 = (dists_xyz21.squeeze() < 1.0) + 0.0
        # label1 = 0.0
        # label2 = 0.0

        return xyz1, normal1, label1, curvature1, dists1, atom_type_sel1, xyz2, normal2, label2, curvature2, dists2, atom_type_sel2


    def gen_rot(self):
        R = torch.FloatTensor(Rotation.random().as_matrix())
        return R





if __name__ == "__main__":
    dataset = Protein_search(phase = 'train', sample_type = 'uniform')
    dataloader = DataLoader(
        dataset,
        batch_size=4,
    )

    # for xyz, normal, label, curvature, dists, atom_type_sel in tqdm(dataset):
    #     print(type(xyz))
    #     break

    for xyz1, normal1, label1, curvature1, dists1, atom_type_sel1, xyz2, normal2, label2, curvature2, dists2, atom_type_sel2 in tqdm(dataloader):
        print(type(xyz1))
        print(xyz1.shape)
        print(normal1.shape)
        print(label1.shape)
        print(curvature1.shape)
        print(dists1.shape)
        print(atom_type_sel1.shape)
        print()


        # src_pcd = xyz.squeeze().numpy()
        # tgt_pcd = ligand_coord.squeeze().numpy()
        #
        # pcd0 = o3d.geometry.PointCloud()
        # pcd0.points = o3d.utility.Vector3dVector(xyz1)
        # pcd0.paint_uniform_color([1, 0, 0])
        #
        # pcd1 = o3d.geometry.PointCloud()
        # pcd1.points = o3d.utility.Vector3dVector(xyz2)
        # pcd1.paint_uniform_color([0, 0, 1])
        #
        # o3d.visualization.draw_geometries([pcd0, pcd1])
        # break