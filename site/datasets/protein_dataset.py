import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pytorch3d.ops.knn import knn_points
from scipy.spatial.transform import Rotation



# def collate_fn_protein(list_data):
#     batched_xyz = []
#     batched_normal = []
#     batched_label = []
#     batched_curvature = []
#     batched_atom = []
#     batched_atom_type = []
#     atom_batch = []
#
#     for ind, (xyz, normal, label, curvature, atom, atom_type) in enumerate(list_data):
#         batched_xyz.append(torch.from_numpy(xyz).unsqueeze(0))
#         batched_normal.append(torch.from_numpy(normal).unsqueeze(0))
#         batched_label.append(torch.from_numpy(label).unsqueeze(0))
#         batched_curvature.append(torch.from_numpy(curvature).unsqueeze(0))
#         batched_atom.append(torch.from_numpy(atom))
#         batched_atom_type.append(torch.from_numpy(atom_type))
#         atom_batch.append(ind * torch.ones(len(atom)))
#
#     batched_xyz = torch.cat(batched_xyz, dim = 0)
#     batched_normal = torch.cat(batched_normal, dim=0)
#     batched_label = torch.cat(batched_label, dim=0)
#     batched_curvature = torch.cat(batched_curvature, dim=0)
#     batched_atom = torch.cat(batched_atom, dim=0)
#     batched_atom_type = torch.cat(batched_atom_type, dim=0)
#     atom_batch = torch.cat(atom_batch, dim=0)
#
#     return batched_xyz, batched_normal, batched_label, batched_curvature, \
#            batched_atom, batched_atom_type, atom_batch


class Protein(Dataset):
    def __init__(self, data_root = '/media/ymz/11c16692-3687-420b-9787-72a7f9e3bdaf/Protein/site_processed', K = 16,
                 phase = 'train',  rot_aug = True, sample_type = 'uniform', sample_num = 2048, pretrain = False,
                 sample_rate = None):
        # dataset parameters
        assert phase in ['train', 'test', 'val']
        assert sample_type in ['uniform', 'knn', None]

        self.root = os.path.join(data_root, phase)
        self.phase = phase
        self.sample_num = sample_num
        self.sample_type = sample_type
        self.rot_aug = rot_aug
        self.factor = 1
        self.pretrain = pretrain
        self.K = K

        for _, _, files in os.walk(self.root):
            break

        if phase == 'train':
            if sample_rate is not None:
                idx = np.loadtxt('./config/site_{}.txt'.format(sample_rate))
                self.files = []
                for i in idx:
                    self.files.append(files[int(i)])
                files = self.files

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

        if not self.pretrain:
            label = data['label']
        else:
            label = np.zeros([len(xyz),1]).astype('float32')

        if self.sample_type == 'uniform':
            if len(xyz) > self.sample_num:
                idx = np.random.choice(len(xyz), self.sample_num, replace=False)
                xyz = xyz[idx]
                normal = normal[idx]
                label = label[idx]
                curvature = curvature[idx]
            else:
                idx = np.random.choice(len(xyz), self.sample_num, replace=True)
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


        elif self.sample_type == 'knn':
            if len(xyz) <= self.sample_num:
                idx = np.random.choice(len(xyz), self.sample_num, replace=True)
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
            else:
                # idx_kpts = np.random.choice(len(xyz), 1)
                # kpts = torch.from_numpy(xyz[idx_kpts])
                kpts = torch.from_numpy(xyz[(label > 0).squeeze()][np.random.choice(int(label.sum()), 1)])

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
                label = label[idx]
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

            xyz = np.matmul(rot, xyz.T).T
            atom = np.matmul(rot, atom.T).T

            normal = np.matmul(rot, normal.T).T

            atom_center = atom.mean(0)
            xyz = xyz - atom_center
            atom = atom - atom_center


        dists, idx, _ = knn_points(xyz.unsqueeze(0), atom.unsqueeze(0), K=self.K)
        dists = dists.squeeze(0)
        idx = idx.squeeze(0)
        atom_type_sel = atom_type[idx]


        return xyz, normal, label, curvature, dists, atom_type_sel


    def gen_rot(self):
        R = torch.FloatTensor(Rotation.random().as_matrix())
        return R



class ProteinPair(Dataset):
    def __init__(self, ):
        pass

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        pass




if __name__ == "__main__":
    dataset = Protein(phase = 'train', sample_type = 'uniform')
    dataloader = DataLoader(
        dataset,
        batch_size=4,
    )

    # for xyz, normal, label, curvature, dists, atom_type_sel in tqdm(dataset):
    #     print(type(xyz))
    #     break

    for xyz, normal, label, curvature, dists, atom_type_sel in tqdm(dataloader):
        print(type(xyz))
        break


