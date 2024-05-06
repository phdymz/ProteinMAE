import torch
import torch.nn.functional as F
import numpy as np




def compute_loss_symmetry():
    pass








def compute_loss(descs1, descs2, xyz1, xyz2, label1, label2):

    loss = 0
    preds_concates = []
    labels_concates = []

    for i in range(len(label1)):
        idx1 = label1[i] > 0
        idx2 = label2[i] > 0

        pos_descs1 = descs1[i][idx1]
        pos_descs2 = descs2[i][idx2]

        pos_xyz1 = xyz1[i][idx1]
        pos_xyz2 = xyz2[i][idx2]

        pos_xyz_dists = (
            ((pos_xyz1[:, None, :] - pos_xyz2[None, :, :]) ** 2).sum(-1).sqrt()
        )
        pos_desc_dists = torch.matmul(pos_descs1, pos_descs2.T)

        pos_preds = pos_desc_dists[pos_xyz_dists < 1.0]
        pos_labels = torch.ones_like(pos_preds)

        n_desc_sample = 100

        sample_desc2 = torch.randperm(len(descs2[i]))[:n_desc_sample]
        sample_desc2 = descs2[i][sample_desc2]
        neg_preds = torch.matmul(pos_descs1, sample_desc2.T).view(-1)
        neg_labels = torch.zeros_like(neg_preds)

        #
        n_points_sample = len(pos_labels)
        pos_indices = torch.randperm(len(pos_labels))[:n_points_sample]
        neg_indices = torch.randperm(len(neg_labels))[:n_points_sample]

        pos_preds = pos_preds[pos_indices]
        pos_labels = pos_labels[pos_indices]
        neg_preds = neg_preds[neg_indices]
        neg_labels = neg_labels[neg_indices]

        preds_concat = torch.cat([pos_preds, neg_preds])
        labels_concat = torch.cat([pos_labels, neg_labels])

        loss += F.binary_cross_entropy_with_logits(preds_concat, labels_concat)
        preds_concates.append(preds_concat)
        labels_concates.append(labels_concat)

    return loss/len(label1), preds_concates, labels_concates