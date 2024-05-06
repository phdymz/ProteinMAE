import numpy as np
import torch
import torch.nn.functional as F
























def site_loss(preds, labels):

    loss = 0
    preds_concates = []
    labels_concates = []

    for i in range(len(preds)):
        pred = preds[i]
        label = labels[i]

        pos_preds = pred[label == 1]
        pos_labels = label[label == 1]
        neg_preds = pred[label == 0]
        neg_labels = label[label == 0]

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

    return loss/len(preds), preds_concates, labels_concates




