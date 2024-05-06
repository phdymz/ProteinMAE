import time
import torch
import torch.nn as nn
from datasets.protein_dataset import Protein
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pytorch3d.ops.knn import knn_points
from config.Arguments import parser
import  numpy as np
from utils.geometry_processing import tangent_vectors
import torch.nn.functional as F
from math import pi, sqrt
from utils.loss import site_loss
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from models.dmasif import *
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import MultiStepLR
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, recall_score, f1_score


def iterate(
    net,
    dataset,
    optimizer,
    args,
    test=False,
    save_path=None,
    pdb_ids=None,
    summary_writer=None,
    epoch_number=None,
):
    """Goes through one epoch of the dataset, returns information for Tensorboard."""

    time_infer = []

    if test:
        net.eval()
        torch.set_grad_enabled(False)
    else:
        net.train()
        torch.set_grad_enabled(True)

    # Statistics and fancy graphs to summarize the epoch:
    info = []
    total_processed_pairs = 0
    # Loop over one epoch:
    for it, (xyz, normal, label, curvature, dist, atom_type) in enumerate(
        tqdm(dataset)
    ):  # , desc="Test " if test else "Train")):

        # try:
        total_processed_pairs += len(xyz)
        xyz = xyz.to(args.device)
        normal = normal.to(args.device)
        label = label.to(args.device)
        curvature = curvature.to(args.device)
        dist = dist.to(args.device)
        atom_type = atom_type.to(args.device)


        if not test:
            optimizer.zero_grad()

        time_new = time.time()
        outputs = net(xyz, normal, curvature, dist, atom_type)
        time_infer.append(time.time() - time_new)

        loss,sampled_preds, sampled_labels = site_loss(outputs, label)

        # Compute the gradient, update the model weights:
        if not test:
            loss.backward()
            do_step = True
            for param in net.parameters():
                if param.grad is not None:
                    if (1 - torch.isfinite(param.grad).long()).sum() > 0:
                        do_step = False
                        break
            if do_step is True:
                optimizer.step()
            # loss.backward()
            # optimizer.step()

        try:
            if sampled_labels is not None:
                roc_auc = []
                for index, item in enumerate(sampled_labels):
                    roc_auc.append(
                        roc_auc_score(
                        np.rint(item.detach().cpu().view(-1).numpy()),
                        sampled_preds[index].detach().cpu().view(-1).numpy()
                        )
                    )


                roc_auc = np.mean(roc_auc)
            else:
                roc_auc = 0.0
        except Exception as e:
            print("Problem with computing roc-auc")
            print(e)
            continue

        info.append(
            dict(
                {
                    "Loss": loss.item(),
                    "ROC-AUC": roc_auc,
                }
            )
        )
        # except Exception as e:
        #     print("Problem with cuda")
        #     print(e)
        #     continue
     # Turn a list of dicts into a dict of lists:
    newdict = {}
    for k, v in [(key, d[key]) for d in info for key in d]:
        if k not in newdict:
            newdict[k] = [v]
        else:
            newdict[k].append(v)
    info = newdict



    return info








if __name__ == "__main__":
    args = parser.parse_args()
    writer = SummaryWriter("runs/{}".format(args.experiment_name))
    model_path = "models/" + args.experiment_name
    if not Path("models/").exists():
        Path("models/").mkdir(exist_ok=False)

    torch.backends.cudnn.deterministic = True
    seed = 25

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    net = dMaSIF(args)
    net = net.to(args.device)

    if args.checkpoint:
        weight = torch.load(args.checkpoint)['model_state_dict']
        net.load_state_dict(weight)

    trainset = Protein(phase='train', rot_aug = True, sample_type = 'uniform', sample_num = args.downsample_points)
    valset = Protein(phase='val', rot_aug = False, sample_type = 'uniform', sample_num = args.downsample_points)
    testset = Protein(phase='test', rot_aug = False, sample_type = 'uniform', sample_num = args.downsample_points)
    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        valset,
        batch_size=args.batch_size,
    )
    test_loader = DataLoader(
        testset,
        batch_size=args.batch_size,
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=2e-4, amsgrad=True)
    #schedule = MultiStepLR(optimizer, milestones=[28], gamma=0.1)
    best_loss = 1e10
    starting_epoch = 0


    for dataset_type in ["Test"]:

        if dataset_type == "Train":
            test = False
        else:
            test = True

        suffix = dataset_type
        if dataset_type == "Train":
            dataloader = train_loader
        elif dataset_type == "Validation":
            dataloader = val_loader
        elif dataset_type == "Test":
            dataloader = test_loader

        # Perform one pass through the data:
        info = iterate(
            net,
            dataloader,
            optimizer,
            args,
            test=test,
            summary_writer=writer,
            epoch_number=0,
        )


        print(np.mean(info["ROC-AUC"]))





















