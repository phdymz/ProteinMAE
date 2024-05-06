import time
import torch
import torch.nn as nn
from datasets.protein_dataset import Protein_ligand
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pytorch3d.ops.knn import knn_points
from config.Arguments import parser
import  numpy as np
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from models.dmasif import *
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from torch.optim.lr_scheduler import MultiStepLR





def iterate(
    net,
    dataset,
    optimizer,
    args,
    ce_loss,
    test=False,
    save_path=None,
    pdb_ids=None,
    summary_writer=None,
    epoch_number=None,
):
    """Goes through one epoch of the dataset, returns information for Tensorboard."""

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


        total_processed_pairs += len(xyz)
        xyz = xyz.to(args.device)
        normal = normal.to(args.device)
        label = label.to(args.device)
        curvature = curvature.to(args.device)
        dist = dist.to(args.device)
        atom_type = atom_type.to(args.device)

        # label_one_hot = torch.zeros(len(label), 7).\
        #     scatter_(1, label.unsqueeze(-1).long(), 1).long().to(args.device)

        if not test:
            optimizer.zero_grad()

        outputs = net(xyz, normal, curvature, dist, atom_type)
        loss = ce_loss(outputs, label.long())
        pred = outputs.argmax(-1)

        # acc = (pred == label).sum() / float(label.size(0))
        balanced_acc = balanced_accuracy_score(label.cpu().numpy(), pred.cpu().numpy())

        # try:
        #     roc_auc = roc_auc_score(label.cpu().numpy(), F.softmax(outputs, -1).cpu().detach().numpy(), multi_class='ovr')
        # except Exception as e:
        #     print("Problem with computing roc-auc")
        #     print(e)
        #     roc_auc = None

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
        info.append(
                dict(
                    {
                        "Loss": loss.item(),
                        'Balanced-ACC': balanced_acc
                    }
                )
            )

     # Turn a list of dicts into a dict of lists:
    newdict = {}
    for k, v in [(key, d[key]) for d in info for key in d]:
        if k not in newdict:
            newdict[k] = [v]
        else:
            newdict[k].append(v)
    info = newdict

    # Final post-processing:
    return info








if __name__ == "__main__":
    args = parser.parse_args()
    writer = SummaryWriter("runs/{}".format(args.experiment_name))
    model_path = "models/" + args.experiment_name
    if not Path("models/").exists():
        Path("models/").mkdir(exist_ok=False)

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    net = dMaSIF(args)
    net = net.to(args.device)

    trainset = Protein_ligand(phase='train', rot_aug = True, sample_type = 'knn', sample_num = args.downsample_points, balance_sampling=args.balance_sampling)
    valset = Protein_ligand(phase='valid', rot_aug = False, sample_type = 'knn', sample_num = args.downsample_points)
    testset = Protein_ligand(phase='test', rot_aug = False, sample_type = 'knn', sample_num = args.downsample_points)
    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers = 6
    )
    val_loader = DataLoader(
        valset,
        batch_size=args.batch_size,
        num_workers=6
    )
    test_loader = DataLoader(
        testset,
        batch_size=args.batch_size,
        num_workers=6
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, amsgrad=True)
    # schedule = MultiStepLR(optimizer, milestones=[100], gamma=0.1)
    best_loss = 1e10
    starting_epoch = 0
    ce_loss = nn.CrossEntropyLoss()

    for i in range(starting_epoch, args.n_epochs):
        # schedule.step()
        # Train first, Test second:
        for dataset_type in ["Train", "Validation", "Test"]:
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
                epoch_number=i,
                ce_loss = ce_loss
            )

            # Write down the results using a TensorBoard writer:
            for key, val in info.items():
                if key in [
                    "Loss",
                    "ROC-AUC",
                    "ACC",
                    "Balanced-ACC",
                ]:
                    writer.add_scalar(f"{key}/{suffix}", np.mean(val), i)

                if "R_values/" in key:
                    val = np.array(val)
                    writer.add_scalar(f"{key}/{suffix}", np.mean(val[val > 0]), i)

            if dataset_type == "Validation":  # Store validation loss for saving the model
                val_loss = np.mean(info["Loss"])


        print("Validation loss {}, saving model".format(val_loss))
        torch.save(
            {
                "epoch": i,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
            },
            model_path + "_epoch{}".format(i),
        )

        best_loss = val_loss



















